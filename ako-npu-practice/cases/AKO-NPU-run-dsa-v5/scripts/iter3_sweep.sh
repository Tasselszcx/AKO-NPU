#!/bin/bash
# Iter 3: Matmul tiling config sweep
# Tests various tiling configurations and records results
set -eo pipefail

PROJECT_ROOT="/home/hadoop-scale-llm/kernel/autoresearch/AKO-NPU-run-dsa-v5"
BUILD_DIR="${PROJECT_ROOT}/solution/build"
B=1; S_q=1; S_kv=4096
RESULTS_CSV="${PROJECT_ROOT}/iter3_sweep_results.csv"
ASC_FILE="${PROJECT_ROOT}/solution/dsa_indexer.asc"

echo "config_id,baseM,baseN,baseK,traverse,l1_manual,depthA1,stepKa,description,compile_ok,correct,runtime_us" > "$RESULTS_CSV"

# Generate test data once (from solution/build dir)
cd "${BUILD_DIR}"
python3 "${PROJECT_ROOT}/scripts/gen_data.py" $B $S_q $S_kv 2>&1 | tail -1

CONFIG_ID=0

run_config() {
    local baseM=$1 baseN=$2 baseK=$3 traverse=$4 l1_manual=$5 depthA1=$6 stepKa=$7 desc=$8
    CONFIG_ID=$((CONFIG_ID + 1))
    echo ""
    echo "=== Config $CONFIG_ID: $desc (baseM=$baseM baseN=$baseN baseK=$baseK) ==="

    # Restore original file first
    cd "$PROJECT_ROOT"
    git checkout -- "$ASC_FILE" 2>/dev/null || cp "${ASC_FILE}.iter2_backup" "$ASC_FILE"

    # Apply modifications via Python
    python3 - "$baseM" "$baseN" "$baseK" "$traverse" "$l1_manual" "$depthA1" "$stepKa" << 'PYEOF'
import sys

baseM = int(sys.argv[1])
baseN = int(sys.argv[2])
baseK = int(sys.argv[3])
traverse = sys.argv[4]
l1_manual = sys.argv[5]
depthA1 = int(sys.argv[6])
stepKa = int(sys.argv[7])

with open("solution/dsa_indexer.asc", "r") as f:
    code = f.read()

# Replace SetFixSplit
code = code.replace(
    "tilingApi.SetFixSplit(64, 256, -1);",
    f"tilingApi.SetFixSplit({baseM}, {baseN}, {baseK});"
)

# Add traverse setting if requested
if traverse == "FIRSTN":
    code = code.replace(
        f"tilingApi.SetFixSplit({baseM}, {baseN}, {baseK});",
        f"tilingApi.SetFixSplit({baseM}, {baseN}, {baseK});\n        tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);"
    )

# Handle L1 manual params
if l1_manual == "1":
    code = code.replace(
        'if (res == -1) std::cerr << "Matmul tiling failed!" << std::endl;',
        f'if (res == -1) std::cerr << "Matmul tiling failed!" << std::endl;\n        cubeTiling.set_depthA1({depthA1});\n        cubeTiling.set_stepKa({stepKa});'
    )

with open("solution/dsa_indexer.asc", "w") as f:
    f.write(code)
PYEOF

    # Build
    cd "$BUILD_DIR"
    if ! make -j4 2>&1 | tail -5 | tee /tmp/build_out.txt | grep -q "Built target"; then
        if grep -qi "error" /tmp/build_out.txt; then
            echo "$CONFIG_ID,$baseM,$baseN,$baseK,$traverse,$l1_manual,$depthA1,$stepKa,$desc,FAIL,N/A,N/A" >> "$RESULTS_CSV"
            echo "  COMPILE FAILED"
            cd "$PROJECT_ROOT"
            return
        fi
    fi

    # Run
    if ! ./dsa_indexer $B $S_q $S_kv 2>&1 > /tmp/run_out.txt; then
        echo "$CONFIG_ID,$baseM,$baseN,$baseK,$traverse,$l1_manual,$depthA1,$stepKa,$desc,OK,CRASH,N/A" >> "$RESULTS_CSV"
        echo "  RUNTIME CRASH"
        cd "$PROJECT_ROOT"
        return
    fi

    # Verify correctness
    VERIFY_OUT=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" $B $S_q $S_kv 2>&1) || true
    if echo "$VERIFY_OUT" | grep -q "Verification PASSED"; then
        CORRECT="PASS"
    else
        echo "$VERIFY_OUT" | tail -3
        echo "$CONFIG_ID,$baseM,$baseN,$baseK,$traverse,$l1_manual,$depthA1,$stepKa,$desc,OK,FAIL,N/A" >> "$RESULTS_CSV"
        echo "  CORRECTNESS FAILED"
        cd "$PROJECT_ROOT"
        return
    fi

    # Profile
    MSPROF_DIR="${PROJECT_ROOT}/msprof_sweep"
    rm -rf "$MSPROF_DIR"
    mkdir -p "$MSPROF_DIR"
    chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=5 --launch-count=3 --output="$MSPROF_DIR" ./dsa_indexer $B $S_q $S_kv 2>&1 > /dev/null || true

    OPPROF_DIR=$(ls -td ${MSPROF_DIR}/OPPROF_* 2>/dev/null | head -1)
    RUNTIME="N/A"
    if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        RUNTIME=$(python3 -c "
import csv, statistics
durations = []
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td_key = [k for k in row.keys() if 'Task Duration' in k]
        if td_key:
            val = row[td_key[0]].strip()
            if val: durations.append(float(val))
if durations: print(f'{statistics.mean(durations):.2f}')
else: print('N/A')
" 2>/dev/null || echo "N/A")
    fi

    echo "$CONFIG_ID,$baseM,$baseN,$baseK,$traverse,$l1_manual,$depthA1,$stepKa,$desc,OK,$CORRECT,$RUNTIME" >> "$RESULTS_CSV"
    echo "  Result: $CORRECT, Runtime: $RUNTIME us"

    rm -rf "$MSPROF_DIR"
    cd "$PROJECT_ROOT"
}

# ========== Sweep configurations ==========

# Group 1: Different baseN values (baseM=64, baseK=auto, MDL mode)
run_config 64 128 -1 default 0 0 0 "baseN=128"
run_config 64 256 -1 default 0 0 0 "baseN=256_verify"
run_config 64 512 -1 default 0 0 0 "baseN=512"

# Group 2: Explicit baseK=64 (2 K-iterations)
run_config 64 256 64 default 0 0 0 "baseN256_baseK64"
run_config 64 512 64 default 0 0 0 "baseN512_baseK64"

# Group 3: baseK=128 (K full-load, matches HEAD_DIM=128)
run_config 64 256 128 default 0 0 0 "baseN256_baseK128"
run_config 64 512 128 default 0 0 0 "baseN512_baseK128"

# Group 4: SetTraverse FIRSTN (N=4096 is much larger than M=64)
run_config 64 256 -1 FIRSTN 0 0 0 "baseN256_FIRSTN"
run_config 64 256 128 FIRSTN 0 0 0 "baseN256_K128_FIRSTN"

# Group 5: L1 cache manual tuning
# A block: 64*64*2=8KB (baseM=64, baseK=64, bf16)
# depthA1=4: 4*8KB=32KB for A in L1
# L1=512KB, so plenty of room for B
run_config 64 256 -1 default 1 4 2 "L1_dA4_sKa2"
run_config 64 256 -1 default 1 8 2 "L1_dA8_sKa2"
run_config 64 256 128 default 1 4 2 "K128_L1_dA4_sKa2"

echo ""
echo "=== Sweep Complete ==="
cat "$RESULTS_CSV"
