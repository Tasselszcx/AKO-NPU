#!/bin/bash
# Iteration 2: Matmul Tiling Parameter Sweep
# Tests multiple tiling configurations and records Task Duration for each.
set -eo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
SOLUTION_DIR="$PROJECT_ROOT/solution"
ASC_FILE="$SOLUTION_DIR/dsa_indexer.asc"
BUILD_DIR="$SOLUTION_DIR/build"
RESULTS_FILE="$PROJECT_ROOT/iter2_sweep_results.csv"

B=1
S_q=1
S_kv=4096

# Backup original
cp "$ASC_FILE" "$ASC_FILE.bak"

# Generate test data if needed
cd "$BUILD_DIR"
python3 "$PROJECT_ROOT/scripts/gen_data.py" $B $S_q $S_kv 2>/dev/null

echo "config_id,tiling_api,kernel_cfg,baseM,baseN,mdl_tiling,depthA1,stepKa,task_duration_us,status" > "$RESULTS_FILE"

# Function: apply a tiling config, compile, run, measure
run_config() {
    local CONFIG_ID="$1"
    local TILING_API="$2"        # MatmulApiTiling or MultiCoreMatmulTiling
    local KERNEL_CFG="$3"        # CFG_NORM or CFG_MDL
    local BASE_M="$4"
    local BASE_N="$5"
    local MDL_TILING="$6"        # 0=auto, 1=use SetMatmulConfigParams for MDL
    local DEPTH_A1="$7"          # 0=auto (don't set)
    local STEP_KA="$8"           # 0=auto (don't set)

    echo ""
    echo "=== Config $CONFIG_ID: api=$TILING_API cfg=$KERNEL_CFG baseM=$BASE_M baseN=$BASE_N mdl=$MDL_TILING depthA1=$DEPTH_A1 stepKa=$STEP_KA ==="

    # Restore original
    cp "$ASC_FILE.bak" "$ASC_FILE"

    # --- Patch kernel-side template ---
    if [ "$KERNEL_CFG" = "CFG_MDL" ]; then
        sed -i 's/AscendC::Matmul<AType, BType, CType, BiasType, CFG_NORM>/AscendC::Matmul<AType, BType, CType, BiasType, CFG_MDL>/' "$ASC_FILE"
    fi

    # --- Build the new ComputeTiling body ---
    local TILING_CODE=""
    if [ "$TILING_API" = "MatmulApiTiling" ]; then
        TILING_CODE="        matmul_tiling::MatmulApiTiling tilingApi(*ascendcPlatform);"
    else
        TILING_CODE="        matmul_tiling::MultiCoreMatmulTiling tilingApi(*ascendcPlatform);
        tilingApi.SetDim(1);"
    fi

    TILING_CODE="$TILING_CODE
        tilingApi.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, false);
        tilingApi.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16, true);
        tilingApi.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        tilingApi.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        tilingApi.SetOrgShape(N_HEADS, S_kv, HEAD_DIM);
        tilingApi.SetShape(N_HEADS, S_kv, HEAD_DIM);"

    if [ "$BASE_M" != "0" ] && [ "$BASE_N" != "0" ]; then
        TILING_CODE="$TILING_CODE
        tilingApi.SetFixSplit($BASE_M, $BASE_N, -1);"
    fi

    TILING_CODE="$TILING_CODE
        tilingApi.EnableBias(false);
        tilingApi.SetBufferSpace(-1, -1, -1);"

    if [ "$MDL_TILING" = "1" ]; then
        TILING_CODE="$TILING_CODE
        tilingApi.SetMatmulConfigParams(1);"
    fi

    TILING_CODE="$TILING_CODE

        optiling::TCubeTiling cubeTiling;
        int64_t res = tilingApi.GetTiling(cubeTiling);
        if (res == -1) std::cerr << \"Matmul tiling failed!\" << std::endl;"

    # Post-tiling L1 cache params
    if [ "$DEPTH_A1" != "0" ]; then
        TILING_CODE="$TILING_CODE
        cubeTiling.set_depthA1($DEPTH_A1);"
    fi
    if [ "$STEP_KA" != "0" ]; then
        TILING_CODE="$TILING_CODE
        cubeTiling.set_stepKa($STEP_KA);"
    fi

    TILING_CODE="$TILING_CODE
        // Tiling generated successfully
        cubeTiling.SaveToBuffer(tilingBuf + CUSTOM_TILING_ALIGNED, cubeTiling.GetDataSize());"

    # Use Python to do the replacement (more reliable than sed for multiline)
    python3 -c "
import re, sys

with open('$ASC_FILE', 'r') as f:
    content = f.read()

# Find the matmul tiling block
pattern = r'(    // Matmul tiling\n    \{)\n(.*?)(    \})'
match = re.search(pattern, content, re.DOTALL)
if not match:
    print('ERROR: Could not find Matmul tiling block', file=sys.stderr)
    sys.exit(1)

new_body = '''    // Matmul tiling
    {
$TILING_CODE
    }'''

content = content[:match.start()] + new_body + content[match.end():]

with open('$ASC_FILE', 'w') as f:
    f.write(content)
"

    if [ $? -ne 0 ]; then
        echo "$CONFIG_ID,$TILING_API,$KERNEL_CFG,$BASE_M,$BASE_N,$MDL_TILING,$DEPTH_A1,$STEP_KA,N/A,patch_failed" >> "$RESULTS_FILE"
        return
    fi

    # Compile
    cd "$BUILD_DIR"
    if ! make -j4 2>&1 | tail -3 | grep -q "Built target"; then
        # Try harder
        cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
        if ! make -j4 2>&1 | tail -3 | grep -q "Built target"; then
            echo "$CONFIG_ID,$TILING_API,$KERNEL_CFG,$BASE_M,$BASE_N,$MDL_TILING,$DEPTH_A1,$STEP_KA,N/A,compile_failed" >> "$RESULTS_FILE"
            return
        fi
    fi

    # Quick correctness check
    cd "$BUILD_DIR"
    if ! ./dsa_indexer $B $S_q $S_kv > /dev/null 2>&1; then
        echo "$CONFIG_ID,$TILING_API,$KERNEL_CFG,$BASE_M,$BASE_N,$MDL_TILING,$DEPTH_A1,$STEP_KA,N/A,run_failed" >> "$RESULTS_FILE"
        return
    fi

    # Correctness verification
    cd "$BUILD_DIR"
    VERIFY_OUT=$(python3 "$PROJECT_ROOT/scripts/verify_result.py" $B $S_q $S_kv 2>&1 || true)
    if echo "$VERIFY_OUT" | grep -qi "fail\|error\|mismatch"; then
        echo "  Correctness FAILED: $VERIFY_OUT"
        echo "$CONFIG_ID,$TILING_API,$KERNEL_CFG,$BASE_M,$BASE_N,$MDL_TILING,$DEPTH_A1,$STEP_KA,N/A,correctness_failed" >> "$RESULTS_FILE"
        return
    fi

    # Profile with msprof
    cd "$BUILD_DIR"
    MSPROF_DIR="$BUILD_DIR/msprof_sweep_${CONFIG_ID}"
    rm -rf "$MSPROF_DIR"
    mkdir -p "$MSPROF_DIR"
    chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=5 --launch-count=3 --output="$MSPROF_DIR" ./dsa_indexer $B $S_q $S_kv > /dev/null 2>&1 || true

    OPPROF_DIR=$(ls -td ${MSPROF_DIR}/OPPROF_* 2>/dev/null | head -1)
    if [ -z "$OPPROF_DIR" ] || [ ! -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        echo "$CONFIG_ID,$TILING_API,$KERNEL_CFG,$BASE_M,$BASE_N,$MDL_TILING,$DEPTH_A1,$STEP_KA,N/A,profile_failed" >> "$RESULTS_FILE"
        return
    fi

    # Extract Task Duration
    DURATION=$(python3 -c "
import csv, statistics
durations = []
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td_key = [k for k in row.keys() if 'Task Duration' in k]
        if td_key:
            val = row[td_key[0]].strip()
            if val:
                durations.append(float(val))
if durations:
    print(f'{statistics.mean(durations):.2f}')
else:
    print('N/A')
" 2>/dev/null)

    echo "  Task Duration: ${DURATION} us"
    echo "$CONFIG_ID,$TILING_API,$KERNEL_CFG,$BASE_M,$BASE_N,$MDL_TILING,$DEPTH_A1,$STEP_KA,$DURATION,ok" >> "$RESULTS_FILE"

    # Cleanup msprof
    rm -rf "$MSPROF_DIR"
}

echo "=============================="
echo "Iteration 2: Tiling Sweep"
echo "=============================="
echo "Dimensions: M=64 (N_HEADS), N=4096 (S_kv), K=128 (HEAD_DIM)"
echo "bf16 x bf16 -> float32, B transposed"
echo ""

# --- Configuration Grid ---
# Constraints:
# - baseM <= ceil(64/16)*16 = 64
# - baseN: for float output C0_size=8, must be multiple of 8
# - baseM * baseN * sizeof(float) <= L0C_size (128KB) => baseM * baseN <= 32768
# - baseK: only -1 supported (auto)
# So: baseM=64 => baseN <= 512
#     baseM=32 => baseN <= 1024
#     baseM=16 => baseN <= 2048

# Config format: run_config ID TILING_API KERNEL_CFG BASE_M BASE_N MDL_TILING DEPTH_A1 STEP_KA

# Group 1: Baseline-like (MultiCoreMatmulTiling + CFG_NORM, no SetFixSplit)
run_config 01 MultiCoreMatmulTiling CFG_NORM 0 0 0 0 0

# Group 2: MatmulApiTiling + CFG_NORM (single-core API, auto tiling)
run_config 02 MatmulApiTiling CFG_NORM 0 0 0 0 0

# Group 3: Different base block sizes with CFG_NORM
run_config 03 MatmulApiTiling CFG_NORM 64 256 0 0 0
run_config 04 MatmulApiTiling CFG_NORM 64 512 0 0 0
run_config 05 MatmulApiTiling CFG_NORM 32 256 0 0 0
run_config 06 MatmulApiTiling CFG_NORM 32 512 0 0 0
run_config 07 MatmulApiTiling CFG_NORM 32 1024 0 0 0
run_config 08 MatmulApiTiling CFG_NORM 16 256 0 0 0
run_config 09 MatmulApiTiling CFG_NORM 16 512 0 0 0
run_config 10 MatmulApiTiling CFG_NORM 16 1024 0 0 0
run_config 11 MatmulApiTiling CFG_NORM 16 2048 0 0 0

# Group 4: MDL mode - different base blocks
run_config 12 MatmulApiTiling CFG_MDL 0 0 1 0 0
run_config 13 MatmulApiTiling CFG_MDL 64 256 1 0 0
run_config 14 MatmulApiTiling CFG_MDL 64 512 1 0 0
run_config 15 MatmulApiTiling CFG_MDL 32 256 1 0 0
run_config 16 MatmulApiTiling CFG_MDL 32 512 1 0 0
run_config 17 MatmulApiTiling CFG_MDL 32 1024 1 0 0
run_config 18 MatmulApiTiling CFG_MDL 16 256 1 0 0
run_config 19 MatmulApiTiling CFG_MDL 16 512 1 0 0
run_config 20 MatmulApiTiling CFG_MDL 16 1024 1 0 0
run_config 21 MatmulApiTiling CFG_MDL 16 2048 1 0 0

# Group 5: MDL + L1 cache optimization (depthA1, stepKa)
# For M=64, K=128, baseM=64, baseK=auto(~64), K is small -> try full-load
# depthA1 should satisfy depthA1/(stepM*stepKa)=2 for double buffer
run_config 22 MatmulApiTiling CFG_MDL 64 256 1 4 2
run_config 23 MatmulApiTiling CFG_MDL 64 512 1 4 2
run_config 24 MatmulApiTiling CFG_MDL 32 512 1 4 2
run_config 25 MatmulApiTiling CFG_MDL 32 1024 1 4 2

# Group 6: MultiCoreMatmulTiling + CFG_MDL
run_config 26 MultiCoreMatmulTiling CFG_MDL 0 0 1 0 0
run_config 27 MultiCoreMatmulTiling CFG_NORM 64 256 0 0 0
run_config 28 MultiCoreMatmulTiling CFG_NORM 64 512 0 0 0
run_config 29 MultiCoreMatmulTiling CFG_NORM 32 512 0 0 0
run_config 30 MultiCoreMatmulTiling CFG_NORM 32 1024 0 0 0

# Restore original
cp "$ASC_FILE.bak" "$ASC_FILE"

echo ""
echo "=============================="
echo "Sweep Complete. Results:"
echo "=============================="
cat "$RESULTS_FILE"
echo ""

# Find the best config
BEST=$(python3 -c "
import csv
best_id = None
best_dur = float('inf')
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['status'] == 'ok' and row['task_duration_us'] != 'N/A':
            dur = float(row['task_duration_us'])
            if dur < best_dur:
                best_dur = dur
                best_id = row['config_id']
                best_row = dict(row)
if best_id:
    print(f'Best config: {best_id} with Task Duration = {best_dur} us')
    for k, v in best_row.items():
        print(f'  {k}: {v}')
else:
    print('No successful config found!')
")
echo "$BEST"
