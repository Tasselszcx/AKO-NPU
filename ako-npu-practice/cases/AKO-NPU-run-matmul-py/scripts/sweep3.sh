#!/bin/bash
# Sweep 3: Fine-grained UB search + L0C + combined params + socVersion + EnableBias + SetShape
set -eo pipefail
cd "$(dirname "$0")"/..

SOLUTION_DIR="$(pwd)/solution"
OP_NAME="matmul_leakyrelu_custom"
RESULTS_FILE="sweep_results3.csv"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
elif [ -f "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" ]; then
    source "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" 2>/dev/null || true
fi
ASCEND_BASE="$(dirname "${ASCEND_HOME_PATH}")"
ASC_DIR=$(find "${ASCEND_HOME_PATH}" "${ASCEND_BASE}" -name "ASCConfig.cmake" -print -quit 2>/dev/null | xargs dirname 2>/dev/null)

echo "iter,config,correct,runtime_us,speedup" > "$RESULTS_FILE"
BASELINE=228.37
ITER=20

cd "$SOLUTION_DIR"
cp matmul_leakyrelu.asc matmul_leakyrelu.asc.orig3

run_test() {
    local desc="$1"
    ITER=$((ITER + 1))
    echo "=== Iter $ITER: $desc ==="
    cd "$SOLUTION_DIR/build"
    rm -rf CMakeCache.txt CMakeFiles 2>/dev/null
    if ! cmake -DASC_DIR="${ASC_DIR}" .. > /dev/null 2>&1 || ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$desc,compile_fail,," >> "$SOLUTION_DIR/../$RESULTS_FILE"
        return 1
    fi
    python3 ../gen_data.py > /dev/null 2>&1
    rm -f output/output.bin
    ./${OP_NAME} > /dev/null 2>&1 || true
    if [ ! -f output/output.bin ]; then
        echo "$ITER,$desc,runtime_fail,," >> "$SOLUTION_DIR/../$RESULTS_FILE"
        return 1
    fi
    local correct=$(python3 ../verify_result.py output/output.bin output/golden.bin 2>&1 | grep "Overall:" | awk '{print $2}')
    if [ "$correct" != "PASS" ]; then
        echo "$ITER,$desc,incorrect,," >> "$SOLUTION_DIR/../$RESULTS_FILE"
        return 1
    fi
    chmod 755 . 2>/dev/null || true
    rm -rf msprof_output
    local td=$(msprof op --warm-up=10 --output=./msprof_output ./${OP_NAME} 2>&1 | grep "Task Duration" | grep -o '[0-9.]*')
    if [ -n "$td" ]; then
        local sp=$(python3 -c "print(f'{$BASELINE/$td:.3f}')")
        echo "$ITER,$desc,true,$td,$sp"
        echo "$ITER,$desc,true,$td,$sp" >> "$SOLUTION_DIR/../$RESULTS_FILE"
    else
        echo "$ITER,$desc,profile_fail,," >> "$SOLUTION_DIR/../$RESULTS_FILE"
    fi
    return 0
}

restore() {
    cd "$SOLUTION_DIR"
    cp matmul_leakyrelu.asc.orig3 matmul_leakyrelu.asc
}

# === Fine UB search around 96KB (best) ===
for ub_kb in 80 84 88 92 96 100 104 108 112; do
    restore
    ub_bytes=$((ub_kb * 1024))
    sed -i "s/tilingApi.SetBufferSpace(-1, -1, 98304);/tilingApi.SetBufferSpace(-1, -1, $ub_bytes);/" matmul_leakyrelu.asc
    run_test "UB=${ub_kb}KB"
done

# === Combined L1+UB ===
for l1_kb in 256 512; do
    for ub_kb in 88 96 104; do
        restore
        l1_bytes=$((l1_kb * 1024))
        ub_bytes=$((ub_kb * 1024))
        sed -i "s/tilingApi.SetBufferSpace(-1, -1, 98304);/tilingApi.SetBufferSpace($l1_bytes, -1, $ub_bytes);/" matmul_leakyrelu.asc
        run_test "L1=${l1_kb}KB_UB=${ub_kb}KB"
    done
done

# === L0C variations ===
for l0c_kb in 64 128 256; do
    restore
    l0c_bytes=$((l0c_kb * 1024))
    sed -i "s/tilingApi.SetBufferSpace(-1, -1, 98304);/tilingApi.SetBufferSpace(-1, $l0c_bytes, 98304);/" matmul_leakyrelu.asc
    run_test "L0C=${l0c_kb}KB_UB=96KB"
done

# === SocVersion ===
for soc in Ascend910B2 Ascend910B3 Ascend910B4; do
    restore
    sed -i "s/Ascend910B1/$soc/" matmul_leakyrelu.asc
    run_test "soc_${soc}"
done

# === EnableBias instead of SetBias ===
restore
sed -i 's/tilingApi.SetBias(isBias);/tilingApi.EnableBias(isBias);/' matmul_leakyrelu.asc
run_test "EnableBias"

# === SetShape(-1,-1,K) ===
restore
sed -i 's/tilingApi.SetShape(M, N, K);/tilingApi.SetShape(-1, -1, K);/' matmul_leakyrelu.asc
run_test "SetShape_auto"

# === Combined best: SetShape auto + UB=96KB ===
restore
sed -i 's/tilingApi.SetShape(M, N, K);/tilingApi.SetShape(-1, -1, K);/' matmul_leakyrelu.asc
run_test "SetShape_auto_UB96"

# === No SetTraverse (let API auto) ===
restore
sed -i '/tilingApi.SetTraverse/d' matmul_leakyrelu.asc
run_test "no_traverse"

# === No SetFixSplit + UB=96KB (pure auto with UB constraint) ===
restore
sed -i '/tilingApi.SetFixSplit/d' matmul_leakyrelu.asc
run_test "auto_split_UB96"

# === __mix__(1,1) instead of __mix__(1,2) ===
restore
sed -i 's/__mix__(1, 2)/__mix__(1, 1)/' matmul_leakyrelu.asc
run_test "mix_1_1"

# Restore best
restore

echo ""
echo "=== Sweep 3 Complete ==="
sort -t, -k5 -rn "$RESULTS_FILE" | head -20
