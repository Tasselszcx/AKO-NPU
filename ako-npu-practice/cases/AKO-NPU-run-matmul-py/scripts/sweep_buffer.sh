#!/bin/bash
# Sweep 2: Buffer space and advanced configs
set -eo pipefail
cd "$(dirname "$0")"/..

SOLUTION_DIR="$(pwd)/solution"
OP_NAME="matmul_leakyrelu_custom"
RESULTS_FILE="sweep_results2.csv"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
elif [ -f "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" ]; then
    source "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" 2>/dev/null || true
fi

ASCEND_BASE="$(dirname "${ASCEND_HOME_PATH}")"
ASC_DIR=""
for search_dir in "${ASCEND_HOME_PATH}" "${ASCEND_BASE}"; do
    found=$(find "${search_dir}" -name "ASCConfig.cmake" -print -quit 2>/dev/null || true)
    if [ -n "${found}" ]; then
        ASC_DIR="$(dirname "${found}")"
        break
    fi
done

echo "iter,config,numBlocks,singleCoreM,singleCoreN,correct,runtime_us,speedup" > "$RESULTS_FILE"

BASELINE=228.37
ITER=48

# Save original
cd "$SOLUTION_DIR"
cp matmul_leakyrelu.asc matmul_leakyrelu.asc.orig

run_test() {
    local desc="$1"
    ITER=$((ITER + 1))
    echo "=== Iter $ITER: $desc ==="

    cd "$SOLUTION_DIR/build"
    if ! cmake -DASC_DIR="${ASC_DIR}" .. > /dev/null 2>&1 || ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$desc,,,false,,,compile_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
        return 1
    fi

    python3 ../gen_data.py > /dev/null 2>&1
    rm -f output/output.bin
    ./${OP_NAME} > /tmp/kernel_output.txt 2>&1 || true

    local numBlocks=$(grep -o 'numBlocks=[0-9]*' /tmp/kernel_output.txt | head -1 | cut -d= -f2)
    local scM=$(grep -o 'singleCoreM=[0-9]*' /tmp/kernel_output.txt | head -1 | cut -d= -f2)
    local scN=$(grep -o 'singleCoreN=[0-9]*' /tmp/kernel_output.txt | head -1 | cut -d= -f2)

    if [ ! -f output/output.bin ]; then
        echo "$ITER,$desc,$numBlocks,$scM,$scN,false,,,runtime_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
        return 1
    fi

    local correct=$(python3 ../verify_result.py output/output.bin output/golden.bin 2>&1 | grep "Overall:" | awk '{print $2}')
    if [ "$correct" != "PASS" ]; then
        echo "$ITER,$desc,$numBlocks,$scM,$scN,false,,," >> "$SOLUTION_DIR/../$RESULTS_FILE"
        return 1
    fi

    chmod 755 . 2>/dev/null || true
    rm -rf msprof_output
    local msprof_out=$(msprof op --warm-up=10 --output=./msprof_output ./${OP_NAME} 2>&1)
    local task_dur=$(echo "$msprof_out" | grep "Task Duration" | grep -o '[0-9.]*')

    if [ -n "$task_dur" ]; then
        local speedup=$(python3 -c "print(f'{$BASELINE / $task_dur:.3f}')")
        echo "$ITER,$desc,$numBlocks,$scM,$scN,true,$task_dur,$speedup"
        echo "$ITER,$desc,$numBlocks,$scM,$scN,true,$task_dur,$speedup" >> "$SOLUTION_DIR/../$RESULTS_FILE"
    else
        echo "$ITER,$desc,$numBlocks,$scM,$scN,true,,profile_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
    fi
    return 0
}

restore() {
    cd "$SOLUTION_DIR"
    cp matmul_leakyrelu.asc.orig matmul_leakyrelu.asc
}

# === Sweep buffer space: L1, L0C, UB ===
# Current: SetBufferSpace(-1, -1, -1) = auto all
# Try restricting UB to leave more for Matmul internals

# Iter 49-52: Vary UB space with best tiling (256x128)
for ub_kb in 64 96 128 160; do
    restore
    ub_bytes=$((ub_kb * 1024))
    sed -i "s/tilingApi.SetBufferSpace(-1, -1, -1);/tilingApi.SetBufferSpace(-1, -1, $ub_bytes);/" matmul_leakyrelu.asc
    run_test "UB=${ub_kb}KB_baseM256_baseN128"
done

# Iter 53-55: Vary L1 space
for l1_kb in 256 512 1024; do
    restore
    l1_bytes=$((l1_kb * 1024))
    sed -i "s/tilingApi.SetBufferSpace(-1, -1, -1);/tilingApi.SetBufferSpace($l1_bytes, -1, -1);/" matmul_leakyrelu.asc
    run_test "L1=${l1_kb}KB_baseM256_baseN128"
done

# Iter 56-58: stepM/stepN variations (remove override)
restore
# Remove stepM/stepN override entirely
sed -i '/tilingData.stepM = 1;/d' matmul_leakyrelu.asc
sed -i '/tilingData.stepN = 1;/d' matmul_leakyrelu.asc
run_test "no_step_override_256x128"

restore
# stepM=2 allows 2 M-tiles before switching N
sed -i 's/tilingData.stepM = 1;/tilingData.stepM = 2;/' matmul_leakyrelu.asc
run_test "stepM2_stepN1_256x128"

restore
# stepN=2
sed -i 's/tilingData.stepN = 1;/tilingData.stepN = 2;/' matmul_leakyrelu.asc
run_test "stepM1_stepN2_256x128"

# Iter 59-61: Try different socVersion strings
for soc in Ascend910B2 Ascend910B3 Ascend910B4; do
    restore
    sed -i "s/Ascend910B1/$soc/" matmul_leakyrelu.asc
    run_test "soc_${soc}_256x128"
done

# Iter 62-63: Try EnableBias vs SetBias
restore
sed -i 's/tilingApi.SetBias(isBias);/tilingApi.EnableBias(isBias);/' matmul_leakyrelu.asc
run_test "EnableBias_256x128"

# Iter 64: Try SetShape(-1,-1,K) like matmul_perf example
restore
sed -i 's/tilingApi.SetShape(M, N, K);/tilingApi.SetShape(-1, -1, K);/' matmul_leakyrelu.asc
run_test "SetShape_auto_256x128"

# Restore to best
restore
# Apply best config: 256x128
sed -i 's/tilingApi.SetFixSplit(128, 128, -1);/tilingApi.SetFixSplit(256, 128, -1);/' matmul_leakyrelu.asc

echo ""
echo "=== Sweep 2 Complete ==="
echo "Results saved to $RESULTS_FILE"
echo ""
echo "All results:"
cat "$RESULTS_FILE"
