#!/bin/bash
# Sweep tiling parameters: vary baseM, baseN, usedCoreNum, traverse order
# Outputs a CSV with results for each configuration
set -eo pipefail
cd "$(dirname "$0")"/..

SOLUTION_DIR="$(pwd)/solution"
OP_NAME="matmul_leakyrelu_custom"
RESULTS_FILE="sweep_results.csv"

# Setup CANN env
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
elif [ -f "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" ]; then
    source "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" 2>/dev/null || true
fi

# Find ASC cmake config
ASCEND_BASE="$(dirname "${ASCEND_HOME_PATH}")"
ASC_DIR=""
for search_dir in "${ASCEND_HOME_PATH}" "${ASCEND_BASE}"; do
    found=$(find "${search_dir}" -name "ASCConfig.cmake" -print -quit 2>/dev/null || true)
    if [ -n "${found}" ]; then
        ASC_DIR="$(dirname "${found}")"
        break
    fi
done

echo "iter,baseM,baseN,cores,traverse,numBlocks,singleCoreM,singleCoreN,correct,runtime_us,speedup" > "$RESULTS_FILE"

BASELINE=228.37
ITER=5  # Continue from iter 5

run_config() {
    local baseM=$1
    local baseN=$2
    local cores=$3
    local traverse=$4
    ITER=$((ITER + 1))

    echo "=== Iter $ITER: baseM=$baseM baseN=$baseN cores=$cores traverse=$traverse ==="

    # Patch the source file
    cd "$SOLUTION_DIR"

    # Create patched version using sed
    cp matmul_leakyrelu.asc matmul_leakyrelu.asc.bak

    # Replace SetFixSplit line
    sed -i "s/tilingApi.SetFixSplit([0-9]*, [0-9]*, -1);/tilingApi.SetFixSplit($baseM, $baseN, -1);/" matmul_leakyrelu.asc

    # Replace usedCoreNum
    sed -i "s/int usedCoreNum = [0-9]*;/int usedCoreNum = $cores;/" matmul_leakyrelu.asc

    # Replace traverse order
    if [ "$traverse" = "FIRSTN" ]; then
        sed -i "s/MatrixTraverse::FIRSTM/MatrixTraverse::FIRSTN/" matmul_leakyrelu.asc
    fi

    # Build
    cd build
    if ! cmake -DASC_DIR="${ASC_DIR}" .. > /dev/null 2>&1; then
        echo "$ITER,$baseM,$baseN,$cores,$traverse,,,false,,,compile_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
        cd "$SOLUTION_DIR"
        cp matmul_leakyrelu.asc.bak matmul_leakyrelu.asc
        return
    fi
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$baseM,$baseN,$cores,$traverse,,,false,,,compile_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
        cd "$SOLUTION_DIR"
        cp matmul_leakyrelu.asc.bak matmul_leakyrelu.asc
        return
    fi

    # Generate data
    python3 ../gen_data.py > /dev/null 2>&1

    # Run correctness check
    rm -f output/output.bin
    ./${OP_NAME} > /tmp/kernel_output.txt 2>&1 || true

    # Extract numBlocks etc from output
    local numBlocks=$(grep -o 'numBlocks=[0-9]*' /tmp/kernel_output.txt | head -1 | cut -d= -f2)
    local scM=$(grep -o 'singleCoreM=[0-9]*' /tmp/kernel_output.txt | head -1 | cut -d= -f2)
    local scN=$(grep -o 'singleCoreN=[0-9]*' /tmp/kernel_output.txt | head -1 | cut -d= -f2)

    if [ ! -f output/output.bin ]; then
        echo "$ITER,$baseM,$baseN,$cores,$traverse,$numBlocks,$scM,$scN,false,,,runtime_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
        cd "$SOLUTION_DIR"
        cp matmul_leakyrelu.asc.bak matmul_leakyrelu.asc
        return
    fi

    local correct_result=$(python3 ../verify_result.py output/output.bin output/golden.bin 2>&1 | grep "Overall:" | awk '{print $2}')

    if [ "$correct_result" != "PASS" ]; then
        echo "$ITER,$baseM,$baseN,$cores,$traverse,$numBlocks,$scM,$scN,false,,," >> "$SOLUTION_DIR/../$RESULTS_FILE"
        cd "$SOLUTION_DIR"
        cp matmul_leakyrelu.asc.bak matmul_leakyrelu.asc
        return
    fi

    # Profile
    chmod 755 . 2>/dev/null || true
    rm -rf msprof_output
    local msprof_out=$(msprof op --warm-up=10 --output=./msprof_output ./${OP_NAME} 2>&1)

    local task_dur=$(echo "$msprof_out" | grep "Task Duration" | grep -o '[0-9.]*')

    if [ -n "$task_dur" ]; then
        local speedup=$(python3 -c "print(f'{$BASELINE / $task_dur:.3f}')")
        echo "$ITER,$baseM,$baseN,$cores,$traverse,$numBlocks,$scM,$scN,true,$task_dur,$speedup"
        echo "$ITER,$baseM,$baseN,$cores,$traverse,$numBlocks,$scM,$scN,true,$task_dur,$speedup" >> "$SOLUTION_DIR/../$RESULTS_FILE"
    else
        echo "$ITER,$baseM,$baseN,$cores,$traverse,$numBlocks,$scM,$scN,true,,profile_fail" >> "$SOLUTION_DIR/../$RESULTS_FILE"
    fi

    # Restore original
    cd "$SOLUTION_DIR"
    cp matmul_leakyrelu.asc.bak matmul_leakyrelu.asc
}

# === Sweep 1: Vary baseM with baseN=128, 2 cores, FIRSTM ===
for bm in 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 384 512; do
    run_config $bm 128 2 FIRSTM
done

# === Sweep 2: Vary baseN with baseM=128, 2 cores, FIRSTM ===
for bn in 32 64 96 128 160 192 256 320; do
    run_config 128 $bn 2 FIRSTM
done

# === Sweep 3: Try FIRSTN traverse ===
for bm in 128 256; do
    for bn in 128 256; do
        run_config $bm $bn 2 FIRSTN
    done
done

# === Sweep 4: Vary core count ===
for cores in 1 2 4 8; do
    run_config 128 128 $cores FIRSTM
done

# === Sweep 5: Best candidates with different baseK influence ===
# (baseK is auto, but different baseM*baseN combinations affect API's choice)
for bm in 128 192 256; do
    for bn in 64 128 192; do
        run_config $bm $bn 2 FIRSTM
    done
done

echo ""
echo "=== Sweep Complete ==="
echo "Results saved to $RESULTS_FILE"
echo ""
echo "Top 10 by speedup:"
sort -t, -k11 -rn "$RESULTS_FILE" | head -11
