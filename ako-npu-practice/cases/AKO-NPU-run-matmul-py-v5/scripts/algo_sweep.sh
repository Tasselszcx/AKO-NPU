#!/bin/bash
# Algorithmic variations sweep
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="algo_sweep_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=48

quick_test() {
    local DESC=$1
    cd solution/build
    cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,FAIL,FAIL,FAIL,compile_fail" >> "../../$RESULTS_FILE"
        echo "  Iter $ITER ($DESC): COMPILE FAIL"
        cd ../..
        ITER=$((ITER + 1))
        return
    fi
    python3 ../scripts/gen_data.py > /dev/null 2>&1
    MSPROF_DIR="msprof_algo"
    rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=3 --launch-count=2 --output="$MSPROF_DIR" ./matmul_leakyrelu > /dev/null 2>&1 || true
    CORRECT="true"
    if ! python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin 2>&1 | grep -q "test pass"; then
        CORRECT="false"
    fi
    OPPROF_BASE=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_BASE" ]; then
        PERF=$(python3 -c "
import os, csv, glob, statistics
csv_files = glob.glob(os.path.join('$OPPROF_BASE', '**/OpBasicInfo*.csv'), recursive=True)
durations = {}
for f in csv_files:
    with open(f, 'r') as fh:
        for row in csv.DictReader(fh):
            name = row.get('Op Name', '')
            td = row.get('Task Duration(us)', '')
            if td:
                durations.setdefault(name, []).append(float(td))
matmul = statistics.mean(durations.get('matmul_leakyrelu_custom', [0]))
relu = statistics.mean(durations.get('leakyrelu_kernel', [0]))
print(f'{matmul:.2f},{relu:.2f},{matmul+relu:.2f}')
" 2>/dev/null || echo "0,0,0")
        echo "$ITER,$DESC,$PERF,$CORRECT" >> "../../$RESULTS_FILE"
        echo "  Iter $ITER ($DESC): $PERF correct=$CORRECT"
    fi
    rm -rf "$MSPROF_DIR"
    cd ../..
    ITER=$((ITER + 1))
}

set_tile_cores() {
    sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = $1/" "$ASC_FILE"
    sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = $2/" "$ASC_FILE"
}

# Reference baseline (5 runs for stability)
for i in 1 2 3 4 5; do
    set_tile_cores 16384 20
    quick_test "ref_baseline_$i"
done

# === Algorithmic variant 1: Try different MatmulConfig field orderings ===
# enableSetBias=false (since we always have bias, this disables the check)
python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    '''constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetBias = false,
    .enableSetOrgShape = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};''',
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"
quick_test "enableSetBias_false"

# Restore
python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    '''constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetOrgShape = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};''',
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"

# === Try enableSetTail=false ===
python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    '''constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetOrgShape = false,
    .enableSetTail = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};''',
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"
quick_test "enableSetTail_false"

# Restore
python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    '''constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetOrgShape = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};''',
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"

# === Try enableInit=false ===
python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    '''constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .enableInit = false,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetOrgShape = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};''',
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"
quick_test "enableInit_false"

# Restore
python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    '''constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetOrgShape = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};''',
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"

# === Batch: Multiple tile/core combos for remaining iters ===
# Systematic grid: tile from 10K-20K step 2K, cores from 15-25 step 5
for TILE in 10240 12288 14336 16384 18432 20480; do
    for CORES in 15 20 25; do
        set_tile_cores $TILE $CORES
        quick_test "grid_t${TILE}_c${CORES}"
    done
done

# === More specific combos ===
for TILE in 11264 13312 15360 17408 19456; do
    for CORES in 18 20 22; do
        set_tile_cores $TILE $CORES
        quick_test "grid2_t${TILE}_c${CORES}"
    done
done

# === Repeat best configs for stability ===
set_tile_cores 16384 20
for i in 1 2 3 4 5 6 7 8 9 10; do
    quick_test "stability_$i"
done

# === Double buffer grid ===
for TILE in 4096 6144 8192 10240; do
    for CORES in 15 20 25; do
        set_tile_cores $TILE $CORES
        sed -i 's/TQue<AscendC::TPosition::VECIN, 1>/TQue<AscendC::TPosition::VECIN, 2>/' "$ASC_FILE"
        sed -i 's/TQue<AscendC::TPosition::VECOUT, 1>/TQue<AscendC::TPosition::VECOUT, 2>/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(inQueue, 1,/pipe.InitBuffer(inQueue, 2,/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(outQueue, 1,/pipe.InitBuffer(outQueue, 2,/' "$ASC_FILE"
        quick_test "db_grid_t${TILE}_c${CORES}"
        sed -i 's/TQue<AscendC::TPosition::VECIN, 2>/TQue<AscendC::TPosition::VECIN, 1>/' "$ASC_FILE"
        sed -i 's/TQue<AscendC::TPosition::VECOUT, 2>/TQue<AscendC::TPosition::VECOUT, 1>/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(inQueue, 2,/pipe.InitBuffer(inQueue, 1,/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(outQueue, 2,/pipe.InitBuffer(outQueue, 1,/' "$ASC_FILE"
    done
done

# === FIRSTN with specific dim values ===
for DIM in 8 10 12 14 16 18 20 22; do
    set_tile_cores 16384 20
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN)/" "$ASC_FILE"
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test "firstn_dim_$DIM"
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM)/" "$ASC_FILE"
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done

# === FIRSTM with specific dim values ===
for DIM in 8 10 12 14 16 18 20 22; do
    set_tile_cores 16384 20
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test "firstm_dim_$DIM"
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done

# Restore optimal config
set_tile_cores 16384 20

echo "=== Algo sweep complete. Total: $((ITER - 48)) experiments ==="
cat "$RESULTS_FILE" | wc -l
