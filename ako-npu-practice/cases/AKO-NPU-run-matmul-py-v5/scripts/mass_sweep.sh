#!/bin/bash
# Mass sweep: test many variations rapidly
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="mass_sweep_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"

quick_test() {
    local ITER=$1 DESC=$2
    cd solution/build
    cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,FAIL,FAIL,FAIL,compile_fail" >> "../../$RESULTS_FILE"
        echo "  Iter $ITER ($DESC): COMPILE FAIL"
        cd ../..
        return
    fi
    python3 ../scripts/gen_data.py > /dev/null 2>&1
    MSPROF_DIR="msprof_mass"
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
}

# === LeakyReLU tile size sweep: every 1024 from 8192 to 24576 ===
for TILE in 8192 9216 10240 11264 12288 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576; do
    ITER=$((24 + (TILE - 8192) / 1024))
    sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = $TILE/" "$ASC_FILE"
    quick_test $ITER "tile_$TILE"
done

# Restore best tile
sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = 16384/" "$ASC_FILE"

# === LeakyReLU core sweep: every core count from 8 to 30 ===
for CORES in 8 10 12 14 16 17 18 19 20 21 22 24 26 28 30; do
    ITER=$((41 + CORES - 8))
    sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = $CORES/" "$ASC_FILE"
    quick_test $ITER "cores_$CORES"
done

# Restore best cores
sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = 20/" "$ASC_FILE"

# === MatMul traverse + core combos ===
# Try FIRSTN with various dim counts
for DIM in 10 15 20 22; do
    ITER=$((56 + DIM - 10))
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN)/" "$ASC_FILE"
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test $ITER "firstn_dim$DIM"
done

# Restore FIRSTM and max dim
sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM)/" "$ASC_FILE"
sed -i "s/SetDim([0-9]*)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"

echo "=== Mass sweep complete ==="
cat "$RESULTS_FILE"
