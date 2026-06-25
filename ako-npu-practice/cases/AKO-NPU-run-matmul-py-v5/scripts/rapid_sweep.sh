#!/bin/bash
# Rapid sweep: test many LeakyReLU + MatMul parameter combinations
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="rapid_sweep_results.csv"
SOLUTION_DIR="solution"

echo "iter,description,tile_size,relu_cores,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"

run_test() {
    local ITER=$1 DESC=$2 TILE=$3 CORES=$4
    # Modify tile size and core count
    sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = $TILE/" "$ASC_FILE"
    sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = $CORES/" "$ASC_FILE"

    cd "$SOLUTION_DIR/build"
    cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,$TILE,$CORES,FAIL,FAIL,FAIL,false" >> "../../$RESULTS_FILE"
        cd ../..
        return
    fi
    python3 ../scripts/gen_data.py > /dev/null 2>&1

    MSPROF_DIR="msprof_rapid"
    rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=5 --launch-count=3 --output="$MSPROF_DIR" ./matmul_leakyrelu > /dev/null 2>&1 || true

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
        echo "$ITER,$DESC,$TILE,$CORES,$PERF,$CORRECT" >> "../../$RESULTS_FILE"
        echo "  Iter $ITER ($DESC): tile=$TILE cores=$CORES -> $PERF correct=$CORRECT"
    fi
    rm -rf "$MSPROF_DIR"
    cd ../..
}

# Fine-grained tile size sweep around 16384
run_test 19 "tile_14336" 14336 20
run_test 20 "tile_15360" 15360 20
run_test 21 "tile_16384" 16384 20
run_test 22 "tile_17408" 17408 20
run_test 23 "tile_18432" 18432 20

# Core count fine sweep around 20
run_test 24 "cores_16" 16384 16
run_test 25 "cores_18" 16384 18
run_test 26 "cores_20" 16384 20
run_test 27 "cores_22" 16384 22

# Combined: try best tile sizes with different core counts
run_test 28 "t14336_c18" 14336 18
run_test 29 "t14336_c22" 14336 22
run_test 30 "t15360_c18" 15360 18
run_test 31 "t15360_c22" 15360 22
run_test 32 "t17408_c18" 17408 18
run_test 33 "t17408_c22" 17408 22

# Very small tiles (high iteration count, better pipeline overlap)
run_test 34 "tile_4096" 4096 20
run_test 35 "tile_6144" 6144 20

# Tile = exact per-core data (32768 elem / 20 cores = 32768)
# With 20 cores, each processes 32768 elems. Try tile=32768 but need single buffer only
# Can't fit 32768 with 2 buffers. Skip.

# More core counts with small tiles
run_test 36 "t8192_c12" 8192 12
run_test 37 "t8192_c16" 8192 16
run_test 38 "t8192_c24" 8192 24

# Restore best config
sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = 16384/" "$ASC_FILE"
sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = 20/" "$ASC_FILE"

echo "=== Rapid sweep complete ==="
cat "$RESULTS_FILE"
