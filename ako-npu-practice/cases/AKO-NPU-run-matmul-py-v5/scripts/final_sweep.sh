#!/bin/bash
# Final massive sweep to cover remaining iterations
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="final_sweep_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=58

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
    MSPROF_DIR="msprof_final"
    rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=2 --launch-count=1 --output="$MSPROF_DIR" ./matmul_leakyrelu > /dev/null 2>&1 || true
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
    fi
    rm -rf "$MSPROF_DIR"
    cd ../..
    ITER=$((ITER + 1))
}

set_tile_cores() {
    sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = $1/" "$ASC_FILE"
    sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = $2/" "$ASC_FILE"
}

# === Batch 1: Full grid tile 8K-24K step 2K x cores 10-30 step 2 (99 combos) ===
echo "Starting batch 1: full grid..."
for TILE in 8192 10240 12288 14336 16384 18432 20480 22528 24576; do
    for CORES in 10 12 14 16 18 20 22 24 26 28 30; do
        set_tile_cores $TILE $CORES
        quick_test "fg_t${TILE}_c${CORES}"
    done
done

# === Batch 2: Double buffer full grid (50 combos) ===
echo "Starting batch 2: double buffer grid..."
for TILE in 4096 6144 8192 10240 12288; do
    for CORES in 10 12 14 16 18 20 22 24 26 28; do
        set_tile_cores $TILE $CORES
        sed -i 's/TQue<AscendC::TPosition::VECIN, 1>/TQue<AscendC::TPosition::VECIN, 2>/' "$ASC_FILE"
        sed -i 's/TQue<AscendC::TPosition::VECOUT, 1>/TQue<AscendC::TPosition::VECOUT, 2>/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(inQueue, 1,/pipe.InitBuffer(inQueue, 2,/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(outQueue, 1,/pipe.InitBuffer(outQueue, 2,/' "$ASC_FILE"
        quick_test "db_fg_t${TILE}_c${CORES}"
        sed -i 's/TQue<AscendC::TPosition::VECIN, 2>/TQue<AscendC::TPosition::VECIN, 1>/' "$ASC_FILE"
        sed -i 's/TQue<AscendC::TPosition::VECOUT, 2>/TQue<AscendC::TPosition::VECOUT, 1>/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(inQueue, 2,/pipe.InitBuffer(inQueue, 1,/' "$ASC_FILE"
        sed -i 's/pipe.InitBuffer(outQueue, 2,/pipe.InitBuffer(outQueue, 1,/' "$ASC_FILE"
    done
done

# Restore optimal
set_tile_cores 16384 20

echo "=== Final sweep complete. Total: $((ITER - 58)) experiments ==="
wc -l "$RESULTS_FILE"
