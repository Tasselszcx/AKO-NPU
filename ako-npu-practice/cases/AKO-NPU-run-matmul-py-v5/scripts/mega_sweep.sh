#!/bin/bash
# Mega sweep: massive parameter exploration
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="mega_sweep_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=28

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
    MSPROF_DIR="msprof_mega"
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

# Helper to set tile and cores
set_tile_cores() {
    sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = $1/" "$ASC_FILE"
    sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = $2/" "$ASC_FILE"
}

# === Batch 1: Odd tile sizes that might align better with hardware ===
for TILE in 7680 8704 9728 10752 11776 12800 13824 14848 15872 16896 17920 18944; do
    set_tile_cores $TILE 20
    quick_test "tile_${TILE}_c20"
done

# === Batch 2: Powers of 2 ===
for TILE in 2048 4096 8192 16384; do
    set_tile_cores $TILE 20
    quick_test "pow2_tile_${TILE}"
done

# === Batch 3: More core count variations ===
for CORES in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    set_tile_cores 16384 $CORES
    quick_test "c${CORES}_t16384"
done

# === Batch 4: Large tile with few iterations ===
set_tile_cores 24576 20
quick_test "tile_24576_c20"
set_tile_cores 22528 20
quick_test "tile_22528_c20"

# === Batch 5: Double buffer with various sizes ===
for TILE in 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288; do
    set_tile_cores $TILE 20
    # Change to double buffer
    sed -i 's/TQue<AscendC::TPosition::VECIN, 1>/TQue<AscendC::TPosition::VECIN, 2>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 1>/TQue<AscendC::TPosition::VECOUT, 2>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 1,/pipe.InitBuffer(inQueue, 2,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 1,/pipe.InitBuffer(outQueue, 2,/' "$ASC_FILE"
    quick_test "db_tile_${TILE}"
    # Restore single buffer
    sed -i 's/TQue<AscendC::TPosition::VECIN, 2>/TQue<AscendC::TPosition::VECIN, 1>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 2>/TQue<AscendC::TPosition::VECOUT, 1>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 2,/pipe.InitBuffer(inQueue, 1,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 2,/pipe.InitBuffer(outQueue, 1,/' "$ASC_FILE"
done

# === Batch 6: Core counts with best double buffer tile ===
for CORES in 10 15 20 25 30; do
    set_tile_cores 8192 $CORES
    sed -i 's/TQue<AscendC::TPosition::VECIN, 1>/TQue<AscendC::TPosition::VECIN, 2>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 1>/TQue<AscendC::TPosition::VECOUT, 2>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 1,/pipe.InitBuffer(inQueue, 2,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 1,/pipe.InitBuffer(outQueue, 2,/' "$ASC_FILE"
    quick_test "db_c${CORES}_t8192"
    sed -i 's/TQue<AscendC::TPosition::VECIN, 2>/TQue<AscendC::TPosition::VECIN, 1>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 2>/TQue<AscendC::TPosition::VECOUT, 1>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 2,/pipe.InitBuffer(inQueue, 1,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 2,/pipe.InitBuffer(outQueue, 1,/' "$ASC_FILE"
done

# === Batch 7: Very large tiles with single buffer ===
for TILE in 21504 22528 23552 24576; do
    for CORES in 15 20 25; do
        set_tile_cores $TILE $CORES
        quick_test "large_t${TILE}_c${CORES}"
    done
done

# === Batch 8: Combined matmul core + leakyrelu core sweeps ===
for DIM in 10 15 20 22; do
    for RCORES in 15 20 25; do
        set_tile_cores 16384 $RCORES
        sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
        quick_test "mdim${DIM}_rc${RCORES}"
        sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
    done
done

# === Batch 9: BufferSpace variations ===
for L1 in 65536 131072 262144; do
    set_tile_cores 16384 20
    sed -i "s/SetBufferSpace(-1, -1, -1)/SetBufferSpace($L1, -1, -1)/" "$ASC_FILE"
    quick_test "l1_${L1}"
    sed -i "s/SetBufferSpace($L1, -1, -1)/SetBufferSpace(-1, -1, -1)/" "$ASC_FILE"
done

# Restore optimal config
set_tile_cores 16384 20

echo "=== Mega sweep complete. Total: $((ITER - 28)) experiments ==="
echo "Results saved to $RESULTS_FILE"
wc -l "$RESULTS_FILE"
