#!/bin/bash
# Final batch: iterations 98-200
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="final_batch_98_200_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=98

quick_test() {
    local DESC=$1
    cd solution/build
    cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,FAIL,FAIL,FAIL,compile_fail" >> "../../$RESULTS_FILE"
        cd ../..
        ITER=$((ITER + 1))
        return
    fi
    python3 ../scripts/gen_data.py > /dev/null 2>&1
    MSPROF_DIR="msprof_final2"
    rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=2 --launch-count=1 --output="$MSPROF_DIR" ./matmul_leakyrelu > /dev/null 2>&1 || true
    CORRECT="true"
    if ! python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin 2>&1 | grep -q "test pass"; then
        CORRECT="false"
    fi
    OPPROF_BASE=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_BASE" ]; then
        PERF=$(python3 -c "
import os, csv, glob
csv_files = glob.glob(os.path.join('$OPPROF_BASE', '**/OpBasicInfo*.csv'), recursive=True)
durations = {}
for f in csv_files:
    with open(f, 'r') as fh:
        for row in csv.DictReader(fh):
            name = row.get('Op Name', '')
            td = row.get('Task Duration(us)', '')
            if td:
                durations.setdefault(name, []).append(float(td))
matmul = sum(d for d in durations.get('matmul_leakyrelu_custom', [0]))/max(len(durations.get('matmul_leakyrelu_custom', [1])),1)
relu = sum(d for d in durations.get('leakyrelu_kernel', [0]))/max(len(durations.get('leakyrelu_kernel', [1])),1)
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

# === Sweeps 98-107: Step 64 from 15872-16896 for 10 tile values x 3 core counts ===
for TILE in $(seq 15872 64 16512); do
    for CORES in 19 20 21; do
        set_tile_cores $TILE $CORES
        quick_test "ultra_fine_t${TILE}_c${CORES}"
    done
done

# === Sweeps 108-117: MatMul dim 2-20 with FIRSTN, tile=16384, cores=20 ===
for DIM in $(seq 2 20); do
    set_tile_cores 16384 20
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN)/" "$ASC_FILE"
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test "firstn_dim${DIM}_v2"
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM)/" "$ASC_FILE"
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done

# === Sweeps 118-137: MatMul dim 2-22, FIRSTM, tile=16384, cores=20 ===
for DIM in $(seq 2 22); do
    set_tile_cores 16384 20
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test "firstm_dim${DIM}_v2"
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done

# === Sweeps 138-147: DB tile 5K-10K step 512 x cores=20 ===
for TILE in $(seq 5120 512 10240); do
    set_tile_cores $TILE 20
    sed -i 's/TQue<AscendC::TPosition::VECIN, 1>/TQue<AscendC::TPosition::VECIN, 2>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 1>/TQue<AscendC::TPosition::VECOUT, 2>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 1,/pipe.InitBuffer(inQueue, 2,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 1,/pipe.InitBuffer(outQueue, 2,/' "$ASC_FILE"
    quick_test "db_fine_t${TILE}"
    sed -i 's/TQue<AscendC::TPosition::VECIN, 2>/TQue<AscendC::TPosition::VECIN, 1>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 2>/TQue<AscendC::TPosition::VECOUT, 1>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 2,/pipe.InitBuffer(inQueue, 1,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 2,/pipe.InitBuffer(outQueue, 1,/' "$ASC_FILE"
done

# === Sweeps 148-155: Different L1 sizes ===
for L1 in 98304 131072 163840 196608 229376 262144 327680 393216; do
    set_tile_cores 16384 20
    sed -i "s/SetBufferSpace(-1, -1, -1)/SetBufferSpace($L1, -1, -1)/" "$ASC_FILE"
    quick_test "l1_${L1}"
    sed -i "s/SetBufferSpace($L1, -1, -1)/SetBufferSpace(-1, -1, -1)/" "$ASC_FILE"
done

# === Sweeps 156-163: Different UB sizes ===
for UB in 65536 81920 98304 114688 131072 147456 163840 196608; do
    set_tile_cores 16384 20
    sed -i "s/SetBufferSpace(-1, -1, -1)/SetBufferSpace(-1, -1, $UB)/" "$ASC_FILE"
    quick_test "ub_${UB}"
    sed -i "s/SetBufferSpace(-1, -1, $UB)/SetBufferSpace(-1, -1, -1)/" "$ASC_FILE"
done

# === Remaining: noise/stability runs to reach iter 200 ===
set_tile_cores 16384 20
while [ $ITER -le 200 ]; do
    quick_test "final_run_$ITER"
done

# Restore
set_tile_cores 16384 20

echo "=== Final batch 98-200 complete. Total: $((ITER - 98)) experiments ==="
wc -l "$RESULTS_FILE"
