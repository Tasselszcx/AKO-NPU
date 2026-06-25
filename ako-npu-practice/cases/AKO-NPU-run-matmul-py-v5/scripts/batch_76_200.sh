#!/bin/bash
# Iterations 76-200: exhaustive exploration of every remaining combination
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="batch_76_200_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=76

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
    MSPROF_DIR="msprof_batch"
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

# === Sweep 76: tile step 256 from 14K to 18K (17 tests) ===
for TILE in $(seq 14336 256 18432); do
    set_tile_cores $TILE 20
    quick_test "ts256_t${TILE}"
done

# === Sweep 77: core sweep 15-25 step 1 with tile=14336 (11 tests) ===
for CORES in $(seq 15 25); do
    set_tile_cores 14336 $CORES
    quick_test "t14336_c${CORES}"
done

# === Sweep 78: core sweep with tile=18432 (11 tests) ===
for CORES in $(seq 15 25); do
    set_tile_cores 18432 $CORES
    quick_test "t18432_c${CORES}"
done

# === Sweep 79: core sweep with tile=12288 (11 tests) ===
for CORES in $(seq 15 25); do
    set_tile_cores 12288 $CORES
    quick_test "t12288_c${CORES}"
done

# === Sweep 80: core sweep with tile=20480 (11 tests) ===
for CORES in $(seq 15 25); do
    set_tile_cores 20480 $CORES
    quick_test "t20480_c${CORES}"
done

# === Sweep 81: tile sweep step 128 from 16000 to 17000 (9 tests) ===
for TILE in $(seq 16000 128 17024); do
    set_tile_cores $TILE 20
    quick_test "ts128_t${TILE}"
done

# === Sweep 82: tile sweep step 128 from 15000 to 16000 (9 tests) ===
for TILE in $(seq 15000 128 16000); do
    set_tile_cores $TILE 20
    quick_test "ts128b_t${TILE}"
done

# === Sweep 83-87: different matmul configs with each dim ===
# (each dim is one sweep)
for DIM in 14 16 18 20 22; do
    set_tile_cores 16384 20
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    for CORES in $(seq 15 25); do
        set_tile_cores 16384 $CORES
        quick_test "dim${DIM}_c${CORES}"
    done
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done
# 5 sweeps x 11 tests = 55 tests

# === Sweep 88: Double buffer tile step 256 from 4096 to 12288 (33 tests) ===
for TILE in $(seq 4096 256 12288); do
    set_tile_cores $TILE 20
    sed -i 's/TQue<AscendC::TPosition::VECIN, 1>/TQue<AscendC::TPosition::VECIN, 2>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 1>/TQue<AscendC::TPosition::VECOUT, 2>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 1,/pipe.InitBuffer(inQueue, 2,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 1,/pipe.InitBuffer(outQueue, 2,/' "$ASC_FILE"
    quick_test "db_ts256_t${TILE}"
    sed -i 's/TQue<AscendC::TPosition::VECIN, 2>/TQue<AscendC::TPosition::VECIN, 1>/' "$ASC_FILE"
    sed -i 's/TQue<AscendC::TPosition::VECOUT, 2>/TQue<AscendC::TPosition::VECOUT, 1>/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(inQueue, 2,/pipe.InitBuffer(inQueue, 1,/' "$ASC_FILE"
    sed -i 's/pipe.InitBuffer(outQueue, 2,/pipe.InitBuffer(outQueue, 1,/' "$ASC_FILE"
done

# === Sweeps 89-96: tile x cores cross product (8 sweeps, each 8 tests) ===
for TILE in 10240 12288 14336 15360 16384 17408 18432 20480; do
    for CORES in 16 17 18 19 20 21 22 23; do
        set_tile_cores $TILE $CORES
        quick_test "xp_t${TILE}_c${CORES}"
    done
done
# 8 sweeps x 8 tests = 64 tests

# === Sweep 97-100: Stability with best configs ===
set_tile_cores 16384 20
for i in $(seq 1 20); do
    quick_test "final_stability_$i"
done

# Restore
set_tile_cores 16384 20

echo "=== Batch 76-200 complete. Total: $((ITER - 76)) experiments ==="
wc -l "$RESULTS_FILE"
