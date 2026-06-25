#!/bin/bash
# Creative sweeps: try unusual combinations and configurations
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="creative_sweep_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=64

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
    MSPROF_DIR="msprof_creative"
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

# === Batch 1: Tile sizes that are exact multiples of N=640 ===
for MULT in 1 2 4 5 8 10 16 20 25 32; do
    TILE=$((640 * MULT))
    if [ $TILE -le 24576 ]; then
        set_tile_cores $TILE 20
        quick_test "n_mult_${MULT}_tile${TILE}"
    fi
done

# === Batch 2: Tile sizes that are exact multiples of singleCoreN=320 ===
for MULT in 1 2 4 8 16 32 48 64 76; do
    TILE=$((320 * MULT))
    if [ $TILE -le 24576 ] && [ $TILE -ge 1024 ]; then
        set_tile_cores $TILE 20
        quick_test "scn_mult_${MULT}_tile${TILE}"
    fi
done

# === Batch 3: Tile sizes based on baseM*baseN from tiling (96*256=24576) ===
for DIV in 1 2 3 4 6 8 12 16 24; do
    TILE=$((24576 / DIV))
    if [ $TILE -ge 1024 ]; then
        set_tile_cores $TILE 20
        quick_test "bmbn_div${DIV}_tile${TILE}"
    fi
done

# === Batch 4: Tile sizes that are powers of 2 multiples of 8 (alignment) ===
for TILE in 1024 2048 4096 8192 16384 24576; do
    for CORES in 20; do
        set_tile_cores $TILE $CORES
        quick_test "pow2x8_tile${TILE}"
    done
done

# === Batch 5: Exact divisors of per-core data (32768 / 20 cores = varies) ===
# Per core data = 1024*640/20 = 32768 elements
for DIV in 1 2 4 8 16 32; do
    TILE=$((32768 / DIV))
    if [ $TILE -le 24576 ] && [ $TILE -ge 1024 ]; then
        set_tile_cores $TILE 20
        quick_test "percore_div${DIV}_tile${TILE}"
    fi
done

# === Batch 6: Core counts that evenly divide total elements (655360) ===
for CORES in 1 2 4 5 8 10 16 20 32 40; do
    set_tile_cores 16384 $CORES
    quick_test "even_div_c${CORES}"
done

# === Batch 7: Core counts that evenly divide total elements (655360) with tile=8192 ===
for CORES in 1 2 4 5 8 10 16 20 32 40; do
    set_tile_cores 8192 $CORES
    quick_test "even_div_t8k_c${CORES}"
done

# === Batch 8: Tile = exact per-core elements for various core counts ===
for CORES in 10 16 20 32; do
    TILE=$((655360 / CORES))
    if [ $TILE -le 24576 ]; then
        set_tile_cores $TILE $CORES
        quick_test "exact_percore_c${CORES}_t${TILE}"
    fi
done

# === Batch 9: Random noise tests (same config, different orderings) ===
set_tile_cores 16384 20
for i in $(seq 1 30); do
    quick_test "noise_run_$i"
done

# === Batch 10: Tile sizes aligned to 512B (128 float elements) ===
for MULT in $(seq 8 8 192); do
    TILE=$((MULT * 128))
    if [ $TILE -ge 1024 ] && [ $TILE -le 24576 ]; then
        set_tile_cores $TILE 20
        quick_test "align512_tile${TILE}"
    fi
done

# === Batch 11: MatMul SetDim sweep with all core counts ===
for DIM in $(seq 2 2 22); do
    set_tile_cores 16384 20
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test "matmul_dim_$DIM"
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done

# === Batch 12: FIRSTN with various dims and core counts ===
for DIM in $(seq 2 2 22); do
    set_tile_cores 16384 20
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN)/" "$ASC_FILE"
    sed -i "s/SetDim(ascendcPlatform->GetCoreNumAic())/SetDim($DIM)/" "$ASC_FILE"
    quick_test "firstn_dim$DIM"
    sed -i "s/SetTraverse(matmul_tiling::MatrixTraverse::[A-Z]*)/SetTraverse(matmul_tiling::MatrixTraverse::FIRSTM)/" "$ASC_FILE"
    sed -i "s/SetDim($DIM)/SetDim(ascendcPlatform->GetCoreNumAic())/" "$ASC_FILE"
done

# Restore optimal
set_tile_cores 16384 20

echo "=== Creative sweep complete. Total: $((ITER - 64)) experiments ==="
wc -l "$RESULTS_FILE"
