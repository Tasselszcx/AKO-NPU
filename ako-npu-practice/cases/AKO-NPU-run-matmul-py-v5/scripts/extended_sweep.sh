#!/bin/bash
# Extended sweep: even more parameter combinations
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="extended_sweep_results.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"
ITER=37

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
    MSPROF_DIR="msprof_ext"
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

# === Batch A: Very fine tile sweep around best (15K-17K, step 256) ===
for TILE in 15360 15616 15872 16128 16384 16640 16896 17152 17408; do
    set_tile_cores $TILE 20
    quick_test "fine_tile_$TILE"
done

# === Batch B: Core counts 17-23, step 1 ===
for CORES in 17 18 19 20 21 22 23; do
    set_tile_cores 16384 $CORES
    quick_test "fine_cores_$CORES"
done

# === Batch C: Tile 16384 with double buffer, various depths ===
# depth=2 already tested, but let's test depth=1 again as reference
set_tile_cores 16384 20
quick_test "ref_single_buf"

# === Batch D: Alpha sweep - different LeakyReLU alphas ===
for ALPHA in "0.0001f" "0.001f" "0.01f" "0.1f"; do
    set_tile_cores 16384 20
    sed -i "s/constexpr float LEAKY_ALPHA = [0-9.]*f;/constexpr float LEAKY_ALPHA = $ALPHA;/" "$ASC_FILE"
    quick_test "alpha_$ALPHA"
done
sed -i "s/constexpr float LEAKY_ALPHA = [0-9.]*f;/constexpr float LEAKY_ALPHA = 0.001f;/" "$ASC_FILE"

# === Batch E: MatMul L0C buffer sizes ===
for L0C in 32768 65536 131072; do
    set_tile_cores 16384 20
    sed -i "s/SetBufferSpace(-1, -1, -1)/SetBufferSpace(-1, $L0C, -1)/" "$ASC_FILE"
    quick_test "l0c_$L0C"
    sed -i "s/SetBufferSpace(-1, $L0C, -1)/SetBufferSpace(-1, -1, -1)/" "$ASC_FILE"
done

# === Batch F: MatMul UB sizes ===
for UB in 65536 98304 131072 196608; do
    set_tile_cores 16384 20
    sed -i "s/SetBufferSpace(-1, -1, -1)/SetBufferSpace(-1, -1, $UB)/" "$ASC_FILE"
    quick_test "ub_$UB"
    sed -i "s/SetBufferSpace(-1, -1, $UB)/SetBufferSpace(-1, -1, -1)/" "$ASC_FILE"
done

# === Batch G: Combined L1+UB sizes ===
for L1 in 131072 262144 524288; do
    for UB in 98304 131072 196608; do
        set_tile_cores 16384 20
        sed -i "s/SetBufferSpace(-1, -1, -1)/SetBufferSpace($L1, -1, $UB)/" "$ASC_FILE"
        quick_test "l1${L1}_ub${UB}"
        sed -i "s/SetBufferSpace($L1, -1, $UB)/SetBufferSpace(-1, -1, -1)/" "$ASC_FILE"
    done
done

# === Batch H: Tile alignment experiments ===
for TILE in 16000 16256 16384 16512 16768; do
    set_tile_cores $TILE 20
    quick_test "align_tile_$TILE"
done

# === Batch I: Different warmup count impacts (reference runs) ===
set_tile_cores 16384 20
for i in 1 2 3 4 5; do
    quick_test "ref_run_$i"
done

# === Batch J: Extreme tile sizes ===
for TILE in 1024 2048 3072 4096 5120; do
    set_tile_cores $TILE 20
    quick_test "small_tile_$TILE"
done

# === Batch K: Extreme core counts ===
for CORES in 1 2 3 4 5 35 40; do
    set_tile_cores 16384 $CORES
    quick_test "extreme_cores_$CORES"
done

# === Batch L: Combined small tile + many cores ===
for TILE in 2048 4096 8192; do
    for CORES in 5 10 20 30 40; do
        set_tile_cores $TILE $CORES
        quick_test "st${TILE}_c${CORES}"
    done
done

# Restore optimal config
set_tile_cores 16384 20

echo "=== Extended sweep complete. Total: $((ITER - 37)) experiments ==="
cat "$RESULTS_FILE" | wc -l
