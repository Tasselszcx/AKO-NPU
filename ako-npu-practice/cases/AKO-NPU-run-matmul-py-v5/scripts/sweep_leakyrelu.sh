#!/bin/bash
# Sweep LeakyReLU tile sizes and core counts
# Usage: bash scripts/sweep_leakyrelu.sh
set -eo pipefail
cd "$(dirname "$0")/.."

SOLUTION_DIR="solution"
ASC_FILE="$SOLUTION_DIR/matmul_leakyrelu.asc"
RESULTS_FILE="sweep_results.csv"

echo "tile_size,relu_cores,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"

for TILE in 8192 12288 16384 20480 24576; do
  for CORES in 15 20 25 30; do
    echo "=== Testing TILE=$TILE, CORES=$CORES ==="

    # Modify tile size
    sed -i "s/constexpr uint32_t TILE_SIZE = [0-9]*/constexpr uint32_t TILE_SIZE = $TILE/" "$ASC_FILE"
    # Modify core count
    sed -i "s/uint32_t reluBlocks = [0-9]*/uint32_t reluBlocks = $CORES/" "$ASC_FILE"

    # Build
    cd "$SOLUTION_DIR/build"
    cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake 2>&1 | tail -1
    if ! make -j4 2>&1 | tail -3; then
      echo "$TILE,$CORES,FAIL,FAIL,FAIL,compile_error" >> "../../$RESULTS_FILE"
      cd ../..
      continue
    fi

    # Generate data
    python3 ../scripts/gen_data.py 2>/dev/null

    # Run with msprof (fewer warmup for speed)
    MSPROF_DIR="msprof_sweep"
    rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=5 --launch-count=3 --output="$MSPROF_DIR" ./matmul_leakyrelu 2>&1 | tail -5

    # Check accuracy
    CORRECT="true"
    if ! python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin 2>&1 | grep -q "test pass"; then
      CORRECT="false"
    fi

    # Extract perf
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
matmul = sum(durations.get('matmul_leakyrelu_custom', [0]))/max(len(durations.get('matmul_leakyrelu_custom', [1])),1)
relu = sum(durations.get('leakyrelu_kernel', [0]))/max(len(durations.get('leakyrelu_kernel', [1])),1)
print(f'{matmul:.2f},{relu:.2f},{matmul+relu:.2f}')
" 2>/dev/null)
      echo "$TILE,$CORES,$PERF,$CORRECT" >> "../../$RESULTS_FILE"
      echo "  Result: $PERF correct=$CORRECT"
    fi

    rm -rf "$MSPROF_DIR"
    cd ../..
  done
done

echo "=== Sweep complete. Results in $RESULTS_FILE ==="
cat "$RESULTS_FILE"
