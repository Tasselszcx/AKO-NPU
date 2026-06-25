#!/bin/bash
set -eo pipefail
SOLUTION_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SOLUTION_DIR"

echo "=== [1/5] Compile ==="
mkdir -p build && cd build
cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake 2>&1 | tail -5
make -j4 2>&1 | tail -10
if [ ! -f "./matmul_leakyrelu" ]; then echo "COMPILE FAILED"; exit 1; fi
echo "Compilation OK"
cd "$SOLUTION_DIR"

echo "=== [2/5] Generate test data ==="
cd build && python3 ../scripts/gen_data.py && cd "$SOLUTION_DIR"

echo "=== [3/5] Run with msprof ==="
cd build
MSPROF_DIR="$SOLUTION_DIR/build/msprof_output"
rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_DIR" ./matmul_leakyrelu 2>&1 | tail -30
cd "$SOLUTION_DIR"

echo "=== [4/5] Verify accuracy ==="
cd build
python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin
if [ $? -ne 0 ]; then echo "ACCURACY FAILED"; exit 1; fi
cd "$SOLUTION_DIR"

echo "=== [5/5] Extract performance ==="
cd build
python3 -c "
import os, csv, statistics, glob

opprof_base = sorted(glob.glob('$MSPROF_DIR/OPPROF_*'))[-1]

# Find all OpBasicInfo CSVs recursively
csv_files = glob.glob(os.path.join(opprof_base, '**/OpBasicInfo*.csv'), recursive=True)

all_durations = {}
for f in csv_files:
    with open(f, 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get('Op Name', 'unknown')
            td = row.get('Task Duration(us)', '')
            if td:
                all_durations.setdefault(name, []).append(float(td))

total_duration = 0
print('=== PERFORMANCE SUMMARY ===')
for name, durations in sorted(all_durations.items()):
    mean_d = statistics.mean(durations)
    total_duration += mean_d
    print(f'  {name}: mean={mean_d:.2f}us, min={min(durations):.2f}us, max={max(durations):.2f}us ({len(durations)} runs)')
print(f'Total kernel time (sum of means): {total_duration:.2f} us')
"

echo "=== BENCH COMPLETE ==="
