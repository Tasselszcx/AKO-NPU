#!/bin/bash
# Inner benchmark: compile, run with msprof, verify, report timing
set -eo pipefail

SOLUTION_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SOLUTION_DIR"

echo "=== [1/5] Compile ==="
mkdir -p build && cd build
cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake 2>&1 | tail -5
make -j4 2>&1 | tail -10
BINARY="./matmul_leakyrelu"
if [ ! -f "$BINARY" ]; then
    echo "COMPILE FAILED"
    exit 1
fi
echo "Compilation OK"
cd "$SOLUTION_DIR"

echo "=== [2/5] Generate test data ==="
cd build
python3 ../scripts/gen_data.py
cd "$SOLUTION_DIR"

echo "=== [3/5] Run with msprof ==="
cd build
rm -rf msprof_output
msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./matmul_leakyrelu 2>&1 | tail -20
cd "$SOLUTION_DIR"

echo "=== [4/5] Verify accuracy ==="
cd build
python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo "ACCURACY FAILED"
    exit 1
fi
cd "$SOLUTION_DIR"

echo "=== [5/5] Extract performance ==="
cd build
OPPROF_DIR=$(ls -td msprof_output/OPPROF_* 2>/dev/null | head -1)
if [ -z "$OPPROF_DIR" ]; then
    echo "No OPPROF directory found"
    exit 1
fi

# Extract Task Duration from OpBasicInfo.csv
if [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
    echo "--- OpBasicInfo.csv ---"
    cat "$OPPROF_DIR/OpBasicInfo.csv"
    echo ""
    # Parse Task Duration values (in us)
    python3 -c "
import csv, sys
durations = []
with open('$OPPROF_DIR/OpBasicInfo.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td = row.get('Task Duration(us)', '')
        if td:
            durations.append(float(td))
if durations:
    import statistics
    print('=== PERFORMANCE SUMMARY ===')
    print(f'Runs: {len(durations)}')
    print(f'Task Duration (us): mean={statistics.mean(durations):.2f}, min={min(durations):.2f}, max={max(durations):.2f}')
    if len(durations) > 1:
        print(f'  stddev={statistics.stdev(durations):.2f}')
else:
    print('No Task Duration data found')
    sys.exit(1)
"
fi

echo "=== BENCH COMPLETE ==="
