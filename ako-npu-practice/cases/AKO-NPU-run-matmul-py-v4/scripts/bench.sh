#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
(
set -eo pipefail

SOLUTION_DIR="solution"
OP_NAME="matmul_leakyrelu_custom"
WARMUP=10
LAUNCH_COUNT=5

echo "=========================================="
echo "AKO Benchmark: matmul_leakyrelu"
echo "=========================================="

# Step 0: Environment
echo "--- [0/5] Environment Setup ---"
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "FAIL: ASCEND_HOME_PATH not set"; exit 1; }
SET_ENV=""
for p in "${ASCEND_HOME_PATH}/set_env.sh" \
         "${ASCEND_HOME_PATH}/../set_env.sh" \
         "${ASCEND_HOME_PATH}/../../set_env.sh"; do
    if [ -f "$p" ]; then SET_ENV="$p"; break; fi
done
[ -n "$SET_ENV" ] && source "$SET_ENV"

if [ -z "${ASC_DIR:-}" ]; then
    ASCEND_BASE="$(cd "${ASCEND_HOME_PATH}/.." && pwd)"
    ASC_FOUND=$(find "${ASCEND_BASE}" -maxdepth 6 -name "ASCConfig.cmake" -path "*/ascendc_kernel_cmake/*" 2>/dev/null | head -1 || true)
    if [ -n "${ASC_FOUND}" ]; then
        export ASC_DIR="$(dirname "${ASC_FOUND}")"
    fi
fi

# Step 1: Compile
echo "--- [1/5] Compile ---"
cd "$SOLUTION_DIR"
mkdir -p build
cd build
cmake ${ASC_DIR:+-DASC_DIR=$ASC_DIR} .. 2>&1 || { echo "FAIL: cmake failed"; exit 1; }
make -j4 2>&1 || { echo "FAIL: make failed"; exit 1; }
echo "Compile: PASS"

# Step 2: Generate test data
echo "--- [2/5] Generate Test Data ---"
python3 ../../scripts/gen_data.py 2>&1 || { echo "FAIL: gen_data.py failed"; exit 1; }

# Step 3: Run and verify correctness
echo "--- [3/5] Correctness Check ---"
rm -f output/output.bin
"./${OP_NAME}" 2>&1 || { echo "FAIL: Kernel execution failed"; exit 1; }
[ -f output/output.bin ] || { echo "FAIL: output.bin not found"; exit 1; }
python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo "Correctness: FAIL"
    exit 1
fi
echo "Correctness: PASS"

# Step 4: Performance profiling with msprof
echo "--- [4/5] Performance Profiling (msprof) ---"
MSPROF_OUT="./msprof_output"
rm -rf "$MSPROF_OUT"
msprof op --warm-up=${WARMUP} --launch-count=${LAUNCH_COUNT} --output="$MSPROF_OUT" "./${OP_NAME}" 2>&1
MSPROF_EXIT=$?
if [ $MSPROF_EXIT -ne 0 ]; then
    echo "WARN: msprof failed (exit $MSPROF_EXIT), trying without launch-count..."
    rm -rf "$MSPROF_OUT"
    msprof op --warm-up=${WARMUP} --output="$MSPROF_OUT" "./${OP_NAME}" 2>&1 || true
fi

# Step 5: Parse performance results
echo "--- [5/5] Performance Summary ---"
OPPROF_DIR=$(ls -td ${MSPROF_OUT}/OPPROF_* 2>/dev/null | head -1)
if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
    echo "OpBasicInfo:"
    cat "$OPPROF_DIR/OpBasicInfo.csv"
    echo ""

    # Extract Task Duration from OpBasicInfo.csv
    python3 -c "
import csv, sys
try:
    with open('$OPPROF_DIR/OpBasicInfo.csv', 'r') as f:
        reader = csv.DictReader(f)
        durations = []
        for row in reader:
            for key in row:
                if 'Task Duration' in key or 'task_duration' in key.lower():
                    try:
                        durations.append(float(row[key]))
                    except:
                        pass
                elif 'Duration' in key and 'Task' not in key:
                    pass
        if durations:
            import statistics
            mean_d = statistics.mean(durations)
            min_d = min(durations)
            max_d = max(durations)
            print(f'Task Duration: mean={mean_d:.2f}us, min={min_d:.2f}us, max={max_d:.2f}us, count={len(durations)}')
        else:
            print('Could not find Task Duration field')
            # Print all column names for debugging
            with open('$OPPROF_DIR/OpBasicInfo.csv', 'r') as f2:
                reader2 = csv.reader(f2)
                header = next(reader2)
                print(f'Available columns: {header}')
except Exception as e:
    print(f'Parse error: {e}')
" 2>&1

    if [ -f "$OPPROF_DIR/PipeUtilization.csv" ]; then
        echo ""
        echo "PipeUtilization:"
        cat "$OPPROF_DIR/PipeUtilization.csv"
    fi
else
    echo "WARN: No profiling data found"
fi

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
) 2>&1 | tee _bench_output.txt
BENCH_EXIT=${PIPESTATUS[0]}
set -e
# --- End bench command ---

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
