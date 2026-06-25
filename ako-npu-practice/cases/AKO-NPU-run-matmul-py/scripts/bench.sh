#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")"/..

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
(
set -eo pipefail

PROJECT_DIR="$(pwd)"
SOLUTION_DIR="${PROJECT_DIR}/solution"
OP_NAME="matmul_leakyrelu_custom"

echo "=== [1/5] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true
elif [ -f "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" ]; then
    source "$(dirname "${ASCEND_HOME_PATH}")/set_env.sh" 2>/dev/null || true
fi

echo "=== [2/5] Build ==="
cd "${SOLUTION_DIR}"
# Fix permissions for msprof (requires no group/other write on output parent dir)
chmod 755 "${SOLUTION_DIR}" 2>/dev/null || true
mkdir -p build
chmod 755 build 2>/dev/null || true
cd build

# Find ASC cmake config
ASCEND_BASE="$(dirname "${ASCEND_HOME_PATH}")"
ASC_DIR=""
for search_dir in "${ASCEND_HOME_PATH}" "${ASCEND_BASE}"; do
    found=$(find "${search_dir}" -name "ASCConfig.cmake" -print -quit 2>/dev/null || true)
    if [ -n "${found}" ]; then
        ASC_DIR="$(dirname "${found}")"
        break
    fi
done
if [ -n "${ASC_DIR}" ]; then
    cmake -DASC_DIR="${ASC_DIR}" .. 2>&1
else
    cmake .. 2>&1
fi
make -j4 2>&1
echo "BUILD: PASS"

echo "=== [3/5] Generate test data ==="
python3 ../gen_data.py 2>&1
echo "DATA_GEN: PASS"

echo "=== [4/5] Run correctness check ==="
rm -f output/output.bin
"./${OP_NAME}" 2>&1
[ -f output/output.bin ] || { echo "CORRECTNESS: FAIL (no output)"; exit 1; }
python3 ../verify_result.py output/output.bin output/golden.bin 2>&1
echo "CORRECTNESS: PASS"

echo "=== [5/5] Performance measurement ==="
# Use msprof op with warm-up for stable timing
MSPROF_DIR="./msprof_output"
rm -rf "${MSPROF_DIR}"
msprof op --warm-up=10 --launch-count=5 --output="${MSPROF_DIR}" "./${OP_NAME}" 2>&1

# Extract Task Duration from OpBasicInfo.csv
OPPROF_DIR=$(ls -td "${MSPROF_DIR}"/OPPROF_* 2>/dev/null | head -1)
if [ -n "${OPPROF_DIR}" ] && [ -f "${OPPROF_DIR}/OpBasicInfo.csv" ]; then
    echo ""
    echo "=== Performance Results ==="
    # Parse Task Duration values from CSV
    python3 -c "
import csv, sys

durations = []
with open('${OPPROF_DIR}/OpBasicInfo.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td = row.get('Task Duration(us)', '')
        if td:
            durations.append(float(td))

if durations:
    import statistics
    mean_d = statistics.mean(durations)
    min_d = min(durations)
    max_d = max(durations)
    print(f'Runtime: {mean_d:.2f} us (mean), {min_d:.2f} ~ {max_d:.2f} us (min ~ max)')
    print(f'Samples: {len(durations)}')
    # Speedup relative to original baseline (228.37 us)
    BASELINE=228.37
    speedup = BASELINE / mean_d
    speedup_min = BASELINE / max_d
    speedup_max = BASELINE / min_d
    print(f'Speedup: {speedup:.3f}x (mean), {speedup_min:.3f} ~ {speedup_max:.3f}x (min ~ max)')
else:
    print('WARNING: No Task Duration data found')
    sys.exit(1)
" 2>&1
else
    echo "WARNING: msprof output not found, falling back to simple timing"
    # Simple timing fallback: run 5 times
    for i in 1 2 3 4 5; do
        "./${OP_NAME}" 2>&1
    done
fi

echo ""
echo "BENCH: COMPLETE"

) 2>&1 | tee _bench_output.txt
BENCH_EXIT=$?
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
