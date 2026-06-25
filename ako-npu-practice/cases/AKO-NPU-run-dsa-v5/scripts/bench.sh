#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
set +e
(
set -euo pipefail

B=1
S_q=1
S_kv=4096

echo "=== DSA Indexer Benchmark: B=${B}, S_q=${S_q}, S_kv=${S_kv} ==="
echo "Label: ${LABEL:-none}"

# [1] Build
echo "=== [1/5] Build ==="
cd "${PROJECT_ROOT}/solution"
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
make -j4 2>&1 | tail -3

# [2] Generate data
echo "=== [2/5] Generate test data ==="
python3 "${PROJECT_ROOT}/scripts/gen_data.py" ${B} ${S_q} ${S_kv}

# [3] Run kernel directly first (for correctness check)
echo "=== [3/5] Run kernel ==="
./dsa_indexer ${B} ${S_q} ${S_kv}

# [4] Verify correctness
echo "=== [4/5] Verify correctness ==="
python3 "${PROJECT_ROOT}/scripts/verify_result.py" ${B} ${S_q} ${S_kv}

# [5] Run with msprof for timing
echo "=== [5/5] Profile with msprof ==="
MSPROF_DIR="${PROJECT_ROOT}/msprof_output"
rm -rf "${MSPROF_DIR}"
mkdir -p "${MSPROF_DIR}"
chmod 700 "${MSPROF_DIR}"
msprof op --warm-up=5 --launch-count=3 --output="${MSPROF_DIR}" ./dsa_indexer ${B} ${S_q} ${S_kv} 2>&1 || echo "msprof failed, trying without output dir..."

OPPROF_DIR=$(ls -td ${MSPROF_DIR}/OPPROF_* 2>/dev/null | head -1)
if [ -z "$OPPROF_DIR" ]; then
    # Try running msprof without --output flag
    msprof op --warm-up=5 --launch-count=3 ./dsa_indexer ${B} ${S_q} ${S_kv} 2>&1 || true
    OPPROF_DIR=$(ls -td OPPROF_* 2>/dev/null | head -1)
fi

if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
    echo "--- Performance Results ---"
    python3 -c "
import csv, sys, statistics
durations = []
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td_key = [k for k in row.keys() if 'Task Duration' in k]
        if td_key:
            val = row[td_key[0]].strip()
            if val:
                durations.append(float(val))
if durations:
    print(f'Task Duration: mean={statistics.mean(durations):.2f} us, min={min(durations):.2f} us, max={max(durations):.2f} us, n={len(durations)}')
else:
    print('Could not extract Task Duration')
"
    if [ -f "$OPPROF_DIR/PipeUtilization.csv" ]; then
        echo ""
        echo "--- PipeUtilization (first 3 lines) ---"
        head -3 "$OPPROF_DIR/PipeUtilization.csv"
    fi
else
    echo "WARNING: msprof output not found"
    echo "Running simple timing with multiple invocations..."
    for i in 1 2 3 4 5; do
        time ./dsa_indexer ${B} ${S_q} ${S_kv} 2>&1 | grep -E "Duration|Time|us"
    done
fi

cd "${PROJECT_ROOT}"
echo "=== Benchmark Complete ==="
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
