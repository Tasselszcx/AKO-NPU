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
set -euo pipefail
SOLUTION_DIR="solution"
BUILD_DIR="${SOLUTION_DIR}/build"
OP_NAME="vae_residual_block"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }

echo "=== [1/5] Build ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. 2>&1
make -j4 2>&1
cd ../..

echo "=== [2/5] Generate test data ==="
cd "${BUILD_DIR}"
python3 ../../scripts/gen_data.py 2>&1

echo "=== [3/5] Run correctness check ==="
rm -f output/output.bin
./${OP_NAME} 2>&1
[ -f output/output.bin ] || { echo "ERROR: output.bin not found"; exit 1; }
python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1
cd ../..

echo "=== [4/5] msprof performance profiling ==="
cd "${BUILD_DIR}"
MSPROF_OUT="/tmp/msprof_vae_$$"
rm -rf "$MSPROF_OUT"
mkdir -p "$MSPROF_OUT"
chmod 700 "$MSPROF_OUT"
msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_OUT" ./${OP_NAME} 2>&1
# Copy results back
rm -rf msprof_output
cp -r "$MSPROF_OUT" msprof_output 2>/dev/null || true
cd ../..

echo "=== [5/5] Performance summary ==="
OPPROF_DIR=$(ls -td ${BUILD_DIR}/msprof_output/OPPROF_* 2>/dev/null | head -1)
if [ -n "$OPPROF_DIR" ]; then
    echo "OPPROF directory: $OPPROF_DIR"
    # Print task duration summary from CSV
    CSV_DIR=$(find "$OPPROF_DIR" -name "TaskTimeInfo.csv" -path "*/op/*.csv" | head -1)
    if [ -n "$CSV_DIR" ]; then
        echo "--- TaskTimeInfo ---"
        head -5 "$CSV_DIR"
    fi
    # Also check all CSVs
    echo "--- Available CSVs ---"
    find "$OPPROF_DIR" -name "*.csv" -path "*/op/*.csv" 2>/dev/null
fi

echo "=== BENCH COMPLETE ==="
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
