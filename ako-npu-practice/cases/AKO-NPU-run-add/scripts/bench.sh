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
cd solution

# Set environment
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
elif [ -f "${ASCEND_HOME_PATH}/../set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/../set_env.sh"
fi

# Build
echo "=== Building ==="
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake 2>&1 | tail -3
make -j4 2>&1 | tail -5
echo "Build OK"
# Fix permissions for msprof
cd ..
find . -type d -exec chmod 755 {} + 2>/dev/null || true
find . -type f -exec chmod 644 {} + 2>/dev/null || true
chmod 755 build/add_custom 2>/dev/null || true
cd build

# Generate data
echo "=== Generating data ==="
python3 ../scripts/gen_data.py

# Run with msprof for profiling
echo "=== Running benchmark with msprof ==="
rm -f output/output.bin
msprof op --warm-up=10 --launch-count=10 --output=./msprof_output ./add_custom 2>&1

# Check output exists
[ -f output/output.bin ] || { echo "ERROR: output.bin not found"; exit 1; }

# Verify correctness
echo "=== Verifying correctness ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin

# Parse msprof results
echo "=== Performance Results ==="
OPPROF_DIR=$(ls -td ./msprof_output/OPPROF_* 2>/dev/null | head -1)
if [ -n "$OPPROF_DIR" ]; then
    echo "OPPROF dir: $OPPROF_DIR"
    if [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        echo "--- OpBasicInfo ---"
        cat "$OPPROF_DIR/OpBasicInfo.csv"
    fi
    if [ -f "$OPPROF_DIR/PipeUtilization.csv" ]; then
        echo "--- PipeUtilization ---"
        cat "$OPPROF_DIR/PipeUtilization.csv"
    fi
    if [ -f "$OPPROF_DIR/Memory.csv" ]; then
        echo "--- Memory ---"
        cat "$OPPROF_DIR/Memory.csv"
    fi
fi

echo "=== Benchmark complete ==="
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
