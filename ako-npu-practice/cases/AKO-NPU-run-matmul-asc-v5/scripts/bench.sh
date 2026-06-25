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

echo "=== Building ==="
cd solution
mkdir -p build && cd build
cmake -DASC_DIR=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake .. 2>&1
make -j 2>&1

echo "=== Generating test data ==="
python3 ../../scripts/gen_data.py 2>&1

echo "=== Running demo (correctness check) ==="
./demo 2>&1

echo "=== Verifying correctness ==="
python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1

echo "=== Running with msprof (warm-up=10, launch-count=5) ==="
MSPROF_OUT=$(mktemp -d /tmp/msprof_XXXXXX)
chmod 700 "$MSPROF_OUT"
msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_OUT" ./demo 2>&1
OPPROF_DIR=$(ls -td "$MSPROF_OUT"/OPPROF_* 2>/dev/null | head -1)

echo "=== Extracting performance ==="
if [ -n "$OPPROF_DIR" ]; then
    echo "OPPROF directory: $OPPROF_DIR"
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
    if [ -f "$OPPROF_DIR/ArithmeticUtilization.csv" ]; then
        echo "--- ArithmeticUtilization ---"
        cat "$OPPROF_DIR/ArithmeticUtilization.csv"
    fi
    # Copy msprof output to build dir for archival
    cp -r "$OPPROF_DIR" ./msprof_latest 2>/dev/null || true
fi
rm -rf "$MSPROF_OUT"

echo "=== BENCH DONE ==="
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
