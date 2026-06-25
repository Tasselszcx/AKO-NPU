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

    # Set CANN environment
    [ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set" >&2; exit 1; }
    # Source set_env.sh if it exists; environment may already be configured
    if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
        source "${ASCEND_HOME_PATH}/set_env.sh"
    elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    fi

    # Build
    echo "=== Building ==="
    export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH:-}"
    mkdir -p build && chmod 755 build && cd build
    cmake .. && make -j4
    chmod 755 .
    cd ..

    # Generate test data
    echo "=== Generating test data ==="
    cd build
    python3 ../scripts/gen_data.py

    # Run and verify correctness
    echo "=== Running kernel ==="
    rm -f output/output.bin
    ./demo
    [ -f output/output.bin ] || { echo "ERROR: output.bin not found" >&2; exit 1; }

    echo "=== Verifying correctness ==="
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin

    # Performance measurement with msprof
    echo "=== Performance measurement (msprof) ==="
    MSPROF_DIR="/tmp/msprof_matmul_$$"
    rm -rf "$MSPROF_DIR"
    mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_DIR" ./demo

    echo "=== msprof results ==="
    OPPROF_DIR=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        echo "--- OpBasicInfo ---"
        cat "$OPPROF_DIR/OpBasicInfo.csv"
        echo ""
        echo "--- PipeUtilization ---"
        cat "$OPPROF_DIR/PipeUtilization.csv"
    else
        echo "WARNING: msprof output not found"
    fi
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
