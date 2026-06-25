#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
(
set -e

echo "=== [0/5] Setting up CANN environment ==="
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export ASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake

echo "=== [1/5] Compiling ==="
cd "$PROJECT_ROOT/solution"
mkdir -p build && cd build
cmake -DASC_DIR=$ASC_DIR .. 2>&1
make -j 2>&1

echo "=== [2/5] Generating test data ==="
python3 "$PROJECT_ROOT/scripts/gen_data.py" 2>&1

echo "=== [3/5] Running with msprof (warm-up=10, launch-count=5) ==="
# msprof requires strict directory permissions
MSPROF_OUT="$PROJECT_ROOT/msprof_output"
rm -rf "$MSPROF_OUT"
mkdir -p "$MSPROF_OUT"
chmod 750 "$MSPROF_OUT"
chmod 750 "$PROJECT_ROOT/solution/build"
msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_OUT" ./demo 2>&1

echo "=== [4/5] Verifying precision ==="
python3 "$PROJECT_ROOT/scripts/verify_result.py" output/output.bin output/golden.bin 2>&1

echo "=== [5/5] Profiling summary ==="
OPPROF_DIR=$(ls -td "$MSPROF_OUT"/OPPROF_* 2>/dev/null | head -1)
if [ -n "$OPPROF_DIR" ]; then
    echo "OPPROF dir: $OPPROF_DIR"
    for csv in OpBasicInfo PipeUtilization Memory ArithmeticUtilization ResourceConflictRatio L2Cache MemoryUB MemoryL0; do
        if [ -f "$OPPROF_DIR/${csv}.csv" ]; then
            echo "--- ${csv} ---"
            cat "$OPPROF_DIR/${csv}.csv"
        fi
    done
else
    echo "WARNING: No OPPROF directory found"
fi

echo "=== BENCH COMPLETE ==="
) 2>&1 | tee "$PROJECT_ROOT/_bench_output.txt"
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
