#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Config ---
BATCH=4
SEQ_Q=256
SEQ_KV=256
SOLUTION_DIR="solution"
BUILD_DIR="${SOLUTION_DIR}/build"
PERF_SKILL=".claude/skills/ops-profiling/scripts/perf_summary.py"

# CANN environment
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export CMAKE_PREFIX_PATH="/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH:-}"

echo "============================================="
echo " attention_backward Benchmark"
echo " Shape: batch=${BATCH}, seq_q=${SEQ_Q}, seq_kv=${SEQ_KV}"
echo " Label: ${LABEL:-none}"
echo "============================================="

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
(
set -euo pipefail

# Step 1: Build
echo ""
echo "[1/5] Building..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. 2>&1 | tail -3
make -j$(nproc) 2>&1 | tail -5
cd ../..

# Step 2: Generate data
echo ""
echo "[2/5] Generating test data..."
cd "${BUILD_DIR}"
mkdir -p input output
python3 ../scripts/gen_data.py ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1 | tail -3
cd ../..

# Step 3: Run with msprof
echo ""
echo "[3/5] Running with msprof (warm-up=5, launch-count=3)..."
MSPROF_OUT="$(pwd)/msprof_output"
rm -rf "${MSPROF_OUT}"
mkdir -p "${MSPROF_OUT}" && chmod 700 "${MSPROF_OUT}"
cd "${BUILD_DIR}"
chmod 700 .
msprof op --warm-up=5 --launch-count=3 --output="${MSPROF_OUT}" ./attention_backward_custom ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1
cd ../..

# Step 4: Verify precision
echo ""
echo "[4/5] Verifying precision..."
cd "${BUILD_DIR}"
python3 ../scripts/verify_result.py ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1
cd ../..

# Step 5: Archive perf data
echo ""
echo "[5/5] Archiving performance data..."
OPPROF_DIR=$(ls -td msprof_output/OPPROF_* 2>/dev/null | head -1)
if [ -n "$OPPROF_DIR" ]; then
    python3 "${PERF_SKILL}" "$OPPROF_DIR" "${SOLUTION_DIR}" 2>&1
    # Print summary
    LATEST_ROUND=$(ls -td ${SOLUTION_DIR}/docs/perf/round_* 2>/dev/null | head -1)
    if [ -n "$LATEST_ROUND" ] && [ -f "$LATEST_ROUND/summary.txt" ]; then
        echo ""
        echo "=== Performance Summary ==="
        cat "$LATEST_ROUND/summary.txt"
    fi
fi

echo ""
echo "============================================="
echo " Benchmark Complete"
echo "============================================="
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
