#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
set +e
(
set -eo pipefail
cd solution

echo "============================================================"
echo "  AKO-NPU Benchmark: attention_backward"
echo "  Label: ${LABEL:-<none>}"
echo "  Time: $(date)"
echo "============================================================"

OP_NAME="attention_backward"

# --- Environment ---
echo ""
echo "=== [1/4] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }
source "${ASCEND_HOME_PATH}/../set_env.sh" 2>/dev/null || source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true

# --- Build ---
echo ""
echo "=== [2/4] Build ==="
mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake .. 2>&1 || { echo "ERROR: cmake failed"; exit 1; }
make -j4 2>&1 || { echo "ERROR: make failed"; exit 1; }
echo "Build OK"

# --- Test shapes ---
echo ""
echo "=== [3/4] Correctness verification (all shapes) ==="

SHAPES=(
    "4 256 256"
    "8 373 449"
    "4 1024 2048"
)
SHAPE_NAMES=(
    "small_aligned"
    "medium_unaligned"
    "large_long_seq"
)

ALL_CORRECT=true

for i in "${!SHAPES[@]}"; do
    read -r BATCH SEQ_Q SEQ_KV <<< "${SHAPES[$i]}"
    SNAME="${SHAPE_NAMES[$i]}"
    echo ""
    echo "--- Shape: ${SNAME} (B=${BATCH}, sq=${SEQ_Q}, skv=${SEQ_KV}) ---"

    python3 ../gen_data.py ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1 || { echo "ERROR: gen_data.py failed for ${SNAME}"; ALL_CORRECT=false; continue; }

    rm -f output/grad_attn_scores.bin output/grad_value_states.bin
    ./${OP_NAME} ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1 || { echo "ERROR: kernel failed for ${SNAME}"; ALL_CORRECT=false; continue; }

    python3 ../verify_result.py 2>&1 || { echo "ERROR: precision failed for ${SNAME}"; ALL_CORRECT=false; continue; }
    echo "Shape ${SNAME}: PASSED"
done

if [ "$ALL_CORRECT" = false ]; then
    echo ""
    echo "ERROR: Some shapes failed precision verification"
fi

# --- Performance measurement ---
echo ""
echo "=== [4/4] Performance measurement ==="
echo ""
echo "Performance is measured via ACL event timing built into the kernel (warmup=5, runs=10)."
echo "The PERF line in each shape's output above shows: avg, min, max (ms)."

echo ""
echo "=== Benchmark Summary ==="
echo "All shapes correct: $ALL_CORRECT"
echo ""

if [ "$ALL_CORRECT" = false ]; then
    exit 1
fi
) 2>&1 | tee _bench_output.txt
BENCH_EXIT=${PIPESTATUS[0]}
set -e
# --- End bench command ---

# --- Trajectory (disabled to save disk) ---
# if [ -n "$LABEL" ]; then
#     TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
# else
#     TRAJ_DIR="trajectory/${TIMESTAMP}"
# fi
# mkdir -p "$TRAJ_DIR"
# cp -r solution/* "$TRAJ_DIR/"
# [ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
# echo "Trajectory saved to: $TRAJ_DIR"
rm -f _bench_output.txt

exit $BENCH_EXIT
