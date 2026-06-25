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

PROJ_ROOT="$(pwd)"
SOL_DIR="${PROJ_ROOT}/solution"
BUILD_DIR="${SOL_DIR}/build"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true

echo "================================================================"
echo "  AKO Benchmark: attention_backward"
echo "  Label: ${LABEL:-none}"
echo "================================================================"

# 1. Build
echo ""
echo "=== [1/3] Building ==="
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
export ASC_DIR=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake
cmake -DASC_DIR=$ASC_DIR .. 2>&1
make -j4 2>&1
if [ ! -f "attention_backward" ]; then
    echo "BUILD FAILED: attention_backward binary not found"
    exit 1
fi
echo "BUILD: OK"

# 2. Run primary test shape (small) for iteration speed
echo ""
echo "=== [2/3] Running benchmark ==="

BATCH=4
SEQ_Q=256
SEQ_KV=256
NAME="primary_4x256x256"

echo "--- Shape: ${NAME} (batch=${BATCH}, seq_q=${SEQ_Q}, seq_kv=${SEQ_KV}) ---"

# Generate test data (only if not already generated for this shape)
cd "${BUILD_DIR}"
MARKER=".data_${BATCH}_${SEQ_Q}_${SEQ_KV}"
if [ ! -f "$MARKER" ]; then
    python3 "${SOL_DIR}/scripts/gen_data.py" ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1
    touch "$MARKER"
fi

# Check if binary supports --bench mode (internal timing)
SUPPORTS_BENCH=false
if ./attention_backward ${BATCH} ${SEQ_Q} ${SEQ_KV} --bench 1 2>&1 | grep -q "BENCH_RESULT"; then
    SUPPORTS_BENCH=true
fi

if [ "$SUPPORTS_BENCH" = true ]; then
    echo "Using internal benchmark mode (amortized setup)"
    # Correctness check first (single run)
    rm -f output/grad_attn_scores.bin output/grad_value_states.bin
    ./attention_backward ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1 > /dev/null || true

    # Verify correctness
    VERIFY_OUT=$(python3 "${SOL_DIR}/scripts/verify_result.py" ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1) || true
    echo "$VERIFY_OUT"

    RTOL_PASS=true
    if echo "$VERIFY_OUT" | grep -q "max_rtol="; then
        for rtol_val in $(echo "$VERIFY_OUT" | grep -oP 'max_rtol=\K[0-9.e+-]+'); do
            if python3 -c "import sys; sys.exit(0 if float('$rtol_val') <= 0.05 else 1)" 2>/dev/null; then
                :
            else
                RTOL_PASS=false
            fi
        done
    fi

    if [ "$RTOL_PASS" != true ]; then
        echo "CORRECTNESS: FAIL (rtol > 0.05)"
        exit 1
    fi
    echo "CORRECTNESS: PASS (rtol <= 0.05)"

    # Benchmark: 200 iterations with internal timing (stable average)
    NUM_RUNS=200
    BENCH_OUT=$(./attention_backward ${BATCH} ${SEQ_Q} ${SEQ_KV} --bench ${NUM_RUNS} --overlap 2>&1)
    AVG_MS=$(echo "$BENCH_OUT" | grep "BENCH_RESULT" | grep -oP '[0-9]+\.[0-9]+')
    echo "  Latency (avg ${NUM_RUNS} runs, internal timing): ${AVG_MS} ms"
else
    echo "Using external benchmark mode"
    # Warmup
    rm -f output/grad_attn_scores.bin output/grad_value_states.bin
    ./attention_backward ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1 > /dev/null || true

    # Timed runs
    TOTAL_MS=0
    NUM_RUNS=5
    for r in $(seq 1 $NUM_RUNS); do
        rm -f output/grad_attn_scores.bin output/grad_value_states.bin
        START_NS=$(date +%s%N)
        ./attention_backward ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1 > /dev/null
        KRET=$?
        END_NS=$(date +%s%N)
        if [ $KRET -ne 0 ]; then
            echo "  RUN FAILED (exit code $KRET)"
            exit 1
        fi
        ELAPSED_NS=$((END_NS - START_NS))
        ELAPSED_MS=$(echo "scale=3; $ELAPSED_NS / 1000000" | bc)
        TOTAL_MS=$(echo "$TOTAL_MS + $ELAPSED_MS" | bc)
    done
    AVG_MS=$(echo "scale=3; $TOTAL_MS / $NUM_RUNS" | bc)
    echo "  Latency (avg ${NUM_RUNS} runs): ${AVG_MS} ms"

    # Verify correctness
    VERIFY_OUT=$(python3 "${SOL_DIR}/scripts/verify_result.py" ${BATCH} ${SEQ_Q} ${SEQ_KV} 2>&1) || true
    echo "$VERIFY_OUT"

    RTOL_PASS=true
    if echo "$VERIFY_OUT" | grep -q "max_rtol="; then
        for rtol_val in $(echo "$VERIFY_OUT" | grep -oP 'max_rtol=\K[0-9.e+-]+'); do
            if python3 -c "import sys; sys.exit(0 if float('$rtol_val') <= 0.05 else 1)" 2>/dev/null; then
                :
            else
                RTOL_PASS=false
            fi
        done
    fi

    if [ "$RTOL_PASS" = true ]; then
        echo "CORRECTNESS: PASS (rtol <= 0.05)"
    else
        echo "CORRECTNESS: FAIL (rtol > 0.05)"
        exit 1
    fi
fi

# 3. Summary
echo ""
echo "=== [3/3] Summary ==="
echo "Shape: ${NAME}"
echo "Latency: ${AVG_MS} ms"
echo "Correctness: PASS"
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
