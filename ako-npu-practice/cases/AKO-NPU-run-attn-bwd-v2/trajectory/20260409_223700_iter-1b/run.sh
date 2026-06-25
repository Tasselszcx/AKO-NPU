#!/bin/bash
# ============================================================================
# attention_backward - build, generate data, run, verify
# Usage:
#   bash run.sh [batch_size seq_q seq_kv]    # full pipeline
#   bash run.sh --skip-build [batch seq_q seq_kv]  # skip compile
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="attention_backward_custom"

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
    shift
fi

# Default test shape
BATCH=${1:-4}
SEQ_Q=${2:-256}
SEQ_KV=${3:-256}

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/5] Setting CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} missing"
    echo "=== [2/5] Skipping build ==="
else
    echo "=== [2/5] Building ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/5] Generating test data (batch=${BATCH}, sq=${SEQ_Q}, skv=${SEQ_KV}) ==="
cd build
python3 ../scripts/gen_data.py ${BATCH} ${SEQ_Q} ${SEQ_KV} || die "gen_data.py failed"

echo "=== [4/5] Running kernel ==="
rm -f output/grad_attn_scores.bin output/grad_value_states.bin
"./${OP_NAME}" ${BATCH} ${SEQ_Q} ${SEQ_KV} || die "Kernel run failed"
[ -f output/grad_attn_scores.bin ] || die "grad_attn_scores.bin not generated"
[ -f output/grad_value_states.bin ] || die "grad_value_states.bin not generated"

echo "=== [5/5] Precision verification ==="
python3 ../scripts/verify_result.py ${BATCH} ${SEQ_Q} ${SEQ_KV} \
    || die "Precision verification failed"

echo "=== Done ==="
exit 0
