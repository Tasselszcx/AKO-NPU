#!/bin/bash
# ============================================================================
# attention_backward: Build, generate data, run kernel, verify precision
# Usage:
#   bash run.sh [batch seq_q seq_kv]
#   bash run.sh --skip-build [batch seq_q seq_kv]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="attention_backward"
BATCH=${1:-4}
SEQ_Q=${2:-256}
SEQ_KV=${3:-256}

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
    BATCH=${2:-4}
    SEQ_Q=${3:-256}
    SEQ_KV=${4:-256}
fi

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
source "${ASCEND_HOME_PATH}/../set_env.sh" 2>/dev/null || source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [2/4] Skip build (reuse existing) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake -DCMAKE_PREFIX_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generate test data (B=${BATCH}, sq=${SEQ_Q}, skv=${SEQ_KV}) ==="
cd build
python3 ../scripts/gen_data.py ${BATCH} ${SEQ_Q} ${SEQ_KV} || die "gen_data.py failed"

echo "=== [4/4] Run kernel ==="
rm -f output/grad_attn_scores.bin output/grad_value_states.bin
./${OP_NAME} ${BATCH} ${SEQ_Q} ${SEQ_KV} || die "Kernel execution failed"
[ -f output/grad_attn_scores.bin ] || die "grad_attn_scores.bin not found"
[ -f output/grad_value_states.bin ] || die "grad_value_states.bin not found"

echo "=== Precision verification ==="
python3 ../scripts/verify_result.py || die "Precision verification failed"

echo "=== DONE ==="
exit 0
