#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Default shape
BATCH=${1:-4}
SEQ_Q=${2:-256}
SEQ_KV=${3:-256}

echo "=== Testing attention_backward: batch=$BATCH, seq_q=$SEQ_Q, seq_kv=$SEQ_KV ==="

# Setup CANN env
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "${ASCEND_HOME_PATH}/set_env.sh" || true

# Build if needed
if [ ! -f "build/attention_backward" ]; then
    echo "=== Building ==="
    cd build
    export ASC_DIR=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake
    cmake -DASC_DIR=$ASC_DIR ..
    make -j4
    cd ..
fi

echo "=== Generating test data ==="
cd build
python3 ../scripts/gen_data.py $BATCH $SEQ_Q $SEQ_KV

echo "=== Running kernel ==="
rm -f output/grad_attn_scores.bin output/grad_value_states.bin
./attention_backward $BATCH $SEQ_Q $SEQ_KV

echo "=== Verifying results ==="
python3 ../scripts/verify_result.py $BATCH $SEQ_Q $SEQ_KV
