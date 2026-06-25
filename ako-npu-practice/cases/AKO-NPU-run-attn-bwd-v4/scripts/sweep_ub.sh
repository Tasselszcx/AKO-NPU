#!/bin/bash
# Parameter sweep: UB_BUF_SIZE
set -eo pipefail
cd "$(dirname "$0")/.."

KERNEL="solution/attention_backward.asc"
BUILD_DIR="solution/build"

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set"; exit 1; }
source "${ASCEND_HOME_PATH}/../set_env.sh" 2>/dev/null || source "${ASCEND_HOME_PATH}/set_env.sh" 2>/dev/null || true

# Generate data for small shape first
cd solution/build
python3 ../../solution/gen_data.py 4 256 256 2>/dev/null

echo "UB_KB,small_ms,medium_ms,large_ms"

for UB_KB in 16 24 32 48 64 96 128 160; do
    UB_BYTES=$((UB_KB * 1024))
    # Patch the kernel
    cd /home/hadoop-scale-llm/kernel/autoresearch/AKO-NPU-run-attn-bwd-v4
    sed -i "s|constexpr uint32_t UB_BUF_SIZE = [0-9]* \* 1024;|constexpr uint32_t UB_BUF_SIZE = ${UB_BYTES};|" "$KERNEL"
    # Rewrite to use raw bytes value
    sed -i "s|constexpr uint32_t UB_BUF_SIZE = ${UB_BYTES};|constexpr uint32_t UB_BUF_SIZE = ${UB_KB} * 1024; // ${UB_KB}KB sweep|" "$KERNEL"

    cd "$BUILD_DIR"
    make -j4 2>&1 | tail -1

    # Test small shape
    python3 ../../solution/gen_data.py 4 256 256 2>/dev/null
    SMALL=$(./attention_backward 4 256 256 2>&1 | grep "^PERF:" | sed 's/.*avg=\([0-9.]*\).*/\1/')

    # Test medium shape
    python3 ../../solution/gen_data.py 8 373 449 2>/dev/null
    MED=$(./attention_backward 8 373 449 2>&1 | grep "^PERF:" | sed 's/.*avg=\([0-9.]*\).*/\1/')

    # Test large shape
    python3 ../../solution/gen_data.py 4 1024 2048 2>/dev/null
    LARGE=$(./attention_backward 4 1024 2048 2>&1 | grep "^PERF:" | sed 's/.*avg=\([0-9.]*\).*/\1/')

    echo "${UB_KB},${SMALL},${MED},${LARGE}"
done

# Restore to best value (will be determined by results)
