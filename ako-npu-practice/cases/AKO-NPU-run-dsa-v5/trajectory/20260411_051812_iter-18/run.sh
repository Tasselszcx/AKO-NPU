#!/bin/bash
# ============================================================================
# DSA Indexer -- Build, Run, and Verify
# Usage:
#   bash run.sh              # Full flow (compile + run + verify) for decode scenario
#   bash run.sh --skip-build # Skip compilation, reuse existing binary
#   bash run.sh B S_q S_kv   # Custom shape
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="dsa_indexer"
B=1
S_q=1
S_kv=4096
SKIP_BUILD=0

# Parse arguments
for arg in "$@"; do
    if [ "$arg" == "--skip-build" ]; then
        SKIP_BUILD=1
    fi
done

# If positional args (non-flag) are provided, use as B S_q S_kv
args=()
for arg in "$@"; do
    if [ "$arg" != "--skip-build" ]; then
        args+=("$arg")
    fi
done
if [ ${#args[@]} -ge 3 ]; then
    B=${args[0]}
    S_q=${args[1]}
    S_kv=${args[2]}
fi

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== DSA Indexer: B=${B}, S_q=${S_q}, S_kv=${S_kv} ==="

echo "=== [1/4] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || {
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    else
        die "ASCEND_HOME_PATH not set and set_env.sh not found"
    fi
}

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} not found"
    echo "=== [2/4] Skip build (reusing existing binary) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generate test data ==="
cd build
python3 ../scripts/gen_data.py ${B} ${S_q} ${S_kv} || die "gen_data.py failed"

echo "=== [4/4] Run kernel ==="
rm -f output/output_topk_indices.bin output/output_index_score.bin
./${OP_NAME} ${B} ${S_q} ${S_kv} || die "Kernel execution failed"
[ -f output/output_topk_indices.bin ] || die "output_topk_indices.bin not produced"
[ -f output/output_index_score.bin ] || die "output_index_score.bin not produced"

echo "=== Verify results ==="
python3 ../scripts/verify_result.py ${B} ${S_q} ${S_kv} || die "Verification failed"

echo "=== Done ==="
exit 0
