#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="vae_residual_block"

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
fi

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setup CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
# source env if set_env.sh exists
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" || true
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [2/4] Skipping build ==="
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
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/4] Run Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel execution failed (exit code $?)"
[ -f output/output.bin ] || die "output.bin not found after kernel execution"

echo "=== Accuracy verification ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin     || die "Accuracy verification failed"

echo "=== DONE ==="
exit 0
