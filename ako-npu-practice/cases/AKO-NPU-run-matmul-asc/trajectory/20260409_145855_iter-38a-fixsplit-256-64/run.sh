#!/bin/bash
# ============================================================================
# Matmul LeakyRelu Kernel - Build, Run, Verify
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="demo"

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
fi

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] Setting CANN environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh failed"
elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    source "/usr/local/Ascend/ascend-toolkit/set_env.sh" || die "set_env.sh failed"
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build specified but build/${OP_NAME} not found"
    echo "=== [2/4] Skipping build ==="
else
    echo "=== [2/4] Building ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generating test data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/4] Running Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel run failed"
[ -f output/output.bin ] || die "output.bin not found after kernel run"

echo "=== Verifying correctness ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "Correctness verification failed"

echo "=== Done ==="
exit 0
