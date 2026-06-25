#!/bin/bash
# ============================================================================
# matmul_leakyrelu - Build, Generate Data, Run, Verify
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="matmul_leakyrelu_custom"

SKIP_BUILD=0
if [ "${1:-}" == "--skip-build" ]; then
    SKIP_BUILD=1
fi

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] CANN Environment ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH not set"
SET_ENV=""
for p in "${ASCEND_HOME_PATH}/set_env.sh" \
         "${ASCEND_HOME_PATH}/../set_env.sh" \
         "${ASCEND_HOME_PATH}/../../set_env.sh"; do
    if [ -f "$p" ]; then SET_ENV="$p"; break; fi
done
[ -n "$SET_ENV" ] && source "$SET_ENV"

if [ -z "${ASC_DIR:-}" ]; then
    ASCEND_BASE="$(cd "${ASCEND_HOME_PATH}/.." && pwd)"
    ASC_FOUND=$(find "${ASCEND_BASE}" -maxdepth 6 -name "ASCConfig.cmake" -path "*/ascendc_kernel_cmake/*" 2>/dev/null | head -1 || true)
    if [ -n "${ASC_FOUND}" ]; then
        export ASC_DIR="$(dirname "${ASC_FOUND}")"
    fi
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build but build/${OP_NAME} not found"
    echo "=== [2/4] Skip build (reuse existing) ==="
else
    echo "=== [2/4] Build ==="
    mkdir -p build
    cd build
    cmake ${ASC_DIR:+-DASC_DIR=$ASC_DIR} .. || die "cmake failed"
    make -j4 || die "make failed"
    cd ..
fi

echo "=== [3/4] Generate Test Data ==="
cd build
python3 ../scripts/gen_data.py || die "gen_data.py failed"

echo "=== [4/4] Run Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" || die "Kernel execution failed (exit code $?)"
[ -f output/output.bin ] || die "output.bin not found after execution"

echo "=== Verify Results ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "Verification failed"

echo "=== Done ==="
exit 0
