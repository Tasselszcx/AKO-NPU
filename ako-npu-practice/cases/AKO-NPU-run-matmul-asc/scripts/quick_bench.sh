#!/bin/bash
# Quick benchmark without trajectory - just build, verify, and time
set -eo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:-quick}"

(
    set -euo pipefail
    cd solution
    [ -n "${ASCEND_HOME_PATH:-}" ] || { echo "ERROR: ASCEND_HOME_PATH not set" >&2; exit 1; }
    if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
        source "${ASCEND_HOME_PATH}/set_env.sh"
    elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source "/usr/local/Ascend/ascend-toolkit/set_env.sh"
    fi
    export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake:${CMAKE_PREFIX_PATH:-}"
    mkdir -p build && chmod 755 build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1 && make -j4 > /dev/null 2>&1
    chmod 755 .
    cd ..
    cd build
    python3 ../scripts/gen_data.py > /dev/null 2>&1
    rm -f output/output.bin
    ./demo > /dev/null 2>&1
    [ -f output/output.bin ] || { echo "FAIL: no output" >&2; exit 1; }
    echo "=== Verify ==="
    python3 ../scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | tail -3
    echo "=== msprof ==="
    MSPROF_DIR="/tmp/msprof_quick_$$"
    rm -rf "$MSPROF_DIR"
    mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_DIR" ./demo 2>&1 | grep -E "Task Duration|Op Name"
    OPPROF_DIR=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        cat "$OPPROF_DIR/OpBasicInfo.csv" | tail -1
    fi
    rm -rf "$MSPROF_DIR"
) 2>&1
