#!/bin/bash
# Quick test: build, run, verify, time - with timeout
set -eo pipefail
cd "$(dirname "$0")/.."
LABEL="${1:-test}"
TIMEOUT="${2:-120}"

timeout_handler() {
    echo "TIMEOUT after ${TIMEOUT}s"
    pkill -f "./demo" 2>/dev/null
    pkill -f "msprof" 2>/dev/null
    exit 1
}

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

    # Build
    mkdir -p build && chmod 755 build && cd build
    if ! cmake .. > /dev/null 2>&1; then echo "CMAKE_FAIL"; exit 1; fi
    if ! make -j4 > /dev/null 2>&1; then echo "COMPILE_FAIL"; exit 1; fi
    chmod 755 .
    cd ..
    cd build
    python3 ../scripts/gen_data.py > /dev/null 2>&1

    # Run with timeout
    rm -f output/output.bin
    timeout 30 ./demo > /dev/null 2>&1 || { echo "RUN_FAIL_OR_TIMEOUT"; exit 1; }
    [ -f output/output.bin ] || { echo "NO_OUTPUT"; exit 1; }

    # Verify
    VERIFY=$(python3 ../scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | tail -2)
    if echo "$VERIFY" | grep -q "test pass"; then
        echo "CORRECT"
    else
        echo "INCORRECT"
        echo "$VERIFY"
        exit 1
    fi

    # Profile with timeout
    MSPROF_DIR="/tmp/msprof_sweep_$$"
    rm -rf "$MSPROF_DIR"
    mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    timeout 60 msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_DIR" ./demo 2>&1 | grep "Task Duration" || echo "MSPROF_FAIL"

    OPPROF_DIR=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        echo "RESULT: $(tail -1 "$OPPROF_DIR/OpBasicInfo.csv")"
    fi
    rm -rf "$MSPROF_DIR"
) 2>&1
