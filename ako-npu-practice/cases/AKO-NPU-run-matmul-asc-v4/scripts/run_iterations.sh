#!/bin/bash
# Quick bench helper: compile + run + extract duration + precision
# Usage: source this file, then call quick_bench
# Returns: sets BENCH_DURATION and BENCH_PRECISION globals

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASC_FILE="$PROJECT_ROOT/solution/matmul_leakyrelu.asc"
BUILD_DIR="$PROJECT_ROOT/solution/build"
ASC_DIR="/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"

quick_bench() {
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    local label="${1:-test}"

    # Build
    cd "$BUILD_DIR"
    cmake -DASC_DIR=$ASC_DIR .. >/dev/null 2>&1
    make -j 2>&1 | tail -3
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "BUILD_FAILED"
        BENCH_DURATION="BUILD_FAILED"
        BENCH_PRECISION="N/A"
        cd "$PROJECT_ROOT"
        return 1
    fi

    # Gen data
    python3 "$PROJECT_ROOT/scripts/gen_data.py" 2>/dev/null

    # Run msprof
    local MSPROF_OUT="$PROJECT_ROOT/msprof_quick"
    rm -rf "$MSPROF_OUT"
    mkdir -p "$MSPROF_OUT" && chmod 750 "$MSPROF_OUT" && chmod 750 "$BUILD_DIR"
    local output=$(msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_OUT" ./demo 2>&1)

    # Extract duration
    BENCH_DURATION=$(echo "$output" | grep -oP 'Task Duration\(us\):\s*\K[\d.]+')

    # Check precision
    local prec_output=$(python3 "$PROJECT_ROOT/scripts/verify_result.py" output/output.bin output/golden.bin 2>&1)
    if echo "$prec_output" | grep -qi "test pass"; then
        BENCH_PRECISION="PASS"
    else
        BENCH_PRECISION="FAIL"
    fi

    echo "Duration: ${BENCH_DURATION}us, Precision: ${BENCH_PRECISION}"
    cd "$PROJECT_ROOT"
    return 0
}
