#!/bin/bash
# Sweep CHUNK_SIZE for LeakyRelu
set -eo pipefail
cd "$(dirname "$0")/.."

# Backup
cp solution/matmul_leakyrelu.asc solution/matmul_leakyrelu.asc.sweep_bak

for CHUNK in 8192 10240 14336 6144 4096; do
    echo "=============================================="
    echo "Testing CHUNK_SIZE = $CHUNK"
    echo "=============================================="

    # Restore from backup and apply change
    cp solution/matmul_leakyrelu.asc.sweep_bak solution/matmul_leakyrelu.asc
    sed -i "s/const uint32_t CHUNK_SIZE = [0-9]*/const uint32_t CHUNK_SIZE = ${CHUNK}/" solution/matmul_leakyrelu.asc

    rm -rf solution/build

    # Build & run
    set +e
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
        cmake .. 2>&1 | tail -3
        make -j4 2>&1 | tail -5
        chmod 755 .
        cd ..

        cd build
        python3 ../scripts/gen_data.py 2>&1 | tail -1
        rm -f output/output.bin
        ./demo 2>&1 | tail -3
        [ -f output/output.bin ] || { echo "ERROR: output.bin not found" >&2; exit 1; }

        python3 ../scripts/verify_result.py output/output.bin output/golden.bin

        MSPROF_DIR="/tmp/msprof_chunk_${CHUNK}_$$"
        rm -rf "$MSPROF_DIR"
        mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
        msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_DIR" ./demo 2>&1 | grep -E "Task Duration|Op Name"

        OPPROF_DIR=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
        if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
            echo "--- OpBasicInfo ---"
            cat "$OPPROF_DIR/OpBasicInfo.csv"
        fi
        rm -rf "$MSPROF_DIR"
    ) 2>&1
    RC=$?
    set -e
    if [ $RC -ne 0 ]; then
        echo "FAILED for CHUNK=$CHUNK (exit code $RC)"
    fi
    echo ""
done

# Restore
cp solution/matmul_leakyrelu.asc.sweep_bak solution/matmul_leakyrelu.asc
rm solution/matmul_leakyrelu.asc.sweep_bak
echo "=== SWEEP DONE ==="
