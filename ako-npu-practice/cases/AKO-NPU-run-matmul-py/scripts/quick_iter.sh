#!/bin/bash
# Quick iteration test: build, verify correctness, profile
# Usage: bash scripts/quick_iter.sh <iter_num> "<description>"
# Assumes solution/ source is already modified
set -o pipefail
cd "$(dirname "$0")"/..

ITER=${1:-0}
DESC=${2:-unknown}
BASELINE=228.37

cd solution/build 2>/dev/null || { mkdir -p solution/build && cd solution/build; }

# Find ASC cmake
ASC_DIR=$(find "${ASCEND_HOME_PATH}" "$(dirname ${ASCEND_HOME_PATH})" -name "ASCConfig.cmake" -print -quit 2>/dev/null | xargs dirname 2>/dev/null)

# Build
rm -rf CMakeCache.txt CMakeFiles 2>/dev/null
if ! cmake -DASC_DIR="$ASC_DIR" .. > /dev/null 2>&1; then
    echo "ITER $ITER | $DESC | COMPILE_FAIL"
    exit 1
fi
if ! make -j4 > /dev/null 2>&1; then
    echo "ITER $ITER | $DESC | COMPILE_FAIL"
    exit 1
fi

# Gen data + run
python3 ../gen_data.py > /dev/null 2>&1
rm -f output/output.bin
./matmul_leakyrelu_custom > /dev/null 2>&1

if [ ! -f output/output.bin ]; then
    echo "ITER $ITER | $DESC | RUNTIME_FAIL"
    exit 1
fi

# Verify
CORRECT=$(python3 ../verify_result.py output/output.bin output/golden.bin 2>&1 | grep "Overall:" | awk '{print $2}')
if [ "$CORRECT" != "PASS" ]; then
    echo "ITER $ITER | $DESC | INCORRECT"
    exit 1
fi

# Profile
chmod 755 . 2>/dev/null || true
rm -rf msprof_output
TD=$(msprof op --warm-up=10 --output=./msprof_output ./matmul_leakyrelu_custom 2>&1 | grep "Task Duration" | grep -o '[0-9.]*')

if [ -n "$TD" ]; then
    SP=$(python3 -c "print(f'{$BASELINE/$TD:.3f}')")
    echo "ITER $ITER | $DESC | PASS | ${TD} us | ${SP}x"
else
    echo "ITER $ITER | $DESC | PROFILE_FAIL"
fi
