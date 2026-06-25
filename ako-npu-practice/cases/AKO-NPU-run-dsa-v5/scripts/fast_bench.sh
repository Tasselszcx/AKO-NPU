#!/bin/bash
# Fast benchmark: compile + run + verify (no msprof, ~30s total)
# Usage: bash scripts/fast_bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
LABEL="${1:-fast}"

echo "=== Fast Bench: ${LABEL} ==="

# Build
cd "${PROJECT_ROOT}/solution"
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
if ! make -j4 2>&1 | tail -3; then
    echo "COMPILE FAILED"
    exit 1
fi

# Generate data
python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1

# Run
echo "--- Running kernel ---"
OUTPUT=$(./dsa_indexer 1 1 4096 2>&1)
echo "$OUTPUT" | grep -E "E2E|Timing|detail|Warmup|Run |Summary"

# Verify
echo "--- Correctness ---"
python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1

echo "=== Fast Bench Complete: ${LABEL} ==="
