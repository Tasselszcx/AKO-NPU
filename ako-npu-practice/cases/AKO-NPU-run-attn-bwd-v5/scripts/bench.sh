#!/bin/bash
# Benchmark script for attention backward kernel
# Usage: bash scripts/bench.sh [iter-label]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Use device 5 since device 4 may be corrupted
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-5}

LABEL="${1:-baseline}"

echo "========================================"
echo "Attention Backward Kernel Benchmark"
echo "Label: $LABEL"
echo "========================================"

python3 scripts/benchmark.py --label "$LABEL" 2>&1 | tee _bench_output.txt

echo ""
echo "Benchmark output saved to _bench_output.txt"
