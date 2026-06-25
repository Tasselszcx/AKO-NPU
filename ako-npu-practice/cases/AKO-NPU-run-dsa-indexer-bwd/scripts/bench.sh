#!/bin/bash
# DSA Indexer Backward - Benchmark wrapper
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LABEL="${1:-baseline}"

cd "$PROJECT_DIR"

python3 scripts/bench.py "$LABEL" 2>&1 | tee _bench_output.txt

# torch_npu may segfault on cleanup; check output for actual pass/fail
if grep -q "CORRECTNESS: PASS" _bench_output.txt; then
    exit 0
elif grep -q "CORRECTNESS: FAIL" _bench_output.txt; then
    exit 1
else
    # Script didn't reach summary - real failure
    exit 1
fi
