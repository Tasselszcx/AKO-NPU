#!/bin/bash
# Automated iteration runner
# Tests a single configuration change and records results
# Usage: bash scripts/auto_iterate.sh <iter_num> <description> [changes_file]
set -eo pipefail
cd "$(dirname "$0")/.."

ITER_NUM="${1:?Usage: $0 <iter_num> <description>}"
DESC="${2:?Usage: $0 <iter_num> <description>}"
CHANGES_FILE="${3:-}"  # Optional file with sed commands

# Apply changes if provided
if [ -n "$CHANGES_FILE" ] && [ -f "$CHANGES_FILE" ]; then
    while IFS= read -r cmd; do
        eval "$cmd"
    done < "$CHANGES_FILE"
fi

# Build
rm -rf solution/build
mkdir -p solution/build
cd solution/build
cmake -DASC_DIR=/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
if ! make -j > /dev/null 2>&1; then
    echo "ITER $ITER_NUM: BUILD FAIL — $DESC"
    exit 1
fi

# Generate data + run
python3 ../../scripts/gen_data.py > /dev/null 2>&1
if ! timeout 10 ./demo > /dev/null 2>&1; then
    echo "ITER $ITER_NUM: DEMO HANG — $DESC"
    exit 1
fi

# Verify
VERIFY=$(python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | tail -1)
if ! echo "$VERIFY" | grep -q "pass"; then
    echo "ITER $ITER_NUM: PRECISION FAIL — $DESC"
    exit 1
fi

# Profile
for i in 1 2; do timeout 5 ./demo > /dev/null 2>&1; done
MSPROF_OUT=$(mktemp -d /tmp/msprof_XXXXXX)
chmod 700 "$MSPROF_OUT"
DUR=$(timeout 90 msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_OUT" ./demo 2>&1 | grep "Task Duration" | sed 's/.*: //' | tr -d ' ')
rm -rf "$MSPROF_OUT"

if [ -z "$DUR" ]; then
    echo "ITER $ITER_NUM: MSPROF TIMEOUT — $DESC"
    exit 1
fi

BASELINE=226.09
SPEEDUP=$(python3 -c "print(f'{$BASELINE / $DUR:.2f}')" 2>/dev/null || echo "N/A")

echo "ITER $ITER_NUM: ${DUR} us (${SPEEDUP}x) — $DESC"
