#!/bin/bash
# Parameter sweep script - tests a single configuration
# Usage: bash scripts/sweep_params.sh <config_name> <workload: W1|W2|W3|W4|ALL>
# Expects solution/ to already have modified code, just builds and runs
set -eo pipefail
cd "$(dirname "$0")/.."

CONFIG_NAME="${1:-test}"
WORKLOAD="${2:-ALL}"
SOLUTION_DIR="solution"
BUILD_DIR="$SOLUTION_DIR/build"

# Build
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake 2>&1 | tail -2
make -j$(nproc) 2>&1 | tail -2
cd - > /dev/null
chmod 700 "$BUILD_DIR"

declare -A WL_PARAMS
WL_PARAMS[W1]="1 128 1"
WL_PARAMS[W2]="1 512 1"
WL_PARAMS[W3]="1 1024 128"
WL_PARAMS[W4]="4 256 1"

if [ "$WORKLOAD" == "ALL" ]; then
    TARGETS="W1 W2 W3 W4"
else
    TARGETS="$WORKLOAD"
fi

mkdir -p "$SOLUTION_DIR/input" "$SOLUTION_DIR/output"

echo "CONFIG: $CONFIG_NAME"
echo "| WL | B | S | ltk | Duration(us) | Correct |"
echo "|----|---|---|-----|--------------| --------|"

for WL in $TARGETS; do
    read -r B S LTK <<< "${WL_PARAMS[$WL]}"

    # Gen data
    python3 scripts/gen_data.py "$B" "$S" "$LTK" --output-dir "$SOLUTION_DIR" > /dev/null 2>&1

    # Run dir
    RUN_DIR="$SOLUTION_DIR/run_$$"
    rm -rf "$RUN_DIR"
    mkdir -m 700 -p "$RUN_DIR"
    cp "$BUILD_DIR/demo" "$RUN_DIR/"
    ln -sfn "$(realpath $SOLUTION_DIR/input)" "$RUN_DIR/input"
    ln -sfn "$(realpath $SOLUTION_DIR/output)" "$RUN_DIR/output"
    rm -f "$SOLUTION_DIR/output/output.bin"

    # Run with msprof (single run for speed)
    MSPROF_OUT="$SOLUTION_DIR/sweep_msprof/$CONFIG_NAME/$WL"
    rm -rf "$MSPROF_OUT"
    mkdir -m 700 -p "$MSPROF_OUT"

    cd "$RUN_DIR"
    msprof op --warm-up=5 --output="$MSPROF_OUT" ./demo "$B" "$S" "$LTK" > /dev/null 2>&1
    cd - > /dev/null
    rm -rf "$RUN_DIR"

    # Get duration
    OPPROF_DIR=$(ls -td "$MSPROF_OUT"/OPPROF_* 2>/dev/null | head -1)
    TD="N/A"
    if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
        TD=$(python3 -c "
import csv
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    for row in csv.DictReader(f):
        print(row.get('Task Duration(us)','N/A')); break
" 2>/dev/null)
    fi

    # Check correctness
    CORRECT="FAIL"
    if [ -f "$SOLUTION_DIR/output/output.bin" ]; then
        python3 scripts/verify_result.py "$SOLUTION_DIR/output/output.bin" "$SOLUTION_DIR/output/golden.bin" > /dev/null 2>&1 && CORRECT="PASS"
    fi

    echo "| $WL | $B | $S | $LTK | $TD | $CORRECT |"
done
