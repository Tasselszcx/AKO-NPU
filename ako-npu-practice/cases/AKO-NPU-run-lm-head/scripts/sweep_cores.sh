#!/bin/bash
# Sweep core count for all workloads
# Usage: bash scripts/sweep_cores.sh
set -eo pipefail
cd "$(dirname "$0")/.."

SOLUTION_DIR="solution"
BUILD_DIR="$SOLUTION_DIR/build"
BACKUP="$SOLUTION_DIR/lm_head_projection.asc.bak"
cp "$SOLUTION_DIR/lm_head_projection.asc" "$BACKUP"

echo "=== Core Count Sweep ==="
echo "| Cores | W1(us) | W2(us) | W3(us) | W4(us) |"
echo "|-------|--------|--------|--------|--------|"

for CORES in 1 2 4 6 8 10 12 16 20 24; do
    # Modify SetDim
    sed -i "s/tilingApi.SetDim(ascendcPlatform->GetCoreNumAic());/tilingApi.SetDim($CORES);/" "$SOLUTION_DIR/lm_head_projection.asc"

    # Build
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake > /dev/null 2>&1
    make -j$(nproc) > /dev/null 2>&1 || { echo "| $CORES | BUILD_FAIL | | | |"; cd - >/dev/null; cp "$BACKUP" "$SOLUTION_DIR/lm_head_projection.asc"; continue; }
    cd - > /dev/null
    chmod 700 "$BUILD_DIR"

    mkdir -p "$SOLUTION_DIR/input" "$SOLUTION_DIR/output"

    declare -A WL_PARAMS
    WL_PARAMS[W1]="1 128 1"
    WL_PARAMS[W2]="1 512 1"
    WL_PARAMS[W3]="1 1024 128"
    WL_PARAMS[W4]="4 256 1"

    RESULTS=""
    for WL in W1 W2 W3 W4; do
        read -r B S LTK <<< "${WL_PARAMS[$WL]}"
        python3 scripts/gen_data.py "$B" "$S" "$LTK" --output-dir "$SOLUTION_DIR" > /dev/null 2>&1

        RUN_DIR="$SOLUTION_DIR/run_$$"
        rm -rf "$RUN_DIR"
        mkdir -m 700 -p "$RUN_DIR"
        cp "$BUILD_DIR/demo" "$RUN_DIR/"
        ln -sfn "$(realpath $SOLUTION_DIR/input)" "$RUN_DIR/input"
        ln -sfn "$(realpath $SOLUTION_DIR/output)" "$RUN_DIR/output"
        rm -f "$SOLUTION_DIR/output/output.bin"

        MSPROF_OUT="$SOLUTION_DIR/sweep_msprof/cores_${CORES}/$WL"
        rm -rf "$MSPROF_OUT"
        mkdir -m 700 -p "$MSPROF_OUT"

        cd "$RUN_DIR"
        msprof op --warm-up=5 --output="$MSPROF_OUT" ./demo "$B" "$S" "$LTK" > /dev/null 2>&1
        cd - > /dev/null
        rm -rf "$RUN_DIR"

        OPPROF_DIR=$(ls -td "$MSPROF_OUT"/OPPROF_* 2>/dev/null | head -1)
        TD="N/A"
        if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
            TD=$(python3 -c "
import csv
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    for row in csv.DictReader(f):
        print(f\"{float(row.get('Task Duration(us)','0')):.1f}\"); break
" 2>/dev/null)
        fi

        # Quick correctness check
        if [ -f "$SOLUTION_DIR/output/output.bin" ]; then
            python3 scripts/verify_result.py "$SOLUTION_DIR/output/output.bin" "$SOLUTION_DIR/output/golden.bin" > /dev/null 2>&1 || TD="${TD}*"
        else
            TD="NO_OUT"
        fi

        RESULTS="$RESULTS $TD"
    done

    echo "| $CORES | $RESULTS |"

    # Restore
    cp "$BACKUP" "$SOLUTION_DIR/lm_head_projection.asc"
done

rm -f "$BACKUP"
echo "=== Done ==="
echo "* = precision failed"
