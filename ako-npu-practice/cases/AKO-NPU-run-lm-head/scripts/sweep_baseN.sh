#!/bin/bash
# Sweep SetFixSplit baseN values for all workloads
# Tests baseN = 128, 192, 256, 320, 384, 512
set -eo pipefail
cd "$(dirname "$0")/.."

SOLUTION_DIR="solution"
BUILD_DIR="$SOLUTION_DIR/build"
ASC_FILE="$SOLUTION_DIR/lm_head_projection.asc"
BACKUP_FILE="$SOLUTION_DIR/lm_head_projection.asc.backup"

# Backup original
cp "$ASC_FILE" "$BACKUP_FILE"

BASE_N_VALUES=(128 192 256 320 384 512)

WORKLOADS=(
    "1 128 1 W1"
    "1 512 1 W2"
    "1 1024 128 W3"
    "4 256 1 W4"
)

echo "baseN,W1_us,W2_us,W3_us,W4_us" > sweep_baseN_results.csv

for BN in "${BASE_N_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing baseN=$BN"
    echo "========================================"

    # Modify the .asc file - replace baseN in SetFixSplit calls
    cp "$BACKUP_FILE" "$ASC_FILE"
    # Replace the SetFixSplit lines
    sed -i "s/tilingApi.SetFixSplit(128, [0-9]*, -1)/tilingApi.SetFixSplit(128, $BN, -1)/g" "$ASC_FILE"
    sed -i "s/tilingApi.SetFixSplit(-1, [0-9]*, -1)/tilingApi.SetFixSplit(-1, $BN, -1)/g" "$ASC_FILE"

    # Build
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    cmake .. -DASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake 2>&1 | tail -1
    make -j 2>&1 | tail -3
    cd - > /dev/null

    RESULTS="$BN"
    ALL_PASS=true

    for wl in "${WORKLOADS[@]}"; do
        read -r B S LTK WL_LABEL <<< "$wl"
        echo "  Running $WL_LABEL: B=$B S=$S ltk=$LTK"

        # Gen data
        python3 scripts/gen_data.py "$B" "$S" "$LTK" --output-dir "$SOLUTION_DIR" 2>/dev/null

        # Run with msprof
        rm -f "$SOLUTION_DIR/output/output.bin"
        WL_OUTPUT="$(realpath $SOLUTION_DIR)/msprof_sweep_${BN}_${WL_LABEL}"
        rm -rf "$WL_OUTPUT"
        mkdir -m 700 -p "$WL_OUTPUT"

        RUN_DIR="$SOLUTION_DIR/run_$$"
        rm -rf "$RUN_DIR"
        mkdir -m 700 -p "$RUN_DIR"
        cp "$BUILD_DIR/demo" "$RUN_DIR/"
        ln -sfn "$(realpath $SOLUTION_DIR/input)" "$RUN_DIR/input"
        ln -sfn "$(realpath $SOLUTION_DIR/output)" "$RUN_DIR/output"

        cd "$RUN_DIR"
        msprof op --warm-up=5 --launch-count=3 --output="$WL_OUTPUT" ./demo "$B" "$S" "$LTK" 2>&1 | tail -3
        cd - > /dev/null
        rm -rf "$RUN_DIR"

        # Verify
        if [[ -f "$SOLUTION_DIR/output/output.bin" ]]; then
            python3 scripts/verify_result.py "$SOLUTION_DIR/output/output.bin" "$SOLUTION_DIR/output/golden.bin" 2>&1 | tail -1 || {
                echo "  [FAIL] Precision check failed for $WL_LABEL with baseN=$BN!"
                ALL_PASS=false
            }
        else
            echo "  [FAIL] No output for $WL_LABEL!"
            ALL_PASS=false
        fi

        # Extract duration
        OPPROF_DIR=$(ls -td "$WL_OUTPUT"/OPPROF_* 2>/dev/null | head -1)
        if [[ -n "$OPPROF_DIR" && -f "$OPPROF_DIR/OpBasicInfo.csv" ]]; then
            TD=$(python3 -c "
import csv
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row.get('Task Duration(us)', 'N/A'))
        break
" 2>/dev/null || echo "N/A")
            echo "  $WL_LABEL Task Duration: ${TD}us"
            RESULTS="$RESULTS,$TD"
        else
            RESULTS="$RESULTS,N/A"
        fi

        rm -rf "$WL_OUTPUT"
    done

    echo "$RESULTS" >> sweep_baseN_results.csv
    if ! $ALL_PASS; then
        echo "  *** PRECISION FAILURE with baseN=$BN ***"
    fi
done

# Restore original
cp "$BACKUP_FILE" "$ASC_FILE"
rm "$BACKUP_FILE"

# Clean up msprof outputs
rm -rf "$SOLUTION_DIR"/msprof_sweep_*

echo ""
echo "========================================"
echo "Sweep Results:"
cat sweep_baseN_results.csv
echo "========================================"
