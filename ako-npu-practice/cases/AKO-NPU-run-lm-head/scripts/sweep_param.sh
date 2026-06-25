#!/bin/bash
# Generic parameter sweep for lm_head_projection
# Usage: bash scripts/sweep_param.sh <param_name> <value1,value2,...> [workload_subset]
# Example: bash scripts/sweep_param.sh baseN 128,192,256,320,384,512
set -eo pipefail
cd "$(dirname "$0")/.."

PARAM_NAME="${1:?Usage: sweep_param.sh <param_name> <values>}"
VALUES_STR="${2:?Usage: sweep_param.sh <param_name> <values>}"
IFS=',' read -ra VALUES <<< "$VALUES_STR"

SOLUTION_DIR="solution"
BUILD_DIR="$SOLUTION_DIR/build"
ASC_FILE="$SOLUTION_DIR/lm_head_projection.asc"
BACKUP_FILE="${ASC_FILE}.sweep_backup"

cp "$ASC_FILE" "$BACKUP_FILE"

WORKLOADS=(
    "1 128 1 W1"
    "1 512 1 W2"
    "1 1024 128 W3"
    "4 256 1 W4"
)

RESULT_FILE="sweep_${PARAM_NAME}_results.csv"
echo "${PARAM_NAME},W1_us,W2_us,W3_us,W4_us,W1_pass,W2_pass,W3_pass,W4_pass" > "$RESULT_FILE"

for VAL in "${VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "Testing ${PARAM_NAME}=${VAL}"
    echo "========================================"

    # Restore from backup and apply parameter
    cp "$BACKUP_FILE" "$ASC_FILE"

    case "$PARAM_NAME" in
        baseN)
            # Replace baseN in both SetFixSplit calls
            sed -i "s/tilingApi.SetFixSplit(128, [0-9]*, -1)/tilingApi.SetFixSplit(128, $VAL, -1)/g" "$ASC_FILE"
            sed -i "s/tilingApi.SetFixSplit(-1, [0-9]*, -1)/tilingApi.SetFixSplit(-1, $VAL, -1)/g" "$ASC_FILE"
            ;;
        baseM_W3)
            # Replace baseM for M>=128 case only
            sed -i "s/tilingApi.SetFixSplit([0-9]*, 256, -1)/tilingApi.SetFixSplit($VAL, 256, -1)/g" "$ASC_FILE"
            ;;
        cores)
            sed -i "s/tilingApi.SetDim(ascendcPlatform->GetCoreNumAic())/tilingApi.SetDim($VAL)/g" "$ASC_FILE"
            ;;
        *)
            echo "Unknown param: $PARAM_NAME"
            cp "$BACKUP_FILE" "$ASC_FILE"
            rm "$BACKUP_FILE"
            exit 1
            ;;
    esac

    # Build
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    if ! cmake .. -DASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake 2>&1 | tail -1; then
        echo "CMAKE FAILED"
        echo "${VAL},FAIL,FAIL,FAIL,FAIL,0,0,0,0" >> "../../$RESULT_FILE"
        cd ../..
        continue
    fi
    if ! make -j 2>&1 | tail -3; then
        echo "MAKE FAILED"
        echo "${VAL},FAIL,FAIL,FAIL,FAIL,0,0,0,0" >> "../../$RESULT_FILE"
        cd ../..
        continue
    fi
    cd ../..

    RESULTS="$VAL"
    PASS_RESULTS=""

    for wl in "${WORKLOADS[@]}"; do
        read -r B S LTK WL_LABEL <<< "$wl"

        python3 scripts/gen_data.py "$B" "$S" "$LTK" --output-dir "$SOLUTION_DIR" 2>/dev/null
        rm -f "$SOLUTION_DIR/output/output.bin"

        WL_OUTPUT="$(realpath $SOLUTION_DIR)/msprof_sweep_$$_${WL_LABEL}"
        rm -rf "$WL_OUTPUT"
        mkdir -m 700 -p "$WL_OUTPUT"

        RUN_DIR="$SOLUTION_DIR/run_$$"
        rm -rf "$RUN_DIR"
        mkdir -m 700 -p "$RUN_DIR"
        cp "$BUILD_DIR/demo" "$RUN_DIR/"
        ln -sfn "$(realpath $SOLUTION_DIR/input)" "$RUN_DIR/input"
        ln -sfn "$(realpath $SOLUTION_DIR/output)" "$RUN_DIR/output"

        cd "$RUN_DIR"
        msprof op --warm-up=5 --launch-count=3 --output="$WL_OUTPUT" ./demo "$B" "$S" "$LTK" 2>&1 | tail -1
        cd - > /dev/null
        rm -rf "$RUN_DIR"

        # Verify
        PASS=1
        if [[ -f "$SOLUTION_DIR/output/output.bin" ]]; then
            if ! python3 scripts/verify_result.py "$SOLUTION_DIR/output/output.bin" "$SOLUTION_DIR/output/golden.bin" 2>&1 | tail -1; then
                PASS=0
            fi
        else
            PASS=0
        fi

        # Extract duration
        OPPROF_DIR=$(ls -td "$WL_OUTPUT"/OPPROF_* 2>/dev/null | head -1)
        TD="N/A"
        if [[ -n "$OPPROF_DIR" && -f "$OPPROF_DIR/OpBasicInfo.csv" ]]; then
            TD=$(python3 -c "
import csv
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row.get('Task Duration(us)', 'N/A'))
        break
" 2>/dev/null || echo "N/A")
        fi

        echo "  ${WL_LABEL}: ${TD}us (pass=${PASS})"
        RESULTS="${RESULTS},${TD}"
        PASS_RESULTS="${PASS_RESULTS},${PASS}"

        rm -rf "$WL_OUTPUT"
    done

    echo "${RESULTS}${PASS_RESULTS}" >> "$RESULT_FILE"
done

# Restore original
cp "$BACKUP_FILE" "$ASC_FILE"
rm "$BACKUP_FILE"

echo ""
echo "========================================"
echo "Sweep Results:"
column -t -s',' "$RESULT_FILE"
echo "========================================"
