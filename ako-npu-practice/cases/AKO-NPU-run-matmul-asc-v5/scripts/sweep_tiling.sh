#!/bin/bash
# Parameter sweep for MatmulLeakyRelu tiling (baseM, baseN) on Ascend910B.
# Produces a CSV with columns: baseM, baseN, task_duration_us, correct
#
# Usage: bash scripts/sweep_tiling.sh [output_csv]
#   output_csv  - path for results CSV (default: sweep_results_<timestamp>.csv)
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOLUTION_DIR="$PROJECT_ROOT/solution"
ASC_FILE="$SOLUTION_DIR/matmul_leakyrelu.asc"
ASC_BACKUP="$SOLUTION_DIR/matmul_leakyrelu.asc.backup"
BUILD_DIR="$SOLUTION_DIR/build"

CMAKE_ASC_DIR="/usr/local/Ascend/ascend-toolkit/8.3.RC1/x86_64-linux/tikcpp/ascendc_kernel_cmake"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_CSV="${1:-$PROJECT_ROOT/sweep_results_${TIMESTAMP}.csv}"

# M=1024, N=640, K=256
# baseM must divide 1024 and be a multiple of 16
# baseN must divide 640  and be a multiple of 16
BASEM_VALUES=(64 128 256 512)
BASEN_VALUES=(64 128 160 320 640)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
cleanup() {
    # Restore original .asc file on exit (normal or error)
    if [ -f "$ASC_BACKUP" ]; then
        cp "$ASC_BACKUP" "$ASC_FILE"
        rm -f "$ASC_BACKUP"
        echo "[sweep] Restored original matmul_leakyrelu.asc"
    fi
}
trap cleanup EXIT

extract_task_duration() {
    # Extract Task Duration (us) from msprof OpBasicInfo.csv
    # The CSV has a header row; we look for the column "Task Duration(us)"
    # and extract the value from the data row(s). If multiple rows, take the
    # first non-header row (there is typically one op).
    local opbasic="$1"
    if [ ! -f "$opbasic" ]; then
        echo ""
        return
    fi

    # Find column index for "Task Duration(us)"
    local header
    header=$(head -1 "$opbasic")
    local col_idx=""
    local idx=1
    while IFS=',' read -r field; do
        # Trim whitespace
        field=$(echo "$field" | xargs)
        if [[ "$field" == *"Task Duration"* ]]; then
            col_idx=$idx
            break
        fi
        idx=$((idx + 1))
    done <<< "$(echo "$header" | tr ',' '\n')"

    if [ -z "$col_idx" ]; then
        echo ""
        return
    fi

    # Get value from second line (first data row)
    local value
    value=$(sed -n '2p' "$opbasic" | cut -d',' -f"$col_idx" | xargs)
    echo "$value"
}

# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------
if [ ! -f "$ASC_FILE" ]; then
    echo "[ERROR] Kernel file not found: $ASC_FILE"
    exit 1
fi

# Back up the original .asc file
cp "$ASC_FILE" "$ASC_BACKUP"

# Write CSV header
echo "baseM,baseN,task_duration_us,correct" > "$OUTPUT_CSV"
echo "[sweep] Results will be written to: $OUTPUT_CSV"
echo "[sweep] baseM values: ${BASEM_VALUES[*]}"
echo "[sweep] baseN values: ${BASEN_VALUES[*]}"
echo "[sweep] Total configurations: $(( ${#BASEM_VALUES[@]} * ${#BASEN_VALUES[@]} ))"
echo ""

config_num=0
total_configs=$(( ${#BASEM_VALUES[@]} * ${#BASEN_VALUES[@]} ))

for bm in "${BASEM_VALUES[@]}"; do
    for bn in "${BASEN_VALUES[@]}"; do
        config_num=$((config_num + 1))
        echo "================================================================"
        echo "[sweep] Config $config_num/$total_configs: baseM=$bm, baseN=$bn"
        echo "================================================================"

        # --- Patch the .asc file ---
        # Restore from backup first (so we always patch from the original)
        cp "$ASC_BACKUP" "$ASC_FILE"
        sed -i "s/int baseM = [0-9]\+;/int baseM = ${bm};/" "$ASC_FILE"
        sed -i "s/int baseN = [0-9]\+;/int baseN = ${bn};/" "$ASC_FILE"

        # Verify the patch applied correctly
        if ! grep -q "int baseM = ${bm};" "$ASC_FILE"; then
            echo "[ERROR] Failed to patch baseM to $bm"
            echo "$bm,$bn,,PATCH_FAIL" >> "$OUTPUT_CSV"
            continue
        fi
        if ! grep -q "int baseN = ${bn};" "$ASC_FILE"; then
            echo "[ERROR] Failed to patch baseN to $bn"
            echo "$bm,$bn,,PATCH_FAIL" >> "$OUTPUT_CSV"
            continue
        fi
        echo "[sweep] Patched: baseM=$bm, baseN=$bn"

        # --- Clean build directory ---
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR"

        # --- Build ---
        echo "[sweep] Building..."
        set +e
        BUILD_LOG=$(cd "$BUILD_DIR" && cmake -DASC_DIR="$CMAKE_ASC_DIR" .. 2>&1 && make -j 2>&1)
        BUILD_RC=$?
        set -e

        if [ $BUILD_RC -ne 0 ]; then
            echo "[ERROR] Build failed for baseM=$bm, baseN=$bn"
            echo "$BUILD_LOG" | tail -20
            echo "$bm,$bn,,BUILD_FAIL" >> "$OUTPUT_CSV"
            continue
        fi
        echo "[sweep] Build succeeded"

        # --- Generate test data ---
        echo "[sweep] Generating test data..."
        set +e
        GEN_LOG=$(cd "$BUILD_DIR" && python3 ../../scripts/gen_data.py 2>&1)
        GEN_RC=$?
        set -e

        if [ $GEN_RC -ne 0 ]; then
            echo "[ERROR] gen_data.py failed for baseM=$bm, baseN=$bn"
            echo "$GEN_LOG" | tail -10
            echo "$bm,$bn,,GENDATA_FAIL" >> "$OUTPUT_CSV"
            continue
        fi

        # --- Run demo (correctness) ---
        echo "[sweep] Running demo..."
        set +e
        DEMO_LOG=$(cd "$BUILD_DIR" && ./demo 2>&1)
        DEMO_RC=$?
        set -e

        if [ $DEMO_RC -ne 0 ]; then
            echo "[ERROR] demo failed for baseM=$bm, baseN=$bn"
            echo "$DEMO_LOG" | tail -10
            echo "$bm,$bn,,DEMO_FAIL" >> "$OUTPUT_CSV"
            continue
        fi
        echo "[sweep] Demo output: $(echo "$DEMO_LOG" | grep -i 'tiling' || true)"

        # --- Verify correctness ---
        echo "[sweep] Verifying correctness..."
        set +e
        VERIFY_LOG=$(cd "$BUILD_DIR" && python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1)
        VERIFY_RC=$?
        set -e

        if [ $VERIFY_RC -ne 0 ]; then
            echo "[WARN] Correctness FAILED for baseM=$bm, baseN=$bn"
            CORRECT="FAIL"
        else
            echo "[sweep] Correctness: PASS"
            CORRECT="PASS"
        fi

        # --- Profiling with msprof ---
        echo "[sweep] Running msprof (warm-up=10, launch-count=5)..."
        MSPROF_OUT=$(mktemp -d /tmp/msprof_XXXXXX)
        chmod 700 "$MSPROF_OUT"

        set +e
        MSPROF_LOG=$(cd "$BUILD_DIR" && msprof op --warm-up=10 --launch-count=5 --output="$MSPROF_OUT" ./demo 2>&1)
        MSPROF_RC=$?
        set -e

        TASK_DURATION=""
        if [ $MSPROF_RC -ne 0 ]; then
            echo "[ERROR] msprof failed for baseM=$bm, baseN=$bn"
            echo "$MSPROF_LOG" | tail -10
        else
            # Find the OPPROF directory
            OPPROF_DIR=$(ls -td "$MSPROF_OUT"/OPPROF_* 2>/dev/null | head -1)
            if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
                echo "[sweep] OpBasicInfo.csv found"
                TASK_DURATION=$(extract_task_duration "$OPPROF_DIR/OpBasicInfo.csv")
                echo "[sweep] Task Duration: ${TASK_DURATION:-N/A} us"
            else
                echo "[WARN] OpBasicInfo.csv not found in msprof output"
                # Try to find it recursively
                OPBASIC_FILE=$(find "$MSPROF_OUT" -name "OpBasicInfo.csv" 2>/dev/null | head -1)
                if [ -n "$OPBASIC_FILE" ]; then
                    TASK_DURATION=$(extract_task_duration "$OPBASIC_FILE")
                    echo "[sweep] Task Duration (found recursively): ${TASK_DURATION:-N/A} us"
                fi
            fi
        fi

        # Clean up msprof temp dir
        rm -rf "$MSPROF_OUT"

        # --- Record result ---
        echo "$bm,$bn,$TASK_DURATION,$CORRECT" >> "$OUTPUT_CSV"
        echo "[sweep] Recorded: baseM=$bm, baseN=$bn, duration=${TASK_DURATION:-N/A}us, correct=$CORRECT"
        echo ""
    done
done

echo "================================================================"
echo "[sweep] Parameter sweep complete!"
echo "[sweep] Results saved to: $OUTPUT_CSV"
echo "================================================================"
echo ""
echo "--- Results ---"
column -t -s',' "$OUTPUT_CSV" 2>/dev/null || cat "$OUTPUT_CSV"
