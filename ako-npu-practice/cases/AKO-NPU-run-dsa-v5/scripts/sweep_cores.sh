#!/bin/bash
# Sweep over core counts and baseN values for multi-core matmul
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

ASC_FILE="solution/dsa_indexer.asc"
ASC_BACKUP="${ASC_FILE}.sweep_backup"
cp "$ASC_FILE" "$ASC_BACKUP"

S_KV=4096
RESULTS_FILE="iter11_sweep_results.csv"
echo "cores,baseN,singleCoreN,depthB1,stepKb,correct,task_duration_us" > "$RESULTS_FILE"

# Configurations to test
# (cores, baseN)
configs=(
    "4 256"
    "4 128"
    "8 256"
    "8 128"
    "16 256"
    "16 128"
    "2 256"
    "2 128"
    "4 512"
)

for cfg in "${configs[@]}"; do
    read -r CORES BASEN <<< "$cfg"
    SCN=$((S_KV / CORES))

    # Check divisibility
    if [ $((SCN % BASEN)) -ne 0 ]; then
        echo "SKIP: cores=$CORES baseN=$BASEN -- singleCoreN=$SCN not divisible by baseN"
        echo "$CORES,$BASEN,$SCN,NA,NA,skip,NA" >> "$RESULTS_FILE"
        continue
    fi

    # Calculate L1 params
    N_BLOCKS=$((SCN / BASEN))
    K_BLOCKS=$((128 / 64))  # baseK=64
    TOTAL_B_BLOCKS=$((N_BLOCKS * K_BLOCKS))
    DEPTHB1=$TOTAL_B_BLOCKS
    if [ $DEPTHB1 -gt 8 ]; then DEPTHB1=8; fi
    STEPKB=2  # K full-load

    echo "=== Testing: cores=$CORES, baseN=$BASEN, singleCoreN=$SCN, depthB1=$DEPTHB1, stepKb=$STEPKB ==="

    # Modify ASC file
    cp "$ASC_BACKUP" "$ASC_FILE"

    # Replace NUM_MATMUL_CORES
    sed -i "s/constexpr int32_t NUM_MATMUL_CORES = [0-9]*/constexpr int32_t NUM_MATMUL_CORES = $CORES/" "$ASC_FILE"

    # Replace basicN in DSA_SHAPE_PARAMS
    sed -i "s/    256,        \/\/ basicN = 256/    $BASEN,        \/\/ basicN = $BASEN/" "$ASC_FILE"

    # Replace depthB1 and stepKb
    sed -i "s/constantCFG.depthB1 = [0-9]*/constantCFG.depthB1 = $DEPTHB1/" "$ASC_FILE"
    sed -i "s/constantCFG.stepKb = [0-9]*/constantCFG.stepKb = $STEPKB/" "$ASC_FILE"

    # Build
    cd solution/build
    if cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1 && make -j4 > /dev/null 2>&1; then
        cd "$PROJECT_ROOT"

        # Generate data
        python3 scripts/gen_data.py 1 1 $S_KV > /dev/null 2>&1

        # Run
        cd solution/build
        if ./dsa_indexer 1 1 $S_KV > /dev/null 2>&1; then
            cd "$PROJECT_ROOT"

            # Check correctness
            CORRECT="FAIL"
            if python3 scripts/verify_result.py 1 1 $S_KV 2>&1 | grep -q "Verification PASSED"; then
                CORRECT="PASS"
            fi

            if [ "$CORRECT" = "PASS" ]; then
                # Profile
                cd solution/build
                MSPROF_DIR="${PROJECT_ROOT}/msprof_sweep"
                rm -rf "$MSPROF_DIR"
                mkdir -p "$MSPROF_DIR"
                chmod 700 "$MSPROF_DIR"
                msprof op --warm-up=5 --launch-count=3 --output="$MSPROF_DIR" ./dsa_indexer 1 1 $S_KV > /dev/null 2>&1 || true
                cd "$PROJECT_ROOT"

                OPPROF_DIR=$(ls -td ${MSPROF_DIR}/OPPROF_* 2>/dev/null | head -1)
                if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
                    TD=$(python3 -c "
import csv
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td_key = [k for k in row.keys() if 'Task Duration' in k]
        if td_key:
            val = row[td_key[0]].strip()
            if val: print(f'{float(val):.2f}'); break
")
                    echo "  Result: $TD us (PASS)"
                    echo "$CORES,$BASEN,$SCN,$DEPTHB1,$STEPKB,PASS,$TD" >> "$RESULTS_FILE"
                else
                    echo "  Profile failed"
                    echo "$CORES,$BASEN,$SCN,$DEPTHB1,$STEPKB,PASS,prof_fail" >> "$RESULTS_FILE"
                fi
            else
                echo "  Correctness FAILED"
                echo "$CORES,$BASEN,$SCN,$DEPTHB1,$STEPKB,FAIL,NA" >> "$RESULTS_FILE"
            fi
        else
            cd "$PROJECT_ROOT"
            echo "  Run failed"
            echo "$CORES,$BASEN,$SCN,$DEPTHB1,$STEPKB,run_fail,NA" >> "$RESULTS_FILE"
        fi
    else
        cd "$PROJECT_ROOT"
        echo "  Build failed"
        echo "$CORES,$BASEN,$SCN,$DEPTHB1,$STEPKB,build_fail,NA" >> "$RESULTS_FILE"
    fi
done

# Restore original
cp "$ASC_BACKUP" "$ASC_FILE"
rm -f "$ASC_BACKUP"

echo ""
echo "=== Sweep Results ==="
cat "$RESULTS_FILE"
