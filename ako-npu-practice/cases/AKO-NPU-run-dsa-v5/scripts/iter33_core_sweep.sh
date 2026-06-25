#!/bin/bash
# Iteration 33: Core count sweep including non-power-of-2 options
# Test: 4, 8, 16, 32 cores (must be factors of 4096 for even N-split)
# Also test 20 and 24 with remainder handling
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

ASC_FILE="${PROJECT_ROOT}/solution/dsa_indexer.asc"
BACKUP="${PROJECT_ROOT}/solution/dsa_indexer.asc.bak33"
cp "$ASC_FILE" "$BACKUP"

RESULTS_FILE="${PROJECT_ROOT}/iter33_core_sweep.csv"
echo "cores,baseN,singleCoreN,compile,correct,runtime_us" > "$RESULTS_FILE"

# Test core counts that evenly divide 4096
for CORES in 4 8 12 16 32; do
    SINGLE_N=$((4096 / CORES))

    # baseN must divide SINGLE_N and satisfy L0C constraint: baseM*baseN*4 <= 256KB
    # baseM=64, so baseN <= 256*1024 / (64*4) = 1024
    # Also baseN must be multiple of 16
    # With SINGLE_N, best baseN is min(SINGLE_N, 256) that divides SINGLE_N
    if [ $SINGLE_N -le 128 ]; then
        BASEN=128
    elif [ $SINGLE_N -le 256 ]; then
        BASEN=128
    else
        BASEN=256
    fi

    # Check SINGLE_N % BASEN == 0 for even division
    if [ $((SINGLE_N % BASEN)) -ne 0 ]; then
        # Try baseN=64
        BASEN=64
    fi

    N_BLOCKS=$((SINGLE_N / BASEN))
    TOTAL_B_BLOCKS=$((N_BLOCKS * 2))  # 2 K blocks

    echo "Testing: CORES=$CORES, SINGLE_N=$SINGLE_N, BASEN=$BASEN, N_BLOCKS=$N_BLOCKS"

    # Update the .asc file
    cp "$BACKUP" "$ASC_FILE"

    # Replace NUM_MATMUL_CORES
    sed -i "s/constexpr int32_t NUM_MATMUL_CORES = [0-9]*/constexpr int32_t NUM_MATMUL_CORES = ${CORES}/" "$ASC_FILE"

    # Replace basicN in DSA_SHAPE_PARAMS
    sed -i "s/[0-9]*,            \/\/ basicN/${BASEN},            \/\/ basicN/" "$ASC_FILE"

    # Update depthB1 to match total B blocks (or cap at 8)
    DEPTH_B1=$TOTAL_B_BLOCKS
    if [ $DEPTH_B1 -gt 8 ]; then
        DEPTH_B1=8
    fi
    if [ $DEPTH_B1 -lt 2 ]; then
        DEPTH_B1=2
    fi
    sed -i "s/constantCFG.depthB1 = [0-9]*/constantCFG.depthB1 = ${DEPTH_B1}/" "$ASC_FILE"

    # Build
    cd "${PROJECT_ROOT}/solution"
    mkdir -p build && cd build
    COMPILE_OK="YES"
    cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
    if ! make -j4 2>&1 | tail -3; then
        COMPILE_OK="NO"
        echo "$CORES,$BASEN,$SINGLE_N,$COMPILE_OK,NO,N/A" >> "$RESULTS_FILE"
        cd "$PROJECT_ROOT"
        continue
    fi

    # Generate data and run
    cd "${PROJECT_ROOT}/solution/build"
    python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1

    OUTPUT=$(./dsa_indexer 1 1 4096 2>&1)

    # Verify correctness
    CORRECT_OK="YES"
    VERIFY=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1) || CORRECT_OK="NO"
    if echo "$VERIFY" | grep -q "FAIL"; then
        CORRECT_OK="NO"
    fi

    # Get E2E timing from output
    E2E_BEST=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*best=\([0-9.]*\).*/\1/' || echo "N/A")

    # Run msprof for kernel timing
    MSPROF_DIR="${PROJECT_ROOT}/msprof_iter33_${CORES}"
    rm -rf "${MSPROF_DIR}"
    mkdir -p "${MSPROF_DIR}"
    chmod 700 "${MSPROF_DIR}"
    KERNEL_TIME="N/A"
    if msprof op --warm-up=5 --launch-count=5 --output="${MSPROF_DIR}" ./dsa_indexer 1 1 4096 > /dev/null 2>&1; then
        OPPROF_DIR=$(ls -td ${MSPROF_DIR}/OPPROF_* 2>/dev/null | head -1)
        if [ -n "$OPPROF_DIR" ] && [ -f "$OPPROF_DIR/OpBasicInfo.csv" ]; then
            KERNEL_TIME=$(python3 -c "
import csv, statistics
durations = []
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td_key = [k for k in row.keys() if 'Task Duration' in k]
        if td_key:
            val = row[td_key[0]].strip()
            if val: durations.append(float(val))
if durations: print(f'{statistics.mean(durations):.2f}')
else: print('N/A')
" 2>/dev/null || echo "N/A")
        fi
    fi

    echo "$CORES,$BASEN,$SINGLE_N,$COMPILE_OK,$CORRECT_OK,$KERNEL_TIME" >> "$RESULTS_FILE"
    echo "  Result: compile=$COMPILE_OK, correct=$CORRECT_OK, kernel=${KERNEL_TIME} us, E2E_best=${E2E_BEST} us"

    cd "$PROJECT_ROOT"
done

# Restore original
cp "$BACKUP" "$ASC_FILE"
rm "$BACKUP"

echo ""
echo "=== Core Sweep Results ==="
cat "$RESULTS_FILE"
