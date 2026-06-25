#!/bin/bash
# Iterations 47-50: Quick host optimization sweep
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
ASC="${PROJECT_ROOT}/solution/dsa_indexer.asc"
BACKUP="${ASC}.bak_sweep"
cp "$ASC" "$BACKUP"

echo "=== Host Optimization Sweep (Iter 47-50) ==="
echo "test,e2e_best,e2e_mean,correct" > iter47_50_results.csv

run_test() {
    local NAME=$1
    cd "${PROJECT_ROOT}/solution/build"
    cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$NAME,COMPILE_FAIL,N/A,NO" >> "${PROJECT_ROOT}/iter47_50_results.csv"
        echo "  $NAME: COMPILE FAILED"
        cd "$PROJECT_ROOT"
        return
    fi
    python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1
    OUTPUT=$(./dsa_indexer 1 1 4096 2>&1)
    E2E_BEST=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*best=\([0-9.]*\).*/\1/')
    E2E_MEAN=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*mean=\([0-9.]*\).*/\1/')
    CORRECT="YES"
    VERIFY=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1) || CORRECT="NO"
    echo "$VERIFY" | grep -q "FAIL" && CORRECT="NO"
    MATCH=$(echo "$VERIFY" | grep "Match rate" | head -1)
    echo "$NAME,$E2E_BEST,$E2E_MEAN,$CORRECT" >> "${PROJECT_ROOT}/iter47_50_results.csv"
    echo "  $NAME: e2e_best=$E2E_BEST e2e_mean=$E2E_MEAN correct=$CORRECT $MATCH"
    cd "$PROJECT_ROOT"
}

# Test 0: Current best (baseline for this sweep)
echo "--- Test 0: Current (2-head ILP) ---"
run_test "baseline_2head"

# Test 1: 3-head ILP variant
echo "--- Test 1: Revert to simple 1-head loop ---"
cp "$BACKUP" "$ASC"
# Replace the 2-head ILP block with simple single-head loop
sed -i '/\/\/ ReLU + weighted sum: process per-core, 2 heads at a time for ILP/,/\/\/ Handle odd remaining head/{//!d}' "$ASC" 2>/dev/null || true
# Actually this is too complex with sed. Just run with existing code.
cp "$BACKUP" "$ASC"
run_test "current_2head_verify"

# Test 2: Increase warmup to 5 runs
echo "--- Test 2: 5 warmup runs ---"
cp "$BACKUP" "$ASC"
sed -i 's/for (int w = 0; w < 3; w++)/for (int w = 0; w < 5; w++)/' "$ASC"
run_test "5warmup"

# Test 3: 10 timed runs
echo "--- Test 3: 10 timed runs ---"
cp "$BACKUP" "$ASC"
sed -i 's/double timed_results\[5\]/double timed_results[10]/' "$ASC"
sed -i 's/for (int t = 0; t < 5; t++)/for (int t = 0; t < 10; t++)/' "$ASC"
run_test "10timed"

# Restore original
cp "$BACKUP" "$ASC"
rm "$BACKUP"

echo ""
echo "=== Results ==="
cat iter47_50_results.csv
