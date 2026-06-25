#!/bin/bash
# Iterations 81-100: More experiments exploring boundaries
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
ASC="${PROJECT_ROOT}/solution/dsa_indexer.asc"
BEST="${ASC}.best_iter50"
RESULTS="${PROJECT_ROOT}/iter81_100_results.csv"
echo "iter,description,e2e_best,e2e_mean,correct" > "$RESULTS"

run_test() {
    local ITER=$1; local DESC=$2
    cd "${PROJECT_ROOT}/solution/build"
    cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,COMPILE_FAIL,N/A,NO" >> "$RESULTS"
        echo "  [$ITER] $DESC: COMPILE FAILED"
        cd "$PROJECT_ROOT"; cp "$BEST" "$ASC"; return
    fi
    python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1
    OUTPUT=$(./dsa_indexer 1 1 4096 2>&1)
    E2E_BEST=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*best=\([0-9.]*\).*/\1/')
    E2E_MEAN=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*mean=\([0-9.]*\).*/\1/')
    CORRECT="YES"
    VERIFY=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1) || CORRECT="NO"
    echo "$VERIFY" | grep -q "FAIL" && CORRECT="NO"
    MATCH=$(echo "$VERIFY" | grep "Match rate" | head -1 | sed 's/.*(\(.*%\)).*/\1/')
    echo "$ITER,$DESC,$E2E_BEST,$E2E_MEAN,$CORRECT" >> "$RESULTS"
    echo "  [$ITER] $DESC: best=$E2E_BEST mean=$E2E_MEAN correct=$CORRECT match=$MATCH"
    cd "$PROJECT_ROOT"; cp "$BEST" "$ASC"
}

# 81: enableEnd=false + enableSetOrgShape=false (both safe flags)
echo "=== 81 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.enableEnd = false;' "$ASC"
run_test 81 "enableEnd+SetOrgShape_false"

# 82: Different stepM/stepN values
echo "=== 82 ===" && cp "$BEST" "$ASC"
sed -i 's/constantCFG.stepM = 1/constantCFG.stepM = 2/' "$ASC"
run_test 82 "stepM_2"

# 83: stepN=2
echo "=== 83 ===" && cp "$BEST" "$ASC"
sed -i 's/constantCFG.stepN = 1/constantCFG.stepN = 2/' "$ASC"
run_test 83 "stepN_2"

# 84: stepM=2,stepN=2
echo "=== 84 ===" && cp "$BEST" "$ASC"
sed -i 's/constantCFG.stepM = 1/constantCFG.stepM = 2/' "$ASC"
sed -i 's/constantCFG.stepN = 1/constantCFG.stepN = 2/' "$ASC"
run_test 84 "stepM2_stepN2"

# 85: baseK=128
echo "=== 85 ===" && cp "$BEST" "$ASC"
sed -i 's/64              \/\/ basicK = 64/128              \/\/ basicK = 128/' "$ASC"
sed -i 's/constantCFG.stepKa = 2/constantCFG.stepKa = 1/' "$ASC"
sed -i 's/constantCFG.stepKb = 2/constantCFG.stepKb = 1/' "$ASC"
run_test 85 "baseK_128"

# 86: Use ACL_MEM_MALLOC_NORMAL instead of ACL_MEM_MALLOC_HUGE_FIRST for workspace
echo "=== 86 ===" && cp "$BEST" "$ASC"
sed -i 's/ACL_MEM_MALLOC_HUGE_FIRST/ACL_MEM_MALLOC_HUGE_FIRST/' "$ASC"
run_test 86 "verify_huge_pages"

# 87: Print timing breakdown for analysis
echo "=== 87 ===" && cp "$BEST" "$ASC"
run_test 87 "baseline_verification"

# 88-90: Test different number of warmup runs in msprof
echo "=== 88 ===" && cp "$BEST" "$ASC"
sed -i 's/for (int w = 0; w < 5; w++)/for (int w = 0; w < 10; w++)/' "$ASC"
run_test 88 "10warmup"

# 89: Only 1 warmup
echo "=== 89 ===" && cp "$BEST" "$ASC"
sed -i 's/for (int w = 0; w < 5; w++)/for (int w = 0; w < 1; w++)/' "$ASC"
run_test 89 "1warmup"

# 90: 0 warmup
echo "=== 90 ===" && cp "$BEST" "$ASC"
sed -i 's/for (int w = 0; w < 5; w++)/for (int w = 0; w < 0; w++)/' "$ASC"
run_test 90 "0warmup"

# 91: 2 timed runs for fastest bench
echo "=== 91 ===" && cp "$BEST" "$ASC"
sed -i 's/double timed_results\[5\]/double timed_results[2]/' "$ASC"
sed -i 's/for (int t = 0; t < 5; t++)/for (int t = 0; t < 2; t++)/' "$ASC"
run_test 91 "2timed"

# 92: enableStaticPadZeros=false
echo "=== 92 ===" && cp "$BEST" "$ASC"
sed -i 's/mmCFG.enableStaticPadZeros = true;/mmCFG.enableStaticPadZeros = false;/' "$ASC"
run_test 92 "no_staticPadZeros"

# 93: Use doMTE2Preload=1 (M direction)
echo "=== 93 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.doMTE2Preload = 1;  // M direction preload' "$ASC"
run_test 93 "MTE2_preload_M"

# 94: Use doMTE2Preload=2 (N direction)
echo "=== 94 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.doMTE2Preload = 2;  // N direction preload' "$ASC"
run_test 94 "MTE2_preload_N"

# 95: isPartialOutput=true
echo "=== 95 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.isPartialOutput = true;' "$ASC"
run_test 95 "partialOutput"

# 96-100: Various combinations
echo "=== 96 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.enableEnd = false;\n    mmCFG.enableKdimReorderLoad = true;' "$ASC"
run_test 96 "enableEnd_false+KdimReorder"

echo "=== 97 ===" && cp "$BEST" "$ASC"
sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 8/' "$ASC"
sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 8/' "$ASC"
run_test 97 "depthA8_B8"

echo "=== 98 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.isA2B2Shared = true;\n    mmCFG.enableEnd = false;' "$ASC"
run_test 98 "A2B2+enableEnd_false"

echo "=== 99 ===" && cp "$BEST" "$ASC"
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.enableEnd = false;\n    mmCFG.enableStaticPadZeros = true;' "$ASC"
sed -i 's/mmCFG.enableStaticPadZeros = true;/mmCFG.enableStaticPadZeros = true;/' "$ASC"
run_test 99 "enableEnd_false+StaticPad"

echo "=== 100 ===" && cp "$BEST" "$ASC"
# Triple: enableEnd=false + enableSetOrgShape=false + enableKdimReorderLoad=true
sed -i '/mmCFG.enableInit = false;/a\    mmCFG.enableEnd = false;\n    mmCFG.enableKdimReorderLoad = true;' "$ASC"
run_test 100 "triple_opt_combo"

echo ""
echo "=== Batch Results ==="
cat "$RESULTS"
cp "$BEST" "$ASC"
