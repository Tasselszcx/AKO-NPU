#!/bin/bash
# Iterations 151-200+: Final comprehensive sweep
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
ASC="${PROJECT_ROOT}/solution/dsa_indexer.asc"
BEST="${ASC}.best_iter50"
RESULTS="${PROJECT_ROOT}/iter151_200_results.csv"
echo "iter,description,e2e_best,e2e_mean,correct" > "$RESULTS"

make_fast() {
    sed -i 's/for (int w = 0; w < [0-9]*; w++)/for (int w = 0; w < 2; w++)/' "$ASC"
    sed -i 's/double timed_results\[[0-9]*\]/double timed_results[3]/' "$ASC"
    sed -i 's/for (int t = 0; t < [0-9]*; t++)/for (int t = 0; t < 3; t++)/' "$ASC"
}

run_test() {
    local ITER=$1; local DESC=$2
    cd "${PROJECT_ROOT}/solution/build"
    cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,COMPILE_FAIL,N/A,NO" >> "$RESULTS"
        echo "  [$ITER] $DESC: COMPILE FAILED"
        cd "$PROJECT_ROOT"; cp "$BEST" "$ASC"; make_fast; return
    fi
    python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1
    OUTPUT=$(timeout 30 ./dsa_indexer 1 1 4096 2>&1) || { echo "$ITER,$DESC,TIMEOUT,N/A,NO" >> "$RESULTS"; echo "  [$ITER] $DESC: TIMEOUT"; cd "$PROJECT_ROOT"; cp "$BEST" "$ASC"; make_fast; return; }
    E2E_BEST=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*best=\([0-9.]*\).*/\1/')
    E2E_MEAN=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*mean=\([0-9.]*\).*/\1/')
    CORRECT="YES"
    VERIFY=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1) || CORRECT="NO"
    echo "$VERIFY" | grep -q "FAIL" && CORRECT="NO"
    MATCH=$(echo "$VERIFY" | grep "Match rate" | head -1 | sed 's/.*(\(.*%\)).*/\1/')
    echo "$ITER,$DESC,$E2E_BEST,$E2E_MEAN,$CORRECT" >> "$RESULTS"
    echo "  [$ITER] $DESC: best=$E2E_BEST mean=$E2E_MEAN correct=$CORRECT match=$MATCH"
    cd "$PROJECT_ROOT"; cp "$BEST" "$ASC"; make_fast
}

cp "$BEST" "$ASC"; make_fast

# Group 1 (151-160): Core count + baseN grid search
for CORES in 2 4 8 16; do
    for BN in 64 128 256; do
        ITER=$((150 + (CORES/2-1)*3 + (BN/64-1) + 1))
        if [ $ITER -gt 160 ]; then continue; fi
        SCN=$((4096 / CORES))
        if [ $((SCN % BN)) -ne 0 ]; then
            echo "$ITER,cores${CORES}_bn${BN},SKIP_INDIVISIBLE,N/A,SKIP" >> "$RESULTS"
            echo "  [$ITER] SKIP: $SCN not divisible by $BN"
            continue
        fi
        cp "$BEST" "$ASC"; make_fast
        sed -i "s/NUM_MATMUL_CORES = 16/NUM_MATMUL_CORES = $CORES/" "$ASC"
        sed -i "s/128,            \/\/ basicN = 128/$BN,            \/\/ basicN = $BN/" "$ASC"
        run_test $ITER "cores${CORES}_bn${BN}"
    done
done

# Group 2 (161-180): Repeat best config 20 more times for statistics
for i in $(seq 161 180); do
    cp "$BEST" "$ASC"; make_fast
    run_test $i "stability_run_$((i-160))"
done

# Group 3 (181-190): Combined flag experiments
for i in $(seq 181 190); do
    cp "$BEST" "$ASC"; make_fast
    case $i in
        181) sed -i '/mmCFG.enableInit = false;/a\    mmCFG.enableEnd = false;' "$ASC"; DESC="enableEnd_false" ;;
        182) sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 2/' "$ASC"; sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 2/' "$ASC"; DESC="depthA2_B2" ;;
        183) sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 8/' "$ASC"; sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 2/' "$ASC"; DESC="depthA8_B2" ;;
        184) sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 2/' "$ASC"; sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 8/' "$ASC"; DESC="depthA2_B8" ;;
        185) sed -i '/mmCFG.enableInit = false;/a\    mmCFG.enableEnd = false;\n    mmCFG.isA2B2Shared = true;' "$ASC"; DESC="end+A2B2" ;;
        186) sed -i 's/NUM_MATMUL_CORES = 16/NUM_MATMUL_CORES = 8/' "$ASC"; sed -i 's/128,            \/\/ basicN = 128/256,            \/\/ basicN = 256/' "$ASC"; sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 4/' "$ASC"; DESC="8cores_bn256_v2" ;;
        187) DESC="verify_187" ;;
        188) DESC="verify_188" ;;
        189) DESC="verify_189" ;;
        190) DESC="verify_190" ;;
    esac
    run_test $i "$DESC"
done

# Group 4 (191-200+): Final verification and convergence assessment
for i in $(seq 191 205); do
    cp "$BEST" "$ASC"; make_fast
    run_test $i "final_verify_$((i-190))"
done

echo ""
echo "=== Batch Complete ==="
wc -l "$RESULTS"
cp "$BEST" "$ASC"
