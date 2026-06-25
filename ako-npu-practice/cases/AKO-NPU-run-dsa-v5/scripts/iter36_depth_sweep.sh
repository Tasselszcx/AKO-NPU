#!/bin/bash
# Iter 36: depthA1/B1 combination sweep
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
ASC="${PROJECT_ROOT}/solution/dsa_indexer.asc"
cp "$ASC" "${ASC}.bak36"

echo "depthA1,depthB1,compile,correct,e2e_best,e2e_mean" > iter36_depth.csv

# All combinations with stepKa=2, stepKb=2 (K full-load)
for DA in 1 2 4 8; do
    for DB in 1 2 4 8; do
        cp "${ASC}.bak36" "$ASC"
        sed -i "s/constantCFG.depthA1 = [0-9]*/constantCFG.depthA1 = ${DA}/" "$ASC"
        sed -i "s/constantCFG.depthB1 = [0-9]*/constantCFG.depthB1 = ${DB}/" "$ASC"

        cd "${PROJECT_ROOT}/solution/build"
        COMPILE="YES"
        cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
        if ! make -j4 > /dev/null 2>&1; then
            COMPILE="NO"
            echo "$DA,$DB,$COMPILE,NO,N/A,N/A" >> "${PROJECT_ROOT}/iter36_depth.csv"
            cd "$PROJECT_ROOT"
            continue
        fi

        python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1
        OUTPUT=$(./dsa_indexer 1 1 4096 2>&1)

        CORRECT="YES"
        VERIFY=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1) || CORRECT="NO"
        echo "$VERIFY" | grep -q "FAIL" && CORRECT="NO"

        E2E_BEST=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*best=\([0-9.]*\).*/\1/')
        E2E_MEAN=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*mean=\([0-9.]*\).*/\1/')

        echo "$DA,$DB,$COMPILE,$CORRECT,$E2E_BEST,$E2E_MEAN" >> "${PROJECT_ROOT}/iter36_depth.csv"
        echo "dA=$DA dB=$DB: correct=$CORRECT e2e_best=$E2E_BEST e2e_mean=$E2E_MEAN"
        cd "$PROJECT_ROOT"
    done
done

cp "${ASC}.bak36" "$ASC"
rm "${ASC}.bak36"

echo ""
echo "=== Depth Sweep Results ==="
cat iter36_depth.csv
