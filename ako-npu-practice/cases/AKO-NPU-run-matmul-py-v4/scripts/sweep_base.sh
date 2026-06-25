#!/bin/bash
# Sweep baseM and baseN parameters for constant tiling
set -eo pipefail
cd "$(dirname "$0")/.."

# Environment setup
[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "FAIL: ASCEND_HOME_PATH not set"; exit 1; }
for p in "${ASCEND_HOME_PATH}/set_env.sh" "${ASCEND_HOME_PATH}/../set_env.sh"; do
    [ -f "$p" ] && source "$p" && break
done
if [ -z "${ASC_DIR:-}" ]; then
    ASCEND_BASE="$(cd "${ASCEND_HOME_PATH}/.." && pwd)"
    ASC_FOUND=$(find "${ASCEND_BASE}" -maxdepth 6 -name "ASCConfig.cmake" -path "*/ascendc_kernel_cmake/*" 2>/dev/null | head -1 || true)
    [ -n "${ASC_FOUND}" ] && export ASC_DIR="$(dirname "${ASC_FOUND}")"
fi

SOLUTION_DIR="solution"
BEST_FILE="solution/matmul_leakyrelu.asc"
BACKUP="solution/matmul_leakyrelu.asc.base_backup"
cp "$BEST_FILE" "$BACKUP"

echo "baseM,baseN,baseK,matmul_us,relu_us,total_us,status" > sweep_base_results.csv

# baseM must be multiple of 16, baseN must be multiple of 16, baseK fixed at 64
# baseM * baseN * 4 must fit in L0C (256KB)
for bM in 32 48 64 96 128; do
for bN in 64 128 192 256 320; do
    # Check L0C constraint: baseM * baseN * sizeof(float) <= 128KB (with factor)
    L0C_size=$(( bM * bN * 4 ))
    if [ $L0C_size -gt 131072 ]; then
        continue
    fi

    # singleCoreM should be >= baseM. Use auto-tiling values matching shapeParams
    # For the sweep, keep singleCoreM=94, singleCoreN=320

    # Patch the .asc file — change the shapeParams line
    cp "$BACKUP" "$BEST_FILE"
    sed -i "s/constexpr MatmulShapeParams shapeParams = {94, 320, CONST_K, .*, .*, 64};/constexpr MatmulShapeParams shapeParams = {94, 320, CONST_K, $bM, $bN, 64};/" "$BEST_FILE"

    # Build
    cd "$SOLUTION_DIR"
    rm -rf build && mkdir -p build && cd build
    cmake ${ASC_DIR:+-DASC_DIR=$ASC_DIR} .. >/dev/null 2>&1
    if ! make -j4 >/dev/null 2>&1; then
        echo "$bM,$bN,64,NA,NA,NA,compile_fail" >> ../../sweep_base_results.csv
        cd ../..
        continue
    fi

    # Generate data
    python3 ../../scripts/gen_data.py >/dev/null 2>&1

    # Run and check correctness
    ./matmul_leakyrelu_custom >/dev/null 2>&1
    if ! python3 ../../scripts/verify_result.py output/output.bin output/golden.bin >/dev/null 2>&1; then
        echo "$bM,$bN,64,NA,NA,NA,precision_fail" >> ../../sweep_base_results.csv
        cd ../..
        continue
    fi

    # Profile
    chmod 750 . 2>/dev/null
    rm -rf msprof_output
    msprof op --warm-up=10 --launch-count=3 --output=./msprof_output ./matmul_leakyrelu_custom >/dev/null 2>&1

    # Extract times
    OPPROF_DIR=$(ls -td msprof_output/OPPROF_* 2>/dev/null | head -1)
    if [ -z "$OPPROF_DIR" ]; then
        echo "$bM,$bN,64,NA,NA,NA,msprof_fail" >> ../../sweep_base_results.csv
        cd ../..
        continue
    fi

    TIMES=$(python3 -c "
import csv, os
matmul_t = relu_t = 0
for subdir in sorted(os.listdir('$OPPROF_DIR')):
    path = os.path.join('$OPPROF_DIR', subdir, '0')
    for f in os.listdir(path):
        if f.startswith('OpBasicInfo'):
            with open(os.path.join(path, f)) as fh:
                r = csv.DictReader(fh)
                for row in r:
                    dur = float(row['Task Duration(us)'])
                    if 'matmul' in row['Op Name']:
                        matmul_t = dur
                    elif 'leakyrelu' in row['Op Name']:
                        relu_t = dur
print(f'{matmul_t:.2f},{relu_t:.2f},{matmul_t+relu_t:.2f}')
" 2>/dev/null)

    if [ -z "$TIMES" ]; then
        echo "$bM,$bN,64,NA,NA,NA,parse_fail" >> ../../sweep_base_results.csv
    else
        echo "$bM,$bN,64,$TIMES,ok" >> ../../sweep_base_results.csv
        echo "bM=$bM bN=$bN → $TIMES"
    fi

    rm -rf msprof_output
    cd ../..
done
done

cp "$BACKUP" "$BEST_FILE"
rm -f "$BACKUP"

echo ""
echo "=== Results ==="
sort -t, -k6 -n sweep_base_results.csv | head -20
