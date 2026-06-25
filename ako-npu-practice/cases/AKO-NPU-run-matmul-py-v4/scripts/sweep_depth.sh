#!/bin/bash
# Sweep depthA1, depthB1, stepKa, stepKb parameters
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
BACKUP="solution/matmul_leakyrelu.asc.sweep_backup"
cp "$BEST_FILE" "$BACKUP"

echo "depthA1,depthB1,stepKa,stepKb,matmul_us,relu_us,total_us,status" > sweep_depth_results.csv

# Parameter space
for dA in 1 2 3 4; do
for dB in 1 2 3 4; do
for sKa in 1 2 4; do
for sKb in 1 2 4; do
    # Skip combos that exceed L1 (512KB)
    # A: dA * 96 * 64 * 2 = dA * 12288 bytes
    # B: dB * 256 * 64 * 2 = dB * 32768 bytes
    L1_total=$(( dA * 12288 + dB * 32768 ))
    if [ $L1_total -gt 524288 ]; then
        continue
    fi

    # Skip if step > depth (invalid)
    if [ $sKa -gt $dA ] || [ $sKb -gt $dB ]; then
        continue
    fi

    # Patch the .asc file
    cp "$BACKUP" "$BEST_FILE"
    sed -i "s/constantCFG.depthA1 = .*;/constantCFG.depthA1 = $dA;/" "$BEST_FILE"
    sed -i "s/constantCFG.depthB1 = .*;/constantCFG.depthB1 = $dB;/" "$BEST_FILE"
    sed -i "s/constantCFG.stepKa = .*;/constantCFG.stepKa = $sKa;/" "$BEST_FILE"
    sed -i "s/constantCFG.stepKb = .*;/constantCFG.stepKb = $sKb;/" "$BEST_FILE"

    # Build
    cd "$SOLUTION_DIR"
    rm -rf build && mkdir -p build && cd build
    cmake ${ASC_DIR:+-DASC_DIR=$ASC_DIR} .. >/dev/null 2>&1
    if ! make -j4 >/dev/null 2>&1; then
        echo "$dA,$dB,$sKa,$sKb,NA,NA,NA,compile_fail" >> ../../sweep_depth_results.csv
        cd ../..
        continue
    fi

    # Generate data
    python3 ../../scripts/gen_data.py >/dev/null 2>&1

    # Run and check correctness
    ./matmul_leakyrelu_custom >/dev/null 2>&1
    if ! python3 ../../scripts/verify_result.py output/output.bin output/golden.bin >/dev/null 2>&1; then
        echo "$dA,$dB,$sKa,$sKb,NA,NA,NA,precision_fail" >> ../../sweep_depth_results.csv
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
        echo "$dA,$dB,$sKa,$sKb,NA,NA,NA,msprof_fail" >> ../../sweep_depth_results.csv
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
        echo "$dA,$dB,$sKa,$sKb,NA,NA,NA,parse_fail" >> ../../sweep_depth_results.csv
    else
        echo "$dA,$dB,$sKa,$sKb,$TIMES,ok" >> ../../sweep_depth_results.csv
        echo "dA=$dA dB=$dB sKa=$sKa sKb=$sKb → $TIMES"
    fi

    rm -rf msprof_output
    cd ../..
done
done
done
done

cp "$BACKUP" "$BEST_FILE"
rm -f "$BACKUP"

echo ""
echo "=== Results ==="
sort -t, -k7 -n sweep_depth_results.csv | head -20
