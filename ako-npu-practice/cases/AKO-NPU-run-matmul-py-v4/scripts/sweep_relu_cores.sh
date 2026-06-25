#!/bin/bash
set -eo pipefail
cd "$(dirname "$0")/.."

SOLUTION_DIR="solution"
ASC_FILE="$SOLUTION_DIR/matmul_leakyrelu.asc"
ORIGINAL=$(cat "$ASC_FILE")

[ -n "${ASCEND_HOME_PATH:-}" ] || { echo "FAIL: ASCEND_HOME_PATH not set"; exit 1; }
SET_ENV=""
for p in "${ASCEND_HOME_PATH}/set_env.sh" "${ASCEND_HOME_PATH}/../set_env.sh" "${ASCEND_HOME_PATH}/../../set_env.sh"; do
    if [ -f "$p" ]; then SET_ENV="$p"; break; fi
done
[ -n "$SET_ENV" ] && source "$SET_ENV"
if [ -z "${ASC_DIR:-}" ]; then
    ASCEND_BASE="$(cd "${ASCEND_HOME_PATH}/.." && pwd)"
    ASC_FOUND=$(find "${ASCEND_BASE}" -maxdepth 6 -name "ASCConfig.cmake" -path "*/ascendc_kernel_cmake/*" 2>/dev/null | head -1 || true)
    [ -n "${ASC_FOUND}" ] && export ASC_DIR="$(dirname "${ASC_FOUND}")"
fi

echo "cores,matmul_us,relu_us,total_us"

for CORES in 8 10 14 16 20 24 32 40; do
    # Update kernel constant
    sed -i "s/constexpr uint32_t usedCoreNum = [0-9]*/constexpr uint32_t usedCoreNum = $CORES/" "$ASC_FILE"
    # Update host tiling
    sed -i "s/reluTiling.usedCoreNum = [0-9]*/reluTiling.usedCoreNum = $CORES/" "$ASC_FILE"

    cd "$SOLUTION_DIR/build"
    cmake ${ASC_DIR:+-DASC_DIR=$ASC_DIR} .. >/dev/null 2>&1
    make -j4 >/dev/null 2>&1 || { echo "$CORES,COMPILE_FAIL,,"; cd ../.. ; continue; }

    rm -rf msprof_sweep
    msprof op --warm-up=10 --launch-count=5 --output=./msprof_sweep ./matmul_leakyrelu_custom >/dev/null 2>&1 || { echo "$CORES,MSPROF_FAIL,,"; cd ../.. ; continue; }

    OPPROF=$(ls -td msprof_sweep/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF" ]; then
        DURATIONS=$(python3 -c "
import csv, glob
files = glob.glob('$OPPROF/**/OpBasicInfo_*.csv', recursive=True)
matmul_d = relu_d = 0
for f in files:
    with open(f) as fh:
        for row in csv.DictReader(fh):
            td = float(row.get('Task Duration(us)','0'))
            if 'matmul' in row.get('Op Name',''):
                matmul_d = td
            elif 'relu' in row.get('Op Name',''):
                relu_d = td
print(f'{matmul_d:.2f},{relu_d:.2f},{matmul_d+relu_d:.2f}')
" 2>/dev/null || echo "PARSE_FAIL,,")
        echo "$CORES,$DURATIONS"
    else
        echo "$CORES,NO_DATA,,"
    fi
    cd ../..
done

echo "$ORIGINAL" > "$ASC_FILE"
echo "Sweep complete, original code restored."
