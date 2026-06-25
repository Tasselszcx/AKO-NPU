#!/bin/bash
# Sweep MatmulConfig parameters
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
RESULTS_FILE="matmul_config_sweep.csv"

echo "iter,description,matmul_us,leakyrelu_us,total_us,correct" > "$RESULTS_FILE"

run_test() {
    local ITER=$1 DESC=$2 CONFIG=$3
    # Replace the MatmulConfig block
    python3 -c "
import re
with open('$ASC_FILE', 'r') as f:
    content = f.read()
# Replace the config block
new_config = '''$CONFIG'''
content = re.sub(
    r'constexpr MatmulConfig mmConfig = \{[^}]+\};',
    new_config.strip(),
    content
)
with open('$ASC_FILE', 'w') as f:
    f.write(content)
"
    cd solution/build
    cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,FAIL,FAIL,FAIL,compile_fail" >> "../../$RESULTS_FILE"
        cd ../..
        return
    fi
    python3 ../scripts/gen_data.py > /dev/null 2>&1
    MSPROF_DIR="msprof_cfg"
    rm -rf "$MSPROF_DIR" && mkdir -p "$MSPROF_DIR" && chmod 700 "$MSPROF_DIR"
    msprof op --warm-up=5 --launch-count=3 --output="$MSPROF_DIR" ./matmul_leakyrelu > /dev/null 2>&1 || true

    CORRECT="true"
    if ! python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin 2>&1 | grep -q "test pass"; then
        CORRECT="false"
    fi

    OPPROF_BASE=$(ls -td "$MSPROF_DIR"/OPPROF_* 2>/dev/null | head -1)
    if [ -n "$OPPROF_BASE" ]; then
        PERF=$(python3 -c "
import os, csv, glob, statistics
csv_files = glob.glob(os.path.join('$OPPROF_BASE', '**/OpBasicInfo*.csv'), recursive=True)
durations = {}
for f in csv_files:
    with open(f, 'r') as fh:
        for row in csv.DictReader(fh):
            name = row.get('Op Name', '')
            td = row.get('Task Duration(us)', '')
            if td:
                durations.setdefault(name, []).append(float(td))
matmul = statistics.mean(durations.get('matmul_leakyrelu_custom', [0]))
relu = statistics.mean(durations.get('leakyrelu_kernel', [0]))
print(f'{matmul:.2f},{relu:.2f},{matmul+relu:.2f}')
" 2>/dev/null || echo "0,0,0")
        echo "$ITER,$DESC,$PERF,$CORRECT" >> "../../$RESULTS_FILE"
        echo "  Iter $ITER ($DESC): $PERF correct=$CORRECT"
    fi
    rm -rf "$MSPROF_DIR"
    cd ../..
}

# Baseline: Norm + ITERATE_MODE_ALL + enableGetTensorC=false + enableQuantVector=false
run_test 22 "baseline_config" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableQuantVector = false,
};'

# Try just CFG_NORM (no custom config)
run_test 23 "default_norm" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
};'

# Try with enUnitFlag=true
run_test 24 "unitflag_true" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .enUnitFlag = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableQuantVector = false,
};'

# Try enableInit=false (disable Init for better constant propagation)
run_test 25 "enableInit_false" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .enableInit = false,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableQuantVector = false,
};'

# Try enableSetTail=false (we have tail handling in CalcOffset)
run_test 26 "enableSetTail_true" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetTail = true,
    .enableQuantVector = false,
};'

# Try enableSetBias=true explicitly
run_test 27 "enableSetBias_true" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetBias = true,
    .enableQuantVector = false,
};'

# Try enableEnd=false with End() removed
run_test 28 "all_disabled" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableSetOrgShape = false,
    .enableQuantVector = false,
    .enableSetDefineData = false,
};'

# Restore baseline
run_test 29 "restore" 'constexpr MatmulConfig mmConfig = {
    .doNorm = true,
    .iterateMode = ITERATE_MODE_ALL,
    .enableGetTensorC = false,
    .enableQuantVector = false,
};'

echo "=== Config sweep complete ==="
cat "$RESULTS_FILE"
