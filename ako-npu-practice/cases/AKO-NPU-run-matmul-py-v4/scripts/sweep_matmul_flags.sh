#!/bin/bash
# Sweep MatmulConfig flags - iter-104
set -eo pipefail
cd "$(dirname "$0")/.."

SOLUTION_DIR="solution"
ASC_FILE="$SOLUTION_DIR/matmul_leakyrelu.asc"
ORIGINAL=$(cat "$ASC_FILE")

# Config templates to test (each replaces the mmCFG lines after enableMixDualMaster)
declare -A CONFIGS
CONFIGS["baseline"]=""
CONFIGS["enableSetDefineData_false"]="mmCFG.enableSetDefineData = false;"
CONFIGS["enableSetBias_true"]="mmCFG.enableSetBias = true;"
CONFIGS["isA2B2Shared_true"]="mmCFG.isA2B2Shared = true;"

echo "flag,matmul_us,relu_us,total_us"

for name in baseline enableSetDefineData_false isA2B2Shared_true; do
    extra="${CONFIGS[$name]}"
    
    # Inject the config
    if [ -n "$extra" ]; then
        sed -i "/mmCFG.enableMixDualMaster = true;/a\\    $extra" "$ASC_FILE"
    fi
    
    # Compile
    cd "$SOLUTION_DIR/build"
    cmake .. > /dev/null 2>&1
    make -j4 > /dev/null 2>&1
    
    # Run correctness check
    python3 ../../scripts/gen_data.py > /dev/null 2>&1
    ./matmul_leakyrelu_custom > /dev/null 2>&1
    CORRECT=$(python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | grep -c "PASSED" || echo 0)
    
    if [ "$CORRECT" -eq 0 ]; then
        echo "$name,FAIL,FAIL,FAIL"
        echo "$ORIGINAL" > "../../$ASC_FILE"
        cd ../..
        continue
    fi
    
    # Profile (3 runs, take median)
    MATMUL_TIMES=()
    RELU_TIMES=()
    for run in 1 2 3; do
        rm -rf msprof_output
        msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./matmul_leakyrelu_custom > /dev/null 2>&1
        OPPROF_DIR=$(ls -td msprof_output/OPPROF_* 2>/dev/null | head -1)
        mm=$(grep -rh "matmul_leakyrelu" "$OPPROF_DIR"/*/0/OpBasicInfo_*.csv 2>/dev/null | grep -v "Op Name" | cut -d, -f3)
        rl=$(grep -rh "leakyrelu_custom,vector" "$OPPROF_DIR"/*/0/OpBasicInfo_*.csv 2>/dev/null | grep -v "Op Name" | cut -d, -f3)
        MATMUL_TIMES+=("$mm")
        RELU_TIMES+=("$rl")
    done
    
    # Calculate median using python
    med_mm=$(python3 -c "import statistics; print(f'{statistics.median([${MATMUL_TIMES[0]},${MATMUL_TIMES[1]},${MATMUL_TIMES[2]}]):.2f}')")
    med_rl=$(python3 -c "import statistics; print(f'{statistics.median([${RELU_TIMES[0]},${RELU_TIMES[1]},${RELU_TIMES[2]}]):.2f}')")
    total=$(python3 -c "print(f'{$med_mm + $med_rl:.2f}')")
    echo "$name,$med_mm,$med_rl,$total"
    
    # Restore original
    echo "$ORIGINAL" > "../../$ASC_FILE"
    cd ../..
done
