#!/bin/bash
# Sweep shapeParams for matmul constant tiling
# Tests different combinations of {singleCoreM, singleCoreN, K, baseM, baseN, baseK}
# keeping singleCoreM and singleCoreN unchanged (to avoid CalcOffset mismatch)
# Only sweeping baseM, baseN, baseK and depth/step params

set -e
cd "$(dirname "$0")/.."

SOLUTION_DIR="solution"
ASC_FILE="$SOLUTION_DIR/matmul_leakyrelu.asc"
BUILD_DIR="$SOLUTION_DIR/build"

# Save original
cp "$ASC_FILE" "${ASC_FILE}.bak"

RESULTS_FILE="scripts/sweep_results.csv"
echo "baseM,baseN,baseK,depthA1,depthB1,stepKa,stepKb,relu_us,matmul_us,total_us,correct" > "$RESULTS_FILE"

# Test combinations
for baseM in 48 64 96 128; do
for baseN in 128 256 320; do
for baseK in 32 64 128; do
for dA in 4; do
for dB in 8; do
for sKa in 2; do
for sKb in 2; do

    echo "--- Testing bM=$baseM bN=$baseN bK=$baseK dA=$dA dB=$dB sKa=$sKa sKb=$sKb ---"

    # Modify the shapeParams line
    sed -i "s/constexpr MatmulShapeParams shapeParams = {[^}]*};/constexpr MatmulShapeParams shapeParams = {94, 320, CONST_K, $baseM, $baseN, $baseK};/" "$ASC_FILE"

    # Modify depth/step
    sed -i "s/constantCFG.depthA1 = [0-9]*/constantCFG.depthA1 = $dA/" "$ASC_FILE"
    sed -i "s/constantCFG.depthB1 = [0-9]*/constantCFG.depthB1 = $dB/" "$ASC_FILE"
    sed -i "s/constantCFG.stepKa = [0-9]*/constantCFG.stepKa = $sKa/" "$ASC_FILE"
    sed -i "s/constantCFG.stepKb = [0-9]*/constantCFG.stepKb = $sKb/" "$ASC_FILE"

    # Build
    cd "$BUILD_DIR"
    if ! (cmake .. > /dev/null 2>&1 && make -j4 > /dev/null 2>&1); then
        echo "COMPILE FAIL"
        echo "$baseM,$baseN,$baseK,$dA,$dB,$sKa,$sKb,,,FAIL,compile_fail" >> "../../$RESULTS_FILE"
        cd ../..
        continue
    fi

    # Generate data and run
    python3 ../../scripts/gen_data.py > /dev/null 2>&1
    if ! ./matmul_leakyrelu_custom > /dev/null 2>&1; then
        echo "RUN FAIL"
        echo "$baseM,$baseN,$baseK,$dA,$dB,$sKa,$sKb,,,FAIL,run_fail" >> "../../$RESULTS_FILE"
        cd ../..
        continue
    fi

    # Check correctness
    VERIFY=$(python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1)
    if echo "$VERIFY" | grep -q "FAILED"; then
        echo "PRECISION FAIL"
        echo "$baseM,$baseN,$baseK,$dA,$dB,$sKa,$sKb,,,FAIL,precision_fail" >> "../../$RESULTS_FILE"
        cd ../..
        continue
    fi

    # Profile
    rm -rf ./msprof_tmp
    PROF_OUT=$(msprof op --warm-up=10 --launch-count=5 --output=./msprof_tmp "./matmul_leakyrelu_custom" 2>&1)
    DURATIONS=$(echo "$PROF_OUT" | grep "Task Duration" | sed 's/.*Task Duration(us): //')
    RELU_US=$(echo "$DURATIONS" | head -1)
    MM_US=$(echo "$DURATIONS" | tail -1)
    TOTAL=$(echo "$RELU_US + $MM_US" | bc 2>/dev/null || echo "N/A")
    rm -rf ./msprof_tmp

    echo "  relu=$RELU_US mm=$MM_US total=$TOTAL"
    echo "$baseM,$baseN,$baseK,$dA,$dB,$sKa,$sKb,$RELU_US,$MM_US,$TOTAL,PASS" >> "../../$RESULTS_FILE"
    cd ../..

done
done
done
done
done
done
done

# Restore original
cp "${ASC_FILE}.bak" "$ASC_FILE"
rm "${ASC_FILE}.bak"

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
echo ""
echo "=== Best (by total) ==="
sort -t, -k10 -n "$RESULTS_FILE" | head -5
