#!/bin/bash
# Iterations 51-80: Batch of rapid experiments
# Each test: modify -> build -> run -> verify -> record -> revert
set -eo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
ASC="${PROJECT_ROOT}/solution/dsa_indexer.asc"
BEST="${ASC}.best_iter50"
RESULTS="${PROJECT_ROOT}/iter51_80_results.csv"

echo "iter,description,e2e_best,e2e_mean,correct" > "$RESULTS"

run_test() {
    local ITER=$1
    local DESC=$2
    cd "${PROJECT_ROOT}/solution/build"
    cmake -DCMAKE_PREFIX_PATH=${ASCEND_HOME_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake .. > /dev/null 2>&1
    if ! make -j4 > /dev/null 2>&1; then
        echo "$ITER,$DESC,COMPILE_FAIL,N/A,NO" >> "$RESULTS"
        echo "  [$ITER] $DESC: COMPILE FAILED"
        cd "$PROJECT_ROOT"
        cp "$BEST" "$ASC"
        return
    fi
    python3 "${PROJECT_ROOT}/scripts/gen_data.py" 1 1 4096 > /dev/null 2>&1
    OUTPUT=$(./dsa_indexer 1 1 4096 2>&1)
    E2E_BEST=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*best=\([0-9.]*\).*/\1/')
    E2E_MEAN=$(echo "$OUTPUT" | grep "E2E Summary" | sed 's/.*mean=\([0-9.]*\).*/\1/')
    CORRECT="YES"
    VERIFY=$(python3 "${PROJECT_ROOT}/scripts/verify_result.py" 1 1 4096 2>&1) || CORRECT="NO"
    echo "$VERIFY" | grep -q "FAIL" && CORRECT="NO"
    MATCH_RATE=$(echo "$VERIFY" | grep "Match rate" | head -1 | sed 's/.*(\(.*%\)).*/\1/')
    echo "$ITER,$DESC,$E2E_BEST,$E2E_MEAN,$CORRECT" >> "$RESULTS"
    echo "  [$ITER] $DESC: best=$E2E_BEST mean=$E2E_MEAN correct=$CORRECT match=$MATCH_RATE"
    cd "$PROJECT_ROOT"
    cp "$BEST" "$ASC"  # Always revert to best
}

# --- Iter 51: baseM=32 (half-M blocks) ---
echo "=== Iter 51: baseM=32 ==="
cp "$BEST" "$ASC"
sed -i 's/64,             \/\/ basicM = 64/32,             \/\/ basicM = 32/' "$ASC"
sed -i 's/N_HEADS,        \/\/ singleCoreM = 64/N_HEADS,        \/\/ singleCoreM = 64/' "$ASC"
run_test 51 "baseM=32"

# --- Iter 52: Reduce warmup to 2 ---
echo "=== Iter 52: 2 warmup ==="
cp "$BEST" "$ASC"
sed -i 's/for (int w = 0; w < 5; w++)/for (int w = 0; w < 2; w++)/' "$ASC"
run_test 52 "2warmup"

# --- Iter 53: baseN=64 ---
echo "=== Iter 53: baseN=64 ==="
cp "$BEST" "$ASC"
sed -i 's/128,            \/\/ basicN = 128/64,            \/\/ basicN = 64/' "$ASC"
# Need to update depthB1 since more B blocks: (256/64)*(128/64)=4*2=8
sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 8/' "$ASC"
run_test 53 "baseN=64"

# --- Iter 54: baseN=256 (fewer, larger blocks) ---
echo "=== Iter 54: baseN=256 ==="
cp "$BEST" "$ASC"
sed -i 's/128,            \/\/ basicN = 128/256,            \/\/ basicN = 256/' "$ASC"
# depthB1: (256/256)*(128/64)=1*2=2
sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 2/' "$ASC"
run_test 54 "baseN=256"

# --- Iter 55: SpecialMDL template ---
echo "=== Iter 55: SpecialMDL ==="
cp "$BEST" "$ASC"
sed -i 's/GetMMConfig<MatmulConfigMode::CONFIG_MDL>/GetMMConfig<MatmulConfigMode::CONFIG_MDL>/' "$ASC"
# Add doSpecialMDL
sed -i 's/mmCFG.enableInit = false;/mmCFG.enableInit = false;\n    mmCFG.doSpecialMDL = true;\n    mmCFG.doMultiDataLoad = false;/' "$ASC"
run_test 55 "SpecialMDL"

# --- Iter 56: Disable UnitFlag ---
echo "=== Iter 56: no UnitFlag ==="
cp "$BEST" "$ASC"
sed -i 's/mmCFG.enUnitFlag = true;/mmCFG.enUnitFlag = false;/' "$ASC"
run_test 56 "no_UnitFlag"

# --- Iter 57: enableEnd=false ---
echo "=== Iter 57: enableEnd=false ==="
cp "$BEST" "$ASC"
sed -i 's/mmCFG.enableInit = false;/mmCFG.enableInit = false;\n    mmCFG.enableEnd = false;/' "$ASC"
run_test 57 "enableEnd_false"

# --- Iter 58: 8 cores (from 16) ---
echo "=== Iter 58: 8 cores ==="
cp "$BEST" "$ASC"
sed -i 's/constexpr int32_t NUM_MATMUL_CORES = 16/constexpr int32_t NUM_MATMUL_CORES = 8/' "$ASC"
sed -i 's/128,            \/\/ basicN = 128/256,            \/\/ basicN = 256/' "$ASC"
run_test 58 "8cores_baseN256"

# --- Iter 59: 4 cores ---
echo "=== Iter 59: 4 cores ==="
cp "$BEST" "$ASC"
sed -i 's/constexpr int32_t NUM_MATMUL_CORES = 16/constexpr int32_t NUM_MATMUL_CORES = 4/' "$ASC"
sed -i 's/128,            \/\/ basicN = 128/256,            \/\/ basicN = 256/' "$ASC"
run_test 59 "4cores_baseN256"

# --- Iter 60: Causal mask check removal (S_q=1, no causal mask) ---
echo "=== Iter 60: skip causal mask check ==="
cp "$BEST" "$ASC"
# The needCausalMask is already 0 for S_q=1, so the check is a branch prediction hit
# Just verify timing is same
run_test 60 "verify_no_causal_mask"

# --- Iter 61: Reduce D2H by only copying first 64 heads * nPerCore per core ---
echo "=== Iter 61: Selective D2H (skip unused workspace) ==="
cp "$BEST" "$ASC"
# Already copying exactly N_HEADS * S_kv * sizeof(float) = 1MB
run_test 61 "verify_minimal_D2H"

# --- Iter 62: Try FIRSTN traverse direction in constant tiling ---
echo "=== Iter 62: FIRSTN traverse ==="
cp "$BEST" "$ASC"
sed -i 's/mmCFG.enableInit = false;/mmCFG.enableInit = false;\n    mmCFG.IterateOrder = IterateOrder::ORDER_N;/' "$ASC"
run_test 62 "FIRSTN_traverse"

# --- Iter 63: enableKdimReorderLoad with 16 cores ---
echo "=== Iter 63: enableKdimReorderLoad ==="
cp "$BEST" "$ASC"
sed -i 's/mmCFG.enableInit = false;/mmCFG.enableInit = false;\n    mmCFG.enableKdimReorderLoad = true;/' "$ASC"
run_test 63 "enableKdimReorderLoad"

# --- Iter 64: Different depthA1 values ---
echo "=== Iter 64: depthA1=2 ==="
cp "$BEST" "$ASC"
sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 2/' "$ASC"
run_test 64 "depthA1_2"

# --- Iter 65: depthA1=8 ---
echo "=== Iter 65: depthA1=8 ==="
cp "$BEST" "$ASC"
sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 8/' "$ASC"
run_test 65 "depthA1_8"

# --- Iter 66: depthB1=8 ---
echo "=== Iter 66: depthB1=8 ==="
cp "$BEST" "$ASC"
sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 8/' "$ASC"
run_test 66 "depthB1_8"

# --- Iter 67: depthA1=1, depthB1=2 (minimal L1 cache) ---
echo "=== Iter 67: minimal L1 ==="
cp "$BEST" "$ASC"
sed -i 's/constantCFG.depthA1 = 4/constantCFG.depthA1 = 1/' "$ASC"
sed -i 's/constantCFG.depthB1 = 4/constantCFG.depthB1 = 2/' "$ASC"
run_test 67 "minimal_L1"

# --- Iter 68: Skip zero-weight heads entirely (pre-filter) ---
echo "=== Iter 68: pre-filter zero weights ==="
cp "$BEST" "$ASC"
# The zero-weight skip is already implemented with `if (w == 0.0f) continue`
run_test 68 "verify_zero_weight_skip"

# --- Iter 69: Use restrict on all hot loop pointers ---
echo "=== Iter 69: restrict pointers ==="
cp "$BEST" "$ASC"
# Already using __restrict__ on coreOut, sc, headRow
run_test 69 "verify_restrict"

# --- Iter 70: Use float instead of double for timing ---
echo "=== Iter 70: float timing ==="
cp "$BEST" "$ASC"
# Timing uses double, which is fine for precision
run_test 70 "verify_timing_precision"

echo ""
echo "=== Batch Results ==="
cat "$RESULTS"

# Ensure best config is restored
cp "$BEST" "$ASC"
