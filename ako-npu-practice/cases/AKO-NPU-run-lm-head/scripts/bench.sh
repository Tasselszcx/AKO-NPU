#!/bin/bash
# Benchmark wrapper with trajectory tracking
# Usage: bash scripts/bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# --- Bench command ---
# Run benchmark without exiting on failure — we need trajectory even for failed runs
set +e
(
set -e
echo "===== AKO Benchmark: lm_head_projection ====="
echo "Timestamp: $TIMESTAMP  Label: $LABEL"

SOLUTION_DIR="solution"
BUILD_DIR="$SOLUTION_DIR/build"

# Step 1: Build
echo ""
echo "[1/3] Building..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake 2>&1
make -j$(nproc) 2>&1
cd - > /dev/null
chmod 700 "$BUILD_DIR"
echo "[1/3] Build complete."

# Step 2: Generate data for all workloads
echo ""
echo "[2/3] Generating test data for all workloads..."

# Workload definitions: "B S logits_to_keep label"
WORKLOADS=(
    "1 128 1 W1_decode_single"
    "1 512 1 W2_decode_long"
    "1 1024 128 W3_prefill_multi"
    "4 256 1 W4_batch_decode"
)

mkdir -p "$SOLUTION_DIR/input" "$SOLUTION_DIR/output"

# Step 3: Run each workload with msprof
echo ""
echo "[3/3] Running workloads with msprof..."

MSPROF_BASE="$SOLUTION_DIR/msprof_output"
rm -rf "$MSPROF_BASE"
mkdir -m 700 -p "$MSPROF_BASE"

ALL_PASS=true

for wl in "${WORKLOADS[@]}"; do
    read -r B S LTK WL_LABEL <<< "$wl"
    echo ""
    echo "--- Workload $WL_LABEL: B=$B S=$S logits_to_keep=$LTK ---"

    # Generate data
    python3 scripts/gen_data.py "$B" "$S" "$LTK" --output-dir "$SOLUTION_DIR"

    rm -f "$SOLUTION_DIR/output/output.bin"

    # Run with msprof - use a clean run directory with proper permissions
    WL_OUTPUT="$(realpath $MSPROF_BASE)/$WL_LABEL"
    mkdir -p "$WL_OUTPUT"
    chmod 700 "$WL_OUTPUT"

    # Create a clean run dir with demo binary and data links
    RUN_DIR="$SOLUTION_DIR/run_$$"
    rm -rf "$RUN_DIR"
    mkdir -m 700 -p "$RUN_DIR"
    cp "$BUILD_DIR/demo" "$RUN_DIR/"
    ln -sfn "$(realpath $SOLUTION_DIR/input)" "$RUN_DIR/input"
    ln -sfn "$(realpath $SOLUTION_DIR/output)" "$RUN_DIR/output"

    cd "$RUN_DIR"
    msprof op --warm-up=10 --launch-count=5 --output="$WL_OUTPUT" ./demo "$B" "$S" "$LTK" 2>&1
    cd - > /dev/null
    rm -rf "$RUN_DIR"

    # Verify correctness
    if [[ -f "$SOLUTION_DIR/output/output.bin" ]]; then
        python3 scripts/verify_result.py "$SOLUTION_DIR/output/output.bin" "$SOLUTION_DIR/output/golden.bin" 2>&1 || {
            echo "[FAIL] Workload $WL_LABEL precision check failed!"
            ALL_PASS=false
        }
    else
        echo "[FAIL] Workload $WL_LABEL: output.bin not found!"
        ALL_PASS=false
    fi

    # Extract Task Duration from OpBasicInfo.csv
    OPPROF_DIR=$(ls -td "$WL_OUTPUT"/OPPROF_* 2>/dev/null | head -1)
    if [[ -n "$OPPROF_DIR" && -f "$OPPROF_DIR/OpBasicInfo.csv" ]]; then
        echo "  msprof data: $OPPROF_DIR"
        # Print summary line from OpBasicInfo
        python3 -c "
import csv, sys
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        td = row.get('Task Duration(us)', 'N/A')
        bd = row.get('Block Dim', 'N/A')
        print(f'  Task Duration: {td} us, Block Dim: {bd}')
" 2>/dev/null || echo "  (Could not parse OpBasicInfo.csv)"
    fi
done

echo ""
echo "===== Summary ====="
if $ALL_PASS; then
    echo "All workloads PASSED correctness check."
else
    echo "Some workloads FAILED correctness check!"
fi

# Print all Task Durations
echo ""
echo "Performance Summary:"
echo "| Workload | B | S | ltk | Task Duration (us) |"
echo "|----------|---|-----|-----|--------------------|"
for wl in "${WORKLOADS[@]}"; do
    read -r B S LTK WL_LABEL <<< "$wl"
    OPPROF_DIR=$(ls -td "$MSPROF_BASE/$WL_LABEL"/OPPROF_* 2>/dev/null | head -1)
    if [[ -n "$OPPROF_DIR" && -f "$OPPROF_DIR/OpBasicInfo.csv" ]]; then
        TD=$(python3 -c "
import csv
with open('$OPPROF_DIR/OpBasicInfo.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row.get('Task Duration(us)', 'N/A'))
        break
" 2>/dev/null || echo "N/A")
        echo "| $WL_LABEL | $B | $S | $LTK | $TD |"
    else
        echo "| $WL_LABEL | $B | $S | $LTK | N/A |"
    fi
done

echo ""
echo "===== Benchmark Complete ====="
) 2>&1 | tee _bench_output.txt
BENCH_EXIT=$?
set -e
# --- End bench command ---

# --- Trajectory ---
if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi
mkdir -p "$TRAJ_DIR"
cp -r solution/* "$TRAJ_DIR/"
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
echo "Trajectory saved to: $TRAJ_DIR"

exit $BENCH_EXIT
