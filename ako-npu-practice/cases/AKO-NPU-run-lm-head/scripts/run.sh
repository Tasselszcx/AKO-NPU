#!/bin/bash
# run.sh - Build, generate data, run kernel, verify result
# Usage:
#   bash run.sh [B] [S] [logits_to_keep]        # Full flow (build + run)
#   bash run.sh --skip-build [B] [S] [ltk]       # Skip build, reuse binary
#
# Default workload: B=1, S=128, logits_to_keep=1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse --skip-build flag
SKIP_BUILD=false
if [[ "$1" == "--skip-build" ]]; then
    SKIP_BUILD=true
    shift
fi

# Workload parameters
B=${1:-1}
S=${2:-128}
LTK=${3:-1}

echo "============================================"
echo "lm_head_projection: B=$B S=$S logits_to_keep=$LTK"
echo "============================================"

# Step 1: Build
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo "[1/4] Building..."
    mkdir -p build
    cd build
    cmake .. -DASC_DIR=/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake 2>&1
    make -j$(nproc) 2>&1
    cd "$SCRIPT_DIR"
    echo "[1/4] Build complete."
else
    echo "[1/4] Build skipped (--skip-build)."
fi

# Step 2: Generate test data
echo "[2/4] Generating test data..."
python3 scripts/gen_data.py "$B" "$S" "$LTK"

# Clean old output
rm -f output/output.bin

# Step 3: Run kernel (symlink input/output into build/ so the binary can find them)
echo "[3/4] Running kernel..."
ln -sfn "$SCRIPT_DIR/input" build/input
ln -sfn "$SCRIPT_DIR/output" build/output
cd build
./demo "$B" "$S" "$LTK"
cd "$SCRIPT_DIR"

# Step 4: Verify result
echo "[4/4] Verifying result..."
if [[ -f output/output.bin ]]; then
    python3 scripts/verify_result.py output/output.bin output/golden.bin
else
    echo "[ERROR] output/output.bin not found!"
    exit 1
fi

echo "============================================"
echo "Done."
