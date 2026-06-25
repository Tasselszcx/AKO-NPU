#!/bin/bash
# Verify top configs with higher launch count
set -eo pipefail
cd "$(dirname "$0")/.."

ASC_FILE="solution/matmul_leakyrelu.asc"
BUILD_DIR="solution/build"
cp "$ASC_FILE" "${ASC_FILE}.bak"

echo "config,matmul_us,relu_us,total_us" > top_configs_results.csv

# Top configs to verify with LC=10
configs=(
  "2 2 1 1"
  "3 3 1 1"
  "4 4 2 2"
  "4 2 2 1"
  "8 8 4 4"
)

for cfg in "${configs[@]}"; do
  read -r dA dB sKa sKb <<< "$cfg"

  cp "${ASC_FILE}.bak" "$ASC_FILE"
  sed -i "s/constantCFG.depthA1 = [0-9]*/constantCFG.depthA1 = $dA/" "$ASC_FILE"
  sed -i "s/constantCFG.depthB1 = [0-9]*/constantCFG.depthB1 = $dB/" "$ASC_FILE"
  sed -i "s/constantCFG.stepKa = [0-9]*/constantCFG.stepKa = $sKa/" "$ASC_FILE"
  sed -i "s/constantCFG.stepKb = [0-9]*/constantCFG.stepKb = $sKb/" "$ASC_FILE"

  cd "$BUILD_DIR"
  make -j4 2>&1 | tail -1
  ./matmul_leakyrelu_custom 2>&1 > /dev/null
  VERIFY=$(python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1 | tail -1)
  if [[ "$VERIFY" != *"PASSED"* ]]; then
    echo "dA=$dA dB=$dB sKa=$sKa sKb=$sKb: FAIL"
    cd ../..
    continue
  fi

  rm -rf msprof_tmp
  PROF_OUT=$(msprof op --warm-up=10 --launch-count=10 --output=./msprof_tmp ./matmul_leakyrelu_custom 2>&1)
  TD_LINES=$(echo "$PROF_OUT" | grep "Task Duration" | grep -oP '[\d.]+')
  RELU_US=$(echo "$TD_LINES" | head -1)
  MATMUL_US=$(echo "$TD_LINES" | tail -1)
  TOTAL=$(python3 -c "print(f'{float(\"${MATMUL_US}\") + float(\"${RELU_US}\"):.2f}')")

  echo "dA=$dA dB=$dB sKa=$sKa sKb=$sKb: mm=${MATMUL_US} relu=${RELU_US} total=${TOTAL}"
  echo "dA=${dA}_dB=${dB}_sKa=${sKa}_sKb=${sKb},$MATMUL_US,$RELU_US,$TOTAL" >> ../../top_configs_results.csv
  cd ../..
  rm -rf "$BUILD_DIR/msprof_tmp"
done

cp "${ASC_FILE}.bak" "$ASC_FILE"
rm "${ASC_FILE}.bak"
echo ""
cat top_configs_results.csv
