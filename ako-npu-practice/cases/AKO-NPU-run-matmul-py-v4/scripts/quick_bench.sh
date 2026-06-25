#!/bin/bash
# Quick benchmark: compile + correctness + 3-run profiling
# Usage: bash scripts/quick_bench.sh [label]
set -eo pipefail
cd "$(dirname "$0")/.."

LABEL="${1:-quick}"
cd solution
mkdir -p build && cd build
cmake ${ASC_DIR:+-DASC_DIR=$ASC_DIR} .. 2>&1 | tail -2 || { echo "FAIL: cmake"; exit 1; }
make -j4 2>&1 | tail -3 || { echo "FAIL: make"; exit 1; }

# Generate test data
python3 ../../scripts/gen_data.py 2>&1 | tail -1 || { echo "FAIL: gen_data"; exit 1; }

# Correctness
rm -f output/output.bin
./matmul_leakyrelu_custom 2>&1 | tail -3 || { echo "FAIL: exec"; exit 1; }
[ -f output/output.bin ] || { echo "FAIL: no output"; exit 1; }
python3 ../../scripts/verify_result.py output/output.bin output/golden.bin 2>&1
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo "Correctness: FAIL"
    exit 1
fi
echo "Correctness: PASS"

# Profile 3 runs
echo "--- Profiling ($LABEL) ---"
for i in 1 2 3; do
    rm -rf msprof_output
    msprof op --warm-up=10 --launch-count=5 --output=./msprof_output ./matmul_leakyrelu_custom 2>/dev/null
    python3 -c "
import csv, glob, os
opprof = sorted(glob.glob('msprof_output/OPPROF_*'))[-1]
results = {}
for kdir in sorted(os.listdir(opprof)):
    if kdir.startswith('_'):
        kpath = os.path.join(opprof, kdir, '0')
        for f in os.listdir(kpath):
            if f.startswith('OpBasicInfo'):
                with open(os.path.join(kpath, f)) as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        dur = [float(v) for k,v in row.items() if 'Duration' in k]
                        name = 'relu' if 'leakyrelu' in kdir and 'matmul' not in kdir else 'matmul'
                        results[name] = dur[0] if dur else 0
print(f'Run {$i}: matmul={results.get(\"matmul\",0):.2f} relu={results.get(\"relu\",0):.2f} total={sum(results.values()):.2f}')
"
done
