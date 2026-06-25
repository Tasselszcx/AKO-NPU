#!/usr/bin/env python3
"""Sweep tiling baseM/baseN while keeping usedCoreNum=2, numBlocks=1."""
import subprocess
import re
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")

# Only sweep baseM and baseN for usedCoreNum=2 (valid configs from sweep_tiling)
# For cores=2, singleCoreM=512, singleCoreN=640
# baseM must <= 512, baseN must <= 640
configs = [
    # (baseM, baseN) - all valid for cores=2 from sweep
    (64, 64),
    (64, 128),
    (64, 256),
    (128, 64),
    (128, 128),
    (128, 256),
    (256, 64),
    (256, 128),    # original baseline
    (512, 64),
]

def modify_asc(baseM, baseN):
    with open(ASC_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'int baseM = \d+;', f'int baseM = {baseM};', content)
    content = re.sub(r'int baseN = \d+;', f'int baseN = {baseN};', content)
    # Ensure usedCoreNum=2 and numBlocks=1
    content = re.sub(r'int usedCoreNum = \d+;', 'int usedCoreNum = 2;', content)
    # Ensure numBlocks=1 (hardcoded)
    content = re.sub(r'uint32_t numBlocks = .*?;', 'uint32_t numBlocks = 1;', content)
    # Ensure original workspace
    content = re.sub(r'size_t workspaceSize = .*?;', 'size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;', content)
    with open(ASC_FILE, 'w') as f:
        f.write(content)

def run_bench():
    result = subprocess.run(
        ['bash', os.path.join(PROJECT_ROOT, 'scripts', 'bench.sh')],
        capture_output=True, text=True, timeout=180, cwd=PROJECT_ROOT
    )
    output = result.stdout + result.stderr
    duration_match = re.search(r'Task Duration\(us\): ([\d.]+)', output)
    duration = float(duration_match.group(1)) if duration_match else None
    passed = 'test pass!' in output
    return duration, passed

results = []
print(f"{'baseM':>5} {'baseN':>5} {'duration_us':>12} {'passed':>6}")
print("-" * 35)

for baseM, baseN in configs:
    modify_asc(baseM, baseN)
    try:
        duration, passed = run_bench()
        status = "PASS" if passed else "FAIL"
        dur_str = f"{duration:.2f}" if duration else "N/A"
        print(f"{baseM:>5} {baseN:>5} {dur_str:>12} {status:>6}")
        results.append((baseM, baseN, duration, passed))
    except Exception as e:
        print(f"{baseM:>5} {baseN:>5} {'ERROR':>12} {'ERR':>6}")
        results.append((baseM, baseN, None, False))

csv_file = os.path.join(PROJECT_ROOT, 'scripts', 'sweep_results2.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['baseM', 'baseN', 'duration_us', 'passed'])
    for r in results:
        writer.writerow(r)

passed_results = [(bm, bn, d) for bm, bn, d, p in results if p and d is not None]
if passed_results:
    best = min(passed_results, key=lambda x: x[2])
    print(f"\nBest: baseM={best[0]}, baseN={best[1]}, duration={best[2]:.2f}us")
else:
    print("\nNo passing configs found!")
