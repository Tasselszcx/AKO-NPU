#!/usr/bin/env python3
"""Sweep tiling parameters and measure performance."""
import subprocess
import re
import csv
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")

# Parameters to sweep (only valid tiling configs from the sweep_tiling analysis)
configs = [
    # (usedCoreNum, baseM, baseN)
    (1, 64, 64),
    (1, 64, 128),
    (1, 64, 256),
    (1, 128, 64),
    (1, 128, 128),
    (1, 128, 256),
    (1, 256, 64),
    (1, 256, 128),
    (1, 512, 64),
]

def modify_asc(cores, baseM, baseN):
    """Modify the .asc file with new tiling parameters."""
    with open(ASC_FILE, 'r') as f:
        content = f.read()

    # Replace usedCoreNum
    content = re.sub(r'int usedCoreNum = \d+;', f'int usedCoreNum = {cores};', content)
    # Replace baseM
    content = re.sub(r'int baseM = \d+;', f'int baseM = {baseM};', content)
    # Replace baseN
    content = re.sub(r'int baseN = \d+;', f'int baseN = {baseN};', content)

    with open(ASC_FILE, 'w') as f:
        f.write(content)

def run_bench():
    """Run bench and return (duration, passed)."""
    env = os.environ.copy()
    result = subprocess.run(
        ['bash', os.path.join(PROJECT_ROOT, 'scripts', 'bench.sh')],
        capture_output=True, text=True, timeout=180, cwd=PROJECT_ROOT, env=env
    )
    output = result.stdout + result.stderr

    # Extract task duration
    duration_match = re.search(r'Task Duration\(us\): ([\d.]+)', output)
    duration = float(duration_match.group(1)) if duration_match else None

    # Check precision
    passed = 'test pass!' in output

    return duration, passed

results = []
print(f"Sweeping {len(configs)} configurations...")
print(f"{'cores':>5} {'baseM':>5} {'baseN':>5} {'duration_us':>12} {'passed':>6}")
print("-" * 45)

for cores, baseM, baseN in configs:
    modify_asc(cores, baseM, baseN)
    try:
        duration, passed = run_bench()
        status = "PASS" if passed else "FAIL"
        dur_str = f"{duration:.2f}" if duration else "N/A"
        print(f"{cores:>5} {baseM:>5} {baseN:>5} {dur_str:>12} {status:>6}")
        results.append((cores, baseM, baseN, duration, passed))
    except Exception as e:
        print(f"{cores:>5} {baseM:>5} {baseN:>5} {'ERROR':>12} {'ERR':>6} ({e})")
        results.append((cores, baseM, baseN, None, False))

# Save results
csv_file = os.path.join(PROJECT_ROOT, 'scripts', 'sweep_results.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['cores', 'baseM', 'baseN', 'duration_us', 'passed'])
    for r in results:
        writer.writerow(r)

print(f"\nResults saved to {csv_file}")

# Print best config
passed_results = [(c, bm, bn, d) for c, bm, bn, d, p in results if p and d is not None]
if passed_results:
    best = min(passed_results, key=lambda x: x[3])
    print(f"\nBest config: cores={best[0]}, baseM={best[1]}, baseN={best[2]}, duration={best[3]:.2f}us")
