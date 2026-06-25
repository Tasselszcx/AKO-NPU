#!/usr/bin/env python3
"""Sweep usedCoreNum with IterateAll approach (numBlocks=1)."""
import subprocess
import re
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")

# Valid configs from earlier sweep_tiling for different core counts
configs = [
    # (usedCoreNum, baseM, baseN)
    (1, 256, 128),
    (2, 256, 128),   # current best
    (4, 256, 64),
    (4, 128, 128),
    (4, 64, 64),
    (8, 256, 64),
    (8, 128, 128),
    (8, 64, 64),
    (16, 128, 128),
    (16, 64, 64),
    (20, 128, 128),
    (20, 64, 64),
]

def modify_asc(cores, baseM, baseN):
    with open(ASC_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'int usedCoreNum = \d+;', f'int usedCoreNum = {cores};', content)
    content = re.sub(r'int baseM = \d+;', f'int baseM = {baseM};', content)
    content = re.sub(r'int baseN = \d+;', f'int baseN = {baseN};', content)
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
    tiling_ok = 'gen tiling failed' not in output
    return duration, passed, tiling_ok

results = []
print(f"{'cores':>5} {'baseM':>5} {'baseN':>5} {'dur_us':>8} {'status':>8}")
print("-" * 40)

for cores, baseM, baseN in configs:
    modify_asc(cores, baseM, baseN)
    try:
        duration, passed, tiling_ok = run_bench()
        if not tiling_ok:
            status = "TLFAIL"
        elif passed:
            status = "PASS"
        else:
            status = "PRFAIL"
        dur_str = f"{duration:.1f}" if duration else "N/A"
        print(f"{cores:>5} {baseM:>5} {baseN:>5} {dur_str:>8} {status:>8}")
        results.append((cores, baseM, baseN, duration, status))
    except Exception as e:
        print(f"{cores:>5} {baseM:>5} {baseN:>5} {'ERR':>8} {'ERR':>8}")
        results.append((cores, baseM, baseN, None, "ERR"))

passed = [(c, bm, bn, d) for c, bm, bn, d, s in results if s == "PASS" and d]
if passed:
    best = min(passed, key=lambda x: x[3])
    print(f"\nBest: cores={best[0]}, baseM={best[1]}, baseN={best[2]}, duration={best[3]:.1f}us")
