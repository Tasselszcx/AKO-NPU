#!/usr/bin/env python3
"""Sweep usedCoreNum with numBlocks=1 (Matmul handles multi-core internally in mix mode)."""
import subprocess
import re
import csv
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")

# Valid configs from sweep_tiling output, all with numBlocks=1
configs = [
    # (usedCoreNum, baseM, baseN)
    (1, 256, 128),   # singleCore = 1024x640
    (1, 128, 128),
    (2, 256, 128),   # original baseline: singleCore = 512x640
    (2, 128, 128),
    (2, 128, 64),
    (2, 64, 128),
    (4, 256, 128),   # singleCore = 512x320
    (4, 128, 128),
    (4, 128, 64),
    (4, 64, 128),
    (4, 64, 64),
    (8, 256, 128),   # singleCore = 256x320
    (8, 128, 128),
    (8, 128, 64),
    (8, 64, 128),
    (8, 64, 64),
    (16, 128, 128),  # singleCore = 128x320
    (16, 64, 128),
    (16, 64, 64),
    (20, 128, 128),  # singleCore = 128x320
    (20, 64, 128),
    (20, 64, 64),
]

def modify_asc(cores, baseM, baseN):
    with open(ASC_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'int usedCoreNum = \d+;', f'int usedCoreNum = {cores};', content)
    content = re.sub(r'int baseM = \d+;', f'int baseM = {baseM};', content)
    content = re.sub(r'int baseN = \d+;', f'int baseN = {baseN};', content)
    content = re.sub(r'uint32_t numBlocks = .*?;', 'uint32_t numBlocks = 1;', content)
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
    dim_match = re.search(r'Mix Block Dim: (\d+)', output)
    mix_dim = int(dim_match.group(1)) if dim_match else None
    passed = 'test pass!' in output
    tiling_ok = 'gen tiling failed' not in output
    return duration, passed, mix_dim, tiling_ok

results = []
print(f"{'cores':>5} {'baseM':>5} {'baseN':>5} {'dur_us':>8} {'mix_dim':>7} {'status':>6}")
print("-" * 45)

for cores, baseM, baseN in configs:
    modify_asc(cores, baseM, baseN)
    try:
        duration, passed, mix_dim, tiling_ok = run_bench()
        if not tiling_ok:
            status = "TLFAIL"
        elif passed:
            status = "PASS"
        else:
            status = "PRFAIL"
        dur_str = f"{duration:.1f}" if duration else "N/A"
        mix_str = str(mix_dim) if mix_dim else "N/A"
        print(f"{cores:>5} {baseM:>5} {baseN:>5} {dur_str:>8} {mix_str:>7} {status:>6}")
        results.append((cores, baseM, baseN, duration, mix_dim, status))
    except Exception as e:
        print(f"{cores:>5} {baseM:>5} {baseN:>5} {'ERR':>8} {'N/A':>7} {'ERR':>6}")
        results.append((cores, baseM, baseN, None, None, "ERR"))

csv_file = os.path.join(PROJECT_ROOT, 'scripts', 'sweep_results3.csv')
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['cores', 'baseM', 'baseN', 'duration_us', 'mix_dim', 'status'])
    for r in results:
        writer.writerow(r)

passed_results = [(c, bm, bn, d, md) for c, bm, bn, d, md, s in results if s == "PASS" and d is not None]
if passed_results:
    best = min(passed_results, key=lambda x: x[3])
    print(f"\nBest: cores={best[0]}, baseM={best[1]}, baseN={best[2]}, duration={best[3]:.1f}us, mix_dim={best[4]}")
