#!/usr/bin/python3
"""
Parameter sweep: try different core counts and base tile sizes.
Modifies the .asc file, recompiles, runs msprof, records results.
"""
import subprocess
import re
import os
import csv
import sys

SOLUTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(SOLUTION_DIR, "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(SOLUTION_DIR, "build")

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def modify_params(content, core_num, base_m, base_n):
    content = re.sub(r'int usedCoreNum = \d+;', f'int usedCoreNum = {core_num};', content)
    content = re.sub(r'int baseM = \d+;', f'int baseM = {base_m};', content)
    content = re.sub(r'int baseN = \d+;', f'int baseN = {base_n};', content)
    # Also update numBlocks to match
    content = re.sub(r'uint32_t numBlocks = \d+;', f'uint32_t numBlocks = {core_num};', content)
    return content

def build_and_run(core_num, base_m, base_n):
    """Returns Task Duration in us, or None if failed."""
    original = read_file(ASC_FILE)
    modified = modify_params(original, core_num, base_m, base_n)
    write_file(ASC_FILE, modified)
    
    try:
        # Build
        os.makedirs(BUILD_DIR, exist_ok=True)
        result = subprocess.run(
            f'cd {BUILD_DIR} && cmake .. -DASC_DIR=$ASCEND_HOME_PATH/compiler/tikcpp/ascendc_kernel_cmake 2>&1 && make -j4 2>&1',
            shell=True, capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  BUILD FAILED: cores={core_num}, baseM={base_m}, baseN={base_n}")
            return None, False
        
        # Gen data
        subprocess.run(f'cd {BUILD_DIR} && python3 ../scripts/gen_data.py', shell=True, timeout=30)
        
        # Run with msprof
        msprof_dir = os.path.join(BUILD_DIR, "msprof_sweep")
        os.makedirs(msprof_dir, exist_ok=True)
        os.chmod(msprof_dir, 0o700)
        result = subprocess.run(
            f'cd {BUILD_DIR} && msprof op --warm-up=5 --output={msprof_dir} ./matmul_leakyrelu 2>&1',
            shell=True, capture_output=True, text=True, timeout=120
        )
        
        # Parse Task Duration from output
        match = re.search(r'Task Duration\(us\):\s+([\d.]+)', result.stdout)
        if not match:
            print(f"  NO DURATION: cores={core_num}, baseM={base_m}, baseN={base_n}")
            return None, False
        
        duration = float(match.group(1))
        
        # Verify accuracy
        verify = subprocess.run(
            f'cd {BUILD_DIR} && python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin 2>&1',
            shell=True, capture_output=True, text=True, timeout=30
        )
        correct = verify.returncode == 0
        
        # Clean msprof
        subprocess.run(f'rm -rf {msprof_dir}/OPPROF_*', shell=True)
        
        return duration, correct
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, False
    finally:
        write_file(ASC_FILE, original)

# Parameter space
# M=1024, N=640, K=256
# Valid core counts: factors that divide M*N grid cleanly
# baseM must divide singleCoreM, baseN must divide singleCoreN
configs = [
    # (cores, baseM, baseN) - must ensure M/cores*baseM and N/baseN divide evenly
    (2, 256, 128),     # baseline
    (4, 256, 128),     # 4 cores
    (4, 128, 128),     # 4 cores, smaller baseM
    (5, 128, 128),     # 5 cores  
    (8, 128, 128),     # 8 cores
    (8, 128, 64),      # 8 cores, smaller baseN
    (10, 128, 128),    # 10 cores
    (10, 128, 64),     # 10 cores
    (16, 64, 64),      # 16 cores
    (20, 64, 64),      # 20 cores
    (20, 128, 64),     # 20 cores
    (4, 256, 64),      # 4 cores, baseN=64
    (2, 128, 128),     # 2 cores, smaller base
    (2, 128, 64),      # 2 cores
    (4, 128, 64),      # 4 cores
    (5, 128, 64),      # 5 cores
]

results = []
print(f"{'Cores':>5} {'baseM':>5} {'baseN':>5} {'Duration(us)':>12} {'Correct':>7} {'Speedup':>7}")
print("-" * 55)

baseline_duration = None
for cores, bm, bn in configs:
    dur, correct = build_and_run(cores, bm, bn)
    if dur is not None:
        if baseline_duration is None:
            baseline_duration = dur
        speedup = baseline_duration / dur if dur > 0 else 0
        status = "PASS" if correct else "FAIL"
        print(f"{cores:5d} {bm:5d} {bn:5d} {dur:12.2f} {status:>7} {speedup:7.2f}x")
        results.append((cores, bm, bn, dur, correct, speedup))
    else:
        print(f"{cores:5d} {bm:5d} {bn:5d} {'FAILED':>12} {'---':>7} {'---':>7}")

# Write CSV
csv_path = os.path.join(SOLUTION_DIR, "sweep_results.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['cores', 'baseM', 'baseN', 'duration_us', 'correct', 'speedup'])
    for r in results:
        writer.writerow(r)

print(f"\nResults saved to {csv_path}")
if results:
    best = min(results, key=lambda x: x[3])
    print(f"Best: cores={best[0]}, baseM={best[1]}, baseN={best[2]}, duration={best[3]:.2f}us, speedup={best[5]:.2f}x")
