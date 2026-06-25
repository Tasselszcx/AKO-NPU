#!/usr/bin/env python3
"""Mega sweep: systematic exploration of all parameter combinations."""
import subprocess, re, os, sys, csv, itertools, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(PROJECT_ROOT, "solution", "build")
ASC_DIR = "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def apply_config(content, config):
    c = content
    for key, val in config.items():
        if key == 'traverse':
            c = re.sub(r'MatrixTraverse::\w+', f'MatrixTraverse::{val}', c)
        elif key == 'usedCoreNum':
            c = re.sub(r'int usedCoreNum = \d+;', f'int usedCoreNum = {val};', c)
        elif key == 'baseM':
            c = re.sub(r'int baseM = \d+;', f'int baseM = {val};', c)
        elif key == 'baseN':
            c = re.sub(r'int baseN = \d+;', f'int baseN = {val};', c)
        elif key == 'tileSize':
            c = re.sub(r'const uint32_t tileSize = \d+;', f'const uint32_t tileSize = {val};', c)
        elif key == 'stepM':
            c = re.sub(r'tilingData\.stepM = \d+;', f'tilingData.stepM = {val};', c)
        elif key == 'stepN':
            c = re.sub(r'tilingData\.stepN = \d+;', f'tilingData.stepN = {val};', c)
        elif key == 'isTransB':
            c = re.sub(r'bool isTransB = (true|false);', f'bool isTransB = {str(val).lower()};', c)
        elif key == 'bufferSpace':
            c = re.sub(r'tilingApi\.SetBufferSpace\([^)]+\)', f'tilingApi.SetBufferSpace({val})', c)
    return c

def build_and_run(content, timeout_s=120):
    write_file(ASC_FILE, content)
    cmd = f"""
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    cd {BUILD_DIR}
    cmake -DASC_DIR={ASC_DIR} .. >/dev/null 2>&1 && make -j 2>&1 | tail -3
    if [ $? -ne 0 ]; then echo "BUILD_FAILED"; exit 1; fi
    python3 {PROJECT_ROOT}/scripts/gen_data.py 2>/dev/null
    MSPROF_OUT={PROJECT_ROOT}/msprof_sweep
    rm -rf $MSPROF_OUT
    mkdir -p $MSPROF_OUT && chmod 750 $MSPROF_OUT && chmod 750 {BUILD_DIR}
    timeout 60 msprof op --warm-up=10 --launch-count=5 --output=$MSPROF_OUT ./demo 2>&1
    timeout 10 python3 {PROJECT_ROOT}/scripts/verify_result.py output/output.bin output/golden.bin 2>&1
    """
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout_s)
        output = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return None, False, "timeout"
    if 'BUILD_FAILED' in output or 'fatal error' in output:
        return None, False, "build_fail"
    dur_match = re.search(r'Task Duration\(us\):\s*([\d.]+)', output)
    duration = float(dur_match.group(1)) if dur_match else None
    prec_pass = 'test pass' in output.lower()
    return duration, prec_pass, output

# Generate all experiments
experiments = []

# 1. Tile size exploration (5 configs)
for ts in [1024, 3072, 5120, 7168, 9216]:
    experiments.append((f"tile{ts}", {"tileSize": ts}))

# 2. Step combinations (8 configs)
for sm, sn in [(1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (3,1), (3,3)]:
    experiments.append((f"step{sm}_{sn}", {"stepM": sm, "stepN": sn}))

# 3. BaseM/BaseN with FIRSTN (6 configs)
for bm, bn in [(64,128), (128,64), (192,128), (256,160), (384,128), (192,192)]:
    experiments.append((f"bM{bm}_bN{bn}", {"baseM": bm, "baseN": bn}))

# 4. Combined best options (10 configs)
experiments.append(("best_combo1", {"stepM": 2, "tileSize": 10240}))
experiments.append(("best_combo2", {"stepM": 2, "tileSize": 12288}))
experiments.append(("best_combo3", {"stepM": 2, "isTransB": True}))
experiments.append(("best_combo4", {"stepM": 2, "tileSize": 6144}))
experiments.append(("best_combo5", {"stepN": 2, "tileSize": 10240}))
experiments.append(("best_combo6", {"stepN": 2, "tileSize": 12288}))
experiments.append(("best_combo7", {"stepM": 2, "stepN": 2, "tileSize": 10240}))
experiments.append(("best_combo8", {"isTransB": True, "tileSize": 10240}))
experiments.append(("best_combo9", {"isTransB": True, "tileSize": 12288}))
experiments.append(("best_combo10", {"stepM": 3, "tileSize": 10240}))

# 5. Buffer space exploration (5 configs)
for bs in ["524288, -1, -1", "786432, -1, -1", "1048576, -1, -1", "-1, 65536, -1", "-1, -1, 65536"]:
    experiments.append((f"buf_{bs.replace(', ', '_').replace('-1', 'auto')}", {"bufferSpace": bs}))

# 6. Core count with M-only split (4 configs)
for cores in [3, 4, 6, 8]:
    experiments.append((f"core{cores}", {"usedCoreNum": cores}))

# 7. FIRSTM with various configs (5 configs)
experiments.append(("FIRSTM_base", {"traverse": "FIRSTM"}))
experiments.append(("FIRSTM_stepM2", {"traverse": "FIRSTM", "stepM": 2}))
experiments.append(("FIRSTM_tile10240", {"traverse": "FIRSTM", "tileSize": 10240}))
experiments.append(("FIRSTM_transB", {"traverse": "FIRSTM", "isTransB": True}))
experiments.append(("FIRSTM_bM128_bN256", {"traverse": "FIRSTM", "baseM": 128, "baseN": 256}))

# 8. Extreme tile sizes (3 configs)
experiments.append(("tile512", {"tileSize": 512}))
experiments.append(("tile256", {"tileSize": 256}))
experiments.append(("tile15360", {"tileSize": 15360}))

# 9. More baseM/baseN (5 configs)
for bm, bn in [(256, 32), (128, 160), (64, 256), (256, 96), (192, 64)]:
    experiments.append((f"bM{bm}_bN{bn}", {"baseM": bm, "baseN": bn}))

# 10. TransB combinations (5 configs)
experiments.append(("transB_bM128_bN128", {"isTransB": True, "baseM": 128, "baseN": 128}))
experiments.append(("transB_bM256_bN64", {"isTransB": True, "baseM": 256, "baseN": 64}))
experiments.append(("transB_stepM2_bM128_bN128", {"isTransB": True, "stepM": 2, "baseM": 128, "baseN": 128}))
experiments.append(("transB_tile6144_stepM2", {"isTransB": True, "tileSize": 6144, "stepM": 2}))
experiments.append(("transB_bM128_bN256", {"isTransB": True, "baseM": 128, "baseN": 256}))

# 11. Step 1,1 with other changes (5 configs)
experiments.append(("step1_1_tile10240", {"stepM": 1, "stepN": 1, "tileSize": 10240}))
experiments.append(("step1_1_tile12288", {"stepM": 1, "stepN": 1, "tileSize": 12288}))
experiments.append(("step1_1_transB", {"stepM": 1, "stepN": 1, "isTransB": True}))
experiments.append(("step1_1_bM128_bN256", {"stepM": 1, "stepN": 1, "baseM": 128, "baseN": 256}))
experiments.append(("step1_1_bM192_bN128", {"stepM": 1, "stepN": 1, "baseM": 192, "baseN": 128}))

# 12. Additional combos (10 configs)
experiments.append(("combo_bM192_bN128_stepM2", {"baseM": 192, "baseN": 128, "stepM": 2}))
experiments.append(("combo_bM192_bN128_tile10240", {"baseM": 192, "baseN": 128, "tileSize": 10240}))
experiments.append(("combo_bM64_bN128_tile10240", {"baseM": 64, "baseN": 128, "tileSize": 10240}))
experiments.append(("combo_bM384_bN128_stepM2", {"baseM": 384, "baseN": 128, "stepM": 2}))
experiments.append(("combo_bM256_bN128_buf786K", {"bufferSpace": "786432, -1, -1"}))
experiments.append(("combo_stepM2_buf786K", {"stepM": 2, "bufferSpace": "786432, -1, -1"}))
experiments.append(("combo_stepM2_stepN3", {"stepM": 2, "stepN": 3}))
experiments.append(("combo_tile10240_stepM2_transB", {"tileSize": 10240, "stepM": 2, "isTransB": True}))
experiments.append(("combo_tile5120_stepM2", {"tileSize": 5120, "stepM": 2}))
experiments.append(("combo_tile3072_stepM2", {"tileSize": 3072, "stepM": 2}))

# 13. More step variations (5 configs)
experiments.append(("step4_1", {"stepM": 4, "stepN": 1}))
experiments.append(("step1_5", {"stepM": 1, "stepN": 5}))
experiments.append(("step2_5", {"stepM": 2, "stepN": 5}))
experiments.append(("step5_1", {"stepM": 5, "stepN": 1}))
experiments.append(("step5_5", {"stepM": 5, "stepN": 5}))

# 14. BaseN that divides 640 evenly (5 configs)
for bn in [32, 40, 64, 80, 160]:
    experiments.append((f"bM256_bN{bn}_div640", {"baseN": bn}))

# 15. Large baseM with suitable baseN (5 configs)
for bm in [512, 1024]:
    for bn in [32, 64]:
        experiments.append((f"bM{bm}_bN{bn}_lg", {"baseM": bm, "baseN": bn}))
experiments.append(("bM512_bN32_stepM2", {"baseM": 512, "baseN": 32, "stepM": 2}))

print(f"Total experiments: {len(experiments)}")

def main():
    original = read_file(ASC_FILE)
    results = []
    total = len(experiments)
    start_time = time.time()

    for i, (name, config) in enumerate(experiments):
        elapsed = time.time() - start_time
        eta = elapsed / (i+1) * (total - i - 1) if i > 0 else 0
        print(f"[{i+1}/{total}] {name}: ", end="", flush=True)

        modified = apply_config(original, config)
        duration, prec_pass, output = build_and_run(modified)

        if duration is None:
            status = output if isinstance(output, str) else "unknown"
            print(f"{status} (ETA: {eta/60:.1f}m)")
            results.append((name, None, False, status, config))
        else:
            status = "pass" if prec_pass else "prec_fail"
            dur_str = f"{duration:.2f}" if duration > 0.5 else "msprof_fail"
            print(f"{dur_str}us {'PASS' if prec_pass else 'FAIL'} (ETA: {eta/60:.1f}m)")
            results.append((name, duration, prec_pass, status, config))

    write_file(ASC_FILE, original)

    # Print sorted results (passing only)
    print("\n=== TOP 20 PASSING RESULTS ===")
    valid = [(n, d, p, s, c) for n, d, p, s, c in results if d and d > 0.5 and p]
    valid.sort(key=lambda x: x[1])
    for i, (name, dur, prec, status, config) in enumerate(valid[:20]):
        print(f"  {i+1:3d}. {name:40s}: {dur:8.2f}us  config={config}")

    # Also show failing
    print("\n=== TOP FAILING (for reference) ===")
    failing = [(n, d, p, s, c) for n, d, p, s, c in results if d and d > 0.5 and not p]
    failing.sort(key=lambda x: x[1])
    for i, (name, dur, prec, status, config) in enumerate(failing[:10]):
        print(f"  {i+1:3d}. {name:40s}: {dur:8.2f}us FAIL  config={config}")

    # Count stats
    pass_count = len([r for r in results if r[2]])
    fail_count = len([r for r in results if not r[2] and r[1] and r[1] > 0.5])
    build_fail = len([r for r in results if r[3] in ('build_fail', 'timeout')])
    print(f"\nStats: {pass_count} pass, {fail_count} prec_fail, {build_fail} build_fail/timeout")

    csv_path = os.path.join(PROJECT_ROOT, "scripts", "mega_sweep_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["name", "duration_us", "precision", "status", "config"])
        for n, d, p, s, c in results:
            w.writerow([n, d, "PASS" if p else "FAIL", s, str(c)])
    print(f"Saved to {csv_path}")

if __name__ == "__main__":
    main()
