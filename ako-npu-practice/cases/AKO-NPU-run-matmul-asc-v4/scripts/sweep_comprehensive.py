#!/usr/bin/env python3
"""Comprehensive parameter sweep for MatmulLeakyRelu optimization."""
import subprocess, re, os, sys, csv, copy

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
    """Apply a config dict to the source code."""
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

def build_and_run(content):
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
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=120)
        output = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return None, False, "timeout"

    if 'BUILD_FAILED' in output or 'fatal error' in output:
        return None, False, "build_fail"

    dur_match = re.search(r'Task Duration\(us\):\s*([\d.]+)', output)
    duration = float(dur_match.group(1)) if dur_match else None
    prec_pass = 'test pass' in output.lower()
    return duration, prec_pass, output

# Experiments to run
EXPERIMENTS = [
    # name, config_overrides
    ("baseline_FIRSTN", {}),  # current best config
    ("FIRSTN_stepM2", {"stepM": 2}),
    ("FIRSTN_stepN2", {"stepN": 2}),
    ("FIRSTN_stepMN2", {"stepM": 2, "stepN": 2}),
    ("FIRSTN_tile4096", {"tileSize": 4096}),
    ("FIRSTN_tile6144", {"tileSize": 6144}),
    ("FIRSTN_tile10240", {"tileSize": 10240}),
    ("FIRSTN_tile12288", {"tileSize": 12288}),
    ("FIRSTN_transB", {"isTransB": True}),
    ("FIRSTN_bM128_bN128", {"baseM": 128, "baseN": 128}),
    ("FIRSTN_bM128_bN256", {"baseM": 128, "baseN": 256}),
    ("FIRSTN_core3", {"usedCoreNum": 3}),
    ("FIRSTN_core4", {"usedCoreNum": 4}),
    ("FIRSTN_core5", {"usedCoreNum": 5}),
    ("FIRSTN_core8", {"usedCoreNum": 8}),
    ("FIRSTN_core10", {"usedCoreNum": 10}),
    ("FIRSTN_core16", {"usedCoreNum": 16}),
    ("FIRSTN_core20", {"usedCoreNum": 20}),
    ("FIRSTN_buf512K", {"bufferSpace": "524288, -1, -1"}),
    ("FIRSTN_buf256K", {"bufferSpace": "262144, -1, -1"}),
]

def main():
    original = read_file(ASC_FILE)
    results = []

    print(f"Running {len(EXPERIMENTS)} experiments...")
    for name, config in EXPERIMENTS:
        print(f"{name}: ", end="", flush=True)
        modified = apply_config(original, config)
        duration, prec_pass, output = build_and_run(modified)

        if duration is None and output == "timeout":
            print("TIMEOUT")
            results.append((name, None, False, "timeout", config))
        elif duration is None:
            print("BUILD_FAILED")
            results.append((name, None, False, "build_fail", config))
        else:
            status = "pass" if prec_pass else "prec_fail"
            dur_str = f"{duration:.2f}" if duration > 0.5 else "msprof_fail"
            print(f"{dur_str}us {'PASS' if prec_pass else 'FAIL'}")
            results.append((name, duration, prec_pass, status, config))

    # Restore original
    write_file(ASC_FILE, original)

    # Print sorted results
    print("\n=== RESULTS (sorted by duration) ===")
    valid = [(n, d, p, s, c) for n, d, p, s, c in results if d and d > 0.5]
    valid.sort(key=lambda x: x[1])
    for name, dur, prec, status, config in valid:
        flag = " <<<" if prec else ""
        print(f"  {name:30s}: {dur:8.2f}us {'PASS' if prec else 'FAIL'}{flag}")

    print("\nFailed:")
    for name, dur, prec, status, config in results:
        if not dur or dur < 0.5:
            print(f"  {name:30s}: {status}")

    # Save CSV
    csv_path = os.path.join(PROJECT_ROOT, "scripts", "sweep_comprehensive_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["name", "duration_us", "precision", "status", "config"])
        for n, d, p, s, c in results:
            w.writerow([n, d, "PASS" if p else "FAIL", s, str(c)])
    print(f"\nSaved to {csv_path}")

if __name__ == "__main__":
    main()
