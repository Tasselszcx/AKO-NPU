#!/usr/bin/env python3
"""V2 sweep for the strided LeakyRelu code."""
import subprocess, re, os, sys, csv

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

EXPERIMENTS = [
    ("v2_baseline", {}),
    # Tile size variations
    ("v2_tile4096", {"tileSize": 4096}),
    ("v2_tile6144", {"tileSize": 6144}),
    ("v2_tile10240", {"tileSize": 10240}),
    ("v2_tile12288", {"tileSize": 12288}),
    ("v2_tile2048", {"tileSize": 2048}),
    # Step variations
    ("v2_stepM2", {"stepM": 2}),
    ("v2_stepN2", {"stepN": 2}),
    ("v2_stepM2N2", {"stepM": 2, "stepN": 2}),
    ("v2_stepM4", {"stepM": 4}),
    ("v2_stepN4", {"stepN": 4}),
    # Traverse
    ("v2_FIRSTM", {"traverse": "FIRSTM"}),
    # TransB
    ("v2_transB", {"isTransB": True}),
    # Different baseM/baseN
    ("v2_bM128_bN128", {"baseM": 128, "baseN": 128}),
    ("v2_bM128_bN256", {"baseM": 128, "baseN": 256}),
    ("v2_bM256_bN64", {"baseM": 256, "baseN": 64}),
    ("v2_bM256_bN256", {"baseM": 256, "baseN": 256}),
    # Combined
    ("v2_transB_stepM2", {"isTransB": True, "stepM": 2}),
    ("v2_tile4096_stepM2", {"tileSize": 4096, "stepM": 2}),
    ("v2_bM128_bN256_stepM2", {"baseM": 128, "baseN": 256, "stepM": 2}),
]

def main():
    original = read_file(ASC_FILE)
    results = []

    print(f"Running {len(EXPERIMENTS)} experiments...")
    for name, config in EXPERIMENTS:
        print(f"{name}: ", end="", flush=True)
        modified = apply_config(original, config)
        duration, prec_pass, output = build_and_run(modified)
        if duration is None:
            status = output if isinstance(output, str) else "unknown"
            print(f"{status}")
            results.append((name, None, False, status))
        else:
            status = "pass" if prec_pass else "prec_fail"
            dur_str = f"{duration:.2f}" if duration > 0.5 else "msprof_fail"
            print(f"{dur_str}us {'PASS' if prec_pass else 'FAIL'}")
            results.append((name, duration, prec_pass, status))

    write_file(ASC_FILE, original)

    print("\n=== RESULTS (sorted) ===")
    valid = [(n, d, p, s) for n, d, p, s in results if d and d > 0.5]
    valid.sort(key=lambda x: x[1])
    for name, dur, prec, status in valid:
        flag = " <<<" if prec else ""
        print(f"  {name:35s}: {dur:8.2f}us {'PASS' if prec else 'FAIL'}{flag}")

    csv_path = os.path.join(PROJECT_ROOT, "scripts", "sweep_v2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["name", "duration_us", "precision", "status"])
        for n, d, p, s in results:
            w.writerow([n, d, "PASS" if p else "FAIL", s])
    print(f"\nSaved to {csv_path}")

if __name__ == "__main__":
    main()
