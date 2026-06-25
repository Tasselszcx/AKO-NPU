#!/usr/bin/env python3
"""Sweep baseM/baseN combinations with FIRSTN traverse - safer version."""
import subprocess, re, os, sys, csv, signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(PROJECT_ROOT, "solution", "build")
ASC_DIR = "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"

# Focused configs - avoid large baseN values that may hang
CONFIGS = [
    (128, 128), (256, 128), (256, 160), (256, 256),
    (512, 128), (512, 160), (1024, 128), (1024, 160),
    (128, 160), (128, 256),
    (256, 64), (512, 64),
]

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def set_config(content, baseM, baseN):
    content = re.sub(r'int baseM = \d+;', f'int baseM = {baseM};', content)
    content = re.sub(r'int baseN = \d+;', f'int baseN = {baseN};', content)
    return content

def build_and_run(baseM, baseN, original):
    modified = set_config(original, baseM, baseN)
    write_file(ASC_FILE, modified)

    cmd = f"""
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    export ASC_DIR={ASC_DIR}
    cd {BUILD_DIR}
    cmake -DASC_DIR=$ASC_DIR .. >/dev/null 2>&1 && make -j 2>&1 | tail -3
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

    if 'BUILD_FAILED' in output:
        return None, False, "build_fail"

    dur_match = re.search(r'Task Duration\(us\):\s*([\d.]+)', output)
    duration = float(dur_match.group(1)) if dur_match else None
    prec_pass = 'test pass' in output.lower()
    return duration, prec_pass, output

def main():
    original = read_file(ASC_FILE)
    results = []

    print(f"Sweeping {len(CONFIGS)} baseM/baseN configs...")
    for baseM, baseN in CONFIGS:
        print(f"baseM={baseM}, baseN={baseN}: ", end="", flush=True)
        duration, prec_pass, output = build_and_run(baseM, baseN, original)
        if duration is None and output == "timeout":
            print("TIMEOUT")
            results.append((baseM, baseN, None, False, "timeout"))
        elif duration is None:
            print("BUILD_FAILED")
            results.append((baseM, baseN, None, False, "build_fail"))
        else:
            status = "pass" if prec_pass else "prec_fail"
            print(f"{duration:.2f}us {'PASS' if prec_pass else 'FAIL'}")
            results.append((baseM, baseN, duration, prec_pass, status))

    # Restore original
    write_file(ASC_FILE, original)

    print("\n=== RESULTS ===")
    best_dur = float('inf')
    best_cfg = None
    for baseM, baseN, dur, prec, status in sorted(results, key=lambda x: x[2] if x[2] and x[2] > 1.0 else 9999):
        dur_str = f"{dur:.2f}" if dur else "N/A"
        prec_str = "PASS" if prec else "FAIL"
        flag = " <<<" if prec and dur and dur > 1.0 and dur < best_dur else ""
        if prec and dur and dur > 1.0 and dur < best_dur:
            best_dur = dur
            best_cfg = (baseM, baseN)
        print(f"  baseM={baseM:>5}, baseN={baseN:>5}: {dur_str:>10}us {prec_str:>5} {status}{flag}")

    if best_cfg:
        print(f"\nBest: baseM={best_cfg[0]}, baseN={best_cfg[1]}, duration={best_dur:.2f}us")

    csv_path = os.path.join(PROJECT_ROOT, "scripts", "sweep_basemn2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["baseM", "baseN", "duration_us", "precision", "status"])
        for r in results:
            w.writerow([r[0], r[1], r[2], "PASS" if r[3] else "FAIL", r[4]])
    print(f"Saved to {csv_path}")

if __name__ == "__main__":
    main()
