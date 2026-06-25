#!/usr/bin/env python3
"""Sweep baseM/baseN combinations with FIRSTN traverse."""
import subprocess, re, os, sys, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(PROJECT_ROOT, "solution", "build")
ASC_DIR = "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"

# Must be multiples of 16 for cube alignment
# M=1024, K=256, N=640
CONFIGS = [
    # (baseM, baseN) - baseM must divide M reasonably, baseN must divide N
    (64, 64), (64, 128), (64, 160), (64, 320), (64, 640),
    (128, 64), (128, 128), (128, 160), (128, 320), (128, 640),
    (256, 64), (256, 128), (256, 160), (256, 320), (256, 640),
    (512, 64), (512, 128), (512, 160), (512, 320), (512, 640),
    (1024, 64), (1024, 128), (1024, 160), (1024, 320), (1024, 640),
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

def build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    r = subprocess.run(
        f"cd {BUILD_DIR} && cmake -DASC_DIR={ASC_DIR} .. 2>&1 && make -j 2>&1",
        shell=True, capture_output=True, text=True, timeout=120
    )
    return r.returncode == 0, r.stdout + r.stderr

def run_bench():
    cmd = f"""
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    cd {BUILD_DIR}
    python3 {PROJECT_ROOT}/scripts/gen_data.py 2>&1
    MSPROF_OUT={PROJECT_ROOT}/msprof_sweep
    rm -rf $MSPROF_OUT
    mkdir -p $MSPROF_OUT && chmod 750 $MSPROF_OUT && chmod 750 {BUILD_DIR}
    msprof op --warm-up=10 --launch-count=5 --output=$MSPROF_OUT ./demo 2>&1
    python3 {PROJECT_ROOT}/scripts/verify_result.py output/output.bin output/golden.bin 2>&1
    """
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=180)
    output = r.stdout + r.stderr
    dur_match = re.search(r'Task Duration\(us\):\s*([\d.]+)', output)
    duration = float(dur_match.group(1)) if dur_match else None
    prec_pass = 'test pass' in output.lower()
    return duration, prec_pass, output

def main():
    original = read_file(ASC_FILE)
    results = []

    print(f"Sweeping {len(CONFIGS)} baseM/baseN configs with FIRSTN...")
    for baseM, baseN in CONFIGS:
        print(f"\n--- baseM={baseM}, baseN={baseN} ---")
        modified = set_config(original, baseM, baseN)
        write_file(ASC_FILE, modified)

        ok, build_out = build()
        if not ok:
            print(f"  BUILD FAILED")
            # Check if tiling failed
            if 'gen tiling failed' in build_out:
                print(f"  Tiling generation failed")
            results.append((baseM, baseN, None, False, "build_fail"))
            continue

        duration, prec_pass, output = run_bench()
        if duration and duration < 1.0:
            # Sometimes msprof returns 0 for failed runs
            status = "suspicious"
        elif prec_pass:
            status = "pass"
        else:
            status = "prec_fail"
        print(f"  Duration: {duration}us, Precision: {'PASS' if prec_pass else 'FAIL'}")
        results.append((baseM, baseN, duration, prec_pass, status))

    # Restore original
    write_file(ASC_FILE, original)

    print("\n\n=== SWEEP RESULTS ===")
    print(f"{'baseM':>6} {'baseN':>6} {'Duration(us)':>14} {'Precision':>10} {'Status':>12}")
    best_dur = float('inf')
    best_cfg = None
    for baseM, baseN, dur, prec, status in results:
        dur_str = f"{dur:.2f}" if dur else "N/A"
        prec_str = "PASS" if prec else "FAIL"
        print(f"{baseM:>6} {baseN:>6} {dur_str:>14} {prec_str:>10} {status:>12}")
        if prec and dur and dur > 1.0 and dur < best_dur:
            best_dur = dur
            best_cfg = (baseM, baseN)

    if best_cfg:
        print(f"\nBest passing: baseM={best_cfg[0]}, baseN={best_cfg[1]}, duration={best_dur:.2f}us")

    csv_path = os.path.join(PROJECT_ROOT, "scripts", "sweep_basemn_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["baseM", "baseN", "duration_us", "precision", "status"])
        for baseM, baseN, dur, prec, status in results:
            w.writerow([baseM, baseN, dur, "PASS" if prec else "FAIL", status])
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
