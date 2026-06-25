#!/usr/bin/env python3
"""Sweep LeakyRelu tile sizes. Modifies the .asc source, rebuilds, runs, parses output."""
import subprocess, re, os, sys, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(PROJECT_ROOT, "solution", "build")
ASC_DIR = "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"

# Tile sizes to sweep (in elements, fp32 = 4 bytes each)
TILE_SIZES = [1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 40960]

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def set_tile_size(content, size):
    return re.sub(
        r'const uint32_t tileSize = \d+;',
        f'const uint32_t tileSize = {size};',
        content
    )

def build():
    """Build the project. Returns True on success."""
    os.makedirs(BUILD_DIR, exist_ok=True)
    r = subprocess.run(
        f"cd {BUILD_DIR} && cmake -DASC_DIR={ASC_DIR} .. 2>&1 && make -j 2>&1",
        shell=True, capture_output=True, text=True, timeout=120
    )
    return r.returncode == 0, r.stdout + r.stderr

def run_bench():
    """Run msprof and precision check. Returns (duration, precision_pass, output)."""
    env = os.environ.copy()
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

    # Parse duration
    dur_match = re.search(r'Task Duration\(us\):\s*([\d.]+)', output)
    duration = float(dur_match.group(1)) if dur_match else None

    # Parse precision
    prec_pass = 'test pass' in output.lower()

    return duration, prec_pass, output

def main():
    original = read_file(ASC_FILE)
    results = []

    print(f"Sweeping {len(TILE_SIZES)} tile sizes...")
    for size in TILE_SIZES:
        print(f"\n--- Tile size: {size} ---")
        modified = set_tile_size(original, size)
        write_file(ASC_FILE, modified)

        ok, build_out = build()
        if not ok:
            print(f"  BUILD FAILED")
            results.append((size, None, False, "build_fail"))
            continue

        duration, prec_pass, output = run_bench()
        status = "pass" if prec_pass else "prec_fail"
        print(f"  Duration: {duration}us, Precision: {'PASS' if prec_pass else 'FAIL'}")
        results.append((size, duration, prec_pass, status))

    # Restore original
    write_file(ASC_FILE, original)

    # Print results table
    print("\n\n=== SWEEP RESULTS ===")
    print(f"{'TileSize':>10} {'Duration(us)':>14} {'Precision':>10} {'Status':>12}")
    best_dur = float('inf')
    best_size = None
    for size, dur, prec, status in results:
        dur_str = f"{dur:.2f}" if dur else "N/A"
        prec_str = "PASS" if prec else "FAIL"
        print(f"{size:>10} {dur_str:>14} {prec_str:>10} {status:>12}")
        if prec and dur and dur < best_dur:
            best_dur = dur
            best_size = size

    print(f"\nBest passing: tileSize={best_size}, duration={best_dur:.2f}us")

    # Save CSV
    csv_path = os.path.join(PROJECT_ROOT, "scripts", "sweep_tilesize_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["tileSize", "duration_us", "precision", "status"])
        for size, dur, prec, status in results:
            w.writerow([size, dur, "PASS" if prec else "FAIL", status])
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    main()
