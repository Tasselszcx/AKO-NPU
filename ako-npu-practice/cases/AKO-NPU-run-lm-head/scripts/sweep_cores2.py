#!/usr/bin/env python3
"""Core count sweep for lm_head_projection. Iter 4 = one parameter sweep."""
import subprocess, os, sys, csv, re, shutil, glob

SOLUTION = "solution"
ASC_FILE = f"{SOLUTION}/lm_head_projection.asc"
BUILD_DIR = f"{SOLUTION}/build"
CMAKE_DIR = "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake"

WORKLOADS = [
    ("W1", 1, 128, 1),
    ("W2", 1, 512, 1),
    ("W3", 1, 1024, 128),
    ("W4", 4, 256, 1),
]

CORE_COUNTS = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24]

def read_file(path):
    with open(path) as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def run(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300, **kw)

def build():
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    r = run(f"cd {BUILD_DIR} && cmake .. -DASC_DIR={CMAKE_DIR} && make -j$(nproc)")
    if r.returncode != 0:
        return False
    run(f"chmod 700 {BUILD_DIR}")
    return True

def run_workload(wl_name, B, S, ltk):
    """Run single workload, return (duration_us, correct)"""
    # Gen data
    run(f"python3 scripts/gen_data.py {B} {S} {ltk} --output-dir {SOLUTION}")

    # Setup run dir
    run_dir = f"{SOLUTION}/run_tmp"
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_dir, mode=0o700, exist_ok=True)
    shutil.copy(f"{BUILD_DIR}/demo", run_dir)

    inp = os.path.realpath(f"{SOLUTION}/input")
    outp = os.path.realpath(f"{SOLUTION}/output")
    os.symlink(inp, f"{run_dir}/input")
    os.symlink(outp, f"{run_dir}/output")

    # Remove old output
    out_bin = f"{SOLUTION}/output/output.bin"
    if os.path.exists(out_bin):
        os.remove(out_bin)

    # msprof
    msprof_dir = f"{SOLUTION}/sweep_tmp/{wl_name}"
    shutil.rmtree(msprof_dir, ignore_errors=True)
    os.makedirs(msprof_dir, mode=0o700, exist_ok=True)

    r = run(f"cd {run_dir} && msprof op --warm-up=5 --output={os.path.realpath(msprof_dir)} ./demo {B} {S} {ltk}")
    shutil.rmtree(run_dir, ignore_errors=True)

    # Parse duration
    duration = None
    opprof_dirs = sorted(glob.glob(f"{msprof_dir}/OPPROF_*"), reverse=True)
    if opprof_dirs:
        csv_path = f"{opprof_dirs[0]}/OpBasicInfo.csv"
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    try:
                        duration = float(row.get('Task Duration(us)', '0'))
                    except:
                        pass
                    break

    # Check correctness
    correct = False
    golden = f"{SOLUTION}/output/golden.bin"
    if os.path.exists(out_bin) and os.path.exists(golden):
        r2 = run(f"python3 scripts/verify_result.py {out_bin} {golden}")
        correct = (r2.returncode == 0)

    return duration, correct

def main():
    original_code = read_file(ASC_FILE)

    print("=== Core Count Sweep ===")
    print(f"{'Cores':>5} | {'W1(us)':>10} | {'W2(us)':>10} | {'W3(us)':>10} | {'W4(us)':>10} | Notes")
    print("-" * 75)

    results = {}

    for cores in CORE_COUNTS:
        # Modify core count
        modified = re.sub(
            r'tilingApi\.SetDim\([^)]+\)',
            f'tilingApi.SetDim({cores})',
            original_code
        )
        write_file(ASC_FILE, modified)

        if not build():
            print(f"{cores:>5} | {'BUILD_FAIL':>10} | {'':>10} | {'':>10} | {'':>10} |")
            write_file(ASC_FILE, original_code)
            continue

        row = {}
        notes = []
        for wl_name, B, S, ltk in WORKLOADS:
            dur, correct = run_workload(wl_name, B, S, ltk)
            row[wl_name] = dur
            if dur is not None and not correct:
                notes.append(f"{wl_name}:FAIL")

        results[cores] = row

        w1 = f"{row.get('W1', 0):.1f}" if row.get('W1') else "N/A"
        w2 = f"{row.get('W2', 0):.1f}" if row.get('W2') else "N/A"
        w3 = f"{row.get('W3', 0):.1f}" if row.get('W3') else "N/A"
        w4 = f"{row.get('W4', 0):.1f}" if row.get('W4') else "N/A"
        note_str = ", ".join(notes) if notes else "all PASS"

        print(f"{cores:>5} | {w1:>10} | {w2:>10} | {w3:>10} | {w4:>10} | {note_str}")

    # Restore original
    write_file(ASC_FILE, original_code)

    print("\n=== Best per workload ===")
    for wl_name, _, _, _ in WORKLOADS:
        best_cores = None
        best_dur = float('inf')
        for cores, row in results.items():
            if row.get(wl_name) and row[wl_name] < best_dur:
                best_dur = row[wl_name]
                best_cores = cores
        if best_cores:
            print(f"  {wl_name}: {best_dur:.1f}us @ {best_cores} cores")

if __name__ == "__main__":
    main()
