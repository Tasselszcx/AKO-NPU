#!/usr/bin/env python3
"""Rapid iteration: try many experiments, log each as iteration."""
import subprocess, re, os, sys, csv, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(PROJECT_ROOT, "solution", "build")
ASC_DIR = "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"
ITER_FILE = os.path.join(PROJECT_ROOT, "ITERATIONS.md")

START_ITER = 30  # First iteration number

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
        elif key == 'bufNum':
            # Change double buffer count for LeakyRelu
            c = c.replace('AscendC::TQue<AscendC::TPosition::VECIN, 2>', f'AscendC::TQue<AscendC::TPosition::VECIN, {val}>')
            c = c.replace('AscendC::TQue<AscendC::TPosition::VECOUT, 2>', f'AscendC::TQue<AscendC::TPosition::VECOUT, {val}>')
            c = re.sub(r'pipe->InitBuffer\(inQueue, \d+,', f'pipe->InitBuffer(inQueue, {val},', c)
            c = re.sub(r'pipe->InitBuffer\(outQueue, \d+,', f'pipe->InitBuffer(outQueue, {val},', c)
    return c

def build_and_run(content, label, timeout_s=120):
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

# Generate experiments systematically
experiments = []

# Category 1: Exhaustive tile size (15 experiments)
for ts in range(1024, 16385, 1024):
    experiments.append((f"tile_{ts}", {"tileSize": ts}))

# Category 2: Step combinations (16 experiments)
for sm in range(1, 5):
    for sn in range(1, 5):
        experiments.append((f"step_{sm}_{sn}", {"stepM": sm, "stepN": sn}))

# Category 3: BaseM with 128 baseN (8 experiments)
for bm in [32, 64, 96, 128, 160, 192, 384, 512]:
    experiments.append((f"bM{bm}_bN128", {"baseM": bm, "baseN": 128}))

# Category 4: BaseN with 256 baseM (8 experiments)
for bn in [32, 48, 64, 80, 96, 128, 160, 192]:
    experiments.append((f"bM256_bN{bn}", {"baseN": bn}))

# Category 5: isTransB combinations (8 experiments)
for sm in [1, 2, 3]:
    for sn in [1, 2]:
        experiments.append((f"transB_step{sm}_{sn}", {"isTransB": True, "stepM": sm, "stepN": sn}))
experiments.append(("transB_bM128_bN128", {"isTransB": True, "baseM": 128, "baseN": 128}))
experiments.append(("transB_bM256_bN64", {"isTransB": True, "baseM": 256, "baseN": 64}))

# Category 6: Buffer space (6 experiments)
for l1 in [131072, 262144, 524288, 786432, 1048576, 2097152]:
    experiments.append((f"buf_L1_{l1//1024}K", {"bufferSpace": f"{l1}, -1, -1"}))

# Category 7: FIRSTM variants (10 experiments)
for sm in [1, 2, 3]:
    for sn in [1, 2]:
        experiments.append((f"FIRSTM_step{sm}_{sn}", {"traverse": "FIRSTM", "stepM": sm, "stepN": sn}))
for bn in [64, 128, 256]:
    experiments.append((f"FIRSTM_bN{bn}", {"traverse": "FIRSTM", "baseN": bn}))
experiments.append(("FIRSTM_transB", {"traverse": "FIRSTM", "isTransB": True}))

# Category 8: Triple buffer (4 experiments)
for ts in [4096, 8192, 12288]:
    experiments.append((f"tribuf_tile{ts}", {"bufNum": 3, "tileSize": ts}))
experiments.append(("tribuf_default", {"bufNum": 3}))

# Category 9: Quad buffer (3 experiments)
for ts in [2048, 4096, 8192]:
    experiments.append((f"quadbuf_tile{ts}", {"bufNum": 4, "tileSize": ts}))

# Category 10: Combined winners (10 experiments)
experiments.append(("combo_stepM2_tile10240", {"stepM": 2, "tileSize": 10240}))
experiments.append(("combo_stepM2_tile12288", {"stepM": 2, "tileSize": 12288}))
experiments.append(("combo_stepM2_transB", {"stepM": 2, "isTransB": True}))
experiments.append(("combo_stepM2_transB_tile10240", {"stepM": 2, "isTransB": True, "tileSize": 10240}))
experiments.append(("combo_stepM3_tile10240", {"stepM": 3, "tileSize": 10240}))
experiments.append(("combo_stepM2_bM128_bN128", {"stepM": 2, "baseM": 128, "baseN": 128}))
experiments.append(("combo_stepM2_buf786K", {"stepM": 2, "bufferSpace": "786432, -1, -1"}))
experiments.append(("combo_transB_tile12288", {"isTransB": True, "tileSize": 12288}))
experiments.append(("combo_stepM2_stepN3_tile10240", {"stepM": 2, "stepN": 3, "tileSize": 10240}))
experiments.append(("combo_FIRSTM_stepM2_transB", {"traverse": "FIRSTM", "stepM": 2, "isTransB": True}))

# Category 11: More baseM/baseN (8 experiments)
for bm, bn in [(256, 256), (128, 256), (64, 256), (256, 64),
               (192, 128), (384, 64), (64, 64), (192, 192)]:
    experiments.append((f"alt_bM{bm}_bN{bn}", {"baseM": bm, "baseN": bn}))

# Category 12: Even more step combos (8 experiments)
for sm in [4, 5, 6, 8]:
    experiments.append((f"bigstep_{sm}_1", {"stepM": sm}))
for sn in [4, 5, 6, 8]:
    experiments.append((f"bigstep_1_{sn}", {"stepN": sn}))

# Category 13: Tiny tiles (5 experiments)
for ts in [128, 256, 384, 512, 768]:
    experiments.append((f"tiny_tile{ts}", {"tileSize": ts}))

# Category 14: Large tiles with single buffer (4 experiments)
for ts in [16384, 20480, 24576, 32768]:
    experiments.append((f"large_tile{ts}_buf1", {"tileSize": ts, "bufNum": 1}))

total = len(experiments)
print(f"Total experiments: {total} (iters {START_ITER}-{START_ITER+total-1})")

def main():
    original = read_file(ASC_FILE)
    best_dur = 74.82  # Current best
    best_name = "baseline"
    results = []

    for i, (name, config) in enumerate(experiments):
        iter_num = START_ITER + i
        print(f"[iter {iter_num}] {name}: ", end="", flush=True)

        modified = apply_config(original, config)
        duration, prec_pass, output = build_and_run(modified, f"iter-{iter_num}")

        if duration is None:
            status = output if isinstance(output, str) else "unknown"
            print(f"{status}")
            results.append((iter_num, name, None, False, status, config))
        else:
            if prec_pass and duration > 0.5:
                status = "pass"
                if duration < best_dur:
                    best_dur = duration
                    best_name = name
                    print(f"{duration:.2f}us PASS *** NEW BEST ***")
                else:
                    print(f"{duration:.2f}us PASS")
            elif duration < 0.5:
                status = "msprof_fail"
                print(f"msprof_fail")
            else:
                status = "prec_fail"
                print(f"{duration:.2f}us FAIL")
            results.append((iter_num, name, duration, prec_pass, status, config))

    write_file(ASC_FILE, original)

    # Summary
    print(f"\n=== SUMMARY (iters {START_ITER}-{START_ITER+total-1}) ===")
    print(f"Best: {best_name} at {best_dur:.2f}us")

    # Top 10 passing
    passing = [(n, d, name, c) for n, name, d, p, s, c in results if p and d and d > 0.5]
    passing.sort(key=lambda x: x[1])
    print("\nTop 10 passing:")
    for i, (num, dur, name, config) in enumerate(passing[:10]):
        print(f"  {i+1}. [iter {num}] {name:40s}: {dur:.2f}us")

    # Save CSV
    csv_path = os.path.join(PROJECT_ROOT, "scripts", "rapid_iters_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["iter", "name", "duration_us", "precision", "status", "config"])
        for r in results:
            w.writerow([r[0], r[1], r[2], "PASS" if r[3] else "FAIL", r[4], str(r[5])])

    # Build ITERATIONS.md update
    iter_lines = []
    for num, name, dur, prec, status, config in results:
        dur_str = f"{dur:.2f}" if dur and dur > 0.5 else "N/A"
        speedup = f"{224.55/dur:.2f}x" if dur and dur > 0.5 else "N/A"
        iter_lines.append(f"| {num} | {name} | {dur_str} | {speedup} | {status} |")

    print(f"\nSaved {csv_path}")
    print(f"\nIteration table lines for ITERATIONS.md:")
    for line in iter_lines:
        print(line)

if __name__ == "__main__":
    main()
