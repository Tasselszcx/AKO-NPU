#!/usr/bin/env python3
"""Rapid iteration batch 2: fine-tuning around the best configs."""
import subprocess, re, os, csv, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASC_FILE = os.path.join(PROJECT_ROOT, "solution", "matmul_leakyrelu.asc")
BUILD_DIR = os.path.join(PROJECT_ROOT, "solution", "build")
ASC_DIR = "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake"

START_ITER = 144

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
            c = c.replace('AscendC::TQue<AscendC::TPosition::VECIN, 2>', f'AscendC::TQue<AscendC::TPosition::VECIN, {val}>')
            c = c.replace('AscendC::TQue<AscendC::TPosition::VECOUT, 2>', f'AscendC::TQue<AscendC::TPosition::VECOUT, {val}>')
            c = re.sub(r'pipe->InitBuffer\(inQueue, \d+,', f'pipe->InitBuffer(inQueue, {val},', c)
            c = re.sub(r'pipe->InitBuffer\(outQueue, \d+,', f'pipe->InitBuffer(outQueue, {val},', c)
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

experiments = []

# Fine-tuned tile sizes around 9216 (8 experiments)
for ts in [8704, 8960, 9216, 9472, 9728, 9984, 10496, 10752]:
    experiments.append((f"finetile_{ts}", {"tileSize": ts}))

# Step combinations with tile 9216 (8 experiments)
for sm in [1, 2]:
    for sn in [1, 2, 3, 4]:
        experiments.append((f"tile9216_step{sm}_{sn}", {"tileSize": 9216, "stepM": sm, "stepN": sn}))

# TransB + tile sizes (6 experiments)
for ts in [8192, 9216, 10240, 11264, 12288, 7168]:
    experiments.append((f"transB_tile{ts}", {"isTransB": True, "tileSize": ts}))

# Triple buffer with fine tile sizes (4 experiments)
for ts in [7168, 8192, 9216, 10240]:
    experiments.append((f"tribuf_tile{ts}", {"bufNum": 3, "tileSize": ts}))

# Buffer space with tile optimization (4 experiments)
for bs in ["786432, -1, -1", "1048576, -1, -1", "1572864, -1, -1", "2097152, -1, -1"]:
    l1k = int(bs.split(',')[0].strip()) // 1024
    experiments.append((f"tile9216_buf{l1k}K", {"tileSize": 9216, "bufferSpace": bs}))

# TransB + step combos (6 experiments)
experiments.append(("transB_step2_1_tile9216", {"isTransB": True, "stepM": 2, "tileSize": 9216}))
experiments.append(("transB_step1_3_tile9216", {"isTransB": True, "stepN": 3, "tileSize": 9216}))
experiments.append(("transB_step2_3_tile9216", {"isTransB": True, "stepM": 2, "stepN": 3, "tileSize": 9216}))
experiments.append(("transB_step1_2_tile10240", {"isTransB": True, "stepN": 2, "tileSize": 10240}))
experiments.append(("transB_step2_2_tile10240", {"isTransB": True, "stepM": 2, "stepN": 2, "tileSize": 10240}))
experiments.append(("transB_step2_3_tile10240", {"isTransB": True, "stepM": 2, "stepN": 3, "tileSize": 10240}))

# Repeat best configs 3x for variance estimation (6 experiments)
for name, cfg in [
    ("repeat1_tile9216", {"tileSize": 9216}),
    ("repeat2_tile9216", {"tileSize": 9216}),
    ("repeat3_tile9216", {"tileSize": 9216}),
    ("repeat1_baseline", {}),
    ("repeat2_baseline", {}),
    ("repeat3_baseline", {}),
]:
    experiments.append((name, cfg))

# Mixed combos (8 experiments)
experiments.append(("tribuf_tile9216_stepM2", {"bufNum": 3, "tileSize": 9216, "stepM": 2}))
experiments.append(("tribuf_tile10240_stepM2", {"bufNum": 3, "tileSize": 10240, "stepM": 2}))
experiments.append(("tile9216_transB_buf1M", {"tileSize": 9216, "isTransB": True, "bufferSpace": "1048576, -1, -1"}))
experiments.append(("tile10240_transB_buf1M", {"tileSize": 10240, "isTransB": True, "bufferSpace": "1048576, -1, -1"}))
experiments.append(("tile9216_buf2M_stepM2", {"tileSize": 9216, "bufferSpace": "2097152, -1, -1", "stepM": 2}))
experiments.append(("tile10240_buf2M_stepM2", {"tileSize": 10240, "bufferSpace": "2097152, -1, -1", "stepM": 2}))
experiments.append(("tile9216_transB_stepM2_buf2M", {"tileSize": 9216, "isTransB": True, "stepM": 2, "bufferSpace": "2097152, -1, -1"}))
experiments.append(("tile10240_transB_stepM2_buf2M", {"tileSize": 10240, "isTransB": True, "stepM": 2, "bufferSpace": "2097152, -1, -1"}))

total = len(experiments)
print(f"Total experiments: {total} (iters {START_ITER}-{START_ITER+total-1})")

def main():
    original = read_file(ASC_FILE)
    best_dur = 71.94
    best_name = "tile_9216"
    results = []

    for i, (name, config) in enumerate(experiments):
        iter_num = START_ITER + i
        print(f"[iter {iter_num}] {name}: ", end="", flush=True)
        modified = apply_config(original, config)
        duration, prec_pass, output = build_and_run(modified)
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

    print(f"\n=== SUMMARY (iters {START_ITER}-{START_ITER+total-1}) ===")
    print(f"Best: {best_name} at {best_dur:.2f}us")

    passing = [(n, d, name, c) for n, name, d, p, s, c in results if p and d and d > 0.5]
    passing.sort(key=lambda x: x[1])
    print("\nTop 10 passing:")
    for i, (num, dur, name, config) in enumerate(passing[:10]):
        print(f"  {i+1}. [iter {num}] {name:40s}: {dur:.2f}us")

    csv_path = os.path.join(PROJECT_ROOT, "scripts", "rapid_iters2_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["iter", "name", "duration_us", "precision", "status", "config"])
        for r in results:
            w.writerow([r[0], r[1], r[2], "PASS" if r[3] else "FAIL", r[4], str(r[5])])
    print(f"Saved to {csv_path}")

if __name__ == "__main__":
    main()
