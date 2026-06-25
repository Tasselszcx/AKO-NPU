#!/usr/bin/env python3
"""Rapid test specific kernel configurations."""

import csv
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOLUTION_DIR = PROJECT_DIR / "solution"
ASC_FILE = SOLUTION_DIR / "matmul_leakyrelu.asc"

CMAKE_PREFIX = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
CMAKE_PREFIX += "/x86_64-linux/tikcpp/ascendc_kernel_cmake"

def read_file():
    return ASC_FILE.read_text()

def write_file(content):
    ASC_FILE.write_text(content)

def patch_tiling(src, *, fix_split=None, traverse="FIRSTN", step_m=2, step_n=1,
                 schedule_type=None, enable_l1_cache_ub=False):
    """Patch the GenerateTiling function."""
    # Fix traverse
    src = re.sub(r'tilingApi\.SetTraverse\([^)]+\)',
                 f'tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::{traverse})', src)

    # Fix split
    if fix_split:
        bm, bn = fix_split
        if 'SetFixSplit' in src:
            src = re.sub(r'tilingApi\.SetFixSplit\([^)]+\)',
                         f'tilingApi.SetFixSplit({bm}, {bn}, -1)', src)
        else:
            src = src.replace('tilingApi.SetBufferSpace',
                              f'tilingApi.SetFixSplit({bm}, {bn}, -1);\n    tilingApi.SetBufferSpace')
    else:
        src = re.sub(r'\s*tilingApi\.SetFixSplit\([^)]+\);\s*\n', '\n', src)

    # stepM/stepN
    src = re.sub(r'tilingData\.stepM\s*=\s*\d+', f'tilingData.stepM = {step_m}', src)
    src = re.sub(r'tilingData\.stepN\s*=\s*\d+', f'tilingData.stepN = {step_n}', src)

    # Schedule type and L1 cache
    if schedule_type or enable_l1_cache_ub:
        sched = schedule_type or "INNER_PRODUCT"
        l1 = "true" if enable_l1_cache_ub else "false"
        config_line = f'    tilingApi.SetMatmulConfigParams(1, {l1}, matmul_tiling::ScheduleType::{sched});\n'
        if 'SetMatmulConfigParams' in src:
            src = re.sub(r'\s*tilingApi\.SetMatmulConfigParams\([^)]+\);\s*\n', '\n' + config_line, src)
        else:
            src = src.replace('tilingApi.SetOrgShape', config_line + '    tilingApi.SetOrgShape')
    else:
        src = re.sub(r'\s*tilingApi\.SetMatmulConfigParams\([^)]+\);\s*\n', '\n', src)

    return src

def patch_hf32(src, enabled=True):
    if enabled:
        if 'SetHF32' not in src:
            src = src.replace('matmulKernel.Process(&pipe);',
                              'matmulKernel.matmulObj.SetHF32(true);\n    matmulKernel.Process(&pipe);')
    else:
        src = re.sub(r'\s*matmulKernel\.matmulObj\.SetHF32\([^)]+\);\s*\n', '\n', src)
    return src

def patch_chunk_size(src, chunk_size=32768):
    src = re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                 f'const uint32_t CHUNK_SIZE = {chunk_size};', src)
    return src

def build():
    build_dir = SOLUTION_DIR / "build"
    build_dir.mkdir(exist_ok=True)
    os.chmod(build_dir, 0o755)
    env = os.environ.copy()
    env["CMAKE_PREFIX_PATH"] = CMAKE_PREFIX + ":" + env.get("CMAKE_PREFIX_PATH", "")
    r = subprocess.run(["cmake", ".."], cwd=build_dir, env=env,
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False, r.stderr
    r = subprocess.run(["make", "-j4"], cwd=build_dir, env=env,
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return False, r.stderr
    return True, ""

def run_test():
    build_dir = SOLUTION_DIR / "build"
    subprocess.run(["python3", "../scripts/gen_data.py"], cwd=build_dir,
                   capture_output=True, timeout=10)
    r = subprocess.run(["./demo"], cwd=build_dir, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False, "demo failed"
    r = subprocess.run(["python3", "../scripts/verify_result.py",
                        "output/output.bin", "output/golden.bin"],
                       cwd=build_dir, capture_output=True, text=True, timeout=10)
    return "test pass" in r.stdout, r.stdout

def profile():
    build_dir = SOLUTION_DIR / "build"
    msprof_dir = f"/tmp/msprof_rapid_{os.getpid()}"
    if os.path.exists(msprof_dir):
        shutil.rmtree(msprof_dir)
    os.makedirs(msprof_dir, mode=0o700)
    r = subprocess.run(
        ["msprof", "op", "--warm-up=10", f"--output={msprof_dir}", "./demo"],
        cwd=build_dir, capture_output=True, text=True, timeout=120)
    duration = None
    m = re.search(r"Task Duration\(us\):\s*([\d.]+)", r.stdout)
    if m:
        duration = float(m.group(1))
    shutil.rmtree(msprof_dir, ignore_errors=True)
    return duration

def test_config(name, **kwargs):
    original = read_file()
    try:
        src = original
        src = patch_tiling(src, **{k: v for k, v in kwargs.items()
                                   if k in ('fix_split','traverse','step_m','step_n',
                                            'schedule_type','enable_l1_cache_ub')})
        if 'hf32' in kwargs:
            src = patch_hf32(src, kwargs['hf32'])
        if 'chunk_size' in kwargs:
            src = patch_chunk_size(src, kwargs['chunk_size'])
        write_file(src)

        shutil.rmtree(SOLUTION_DIR / "build", ignore_errors=True)
        ok, err = build()
        if not ok:
            print(f"  {name}: BUILD FAILED")
            return None

        correct, msg = run_test()
        if not correct:
            print(f"  {name}: INCORRECT")
            return None

        dur = profile()
        if dur:
            print(f"  {name}: {dur:.1f} us")
        else:
            print(f"  {name}: PROFILE FAILED")
        return dur
    except Exception as e:
        print(f"  {name}: ERROR {e}")
        return None
    finally:
        write_file(original)

# =========================================================================
# Test configurations
# =========================================================================
configs = [
    # Baseline best
    ("best-current", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1)),

    # Schedule types
    ("outer-product", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1,
                           schedule_type="OUTER_PRODUCT")),
    ("n-buffer-33", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1,
                         schedule_type="N_BUFFER_33")),

    # L1 cache
    ("l1-cache-ub", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1,
                         enable_l1_cache_ub=True)),
    ("outer+l1cache", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1,
                           schedule_type="OUTER_PRODUCT", enable_l1_cache_ub=True)),

    # stepM variations
    ("stepM=4", dict(fix_split=(256,128), traverse="FIRSTN", step_m=4, step_n=1)),
    ("stepM=1", dict(fix_split=(256,128), traverse="FIRSTN", step_m=1, step_n=1)),

    # HF32 off
    ("no-hf32", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1, hf32=False)),

    # Different tile sizes
    ("tiles-128x128", dict(fix_split=(128,128), traverse="FIRSTN", step_m=2, step_n=1)),
    ("tiles-256x64", dict(fix_split=(256,64), traverse="FIRSTN", step_m=2, step_n=1)),
    ("tiles-128x160", dict(fix_split=(128,160), traverse="FIRSTN", step_m=2, step_n=1)),
    ("tiles-64x128", dict(fix_split=(64,128), traverse="FIRSTN", step_m=2, step_n=1)),

    # FIRSTM with best tiles
    ("firstm-256x128", dict(fix_split=(256,128), traverse="FIRSTM", step_m=2, step_n=1)),

    # LeakyRelu chunk size
    ("chunk-16k", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1, chunk_size=16384)),
    ("chunk-48k", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1, chunk_size=49152)),

    # Combined advanced
    ("outer+l1+stepM4", dict(fix_split=(256,128), traverse="FIRSTN", step_m=4, step_n=1,
                              schedule_type="OUTER_PRODUCT", enable_l1_cache_ub=True)),
    ("nbuf33+l1", dict(fix_split=(256,128), traverse="FIRSTN", step_m=2, step_n=1,
                        schedule_type="N_BUFFER_33", enable_l1_cache_ub=True)),
]

if __name__ == "__main__":
    print("=" * 60)
    print("Rapid configuration testing")
    print("=" * 60)
    results = []
    for name, kwargs in configs:
        dur = test_config(name, **kwargs)
        if dur:
            results.append((name, dur))

    print("\n" + "=" * 60)
    print("Results sorted by duration:")
    print("=" * 60)
    for name, dur in sorted(results, key=lambda x: x[1]):
        speedup = 227.9 / dur
        print(f"  {dur:8.1f} us  {speedup:.2f}x  {name}")
