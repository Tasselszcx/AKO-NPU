#!/usr/bin/env python3
"""Rapid test round 2: more advanced configurations."""

import os
import re
import subprocess
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

def build():
    build_dir = SOLUTION_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    os.chmod(build_dir, 0o755)
    env = os.environ.copy()
    env["CMAKE_PREFIX_PATH"] = CMAKE_PREFIX + ":" + env.get("CMAKE_PREFIX_PATH", "")
    r = subprocess.run(["cmake", ".."], cwd=build_dir, env=env, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False
    r = subprocess.run(["make", "-j4"], cwd=build_dir, env=env, capture_output=True, text=True, timeout=60)
    return r.returncode == 0

def test_correct():
    build_dir = SOLUTION_DIR / "build"
    subprocess.run(["python3", "../scripts/gen_data.py"], cwd=build_dir, capture_output=True, timeout=10)
    r = subprocess.run(["./demo"], cwd=build_dir, capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False
    r = subprocess.run(["python3", "../scripts/verify_result.py", "output/output.bin", "output/golden.bin"],
                       cwd=build_dir, capture_output=True, text=True, timeout=10)
    return "test pass" in r.stdout

def profile():
    build_dir = SOLUTION_DIR / "build"
    msprof_dir = f"/tmp/msprof_rapid2_{os.getpid()}"
    shutil.rmtree(msprof_dir, ignore_errors=True)
    os.makedirs(msprof_dir, mode=0o700)
    r = subprocess.run(["msprof", "op", "--warm-up=10", f"--output={msprof_dir}", "./demo"],
                       cwd=build_dir, capture_output=True, text=True, timeout=120)
    m = re.search(r"Task Duration\(us\):\s*([\d.]+)", r.stdout)
    dur = float(m.group(1)) if m else None
    shutil.rmtree(msprof_dir, ignore_errors=True)
    return dur

def test_variant(name, modifier_fn):
    original = read_file()
    try:
        modified = modifier_fn(original)
        write_file(modified)
        if not build():
            print(f"  {name}: BUILD FAILED")
            return None
        if not test_correct():
            print(f"  {name}: INCORRECT")
            return None
        dur = profile()
        if dur:
            print(f"  {name}: {dur:.1f} us ({227.9/dur:.2f}x)")
        else:
            print(f"  {name}: PROFILE FAILED")
        return dur
    except Exception as e:
        print(f"  {name}: ERROR {e}")
        return None
    finally:
        write_file(original)

def make_usedCoreNum_modifier(n):
    def mod(src):
        src = re.sub(r'int usedCoreNum = \d+;', f'int usedCoreNum = {n};', src)
        return src
    return mod

def make_stepM_modifier(sm):
    def mod(src):
        src = re.sub(r'tilingData\.stepM = \d+;', f'tilingData.stepM = {sm};', src)
        return src
    return mod

def make_fixsplit_modifier(bm, bn):
    def mod(src):
        src = re.sub(r'tilingApi\.SetFixSplit\(\d+,\s*\d+,\s*-1\)',
                     f'tilingApi.SetFixSplit({bm}, {bn}, -1)', src)
        return src
    return mod

def remove_fixsplit(src):
    src = re.sub(r'\s*tilingApi\.SetFixSplit\([^)]+\);\n', '\n', src)
    return src

def remove_l1cache(src):
    src = re.sub(r'\s*tilingApi\.SetMatmulConfigParams\([^)]+\);\s*//[^\n]*\n', '\n', src)
    return src

if __name__ == "__main__":
    print("=" * 60)
    print("Rapid test round 2")
    print("=" * 60)
    results = []

    tests = [
        # Current best baseline
        ("current-best", lambda s: s),

        # Remove L1 cache to compare
        ("no-l1cache", remove_l1cache),

        # Different core counts (will set tiling usedCoreNum)
        ("cores=1", make_usedCoreNum_modifier(1)),

        # Different tile sizes with L1 cache
        ("tiles-256x128-stepM1", lambda s: make_stepM_modifier(1)(s)),
        ("tiles-256x128-stepM4", lambda s: make_stepM_modifier(4)(s)),

        # Auto tiling (no fix split)
        ("auto-tiling", remove_fixsplit),

        # Various fixsplit with L1
        ("tiles-128x128", make_fixsplit_modifier(128, 128)),
        ("tiles-256x64", make_fixsplit_modifier(256, 64)),

        # Combine: auto tiling + no L1 cache
        ("auto-no-l1", lambda s: remove_l1cache(remove_fixsplit(s))),

        # Combine: remove everything (minimal config)
        ("minimal", lambda s: remove_l1cache(remove_fixsplit(make_stepM_modifier(1)(s)))),

        # Different CHUNK_SIZE for LeakyRelu
        ("chunk-8k", lambda s: re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                                       'const uint32_t CHUNK_SIZE = 8192;', s)),
        ("chunk-24k", lambda s: re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                                        'const uint32_t CHUNK_SIZE = 24576;', s)),
        ("chunk-40k", lambda s: re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                                        'const uint32_t CHUNK_SIZE = 40960;', s)),
        ("chunk-44k", lambda s: re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                                        'const uint32_t CHUNK_SIZE = 45056;', s)),
    ]

    for name, mod_fn in tests:
        dur = test_variant(name, mod_fn)
        if dur:
            results.append((name, dur))

    print("\n" + "=" * 60)
    print("Sorted results:")
    print("=" * 60)
    for name, dur in sorted(results, key=lambda x: x[1]):
        print(f"  {dur:8.1f} us  {227.9/dur:.2f}x  {name}")
