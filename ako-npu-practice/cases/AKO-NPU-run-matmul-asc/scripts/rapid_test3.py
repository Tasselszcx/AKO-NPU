#!/usr/bin/env python3
"""Rapid test round 3: comprehensive sweep with current best base."""

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

def profile(launch_count=3):
    build_dir = SOLUTION_DIR / "build"
    msprof_dir = f"/tmp/msprof_rapid3_{os.getpid()}"
    shutil.rmtree(msprof_dir, ignore_errors=True)
    os.makedirs(msprof_dir, mode=0o700)
    r = subprocess.run(["msprof", "op", "--warm-up=10", f"--launch-count={launch_count}",
                         f"--output={msprof_dir}", "./demo"],
                       cwd=build_dir, capture_output=True, text=True, timeout=180)
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
            print(f"  {name}: BUILD FAIL")
            return None
        if not test_correct():
            print(f"  {name}: INCORRECT")
            return None
        dur = profile()
        if dur:
            print(f"  {name}: {dur:.1f} us ({227.9/dur:.2f}x)")
        else:
            print(f"  {name}: PROFILE FAIL")
        return dur
    except Exception as e:
        print(f"  {name}: ERROR {e}")
        return None
    finally:
        write_file(original)

# Helper modifiers
def set_chunk(n):
    return lambda s: re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                             f'const uint32_t CHUNK_SIZE = {n};', s)
def set_step(m, n):
    def mod(s):
        s = re.sub(r'tilingData\.stepM = \d+;', f'tilingData.stepM = {m};', s)
        s = re.sub(r'tilingData\.stepN = \d+;', f'tilingData.stepN = {n};', s)
        return s
    return mod
def set_fixsplit(bm, bn):
    return lambda s: re.sub(r'tilingApi\.SetFixSplit\(\d+,\s*\d+,\s*-1\)',
                             f'tilingApi.SetFixSplit({bm}, {bn}, -1)', s)
def set_traverse(t):
    return lambda s: re.sub(r'tilingApi\.SetTraverse\([^)]+\)',
                             f'tilingApi.SetTraverse(matmul_tiling::MatrixTraverse::{t})', s)
def remove_fixsplit(s):
    return re.sub(r'\s*tilingApi\.SetFixSplit\([^)]+\);\n', '\n', s)
def remove_l1(s):
    return re.sub(r'\s*tilingApi\.SetMatmulConfigParams\([^)]+\);[^\n]*\n', '\n', s)
def compose(*fns):
    def composed(s):
        for f in fns:
            s = f(s)
        return s
    return composed

# Use double buffer for LeakyRelu (2 buffers with TQue instead of TBuf)
def use_double_buffer_leakyrelu(s):
    """Replace TBuf single-buffer LeakyRelu with a simpler but larger single pass."""
    # Just increase chunk size to process all elements in fewer iterations
    return set_chunk(47104)(s)  # 46K elements = ~184KB

# Increase UB utilization
def use_max_chunk(s):
    # 192KB UB / 4 bytes = 49152 float elements max
    # But need headroom, try 48K
    return set_chunk(48128)(s)

if __name__ == "__main__":
    print("=" * 60)
    print("Rapid test round 3: comprehensive")
    print("=" * 60)
    results = []

    tests = [
        # === CHUNK SIZE sweep (most promising from round 2) ===
        ("chunk-36k", set_chunk(36864)),
        ("chunk-40k", set_chunk(40960)),
        ("chunk-44k", set_chunk(45056)),  # current
        ("chunk-46k", set_chunk(47104)),
        ("chunk-48k", set_chunk(49152)),  # max UB

        # === Tile size sweep with L1 cache ===
        ("tiles-256x128", lambda s: s),  # current best
        ("tiles-256x128-noL1", remove_l1),
        ("tiles-auto", remove_fixsplit),
        ("tiles-auto-noL1", compose(remove_fixsplit, remove_l1)),
        ("tiles-128x128", set_fixsplit(128, 128)),
        ("tiles-256x64", set_fixsplit(256, 64)),
        ("tiles-128x64", set_fixsplit(128, 64)),

        # === Traverse sweep ===
        ("firstm", set_traverse("FIRSTM")),
        ("firstm-auto", compose(set_traverse("FIRSTM"), remove_fixsplit)),

        # === Step sweep ===
        ("stepM2", set_step(2, 1)),
        ("stepM2-stepN2", set_step(2, 2)),

        # === Combo: best chunk + auto tiling ===
        ("chunk46k-auto", compose(set_chunk(47104), remove_fixsplit)),
        ("chunk46k-firstm", compose(set_chunk(47104), set_traverse("FIRSTM"))),
        ("chunk46k-noL1", compose(set_chunk(47104), remove_l1)),

        # === Max UB chunk + various configs ===
        ("chunk48k-auto", compose(set_chunk(49152), remove_fixsplit)),
        ("chunk48k-auto-noL1", compose(set_chunk(49152), remove_fixsplit, remove_l1)),
    ]

    for name, mod_fn in tests:
        dur = test_variant(name, mod_fn)
        if dur:
            results.append((name, dur))

    print("\n" + "=" * 60)
    print("ALL RESULTS sorted:")
    print("=" * 60)
    for name, dur in sorted(results, key=lambda x: x[1]):
        print(f"  {dur:8.1f} us  {227.9/dur:.2f}x  {name}")
