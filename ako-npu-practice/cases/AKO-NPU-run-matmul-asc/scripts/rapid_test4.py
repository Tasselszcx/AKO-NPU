#!/usr/bin/env python3
"""Rapid test round 4: radical approaches."""

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
    msprof_dir = f"/tmp/msprof_rapid4_{os.getpid()}"
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

def replace_leakyrelu_with_manual(src):
    """Replace LeakyRelu API call with manual compare+mul for potentially less overhead."""
    old_relu = """        // LeakyRelu
        LeakyRelu(localBuf, localBuf, (float)0.001, alignedSize);"""
    new_relu = """        // Manual LeakyRelu: x > 0 ? x : x * 0.001
        {
            uint32_t maskLen = alignedSize;
            // Use Muls for the negative path: localBuf *= 0.001, then select
            AscendC::LocalTensor<float> tmpBuf = calcBuf.Get<float>();
            // Since we can't easily get a second buffer, use in-place approach:
            // LeakyRelu is already efficient, keep it
            LeakyRelu(localBuf, localBuf, (float)0.001, alignedSize);
        }"""
    return src.replace(old_relu, new_relu)

def use_finer_sync(src):
    """Replace PIPE_ALL with finer barriers."""
    # Replace the 3 PIPE_ALL barriers with more specific ones
    leaky_section = """        DataCopy(localBuf, cGm[offset], alignedSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        // LeakyRelu
        LeakyRelu(localBuf, localBuf, (float)0.001, alignedSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        // Copy back
        DataCopy(cGm[offset], localBuf, alignedSize);
        AscendC::PipeBarrier<PIPE_ALL>();"""

    finer_section = """        DataCopy(localBuf, cGm[offset], alignedSize);
        AscendC::PipeBarrier<PIPE_MTE2>();
        // LeakyRelu
        LeakyRelu(localBuf, localBuf, (float)0.001, alignedSize);
        AscendC::PipeBarrier<PIPE_V>();
        // Copy back
        DataCopy(cGm[offset], localBuf, alignedSize);
        AscendC::PipeBarrier<PIPE_MTE3>();"""
    return src.replace(leaky_section, finer_section)

def use_mix_2_2(src):
    """Change __mix__(1, 2) to __mix__(2, 2) for 2 AIC + 2 AIV."""
    return src.replace("__mix__(1, 2)", "__mix__(2, 2)")

def use_mix_1_1(src):
    """Change __mix__(1, 2) to __mix__(1, 1) for 1 AIC + 1 AIV."""
    src = src.replace("__mix__(1, 2)", "__mix__(1, 1)")
    src = re.sub(r'uint32_t aivBlockNum = \d+;', 'uint32_t aivBlockNum = 1;', src)
    return src

def remove_leakyrelu_pass(src):
    """Remove LeakyRelu entirely (just matmul) to measure matmul overhead."""
    # Find the LeakyRelu section and remove it
    start = src.find("    // LeakyRelu pass:")
    end = src.find("\n}", start)
    if start > 0 and end > 0:
        src = src[:start] + src[end:]
    return src

def use_stepM_4(src):
    return re.sub(r'tilingData\.stepM = \d+;', 'tilingData.stepM = 4;', src)

if __name__ == "__main__":
    print("=" * 60)
    print("Rapid test round 4: radical approaches")
    print("=" * 60)
    results = []

    tests = [
        ("current-best", lambda s: s),
        ("finer-sync", use_finer_sync),
        ("mix-2-2", use_mix_2_2),
        ("mix-1-1", use_mix_1_1),
        ("matmul-only", remove_leakyrelu_pass),
        ("stepM4", use_stepM_4),
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
