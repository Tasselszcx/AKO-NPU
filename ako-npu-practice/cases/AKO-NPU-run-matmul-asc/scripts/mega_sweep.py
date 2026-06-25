#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mega sweep for matmul_leakyrelu NPU kernel optimization.

Iterations 29-200: exhaustive search across tiling, LeakyRelu chunking,
kernel structure, advanced matmul configs, and radical restructuring.

Results -> scripts/mega_sweep_results.csv
"""

import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SOLUTION_DIR = PROJECT_DIR / "solution"
ASC_FILE = SOLUTION_DIR / "matmul_leakyrelu.asc"
RESULTS_CSV = SCRIPT_DIR / "mega_sweep_results.csv"

ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
CMAKE_PREFIX = ASCEND_HOME + "/x86_64-linux/tikcpp/ascendc_kernel_cmake"

GEN_DATA_SCRIPT = SOLUTION_DIR / "scripts" / "gen_data.py"
VERIFY_SCRIPT = SOLUTION_DIR / "scripts" / "verify_result.py"

# If the scripts are in the project-level scripts dir instead
if not GEN_DATA_SCRIPT.exists():
    GEN_DATA_SCRIPT = SCRIPT_DIR / "gen_data.py"
if not VERIFY_SCRIPT.exists():
    VERIFY_SCRIPT = SCRIPT_DIR / "verify_result.py"


def read_asc():
    return ASC_FILE.read_text()

def write_asc(content):
    ASC_FILE.write_text(content)

def get_env():
    env = os.environ.copy()
    existing = env.get("CMAKE_PREFIX_PATH", "")
    env["CMAKE_PREFIX_PATH"] = CMAKE_PREFIX + (":" + existing if existing else "")
    return env

ENV = get_env()


# ---------------------------------------------------------------------------
# Build / Run / Verify / Profile helpers
# ---------------------------------------------------------------------------

def build():
    """Build the project. Returns True on success."""
    build_dir = SOLUTION_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(build_dir), 0o755)

    r = subprocess.run(["cmake", ".."], cwd=str(build_dir), env=ENV,
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return False, f"cmake: {r.stderr[:300]}"
    r = subprocess.run(["make", "-j4"], cwd=str(build_dir), env=ENV,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        return False, f"make: {r.stderr[:300]}"
    if not (build_dir / "demo").exists():
        return False, "no demo binary"
    return True, ""


def gen_data():
    """Generate test data."""
    build_dir = SOLUTION_DIR / "build"
    r = subprocess.run([sys.executable, str(GEN_DATA_SCRIPT)],
                       cwd=str(build_dir), env=ENV,
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def run_demo():
    """Run demo. Returns True if output.bin produced."""
    build_dir = SOLUTION_DIR / "build"
    output_bin = build_dir / "output" / "output.bin"
    output_bin.unlink(missing_ok=True)
    r = subprocess.run(["./demo"], cwd=str(build_dir), env=ENV,
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return False
    return output_bin.exists()


def verify():
    """Verify correctness."""
    build_dir = SOLUTION_DIR / "build"
    output_bin = build_dir / "output" / "output.bin"
    golden_bin = build_dir / "output" / "golden.bin"
    if not output_bin.exists() or not golden_bin.exists():
        return False
    r = subprocess.run([sys.executable, str(VERIFY_SCRIPT),
                        str(output_bin), str(golden_bin)],
                       cwd=str(build_dir), env=ENV,
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0 and "test pass" in r.stdout.lower()


def profile():
    """Profile with msprof. Returns task duration in us or -1."""
    build_dir = SOLUTION_DIR / "build"
    msprof_dir = f"/tmp/msprof_mega_{os.getpid()}_{int(time.time())}"
    os.makedirs(msprof_dir, mode=0o700, exist_ok=True)
    try:
        r = subprocess.run(
            ["msprof", "op", "--warm-up=10", f"--output={msprof_dir}", "./demo"],
            cwd=str(build_dir), env=ENV,
            capture_output=True, text=True, timeout=60)
        text = r.stdout + "\n" + r.stderr
        # Parse Task Duration(us): NNN.NN
        m = re.search(r'Task\s*Duration\s*\(us\)\s*[:\s]+([\d.]+)', text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # Fallback: parse from CSV
        for opdir in sorted(Path(msprof_dir).glob("OPPROF_*"), reverse=True):
            csv_file = opdir / "OpBasicInfo.csv"
            if csv_file.exists():
                import csv as csvmod
                with open(csv_file) as f:
                    reader = csvmod.DictReader(f)
                    for row in reader:
                        for col in ['Task Duration(us)', 'Task Duration (us)',
                                    'TaskDuration(us)', 'Duration(us)']:
                            val = row.get(col, '')
                            if val:
                                try:
                                    return float(val)
                                except ValueError:
                                    pass
        return -1.0
    except subprocess.TimeoutExpired:
        return -1.0
    except Exception:
        return -1.0
    finally:
        shutil.rmtree(msprof_dir, ignore_errors=True)


def test_config(name, modifier_fn, needs_data_regen=False):
    """
    Test a single configuration.
    modifier_fn: takes original source, returns modified source.
    Returns (status, duration_us, details).
    """
    original = read_asc()
    try:
        modified = modifier_fn(original)
        if modified is None:
            return "skipped", -1, "modifier returned None"
        write_asc(modified)

        ok, err = build()
        if not ok:
            return "build_fail", -1, err

        if not gen_data():
            return "gendata_fail", -1, ""

        if not run_demo():
            return "run_fail", -1, ""

        if not verify():
            return "incorrect", -1, ""

        dur = profile()
        if dur > 0:
            return "ok", dur, f"{dur:.2f} us"
        else:
            return "msprof_fail", -1, ""
    except subprocess.TimeoutExpired:
        return "timeout", -1, ""
    except Exception as e:
        return "error", -1, str(e)[:200]
    finally:
        write_asc(original)


# ---------------------------------------------------------------------------
# Source patching helpers
# ---------------------------------------------------------------------------

def patch_traverse(src, traverse):
    return re.sub(
        r'(tilingApi\.SetTraverse\(matmul_tiling::MatrixTraverse::)\w+(\))',
        rf'\g<1>{traverse}\2', src)

def patch_fixsplit(src, fixsplit):
    """fixsplit: None=remove, (M,N)=set, (M,N,K)=set with baseK."""
    if fixsplit is None:
        # Remove SetFixSplit line
        src = re.sub(r'\s*tilingApi\.SetFixSplit\([^)]*\);\n?', '\n', src)
        return src
    if len(fixsplit) == 2:
        baseM, baseN = fixsplit
        baseK = -1
    else:
        baseM, baseN, baseK = fixsplit
    new_line = f'    tilingApi.SetFixSplit({baseM}, {baseN}, {baseK});'
    if 'SetFixSplit' in src:
        src = re.sub(r'(\s*)tilingApi\.SetFixSplit\([^)]*\);',
                     rf'\1tilingApi.SetFixSplit({baseM}, {baseN}, {baseK});', src)
    else:
        src = re.sub(r'(tilingApi\.SetShape\([^)]*\);)',
                     rf'\1\n{new_line}', src)
    return src

def patch_stepmn(src, stepM, stepN):
    src = re.sub(r'tilingData\.stepM\s*=\s*\d+;', f'tilingData.stepM = {stepM};', src)
    src = re.sub(r'tilingData\.stepN\s*=\s*\d+;', f'tilingData.stepN = {stepN};', src)
    return src

def patch_buffer_space(src, l1, l0a, l0b):
    return re.sub(r'tilingApi\.SetBufferSpace\([^)]*\)',
                  f'tilingApi.SetBufferSpace({l1}, {l0a}, {l0b})', src)

def patch_l1cache_ub(src, enable):
    """SetMatmulConfigParams(1, true/false) - second param is L1CacheUB."""
    if enable:
        return re.sub(r'tilingApi\.SetMatmulConfigParams\([^)]*\)',
                      'tilingApi.SetMatmulConfigParams(1, true)', src)
    else:
        return re.sub(r'tilingApi\.SetMatmulConfigParams\([^)]*\)',
                      'tilingApi.SetMatmulConfigParams(1, false)', src)

def patch_chunk_size(src, chunk_size):
    return re.sub(r'const uint32_t CHUNK_SIZE = \d+;',
                  f'const uint32_t CHUNK_SIZE = {chunk_size};', src)

def patch_aiv_block_num(src, num):
    src = re.sub(r'(uint32_t\s+aivBlockNum\s*=\s*)\d+;', rf'\g<1>{num};', src)
    return src

def patch_mix(src, aic, aiv):
    """Change __mix__(aic, aiv)."""
    src = re.sub(r'__mix__\(\s*\d+\s*,\s*\d+\s*\)', f'__mix__({aic}, {aiv})', src)
    return src

def patch_used_core_num(src, n):
    return re.sub(r'(int\s+usedCoreNum\s*=\s*)\d+;', rf'\g<1>{n};', src)

def patch_num_blocks(src, n):
    return re.sub(r'(uint32_t\s+numBlocks\s*=\s*)\d+;', rf'\g<1>{n};', src)

def add_hf32(src):
    """Add SetHF32(true) after REGIST_MATMUL_OBJ if not present."""
    if 'SetHF32' not in src:
        src = re.sub(r'(REGIST_MATMUL_OBJ\([^)]*\);)',
                     r'\1\n    matmulKernel.matmulObj.SetHF32(true);', src)
    return src

def remove_hf32(src):
    """Remove SetHF32 line if present."""
    src = re.sub(r'\s*matmulKernel\.matmulObj\.SetHF32\([^)]*\);\n?', '\n', src)
    return src

def patch_tque_to_tbuf(src, chunk_size):
    """Replace TQue double-buffer LeakyRelu with single TBuf + PIPE_ALL."""
    # Find the LeakyRelu section and replace it
    new_leakyrelu = f"""    // LeakyRelu pass: single TBuf + PIPE_ALL mode
    uint32_t totalElements = tiling.M * tiling.N;
    uint32_t aivBlockIdx = AscendC::GetSubBlockIdx();
    uint32_t aivBlockNum = 2;
    uint32_t elementsPerBlock = totalElements / aivBlockNum;
    uint32_t startElem = aivBlockIdx * elementsPerBlock;
    uint32_t endElem = (aivBlockIdx == aivBlockNum - 1) ? totalElements : startElem + elementsPerBlock;
    uint32_t myElements = endElem - startElem;

    const uint32_t CHUNK_SIZE = {chunk_size};
    AscendC::GlobalTensor<float> cGm;
    cGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(c) + startElem, myElements);

    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    pipe.InitBuffer(calcBuf, CHUNK_SIZE * sizeof(float));

    uint32_t numChunks = (myElements + CHUNK_SIZE - 1) / CHUNK_SIZE;
    for (uint32_t i = 0; i < numChunks; i++) {{
        uint32_t offset = i * CHUNK_SIZE;
        uint32_t curSize = (myElements - offset < CHUNK_SIZE) ? (myElements - offset) : CHUNK_SIZE;
        uint32_t alignedSize = (curSize + 7) / 8 * 8;

        AscendC::LocalTensor<float> localBuf = calcBuf.Get<float>();
        DataCopy(localBuf, cGm[offset], alignedSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        LeakyRelu(localBuf, localBuf, (float)0.001, alignedSize);
        AscendC::PipeBarrier<PIPE_ALL>();
        DataCopy(cGm[offset], localBuf, alignedSize);
        AscendC::PipeBarrier<PIPE_ALL>();
    }}"""

    # Replace everything from "// LeakyRelu pass" to end of function (before closing brace)
    src = re.sub(
        r'    // LeakyRelu pass.*?(?=\n\})',
        new_leakyrelu,
        src, flags=re.DOTALL)
    return src

def patch_tque_1buf(src):
    """Change TQue from 2 buffers to 1 buffer each."""
    src = re.sub(r'pipe\.InitBuffer\(inQueue, 2,', 'pipe.InitBuffer(inQueue, 1,', src)
    src = re.sub(r'pipe\.InitBuffer\(outQueue, 2,', 'pipe.InitBuffer(outQueue, 1,', src)
    return src

def patch_result_format_nz(src):
    """Change result format from ND to NZ."""
    src = re.sub(r'CubeFormat resultFormat = CubeFormat::ND;',
                 'CubeFormat resultFormat = CubeFormat::NZ;', src)
    # Also change the template to NZ
    src = re.sub(
        r'(matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, cType>)',
        r'matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, cType>',
        src)
    return src

def patch_transB(src):
    """Set isTransB = true."""
    src = re.sub(r'bool isTransB = false;', 'bool isTransB = true;', src)
    return src

def patch_input_nz(src, which='both'):
    """Change A/B input format from ND to NZ."""
    if which in ('A', 'both'):
        src = re.sub(r'CubeFormat leftFormat = CubeFormat::ND;',
                     'CubeFormat leftFormat = CubeFormat::NZ;', src)
    if which in ('B', 'both'):
        src = re.sub(r'CubeFormat rightFormat = CubeFormat::ND;',
                     'CubeFormat rightFormat = CubeFormat::NZ;', src)
    return src

def patch_atomic_add(src):
    """Try enabling AtomicAdd in GetTensorC: GetTensorC<true, true>."""
    src = re.sub(r'matmulObj\.template GetTensorC<true>\(cGlobal\)',
                 'matmulObj.template GetTensorC<true, true>(cGlobal)', src)
    return src

def remove_bias_from_matmul(src):
    """Remove bias from matmul, apply bias in LeakyRelu pass instead.
    This is complex - we remove SetBias and the bias global tensor."""
    src = re.sub(r'\s*tilingApi\.SetBias\(isBias\);', '\n    tilingApi.SetBias(false);', src)
    src = re.sub(r'bool isBias = true;', 'bool isBias = false;', src)
    return src

def patch_enable_nd2nz(src):
    """Try adding enVecND2NZ config."""
    # Insert SetMatmulConfigParams to enable ND->NZ conversion
    # This is done by adding a second SetMatmulConfigParams call or modifying the existing one
    if 'SetMatmulConfigParams' in src:
        # Replace to add enVecND2NZ
        src = re.sub(r'tilingApi\.SetMatmulConfigParams\([^)]*\)',
                     'tilingApi.SetMatmulConfigParams(1, true, true)', src)
    return src

def patch_disable_nd2nz(src):
    """Ensure enVecND2NZ is disabled."""
    if 'SetMatmulConfigParams' in src:
        src = re.sub(r'tilingApi\.SetMatmulConfigParams\([^)]*\)',
                     'tilingApi.SetMatmulConfigParams(1, true, false)', src)
    return src

def make_aicore_only(src):
    """Replace __mix__(1,2) with __aicore__ - matmul only, no LeakyRelu."""
    # Change function declaration
    src = re.sub(r'__global__\s+__mix__\(\s*\d+\s*,\s*\d+\s*\)',
                 '__global__ __aicore__', src)
    # Remove everything after matmulKernel.Process(&pipe); up to the closing brace
    src = re.sub(
        r'(matmulKernel\.Process\(&pipe\);)\s*\n\s*// LeakyRelu pass.*?(?=\n\})',
        r'\1',
        src, flags=re.DOTALL)
    return src

def patch_half_output(src):
    """Change output from float to half - measures perf, changes semantics."""
    src = re.sub(r'DataType resultDtype = DataType::DT_FLOAT;',
                 'DataType resultDtype = DataType::DT_FLOAT16;', src)
    # Change template instantiation
    src = re.sub(r'MatmulKernel<half, half, float, float>',
                 'MatmulKernel<half, half, half, float>', src)
    # Change cFileSize
    src = re.sub(r'size_t cFileSize = 655360 \* sizeof\(float\);',
                 'size_t cFileSize = 655360 * sizeof(int16_t);', src)
    # Change LeakyRelu to use half
    src = re.sub(r'AscendC::GlobalTensor<float> cGm;',
                 'AscendC::GlobalTensor<half> cGm;', src)
    src = re.sub(r'reinterpret_cast<__gm__ float \*>\(c\)',
                 'reinterpret_cast<__gm__ half *>(c)', src)
    src = re.sub(r'AllocTensor<float>', 'AllocTensor<half>', src)
    src = re.sub(r'DeQue<float>', 'DeQue<half>', src)
    src = re.sub(r'LeakyRelu\(outBuf, computeBuf, \(float\)0\.001',
                 'LeakyRelu(outBuf, computeBuf, (half)0.001', src)
    src = re.sub(r'CHUNK_SIZE \* sizeof\(float\)', 'CHUNK_SIZE * sizeof(half)', src)
    return src

def patch_single_aiv_leakyrelu(src):
    """Only process LeakyRelu on AIV 0 (skip on AIV 1)."""
    # Wrap the LeakyRelu section in an if (aivBlockIdx == 0) block,
    # but adjust to process all elements
    src = re.sub(
        r'(uint32_t aivBlockIdx = AscendC::GetSubBlockIdx\(\);)\s*\n\s*(uint32_t aivBlockNum = \d+;)',
        r'\1\n    uint32_t aivBlockNum = 1;  // single AIV processes all',
        src)
    # Wrap with if
    src = re.sub(
        r'(uint32_t aivBlockIdx = AscendC::GetSubBlockIdx\(\);)',
        r'\1\n    if (aivBlockIdx != 0) return;  // only AIV 0 does LeakyRelu',
        src)
    return src


# ---------------------------------------------------------------------------
# Configuration generators
# ---------------------------------------------------------------------------

def gen_group_a():
    """Group A: Matmul tiling (iter 29-60)."""
    configs = []

    # fixSplit variations with FIRSTN
    fixsplits = [
        (256, 128),  # baseline
        (128, 128),
        (256, 64),
        (192, 128),
        (384, 128),
        (256, 192),
        (192, 192),
    ]

    # fixSplit + traverse + stepM + stepN combos
    for fs in fixsplits:
        for stepM in [1, 2, 3, 4]:
            for stepN in [1, 2]:
                name = f"A_fs{fs[0]}x{fs[1]}_FIRSTN_sM{stepM}_sN{stepN}"
                def make_fn(fs=fs, stepM=stepM, stepN=stepN):
                    def fn(src):
                        src = patch_fixsplit(src, fs)
                        src = patch_traverse(src, "FIRSTN")
                        src = patch_stepmn(src, stepM, stepN)
                        return src
                    return fn
                configs.append((name, make_fn()))

    # fixSplit (256,128) with FIRSTM, various steps
    for stepM in [1, 2, 3, 4]:
        for stepN in [1, 2]:
            name = f"A_fs256x128_FIRSTM_sM{stepM}_sN{stepN}"
            def make_fn(stepM=stepM, stepN=stepN):
                def fn(src):
                    src = patch_fixsplit(src, (256, 128))
                    src = patch_traverse(src, "FIRSTM")
                    src = patch_stepmn(src, stepM, stepN)
                    return src
                return fn
            configs.append((name, make_fn()))

    # baseK variations
    for baseK in [64, 128, 256]:
        name = f"A_fs256x128xK{baseK}_FIRSTN_sM2_sN1"
        def make_fn(baseK=baseK):
            def fn(src):
                src = patch_fixsplit(src, (256, 128, baseK))
                src = patch_traverse(src, "FIRSTN")
                src = patch_stepmn(src, 2, 1)
                return src
            return fn
        configs.append((name, make_fn()))

    # L1CacheUB false
    name = "A_L1CacheUB_false_fs256x128"
    def fn_l1false(src):
        src = patch_l1cache_ub(src, False)
        return src
    configs.append((name, fn_l1false))

    # Auto tiling (no fixSplit) with various stepM
    for stepM in [1, 2, 4]:
        name = f"A_auto_FIRSTN_sM{stepM}_sN1"
        def make_fn(stepM=stepM):
            def fn(src):
                src = patch_fixsplit(src, None)
                src = patch_traverse(src, "FIRSTN")
                src = patch_stepmn(src, stepM, 1)
                return src
            return fn
        configs.append((name, make_fn()))

    return configs


def gen_group_b():
    """Group B: LeakyRelu chunk tuning (iter 61-80)."""
    configs = []

    # TQue double-buffer mode with different chunk sizes
    for cs in [4096, 6144, 8192, 10240, 12288, 14336, 16384, 20480, 24576]:
        name = f"B_tque_chunk{cs}"
        def make_fn(cs=cs):
            def fn(src):
                src = patch_chunk_size(src, cs)
                return src
            return fn
        configs.append((name, make_fn()))

    # Single TBuf + PIPE_ALL mode with large chunks
    for cs in [32768, 40960, 45056, 49152]:
        name = f"B_tbuf_chunk{cs}"
        def make_fn(cs=cs):
            def fn(src):
                src = patch_tque_to_tbuf(src, cs)
                return src
            return fn
        configs.append((name, make_fn()))

    # TQue with 1 buffer instead of 2
    name = "B_tque_1buf_chunk12288"
    def fn_1buf(src):
        src = patch_tque_1buf(src)
        return src
    configs.append((name, fn_1buf))

    # TQue with 1 buffer + different chunks
    for cs in [16384, 24576]:
        name = f"B_tque_1buf_chunk{cs}"
        def make_fn(cs=cs):
            def fn(src):
                src = patch_chunk_size(src, cs)
                src = patch_tque_1buf(src)
                return src
            return fn
        configs.append((name, make_fn()))

    return configs


def gen_group_c():
    """Group C: Kernel structure (iter 81-120)."""
    configs = []

    # HF32 on
    name = "C_hf32_on"
    configs.append((name, add_hf32))

    # HF32 off (ensure removed)
    name = "C_hf32_off"
    configs.append((name, remove_hf32))

    # enVecND2NZ on
    name = "C_nd2nz_on"
    configs.append((name, patch_enable_nd2nz))

    # enVecND2NZ off
    name = "C_nd2nz_off"
    configs.append((name, patch_disable_nd2nz))

    # C output NZ format
    name = "C_result_NZ"
    configs.append((name, patch_result_format_nz))

    # Single AIV for LeakyRelu
    name = "C_single_aiv_leakyrelu"
    configs.append((name, patch_single_aiv_leakyrelu))

    # Remove bias from matmul
    name = "C_no_bias_matmul"
    configs.append((name, remove_bias_from_matmul))

    # HF32 on + best tiling combos
    for fs in [(256, 128), (192, 128), (384, 128)]:
        for stepM in [2, 3]:
            name = f"C_hf32_fs{fs[0]}x{fs[1]}_sM{stepM}"
            def make_fn(fs=fs, stepM=stepM):
                def fn(src):
                    src = add_hf32(src)
                    src = patch_fixsplit(src, fs)
                    src = patch_stepmn(src, stepM, 1)
                    return src
                return fn
            configs.append((name, make_fn()))

    # nd2nz + best combos
    for fs in [(256, 128), (192, 128)]:
        name = f"C_nd2nz_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = patch_enable_nd2nz(src)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # HF32 + nd2nz combined
    name = "C_hf32_nd2nz"
    def fn_both(src):
        src = add_hf32(src)
        src = patch_enable_nd2nz(src)
        return src
    configs.append((name, fn_both))

    # HF32 + larger chunk
    for cs in [16384, 24576]:
        name = f"C_hf32_chunk{cs}"
        def make_fn(cs=cs):
            def fn(src):
                src = add_hf32(src)
                src = patch_chunk_size(src, cs)
                return src
            return fn
        configs.append((name, make_fn()))

    # nd2nz + larger chunk
    for cs in [16384, 24576]:
        name = f"C_nd2nz_chunk{cs}"
        def make_fn(cs=cs):
            def fn(src):
                src = patch_enable_nd2nz(src)
                src = patch_chunk_size(src, cs)
                return src
            return fn
        configs.append((name, make_fn()))

    # Combined: hf32 + nd2nz + best tiling
    for fs in [(256, 128), (192, 128)]:
        for stepM in [2, 3]:
            name = f"C_hf32_nd2nz_fs{fs[0]}x{fs[1]}_sM{stepM}"
            def make_fn(fs=fs, stepM=stepM):
                def fn(src):
                    src = add_hf32(src)
                    src = patch_enable_nd2nz(src)
                    src = patch_fixsplit(src, fs)
                    src = patch_stepmn(src, stepM, 1)
                    return src
                return fn
            configs.append((name, make_fn()))

    # Different aivBlockNum values
    for aivn in [1, 4]:
        name = f"C_aivBlockNum{aivn}"
        def make_fn(aivn=aivn):
            def fn(src):
                src = patch_aiv_block_num(src, aivn)
                if aivn != 2:
                    src = patch_mix(src, 1, aivn)
                return src
            return fn
        configs.append((name, make_fn()))

    # Various LeakyRelu + tiling combos for the remaining slots
    # Smaller chunk with FIRSTN auto
    for cs in [8192, 10240]:
        name = f"C_auto_chunk{cs}"
        def make_fn(cs=cs):
            def fn(src):
                src = patch_fixsplit(src, None)
                src = patch_chunk_size(src, cs)
                return src
            return fn
        configs.append((name, make_fn()))

    return configs


def gen_group_d():
    """Group D: Advanced matmul (iter 121-160)."""
    configs = []

    # TransB (requires data change - mark as skipped but try anyway)
    name = "D_transB"
    configs.append((name, patch_transB))

    # NZ input format for A
    name = "D_input_NZ_A"
    def fn_nz_a(src):
        return patch_input_nz(src, 'A')
    configs.append((name, fn_nz_a))

    # NZ input format for B
    name = "D_input_NZ_B"
    def fn_nz_b(src):
        return patch_input_nz(src, 'B')
    configs.append((name, fn_nz_b))

    # NZ input for both
    name = "D_input_NZ_both"
    def fn_nz_both(src):
        return patch_input_nz(src, 'both')
    configs.append((name, fn_nz_both))

    # AtomicAdd in GetTensorC
    name = "D_atomic_add"
    configs.append((name, patch_atomic_add))

    # Different SetBufferSpace L1 values
    for l1_val in [131072, 262144, 524288]:
        name = f"D_bufL1_{l1_val}"
        def make_fn(l1=l1_val):
            def fn(src):
                src = patch_buffer_space(src, l1, -1, -1)
                return src
            return fn
        configs.append((name, make_fn()))

    # SetBufferSpace L0A/L0B values
    for l0 in [32768, 65536]:
        name = f"D_bufL0_{l0}"
        def make_fn(l0=l0):
            def fn(src):
                src = patch_buffer_space(src, -1, l0, l0)
                return src
            return fn
        configs.append((name, make_fn()))

    # Combined: buffer space + best tiling
    for l1_val in [131072, 262144, 524288]:
        for fs in [(256, 128), (192, 128)]:
            name = f"D_bufL1_{l1_val}_fs{fs[0]}x{fs[1]}"
            def make_fn(l1=l1_val, fs=fs):
                def fn(src):
                    src = patch_buffer_space(src, l1, -1, -1)
                    src = patch_fixsplit(src, fs)
                    return src
                return fn
            configs.append((name, make_fn()))

    # AtomicAdd + best tiling
    for fs in [(256, 128), (192, 128)]:
        name = f"D_atomic_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = patch_atomic_add(src)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # Buffer space with L0A/L0B + L1
    for l1_val in [131072, 262144]:
        for l0 in [32768, 65536]:
            name = f"D_L1_{l1_val}_L0_{l0}"
            def make_fn(l1=l1_val, l0=l0):
                def fn(src):
                    src = patch_buffer_space(src, l1, l0, l0)
                    return src
                return fn
            configs.append((name, make_fn()))

    # Best combos: buffer + hf32
    for l1_val in [131072, 262144, 524288]:
        name = f"D_hf32_bufL1_{l1_val}"
        def make_fn(l1=l1_val):
            def fn(src):
                src = add_hf32(src)
                src = patch_buffer_space(src, l1, -1, -1)
                return src
            return fn
        configs.append((name, make_fn()))

    # buffer + nd2nz
    for l1_val in [131072, 262144]:
        name = f"D_nd2nz_bufL1_{l1_val}"
        def make_fn(l1=l1_val):
            def fn(src):
                src = patch_enable_nd2nz(src)
                src = patch_buffer_space(src, l1, -1, -1)
                return src
            return fn
        configs.append((name, make_fn()))

    # Combined best: hf32 + buf + tiling
    for l1_val in [131072, 262144]:
        for stepM in [2, 3]:
            name = f"D_hf32_bufL1_{l1_val}_sM{stepM}"
            def make_fn(l1=l1_val, stepM=stepM):
                def fn(src):
                    src = add_hf32(src)
                    src = patch_buffer_space(src, l1, -1, -1)
                    src = patch_stepmn(src, stepM, 1)
                    return src
                return fn
            configs.append((name, make_fn()))

    return configs


def gen_group_e():
    """Group E: Radical restructuring (iter 161-200)."""
    configs = []

    # __aicore__ only matmul (no LeakyRelu) - measures matmul-only floor
    name = "E_aicore_matmul_only"
    configs.append((name, make_aicore_only))

    # Half output (changes semantics)
    name = "E_half_output"
    configs.append((name, patch_half_output))

    # Different numBlocks with matching usedCoreNum
    for nb in [2, 4]:
        name = f"E_numBlocks{nb}_cores{nb}"
        def make_fn(nb=nb):
            def fn(src):
                src = patch_num_blocks(src, nb)
                src = patch_used_core_num(src, nb)
                src = patch_mix(src, 1, nb)
                src = patch_aiv_block_num(src, nb)
                return src
            return fn
        configs.append((name, make_fn()))

    # aicore matmul-only + different tilings
    for fs in [(256, 128), (192, 128), (384, 128), (128, 128)]:
        name = f"E_aicore_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = make_aicore_only(src)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # aicore + auto tiling
    name = "E_aicore_auto"
    def fn_aicore_auto(src):
        src = make_aicore_only(src)
        src = patch_fixsplit(src, None)
        return src
    configs.append((name, fn_aicore_auto))

    # aicore + different stepM
    for stepM in [1, 2, 3, 4]:
        name = f"E_aicore_sM{stepM}"
        def make_fn(stepM=stepM):
            def fn(src):
                src = make_aicore_only(src)
                src = patch_stepmn(src, stepM, 1)
                return src
            return fn
        configs.append((name, make_fn()))

    # aicore + hf32
    name = "E_aicore_hf32"
    def fn_aicore_hf32(src):
        src = make_aicore_only(src)
        src = add_hf32(src)
        return src
    configs.append((name, fn_aicore_hf32))

    # aicore + buffer space
    for l1_val in [131072, 262144]:
        name = f"E_aicore_bufL1_{l1_val}"
        def make_fn(l1=l1_val):
            def fn(src):
                src = make_aicore_only(src)
                src = patch_buffer_space(src, l1, -1, -1)
                return src
            return fn
        configs.append((name, make_fn()))

    # aicore + nd2nz
    name = "E_aicore_nd2nz"
    def fn_aicore_nd2nz(src):
        src = make_aicore_only(src)
        src = patch_enable_nd2nz(src)
        return src
    configs.append((name, fn_aicore_nd2nz))

    # numBlocks=2 + different tiling
    for fs in [(256, 128), (192, 128), (128, 128)]:
        name = f"E_nb2_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = patch_num_blocks(src, 2)
                src = patch_used_core_num(src, 2)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # Combined radical: aicore + best tiling + hf32
    for fs in [(256, 128), (192, 128)]:
        name = f"E_aicore_hf32_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = make_aicore_only(src)
                src = add_hf32(src)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # aicore + best buffer + tiling
    for l1_val in [131072, 262144]:
        name = f"E_aicore_hf32_buf{l1_val}"
        def make_fn(l1=l1_val):
            def fn(src):
                src = make_aicore_only(src)
                src = add_hf32(src)
                src = patch_buffer_space(src, l1, -1, -1)
                return src
            return fn
        configs.append((name, make_fn()))

    # numBlocks=4 + tiling combos
    for fs in [(256, 128), (128, 128)]:
        name = f"E_nb4_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = patch_num_blocks(src, 4)
                src = patch_used_core_num(src, 4)
                src = patch_mix(src, 1, 4)
                src = patch_aiv_block_num(src, 4)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # Half output + best tiling
    name = "E_half_fs256x128"
    def fn_half_tiling(src):
        src = patch_half_output(src)
        src = patch_fixsplit(src, (256, 128))
        return src
    configs.append((name, fn_half_tiling))

    # L1CacheUB false + various combos
    for fs in [(256, 128), (192, 128)]:
        name = f"E_noL1_fs{fs[0]}x{fs[1]}"
        def make_fn(fs=fs):
            def fn(src):
                src = patch_l1cache_ub(src, False)
                src = patch_fixsplit(src, fs)
                return src
            return fn
        configs.append((name, make_fn()))

    # Kitchen sink: hf32 + nd2nz + best buf + best tiling
    name = "E_kitchen_sink"
    def fn_kitchen(src):
        src = add_hf32(src)
        src = patch_enable_nd2nz(src)
        src = patch_buffer_space(src, 262144, -1, -1)
        src = patch_fixsplit(src, (256, 128))
        src = patch_stepmn(src, 2, 1)
        return src
    configs.append((name, fn_kitchen))

    # Kitchen sink v2
    name = "E_kitchen_sink_v2"
    def fn_kitchen2(src):
        src = add_hf32(src)
        src = patch_enable_nd2nz(src)
        src = patch_buffer_space(src, 131072, -1, -1)
        src = patch_fixsplit(src, (192, 128))
        src = patch_stepmn(src, 3, 1)
        return src
    configs.append((name, fn_kitchen2))

    # Kitchen sink v3 with chunk tuning
    name = "E_kitchen_sink_chunk16k"
    def fn_kitchen3(src):
        src = add_hf32(src)
        src = patch_enable_nd2nz(src)
        src = patch_fixsplit(src, (256, 128))
        src = patch_stepmn(src, 2, 1)
        src = patch_chunk_size(src, 16384)
        return src
    configs.append((name, fn_kitchen3))

    return configs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("MEGA SWEEP: matmul_leakyrelu NPU kernel optimization")
    print(f"Starting from iteration 29")
    print("=" * 80)

    # Verify files exist
    if not ASC_FILE.exists():
        print(f"ERROR: {ASC_FILE} not found!")
        sys.exit(1)

    # Save original
    original_src = read_asc()
    backup_path = SOLUTION_DIR / "matmul_leakyrelu.asc.mega_backup"
    backup_path.write_text(original_src)
    print(f"Backup saved to {backup_path}")

    # Generate all configs
    all_configs = []
    ga = gen_group_a()
    gb = gen_group_b()
    gc = gen_group_c()
    gd = gen_group_d()
    ge = gen_group_e()

    all_configs.extend(ga)
    all_configs.extend(gb)
    all_configs.extend(gc)
    all_configs.extend(gd)
    all_configs.extend(ge)

    total = len(all_configs)
    print(f"\nTotal configurations: {total}")
    print(f"  Group A (matmul tiling):    {len(ga)}")
    print(f"  Group B (LeakyRelu chunk):  {len(gb)}")
    print(f"  Group C (kernel structure): {len(gc)}")
    print(f"  Group D (advanced matmul):  {len(gd)}")
    print(f"  Group E (radical):          {len(ge)}")
    print()

    # CSV setup
    csv_fields = ['iter', 'name', 'task_duration_us', 'status', 'details']
    with open(RESULTS_CSV, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        writer.writeheader()

        results = []
        start_time = time.time()

        for idx, (name, modifier_fn) in enumerate(all_configs):
            iter_num = 29 + idx
            elapsed = time.time() - start_time
            eta = (elapsed / max(idx, 1)) * (total - idx)
            print(f"\nIter {iter_num}/{29 + total - 1} [{idx+1}/{total}]: {name}  "
                  f"(elapsed: {elapsed/60:.1f}m, ETA: {eta/60:.1f}m)")

            try:
                status, dur, details = test_config(name, modifier_fn)
                dur_str = f"{dur:.2f}" if dur > 0 else ""

                row = {
                    'iter': iter_num,
                    'name': name,
                    'task_duration_us': dur_str,
                    'status': status,
                    'details': details,
                }
                writer.writerow(row)
                csvf.flush()
                results.append(row)

                if status == 'ok':
                    speedup = 227.9 / dur if dur > 0 else 0
                    print(f"  => {dur:.2f} us ({speedup:.2f}x from baseline)")
                else:
                    print(f"  => {status}: {details[:100]}")

            except Exception as e:
                tb = traceback.format_exc()
                print(f"  => EXCEPTION: {e}")
                row = {
                    'iter': iter_num,
                    'name': name,
                    'task_duration_us': '',
                    'status': 'exception',
                    'details': str(e)[:200],
                }
                writer.writerow(row)
                csvf.flush()
                results.append(row)

    # Restore original
    write_asc(original_src)
    backup_path.unlink(missing_ok=True)
    print(f"\nRestored original matmul_leakyrelu.asc")

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"MEGA SWEEP COMPLETE - {total_time/60:.1f} minutes")
    print(f"{'=' * 80}")

    ok_results = [r for r in results if r['status'] == 'ok' and r['task_duration_us']]
    fail_results = [r for r in results if r['status'] != 'ok']

    if ok_results:
        ok_results.sort(key=lambda r: float(r['task_duration_us']))
        print(f"\nTOP 20 (by task duration):")
        print(f"{'Rank':>4}  {'Iter':>4}  {'Duration(us)':>12}  {'Speedup':>8}  Name")
        print("-" * 80)
        for rank, r in enumerate(ok_results[:20], 1):
            dur = float(r['task_duration_us'])
            speedup = 227.9 / dur
            print(f"{rank:>4}  {r['iter']:>4}  {dur:>12.2f}  {speedup:>7.2f}x  {r['name']}")

        best = ok_results[0]
        print(f"\nBEST: {best['task_duration_us']} us - {best['name']} "
              f"({227.9/float(best['task_duration_us']):.2f}x from 227.9 us baseline)")
    else:
        print("\nNo successful profiling runs!")

    print(f"\nTotal: {len(results)} | OK: {len(ok_results)} | Failed: {len(fail_results)}")
    if fail_results:
        from collections import Counter
        counts = Counter(r['status'] for r in fail_results)
        print("Failure breakdown:")
        for st, cnt in counts.most_common():
            print(f"  {st}: {cnt}")

    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == '__main__':
    main()
