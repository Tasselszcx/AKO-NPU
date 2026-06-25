#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sweep for matmul_leakyrelu NPU kernel.

Generates variants of solution/matmul_leakyrelu.asc with different tiling
parameters, builds, runs, verifies correctness, and profiles with msprof.
Results are written to scripts/sweep_results.csv.
"""

import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
PROJECT_DIR = SCRIPT_DIR.parent                        # project root
SOLUTION_DIR = PROJECT_DIR / "solution"
TEMPLATE_ASC = SOLUTION_DIR / "matmul_leakyrelu.asc"
RESULTS_CSV = SCRIPT_DIR / "sweep_results.csv"

# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
USED_CORE_NUMS = [2]  # Only 2 works with mix(1,2)
STEP_MS = [1, 2, 4]
STEP_NS = [1, 2]
TRAVERSES = ["FIRSTM", "FIRSTN"]
# (baseM, baseN) pairs; None = auto tiling (no SetFixSplit)
FIX_SPLITS = [
    None,              # auto
    (256, 128),        # original baseline
    (128, 128),        # smaller M tiles
    (64, 128),         # even smaller M
    (128, 160),        # divides 640
    (256, 64),         # narrow N
]
SET_HF32S = [True]  # HF32 always on (minimal overhead)
# (L1, L0A, L0B) tuples; None = auto (-1,-1,-1)
BUFFER_SPACES = [
    None,               # auto: SetBufferSpace(-1, -1, -1)
]

DEMO_TIMEOUT = 30  # seconds
MSPROF_TIMEOUT = 120  # seconds


def read_template():
    """Read the template .asc source."""
    return TEMPLATE_ASC.read_text()


# ---------------------------------------------------------------------------
# Source-code patching helpers
# ---------------------------------------------------------------------------

def patch_used_core_num(src: str, value: int) -> str:
    """Replace usedCoreNum = <N>; in GenerateTiling."""
    return re.sub(
        r'(int\s+usedCoreNum\s*=\s*)\d+\s*;',
        rf'\g<1>{value};',
        src,
    )


def patch_traverse(src: str, traverse: str) -> str:
    """Replace SetTraverse(matmul_tiling::MatrixTraverse::XXX)."""
    return re.sub(
        r'(tilingApi\.SetTraverse\(matmul_tiling::MatrixTraverse::)\w+(\))',
        rf'\g<1>{traverse}\2',
        src,
    )


def patch_fix_split(src: str, fix_split) -> str:
    """Insert / remove / replace SetFixSplit line.

    fix_split: None  -> remove any SetFixSplit line
               (M,N) -> insert or replace SetFixSplit(baseM, baseN, -1)
    """
    has_fix_split = 'SetFixSplit' in src

    if fix_split is None:
        # Remove existing SetFixSplit line if present
        if has_fix_split:
            src = re.sub(r'\s*tilingApi\.SetFixSplit\([^)]*\);\n?', '\n', src)
        return src

    baseM, baseN = fix_split
    new_line = f'    tilingApi.SetFixSplit({baseM}, {baseN}, -1);'

    if has_fix_split:
        src = re.sub(
            r'(\s*)tilingApi\.SetFixSplit\([^)]*\);',
            rf'\1tilingApi.SetFixSplit({baseM}, {baseN}, -1);',
            src,
        )
    else:
        # Insert after SetShape line
        src = re.sub(
            r'(tilingApi\.SetShape\([^)]*\);)',
            rf'\1\n{new_line}',
            src,
        )
    return src


def patch_buffer_space(src: str, buf_space) -> str:
    """Replace SetBufferSpace(...) arguments."""
    if buf_space is None:
        l1, l0a, l0b = -1, -1, -1
    else:
        l1, l0a, l0b = buf_space
    return re.sub(
        r'tilingApi\.SetBufferSpace\([^)]*\)',
        f'tilingApi.SetBufferSpace({l1}, {l0a}, {l0b})',
        src,
    )


def patch_step_m_n(src: str, stepM: int, stepN: int) -> str:
    """Replace tilingData.stepM and tilingData.stepN assignments."""
    src = re.sub(
        r'tilingData\.stepM\s*=\s*\d+\s*;',
        f'tilingData.stepM = {stepM};',
        src,
    )
    src = re.sub(
        r'tilingData\.stepN\s*=\s*\d+\s*;',
        f'tilingData.stepN = {stepN};',
        src,
    )
    return src


def patch_set_hf32(src: str, enable: bool) -> str:
    """Enable or disable SetHF32(true) in the kernel body.

    When enable=True:  ensure matmulKernel.matmulObj.SetHF32(true); is present
    When enable=False: remove that line
    """
    has_hf32 = 'SetHF32' in src

    if enable and not has_hf32:
        # Insert after REGIST_MATMUL_OBJ line
        src = re.sub(
            r'(REGIST_MATMUL_OBJ\([^)]*\);)',
            r'\1\n    matmulKernel.matmulObj.SetHF32(true);',
            src,
        )
    elif not enable and has_hf32:
        src = re.sub(r'\s*matmulKernel\.matmulObj\.SetHF32\(true\);\n?', '\n', src)

    return src


def patch_aiv_block_num(src: str, used_core_num: int) -> str:
    """Keep aivBlockNum consistent with __mix__(1, N).

    __mix__(1, N) means N AIV sub-blocks. For correctness the LeakyRelu
    section hard-codes aivBlockNum; update it when usedCoreNum changes.
    Note: numBlocks stays at 1 (the mix mode handles multi-core).
    """
    # The AIV sub-block count in __mix__(1, N) equals N = usedCoreNum
    src = re.sub(
        r'(uint32_t\s+aivBlockNum\s*=\s*)\d+\s*;',
        rf'\g<1>{used_core_num};',
        src,
    )
    # Also patch the __mix__ declaration itself
    src = re.sub(
        r'__mix__\(\s*1\s*,\s*\d+\s*\)',
        f'__mix__(1, {used_core_num})',
        src,
    )
    return src


def generate_variant(template: str, params: dict) -> str:
    """Apply all parameter patches to the template source."""
    src = template
    src = patch_used_core_num(src, params['usedCoreNum'])
    src = patch_traverse(src, params['traverse'])
    src = patch_fix_split(src, params['fixSplit'])
    src = patch_buffer_space(src, params['bufferSpace'])
    src = patch_step_m_n(src, params['stepM'], params['stepN'])
    src = patch_set_hf32(src, params['setHF32'])
    src = patch_aiv_block_num(src, params['usedCoreNum'])
    return src


# ---------------------------------------------------------------------------
# Build / Run / Profile helpers
# ---------------------------------------------------------------------------

def setup_env() -> dict:
    """Return an environment dict with ASCEND / CANN paths configured."""
    env = os.environ.copy()
    ascend_home = env.get('ASCEND_HOME_PATH', '')
    if not ascend_home:
        # Try common default
        for candidate in ['/usr/local/Ascend/ascend-toolkit/latest',
                          '/usr/local/Ascend/ascend-toolkit']:
            if os.path.isdir(candidate):
                ascend_home = candidate
                break
    if not ascend_home:
        print("WARNING: ASCEND_HOME_PATH not set and default not found", file=sys.stderr)

    env['ASCEND_HOME_PATH'] = ascend_home

    # Source set_env.sh by running it in a sub-shell and capturing env
    set_env_sh = None
    for p in [os.path.join(ascend_home, 'set_env.sh'),
              '/usr/local/Ascend/ascend-toolkit/set_env.sh']:
        if os.path.isfile(p):
            set_env_sh = p
            break

    if set_env_sh:
        try:
            cmd = f'source "{set_env_sh}" >/dev/null 2>&1 && env'
            result = subprocess.run(
                ['bash', '-c', cmd],
                capture_output=True, text=True, timeout=10, env=env,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if '=' in line:
                        key, _, val = line.partition('=')
                        env[key] = val
        except Exception:
            pass

    # Set CMAKE_PREFIX_PATH
    cmake_prefix = os.path.join(ascend_home, 'x86_64-linux', 'tikcpp',
                                'ascendc_kernel_cmake')
    existing = env.get('CMAKE_PREFIX_PATH', '')
    env['CMAKE_PREFIX_PATH'] = f"{cmake_prefix}:{existing}" if existing else cmake_prefix

    return env


def build_variant(solution_dir: Path, env: dict) -> bool:
    """Run cmake + make in solution_dir/build. Returns True on success."""
    build_dir = solution_dir / 'build'
    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(str(build_dir), 0o755)

    try:
        r = subprocess.run(
            ['cmake', '..'],
            cwd=str(build_dir), env=env,
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print(f"  cmake FAILED:\n{r.stderr[:500]}", file=sys.stderr)
            return False

        r = subprocess.run(
            ['make', '-j4'],
            cwd=str(build_dir), env=env,
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            print(f"  make FAILED:\n{r.stderr[:500]}", file=sys.stderr)
            return False

        os.chmod(str(build_dir), 0o755)
        demo = build_dir / 'demo'
        if not demo.exists():
            print("  demo binary not found after build", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  build TIMEOUT", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  build exception: {e}", file=sys.stderr)
        return False


def gen_data(build_dir: Path, solution_dir: Path, env: dict) -> bool:
    """Generate test data in build_dir."""
    gen_script = solution_dir / 'scripts' / 'gen_data.py'
    try:
        r = subprocess.run(
            [sys.executable, str(gen_script)],
            cwd=str(build_dir), env=env,
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except Exception as e:
        print(f"  gen_data exception: {e}", file=sys.stderr)
        return False


def run_demo(build_dir: Path, env: dict) -> bool:
    """Run the demo binary. Returns True if output.bin is produced."""
    output_bin = build_dir / 'output' / 'output.bin'
    output_bin.unlink(missing_ok=True)

    try:
        r = subprocess.run(
            ['./demo'],
            cwd=str(build_dir), env=env,
            capture_output=True, text=True, timeout=DEMO_TIMEOUT,
        )
        if r.returncode != 0:
            print(f"  demo FAILED (rc={r.returncode})", file=sys.stderr)
            return False
        if not output_bin.exists():
            print("  output.bin not found after run", file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  demo TIMEOUT", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  demo exception: {e}", file=sys.stderr)
        return False


def verify_correctness(build_dir: Path, solution_dir: Path, env: dict) -> bool:
    """Verify output against golden data."""
    verify_script = solution_dir / 'scripts' / 'verify_result.py'
    output_bin = build_dir / 'output' / 'output.bin'
    golden_bin = build_dir / 'output' / 'golden.bin'

    if not output_bin.exists() or not golden_bin.exists():
        return False

    try:
        r = subprocess.run(
            [sys.executable, str(verify_script), str(output_bin), str(golden_bin)],
            cwd=str(build_dir), env=env,
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            print(f"  verify FAILED", file=sys.stderr)
            return False
        # Check for "test pass!" in output
        if 'test pass' in r.stdout.lower():
            return True
        print(f"  verify: no 'test pass' in output", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  verify exception: {e}", file=sys.stderr)
        return False


def run_msprof(build_dir: Path, env: dict) -> float:
    """Run msprof profiling and extract Task Duration in us.

    Returns duration in us, or -1.0 on failure.
    """
    msprof_dir = Path(tempfile.mkdtemp(prefix='msprof_sweep_'))
    os.chmod(str(msprof_dir), 0o700)

    try:
        cmd = [
            'msprof', 'op',
            '--warm-up=10',
            f'--output={msprof_dir}',
            './demo',
        ]
        r = subprocess.run(
            cmd,
            cwd=str(build_dir), env=env,
            capture_output=True, text=True, timeout=MSPROF_TIMEOUT,
        )

        combined_output = r.stdout + '\n' + r.stderr

        # Method 1: Parse Task Duration from console output
        # msprof prints lines like:  Task Duration(us):  127.4
        duration = _parse_task_duration_from_text(combined_output)
        if duration > 0:
            return duration

        # Method 2: Parse from OpBasicInfo.csv
        duration = _parse_task_duration_from_csv(msprof_dir)
        if duration > 0:
            return duration

        print(f"  msprof: could not extract Task Duration", file=sys.stderr)
        return -1.0
    except subprocess.TimeoutExpired:
        print("  msprof TIMEOUT", file=sys.stderr)
        return -1.0
    except Exception as e:
        print(f"  msprof exception: {e}", file=sys.stderr)
        return -1.0
    finally:
        # Clean up msprof output
        shutil.rmtree(str(msprof_dir), ignore_errors=True)


def _parse_task_duration_from_text(text: str) -> float:
    """Extract Task Duration from msprof console output."""
    # Patterns seen in msprof output:
    #   Task Duration(us):  127.4
    #   Task Duration(us)  127.4
    #   TaskDuration(us): 127.4
    patterns = [
        r'Task\s*Duration\s*\(us\)\s*[:\s]+\s*([\d.]+)',
        r'TaskDuration\s*\(us\)\s*[:\s]+\s*([\d.]+)',
        r'task_duration\s*\(us\)\s*[:\s]+\s*([\d.]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return -1.0


def _parse_task_duration_from_csv(msprof_dir: Path) -> float:
    """Find OpBasicInfo.csv and extract Task Duration."""
    # msprof creates OPPROF_<timestamp>/ directories
    for opprof_dir in sorted(msprof_dir.glob('OPPROF_*'), reverse=True):
        csv_file = opprof_dir / 'OpBasicInfo.csv'
        if not csv_file.exists():
            continue
        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                durations = []
                for row in reader:
                    # Look for the matmul_leakyrelu_custom kernel row
                    op_name = row.get('Op Name', '') or row.get('op_name', '') or ''
                    if 'matmul_leakyrelu' in op_name.lower() or not op_name:
                        # Try various column names for task duration
                        for col in ['Task Duration(us)', 'Task Duration (us)',
                                    'TaskDuration(us)', 'task_duration(us)',
                                    'Duration(us)', 'Duration (us)']:
                            val = row.get(col, '')
                            if val:
                                try:
                                    durations.append(float(val))
                                except ValueError:
                                    continue
                if durations:
                    # Return the average or the first match
                    return sum(durations) / len(durations)
        except Exception:
            continue
    return -1.0


# ---------------------------------------------------------------------------
# Parameter label helpers
# ---------------------------------------------------------------------------

def fix_split_label(fs) -> str:
    if fs is None:
        return "auto"
    return f"{fs[0]}x{fs[1]}"


def buf_space_label(bs) -> str:
    if bs is None:
        return "auto"
    return f"L1={bs[0]}_L0A={bs[1]}_L0B={bs[2]}"


def params_label(p: dict) -> str:
    """Short human-readable label for a parameter combination."""
    parts = [
        f"c{p['usedCoreNum']}",
        f"sM{p['stepM']}",
        f"sN{p['stepN']}",
        p['traverse'],
        f"fs={fix_split_label(p['fixSplit'])}",
        f"hf32={'T' if p['setHF32'] else 'F'}",
        f"buf={buf_space_label(p['bufferSpace'])}",
    ]
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Main sweep logic
# ---------------------------------------------------------------------------

def generate_all_params():
    """Generate all parameter combinations to sweep."""
    combos = list(itertools.product(
        USED_CORE_NUMS,
        STEP_MS,
        STEP_NS,
        TRAVERSES,
        FIX_SPLITS,
        SET_HF32S,
        BUFFER_SPACES,
    ))
    params_list = []
    for (cores, sm, sn, trav, fs, hf32, buf) in combos:
        params_list.append({
            'usedCoreNum': cores,
            'stepM': sm,
            'stepN': sn,
            'traverse': trav,
            'fixSplit': fs,
            'setHF32': hf32,
            'bufferSpace': buf,
        })
    return params_list


def main():
    print("=" * 70)
    print("Matmul LeakyRelu NPU Kernel Parameter Sweep")
    print("=" * 70)

    # Read template
    if not TEMPLATE_ASC.exists():
        print(f"ERROR: Template not found: {TEMPLATE_ASC}", file=sys.stderr)
        sys.exit(1)
    template = read_template()

    # Setup environment once
    env = setup_env()

    # Backup original .asc
    backup_asc = SOLUTION_DIR / 'matmul_leakyrelu.asc.bak'
    shutil.copy2(str(TEMPLATE_ASC), str(backup_asc))

    all_params = generate_all_params()
    total = len(all_params)
    print(f"Total parameter combinations: {total}\n")

    results = []

    # CSV header
    csv_fields = [
        'index', 'label', 'usedCoreNum', 'stepM', 'stepN', 'traverse',
        'fixSplit', 'setHF32', 'bufferSpace',
        'build_ok', 'correct', 'task_duration_us', 'status',
    ]

    # Open CSV for incremental writes
    with open(RESULTS_CSV, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
        writer.writeheader()

        for idx, params in enumerate(all_params, 1):
            label = params_label(params)
            print(f"[{idx}/{total}] {label}")

            row = {
                'index': idx,
                'label': label,
                'usedCoreNum': params['usedCoreNum'],
                'stepM': params['stepM'],
                'stepN': params['stepN'],
                'traverse': params['traverse'],
                'fixSplit': fix_split_label(params['fixSplit']),
                'setHF32': params['setHF32'],
                'bufferSpace': buf_space_label(params['bufferSpace']),
                'build_ok': False,
                'correct': False,
                'task_duration_us': '',
                'status': '',
            }

            try:
                # 1. Generate variant source
                variant_src = generate_variant(template, params)
                TEMPLATE_ASC.write_text(variant_src)

                # 2. Build
                print("  Building...", end=' ', flush=True)
                if not build_variant(SOLUTION_DIR, env):
                    row['status'] = 'build_fail'
                    print("FAIL")
                    writer.writerow(row)
                    csvf.flush()
                    results.append(row)
                    continue
                row['build_ok'] = True
                print("OK", end=' ', flush=True)

                # 3. Generate test data
                build_dir = SOLUTION_DIR / 'build'
                if not gen_data(build_dir, SOLUTION_DIR, env):
                    row['status'] = 'gendata_fail'
                    print("| gendata FAIL")
                    writer.writerow(row)
                    csvf.flush()
                    results.append(row)
                    continue

                # 4. Run demo
                print("| Run...", end=' ', flush=True)
                if not run_demo(build_dir, env):
                    row['status'] = 'run_fail'
                    print("FAIL")
                    writer.writerow(row)
                    csvf.flush()
                    results.append(row)
                    continue
                print("OK", end=' ', flush=True)

                # 5. Verify correctness
                print("| Verify...", end=' ', flush=True)
                if not verify_correctness(build_dir, SOLUTION_DIR, env):
                    row['status'] = 'incorrect'
                    row['correct'] = False
                    print("FAIL")
                    writer.writerow(row)
                    csvf.flush()
                    results.append(row)
                    continue
                row['correct'] = True
                print("OK", end=' ', flush=True)

                # 6. Profile with msprof
                print("| msprof...", end=' ', flush=True)
                duration = run_msprof(build_dir, env)
                if duration > 0:
                    row['task_duration_us'] = f"{duration:.2f}"
                    row['status'] = 'ok'
                    print(f"{duration:.2f} us")
                else:
                    row['task_duration_us'] = ''
                    row['status'] = 'msprof_fail'
                    print("FAIL")

            except Exception as e:
                row['status'] = f'error: {e}'
                print(f"  EXCEPTION: {e}")

            writer.writerow(row)
            csvf.flush()
            results.append(row)

    # Restore original .asc
    if backup_asc.exists():
        shutil.copy2(str(backup_asc), str(TEMPLATE_ASC))
        backup_asc.unlink()
        print("\nRestored original matmul_leakyrelu.asc")

    # Print summary sorted by Task Duration
    print("\n" + "=" * 70)
    print("SWEEP RESULTS SUMMARY (sorted by Task Duration)")
    print("=" * 70)

    # Separate successful and failed runs
    ok_results = [r for r in results if r['status'] == 'ok' and r['task_duration_us']]
    fail_results = [r for r in results if r['status'] != 'ok']

    if ok_results:
        ok_results.sort(key=lambda r: float(r['task_duration_us']))
        print(f"\n{'Rank':>4}  {'Duration(us)':>12}  {'Cores':>5}  {'sM':>3}  {'sN':>3}  "
              f"{'Traverse':>8}  {'FixSplit':>10}  {'HF32':>5}  {'Buffer':>20}  Label")
        print("-" * 120)
        for rank, r in enumerate(ok_results, 1):
            print(f"{rank:>4}  {r['task_duration_us']:>12}  {r['usedCoreNum']:>5}  "
                  f"{r['stepM']:>3}  {r['stepN']:>3}  {r['traverse']:>8}  "
                  f"{r['fixSplit']:>10}  {str(r['setHF32']):>5}  "
                  f"{r['bufferSpace']:>20}  {r['label']}")

        best = ok_results[0]
        print(f"\nBEST: {best['task_duration_us']} us — {best['label']}")
    else:
        print("\nNo successful profiling runs.")

    print(f"\nTotal: {len(results)} | OK: {len(ok_results)} | Failed: {len(fail_results)}")
    print(f"\nFailed breakdown:")
    from collections import Counter
    status_counts = Counter(r['status'] for r in fail_results)
    for status, cnt in status_counts.most_common():
        print(f"  {status}: {cnt}")

    print(f"\nResults written to: {RESULTS_CSV}")


if __name__ == '__main__':
    main()
