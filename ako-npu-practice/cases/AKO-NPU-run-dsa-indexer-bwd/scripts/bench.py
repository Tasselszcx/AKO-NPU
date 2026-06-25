#!/usr/bin/env python3
"""
Benchmark script for DSA Indexer Backward kernel.

Usage:
    python3 scripts/bench.py [label]

Compares solution/ against input/ (reference), checks correctness,
and measures kernel execution time via msprof or torch timing.
"""

import sys
import os
import time
import importlib.util
import argparse

import numpy as np

# Suppress torch_npu warnings
import warnings
warnings.filterwarnings("ignore")

import torch
import torch_npu


def load_module(path, name):
    """Load a Python module from path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_correctness(ref_outputs, sol_outputs, max_atol, max_rtol):
    """Check correctness using allclose formula: |a-b| <= atol + rtol * |b|."""
    names = ["grad_q", "grad_k", "grad_weights"]
    all_pass = True
    for name, ref, sol in zip(names, ref_outputs, sol_outputs):
        ref_cpu = ref.cpu().float()
        sol_cpu = sol.cpu().float()

        if ref_cpu.shape != sol_cpu.shape:
            print(f"  {name}: SHAPE MISMATCH ref={ref_cpu.shape} sol={sol_cpu.shape}")
            all_pass = False
            continue

        abs_diff = torch.abs(ref_cpu - sol_cpu)
        max_abs = abs_diff.max().item()

        # allclose check: |a-b| <= atol + rtol * |b|
        tolerance = max_atol + max_rtol * torch.abs(ref_cpu)
        pass_mask = abs_diff <= tolerance
        n_fail = (~pass_mask).sum().item()
        n_total = ref_cpu.numel()

        ok = (n_fail == 0)
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status} | max_diff={max_abs:.2e} | fail={n_fail}/{n_total}")

        if not ok:
            all_pass = False

    return all_pass


def benchmark_shape(ref_mod, sol_mod, shape, device, n_warmup=3, n_runs=10):
    """Benchmark a single shape."""
    # Generate inputs on device
    ref_inputs = ref_mod.get_inputs(shape, device)
    sol_inputs = sol_mod.get_inputs(shape, device)

    # Copy the same inputs for fair comparison
    for key in ref_inputs:
        sol_inputs[key] = ref_inputs[key].clone()

    # Reference run
    ref_outputs = ref_mod.run(**ref_inputs)

    # Solution run + correctness check
    sol_outputs = sol_mod.run(**sol_inputs)
    max_atol = getattr(sol_mod, "MAX_ATOL", 1e-4)
    max_rtol = getattr(sol_mod, "MAX_RTOL", 0.01)
    correct = check_correctness(ref_outputs, sol_outputs, max_atol, max_rtol)

    if not correct:
        return None, False

    # Warmup
    for _ in range(n_warmup):
        _ = sol_mod.run(**sol_inputs)
    torch.npu.synchronize()

    # Timed runs
    times = []
    for _ in range(n_runs):
        sol_inputs_copy = {k: v.clone() for k, v in sol_inputs.items()}
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = sol_mod.run(**sol_inputs_copy)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    # Reference timing
    ref_times = []
    for _ in range(n_warmup):
        _ = ref_mod.run(**ref_inputs)
    torch.npu.synchronize()

    for _ in range(n_runs):
        ref_inputs_copy = {k: v.clone() for k, v in ref_inputs.items()}
        torch.npu.synchronize()
        t0 = time.perf_counter()
        _ = ref_mod.run(**ref_inputs_copy)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        ref_times.append((t1 - t0) * 1000)

    return {
        "sol_times": times,
        "ref_times": ref_times,
        "sol_mean": np.mean(times),
        "sol_min": np.min(times),
        "sol_max": np.max(times),
        "ref_mean": np.mean(ref_times),
        "ref_min": np.min(ref_times),
        "ref_max": np.max(ref_times),
        "speedup_mean": np.mean(ref_times) / np.mean(times),
        "speedup_min": np.min(ref_times) / np.max(times),
        "speedup_max": np.max(ref_times) / np.min(times),
    }, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", default="baseline")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    ref_path = os.path.join(project_root, "input", "dsa_indexer_backward.py")
    sol_path = os.path.join(project_root, "solution", "dsa_indexer_backward.py")

    if not os.path.exists(ref_path):
        print(f"ERROR: Reference not found: {ref_path}")
        sys.exit(1)
    if not os.path.exists(sol_path):
        print(f"ERROR: Solution not found: {sol_path}")
        sys.exit(1)

    ref_mod = load_module(ref_path, "ref_kernel")
    sol_mod = load_module(sol_path, "sol_kernel")

    device = "npu:0"
    shapes = getattr(sol_mod, "TEST_SHAPES", ref_mod.TEST_SHAPES)

    print(f"=== DSA Indexer Backward Benchmark [{args.label}] ===")
    print(f"Device: {device}")
    print()

    all_correct = True
    all_results = []

    for i, shape in enumerate(shapes):
        shape_str = f"B={shape['batch_size']}, S_q={shape['seq_len_q']}, S_kv={shape['seq_len_kv']}"
        print(f"--- Shape {i+1}: {shape_str} ---")

        result, correct = benchmark_shape(ref_mod, sol_mod, shape, device)

        if not correct:
            print(f"  CORRECTNESS FAILED!")
            all_correct = False
            continue

        print(f"  Solution: {result['sol_mean']:.3f} ms (mean), {result['sol_min']:.3f} ~ {result['sol_max']:.3f} ms")
        print(f"  Reference: {result['ref_mean']:.3f} ms (mean), {result['ref_min']:.3f} ~ {result['ref_max']:.3f} ms")
        print(f"  Speedup: {result['speedup_mean']:.3f}x (mean), {result['speedup_min']:.3f} ~ {result['speedup_max']:.3f}x")
        print()
        all_results.append(result)

    # Summary
    print("=== Summary ===")
    if not all_correct:
        print("CORRECTNESS: FAIL")
        sys.exit(1)
    else:
        print("CORRECTNESS: PASS")

    if all_results:
        avg_speedup = np.mean([r["speedup_mean"] for r in all_results])
        print(f"Average Speedup: {avg_speedup:.3f}x")

        # Overall timing
        sol_total = sum(r["sol_mean"] for r in all_results)
        ref_total = sum(r["ref_mean"] for r in all_results)
        print(f"Total Solution Time: {sol_total:.3f} ms")
        print(f"Total Reference Time: {ref_total:.3f} ms")
        print(f"Overall Speedup: {ref_total/sol_total:.3f}x")

    sys.exit(0)


if __name__ == "__main__":
    main()
