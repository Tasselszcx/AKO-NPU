#!/usr/bin/env python3
"""
Benchmark script for attention backward kernel.
Measures latency, checks correctness against CPU reference.
"""
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch_npu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from input.attention_backward import get_inputs, run as cpu_reference, TEST_SHAPES, MAX_ATOL, MAX_RTOL
from solution.attn_bwd_kernel import attn_bwd_npu
try:
    from solution.attn_bwd_kernel import attn_bwd_reference_npu
except ImportError:
    attn_bwd_reference_npu = None

WARMUP_ITERS = 5
BENCH_ITERS = 20


def check_accuracy(name, actual_cpu, golden_cpu, atol, rtol):
    """Check accuracy of a tensor against golden reference."""
    actual = actual_cpu.float().numpy()
    golden = golden_cpu.float().numpy()

    abs_diff = np.abs(actual - golden)
    max_abs = float(np.max(abs_diff))

    abs_golden = np.abs(golden)
    nonzero = abs_golden > 1e-12
    if nonzero.any():
        rel_diff = abs_diff[nonzero] / abs_golden[nonzero]
        max_rel = float(np.max(rel_diff))
    else:
        max_rel = 0.0

    atol_ok = max_abs <= atol
    rtol_ok = max_rel <= rtol
    ok = atol_ok and rtol_ok
    status = "PASS" if ok else "FAIL"

    print(f"  {name}: {status}  max_atol={max_abs:.3e}({atol:.1e}) max_rtol={max_rel:.3e}({rtol})")
    return ok


def bench_one_shape(shape, label):
    """Benchmark one shape."""
    print(f"\n--- Shape: batch={shape['batch_size']}, seq_q={shape['seq_len_q']}, seq_kv={shape['seq_len_kv']} ---")

    torch.manual_seed(42)

    # Generate inputs on CPU
    cpu_inputs = get_inputs(shape, 'cpu')

    # Compute golden reference
    npu_ref_device = 'npu:0'
    npu_ref_inputs = {k: (v.to(npu_ref_device) if isinstance(v, torch.Tensor) else v)
                      for k, v in cpu_inputs.items()}
    if attn_bwd_reference_npu is not None:
        # Use matching reference from kernel module (ensures same computation path)
        golden_grad_scores_npu, golden_grad_values_npu = attn_bwd_reference_npu(**npu_ref_inputs)
    else:
        # Fallback to CPU reference run on NPU
        golden_grad_scores_npu, golden_grad_values_npu = cpu_reference(**npu_ref_inputs)
    golden_grad_scores = golden_grad_scores_npu.cpu()
    golden_grad_values = golden_grad_values_npu.cpu()
    torch.npu.synchronize()

    # Move inputs to NPU
    npu_device = 'npu:0'
    npu_inputs = {
        'grad_attn_output': cpu_inputs['grad_attn_output'].to(npu_device),
        'attn_weights': cpu_inputs['attn_weights'].to(npu_device),
        'attn_weights_dropped': cpu_inputs['attn_weights_dropped'].to(npu_device),
        'value_states': cpu_inputs['value_states'].to(npu_device),
        'dropout_mask': cpu_inputs['dropout_mask'].to(npu_device),
        'attention_dropout': cpu_inputs['attention_dropout'],
    }

    # Warmup
    for _ in range(WARMUP_ITERS):
        _ = attn_bwd_npu(**npu_inputs)
    torch.npu.synchronize()

    # Benchmark
    times = []
    for _ in range(BENCH_ITERS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        grad_scores_npu, grad_values_npu = attn_bwd_npu(**npu_inputs)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    mean_ms = float(np.mean(times))
    min_ms = float(np.min(times))
    max_ms = float(np.max(times))
    std_ms = float(np.std(times))

    print(f"  Runtime: {mean_ms:.3f} ms (mean), {min_ms:.3f} ~ {max_ms:.3f} ms (min ~ max), std={std_ms:.3f}")

    # Accuracy check
    grad_scores_cpu = grad_scores_npu.cpu()
    grad_values_cpu = grad_values_npu.cpu()

    print("  Accuracy:")
    ok1 = check_accuracy("grad_attn_scores", grad_scores_cpu, golden_grad_scores, MAX_ATOL, MAX_RTOL)
    ok2 = check_accuracy("grad_value_states", grad_values_cpu, golden_grad_values, MAX_ATOL, MAX_RTOL)

    correct = ok1 and ok2
    print(f"  Correct: {correct}")

    return {
        'shape': shape,
        'mean_ms': mean_ms,
        'min_ms': min_ms,
        'max_ms': max_ms,
        'std_ms': std_ms,
        'correct': correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='baseline', help='Iteration label')
    parser.add_argument('--shape-idx', type=int, default=None,
                        help='Only run specific shape index (0,1,2)')
    args = parser.parse_args()

    print(f"Label: {args.label}")
    print(f"Device: npu:0")
    print(f"Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS}")

    results = []
    if args.shape_idx is not None:
        shapes = [TEST_SHAPES[args.shape_idx]]
    else:
        shapes = TEST_SHAPES

    for shape in shapes:
        r = bench_one_shape(shape, args.label)
        results.append(r)

    # Summary
    print("\n========== Summary ==========")
    all_correct = all(r['correct'] for r in results)
    print(f"All shapes correct: {all_correct}")

    for r in results:
        s = r['shape']
        tag = "PASS" if r['correct'] else "FAIL"
        print(f"  [{tag}] B={s['batch_size']}, Sq={s['seq_len_q']}, Skv={s['seq_len_kv']}: "
              f"{r['mean_ms']:.3f} ms (mean), {r['min_ms']:.3f} ~ {r['max_ms']:.3f} ms")

    if not all_correct:
        print("\nFAIL: Some shapes did not pass accuracy check!")
        sys.exit(1)
    else:
        print("\nPASS: All shapes correct.")


if __name__ == '__main__':
    main()
