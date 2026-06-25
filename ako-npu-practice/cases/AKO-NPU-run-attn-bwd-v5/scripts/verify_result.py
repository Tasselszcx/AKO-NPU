#!/usr/bin/env python3
"""Verify kernel output against golden reference."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from input.attention_backward import MAX_ATOL, MAX_RTOL

def load_bin(path, dtype):
    return np.fromfile(path, dtype=dtype)

def check_accuracy(name, actual, golden, atol, rtol):
    if actual.shape != golden.shape:
        print(f"  {name}: SHAPE MISMATCH actual={actual.shape} golden={golden.shape}")
        return False

    abs_diff = np.abs(actual.astype(np.float32) - golden.astype(np.float32))
    abs_golden = np.abs(golden.astype(np.float32))
    max_abs_diff = float(np.max(abs_diff))

    # Relative error: avoid division by zero
    nonzero_mask = abs_golden > 1e-12
    if nonzero_mask.any():
        rel_diff = abs_diff[nonzero_mask] / abs_golden[nonzero_mask]
        max_rel_diff = float(np.max(rel_diff))
        mean_rel_diff = float(np.mean(rel_diff))
    else:
        max_rel_diff = 0.0
        mean_rel_diff = 0.0

    atol_pass = max_abs_diff <= atol
    rtol_pass = max_rel_diff <= rtol

    status = "PASS" if (atol_pass and rtol_pass) else "FAIL"
    print(f"  {name}: {status}")
    print(f"    max_abs_diff={max_abs_diff:.6e} (threshold={atol:.1e}) {'OK' if atol_pass else 'FAIL'}")
    print(f"    max_rel_diff={max_rel_diff:.6e} (threshold={rtol}) {'OK' if rtol_pass else 'FAIL'}")
    print(f"    mean_rel_diff={mean_rel_diff:.6e}")

    return atol_pass and rtol_pass

def main():
    data_dir = os.environ.get('DATA_DIR', './data')

    # Load golden
    golden_grad_attn_scores = load_bin(f'{data_dir}/golden/grad_attn_scores.bin', np.uint16).view(np.float16)
    golden_grad_value_states = load_bin(f'{data_dir}/golden/grad_value_states.bin', np.uint16).view(np.float16)

    # Load actual output
    output_dir = os.environ.get('OUTPUT_DIR', f'{data_dir}/output')
    actual_grad_attn_scores = load_bin(f'{output_dir}/grad_attn_scores.bin', np.uint16).view(np.float16)
    actual_grad_value_states = load_bin(f'{output_dir}/grad_value_states.bin', np.uint16).view(np.float16)

    print("Accuracy check:")
    pass1 = check_accuracy("grad_attn_scores", actual_grad_attn_scores, golden_grad_attn_scores, MAX_ATOL, MAX_RTOL)
    pass2 = check_accuracy("grad_value_states", actual_grad_value_states, golden_grad_value_states, MAX_ATOL, MAX_RTOL)

    if pass1 and pass2:
        print("\nOverall: PASS")
        return 0
    else:
        print("\nOverall: FAIL")
        return 1

if __name__ == '__main__':
    sys.exit(main())
