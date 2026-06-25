#!/usr/bin/env python3
"""Verify attention_backward kernel output against golden data."""

import sys
import os
import numpy as np

def from_bf16_file(filepath, shape):
    """Read bf16 binary file and convert to float32."""
    data = open(filepath, "rb").read()
    n_elements = 1
    for s in shape:
        n_elements *= s
    expected_size = n_elements * 2
    if len(data) != expected_size:
        print(f"ERROR: File {filepath} size {len(data)} != expected {expected_size}")
        return None
    f32_bytes = bytearray()
    for i in range(0, len(data), 2):
        f32_bytes.extend(b'\x00\x00')
        f32_bytes.extend(data[i:i+2])
    return np.frombuffer(bytes(f32_bytes), dtype=np.float32).reshape(shape)

def verify_tensor(name, output_path, golden_path, shape, max_atol=1e-5, max_rtol=0.05):
    """Compare output vs golden, return True if passes."""
    output = from_bf16_file(output_path, shape)
    golden = from_bf16_file(golden_path, shape)

    if output is None or golden is None:
        print(f"FAIL: {name} - file read error")
        return False

    abs_diff = np.abs(output - golden)
    atol = np.max(abs_diff)

    # rtol: |output - golden| / |golden|, excluding zeros
    nonzero_mask = np.abs(golden) > 1e-10
    if nonzero_mask.any():
        rel_diff = abs_diff[nonzero_mask] / np.abs(golden[nonzero_mask])
        rtol = np.max(rel_diff)
    else:
        rtol = 0.0

    mean_abs_diff = np.mean(abs_diff)

    passed = (atol <= max_atol) and (rtol <= max_rtol)
    status = "PASS" if passed else "FAIL"

    print(f"{status}: {name}")
    print(f"  max_atol={atol:.6e} (limit={max_atol})")
    print(f"  max_rtol={rtol:.6e} (limit={max_rtol})")
    print(f"  mean_abs_diff={mean_abs_diff:.6e}")
    print(f"  output range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"  golden range: [{golden.min():.6f}, {golden.max():.6f}]")

    if not passed:
        # Find worst positions
        worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"  worst_pos: {worst_idx}")
        print(f"  output[worst]: {output[worst_idx]:.8f}")
        print(f"  golden[worst]: {golden[worst_idx]:.8f}")

    return passed

def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seq_q = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    seq_kv = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    NUM_HEADS = 80
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    MAX_ATOL = 1e-5
    MAX_RTOL = 0.05

    print(f"Verifying: batch={batch}, seq_q={seq_q}, seq_kv={seq_kv}")
    print(f"Precision requirements: atol<={MAX_ATOL}, rtol<={MAX_RTOL}")
    print()

    all_pass = True

    # Check grad_attn_scores
    shape1 = (batch, NUM_HEADS, seq_q, seq_kv)
    p1 = verify_tensor("grad_attn_scores",
        "output/grad_attn_scores.bin",
        "output/golden_grad_attn_scores.bin",
        shape1, MAX_ATOL, MAX_RTOL)
    all_pass = all_pass and p1

    print()

    # Check grad_value_states
    shape2 = (batch, NUM_KV_HEADS, seq_kv, HEAD_DIM)
    p2 = verify_tensor("grad_value_states",
        "output/grad_value_states.bin",
        "output/golden_grad_value_states.bin",
        shape2, MAX_ATOL, MAX_RTOL)
    all_pass = all_pass and p2

    print()
    if all_pass:
        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
        sys.exit(0)
    else:
        print("=" * 50)
        print("SOME TESTS FAILED")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    main()
