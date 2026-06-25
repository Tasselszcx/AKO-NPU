#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify lm_head_projection operator output against golden reference.
Usage: python verify_result.py <output_file> <golden_file>
"""

import sys
import numpy as np


# Precision thresholds for bf16 matmul
MAX_ATOL = 1e-3
MAX_RTOL = 0.05
ERROR_RATIO_TOL = 1e-3  # Allow 0.1% mismatch elements


def read_bf16_bin(path):
    """Read bfloat16 binary file and return float32 numpy array."""
    raw = np.fromfile(path, dtype=np.uint16)
    # Convert uint16 (bf16) → uint32 → float32
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return f32


def verify_result(output_path, golden_path):
    output = read_bf16_bin(output_path)
    golden = read_bf16_bin(golden_path)

    if output.shape != golden.shape:
        print(f"[ERROR] Shape mismatch: output={output.shape}, golden={golden.shape}")
        return False

    print(f"Total elements: {output.size}")

    # Element-wise comparison using numpy isclose (combines atol and rtol)
    close_mask = np.isclose(output, golden, rtol=MAX_RTOL, atol=MAX_ATOL, equal_nan=True)
    mismatch_indices = np.where(~close_mask)[0]

    # Compute max absolute and relative errors
    abs_diff = np.abs(output - golden)
    max_atol = np.max(abs_diff)

    # Relative error (avoid division by zero)
    nonzero_mask = np.abs(golden) > 1e-6
    if nonzero_mask.any():
        max_rtol = np.max(np.abs((output[nonzero_mask] - golden[nonzero_mask]) / golden[nonzero_mask]))
    else:
        max_rtol = 0.0

    error_ratio = float(mismatch_indices.size) / golden.size

    print(f"max_atol: {max_atol:.6e} (threshold: {MAX_ATOL})")
    print(f"max_rtol: {max_rtol:.6e} (threshold: {MAX_RTOL})")
    print(f"error_ratio: {error_ratio:.6f} (threshold: {ERROR_RATIO_TOL})")

    # Show first few mismatches
    if mismatch_indices.size > 0:
        print(f"\nFirst mismatches (up to 20):")
        for i, idx in enumerate(mismatch_indices[:20]):
            print(f"  [{idx}] golden={golden[idx]:.6f}, output={output[idx]:.6f}, "
                  f"adiff={abs_diff[idx]:.6e}")

    passed = error_ratio <= ERROR_RATIO_TOL
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    try:
        res = verify_result(sys.argv[1], sys.argv[2])
        if not res:
            raise ValueError("[ERROR] Precision verification failed")
        else:
            print("test pass!")
    except Exception as e:
        print(e)
        sys.exit(1)
