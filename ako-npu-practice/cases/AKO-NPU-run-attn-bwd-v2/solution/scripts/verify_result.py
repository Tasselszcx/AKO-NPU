#!/usr/bin/env python3
"""Verify attention_backward kernel output against golden reference."""
import sys
import os
import numpy as np

MAX_ATOL = 1e-5
MAX_RTOL = 0.05

def verify_tensor(name, output_path, golden_path, dtype_str="bf16"):
    """Compare output tensor with golden reference.

    Uses PyTorch-style allclose: |a-b| <= atol + rtol * |b|
    Also reports cosine similarity and relative L2 error.
    """
    if not os.path.exists(output_path):
        print(f"FAIL: {name} - output file not found: {output_path}")
        return False
    if not os.path.exists(golden_path):
        print(f"FAIL: {name} - golden file not found: {golden_path}")
        return False

    # Load as bf16 (stored as uint16, need to convert to float for comparison)
    if dtype_str == "bf16":
        output_u16 = np.fromfile(output_path, dtype=np.uint16)
        golden_u16 = np.fromfile(golden_path, dtype=np.uint16)

        if output_u16.shape != golden_u16.shape:
            print(f"FAIL: {name} - shape mismatch: output={output_u16.shape}, golden={golden_u16.shape}")
            return False

        # Convert bf16 (uint16) to float32 for comparison
        output_f32 = np.zeros(output_u16.shape, dtype=np.float32)
        golden_f32 = np.zeros(golden_u16.shape, dtype=np.float32)
        output_f32.view(np.uint32)[:] = output_u16.astype(np.uint32) << 16
        golden_f32.view(np.uint32)[:] = golden_u16.astype(np.uint32) << 16
    else:
        output_f32 = np.fromfile(output_path, dtype=np.float32)
        golden_f32 = np.fromfile(golden_path, dtype=np.float32)

    # --- Metrics ---
    abs_diff = np.abs(output_f32 - golden_f32)
    max_atol = float(np.max(abs_diff))

    # PyTorch-style allclose: |a-b| <= atol + rtol * |b|
    tolerance = MAX_ATOL + MAX_RTOL * np.abs(golden_f32)
    allclose_mask = abs_diff <= tolerance
    allclose_pct = np.sum(allclose_mask) / abs_diff.size * 100

    # Cosine similarity
    out_flat = output_f32.flatten()
    ref_flat = golden_f32.flatten()
    cos_sim = float(np.dot(out_flat, ref_flat) / (np.linalg.norm(out_flat) * np.linalg.norm(ref_flat) + 1e-30))

    # Relative L2 error
    rel_l2 = float(np.linalg.norm(out_flat - ref_flat) / (np.linalg.norm(ref_flat) + 1e-30))

    # max rtol excluding near-zero values (|golden| > 1e-6)
    nz_mask = np.abs(golden_f32) > 1e-6
    if nz_mask.any():
        rtol_nz = np.max(abs_diff[nz_mask] / np.abs(golden_f32[nz_mask]))
    else:
        rtol_nz = 0.0

    # Pass criteria: allclose >= 99.99% AND cosine_sim >= 0.9999 AND rel_l2 < 0.01
    # Also pass if strict: max_atol <= threshold OR max_rtol_nonzero <= threshold
    strict_pass = (max_atol <= MAX_ATOL) or (rtol_nz <= MAX_RTOL)
    relaxed_pass = (allclose_pct >= 99.99) and (cos_sim >= 0.99999) and (rel_l2 < 0.001)
    passed = strict_pass or relaxed_pass

    status = "PASS" if passed else "FAIL"
    print(f"  {name}: {status}")
    print(f"    max_atol     = {max_atol:.6e} (threshold={MAX_ATOL:.0e})")
    print(f"    max_rtol(nz) = {rtol_nz:.6e} (threshold={MAX_RTOL}, |ref|>1e-6)")
    print(f"    allclose     = {allclose_pct:.4f}% (atol+rtol*|b| criterion)")
    print(f"    cosine_sim   = {cos_sim:.10f}")
    print(f"    rel_L2_err   = {rel_l2:.6e}")
    print(f"    elements     = {output_f32.size}")

    return passed


def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seq_q = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    seq_kv = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    print(f"Verifying: batch={batch_size}, seq_q={seq_q}, seq_kv={seq_kv}")
    print(f"Thresholds: max_atol={MAX_ATOL}, max_rtol={MAX_RTOL}")
    print()

    all_passed = True

    # Verify grad_attn_scores
    passed1 = verify_tensor(
        "grad_attn_scores",
        "output/grad_attn_scores.bin",
        "output/golden_grad_attn_scores.bin",
        "bf16"
    )
    all_passed = all_passed and passed1

    # Verify grad_value_states
    passed2 = verify_tensor(
        "grad_value_states",
        "output/grad_value_states.bin",
        "output/golden_grad_value_states.bin",
        "bf16"
    )
    all_passed = all_passed and passed2

    print()
    if all_passed:
        print("OVERALL: PASS")
        sys.exit(0)
    else:
        print("OVERALL: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
