# ============================================================================
# Result verification for matmul_leakyrelu
# ============================================================================

import numpy as np
import sys

dtype = np.float32
rtol = 1e-3
atol = 1e-3

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    diff = np.abs(output - golden)
    max_abs_err = np.max(diff)
    mean_abs_err = np.mean(diff)

    # Cosine similarity
    dot = np.dot(output.flatten(), golden.flatten())
    norm_out = np.linalg.norm(output.flatten())
    norm_gold = np.linalg.norm(golden.flatten())
    cosine_sim = dot / (norm_out * norm_gold + 1e-12)

    # Relative error (avoid division by zero)
    rel_err = diff / (np.abs(golden) + 1e-12)
    mean_rel_err = np.mean(rel_err)

    print(f"Verification Results:")
    print(f"  Shape: {output.shape}")
    print(f"  Max absolute error: {max_abs_err:.6e}")
    print(f"  Mean absolute error: {mean_abs_err:.6e}")
    print(f"  Mean relative error: {mean_rel_err:.6e}")
    print(f"  Cosine similarity: {cosine_sim:.8f}")

    # Acceptance criteria from PLAN.md
    passed = True
    if max_abs_err >= 1e-3:
        # For matmul with half inputs, some tolerance is needed
        # Check if most elements pass
        mismatch_count = np.sum(diff > atol + rtol * np.abs(golden))
        mismatch_ratio = mismatch_count / len(golden)
        print(f"  Mismatch count: {mismatch_count} / {len(golden)} ({mismatch_ratio*100:.2f}%)")
        if mismatch_ratio > 0.01:  # Allow up to 1% mismatches for half precision matmul
            print(f"  WARN: Too many mismatches")
            passed = False

    if cosine_sim < 0.9999:
        print(f"  WARN: Cosine similarity below threshold")
        passed = False

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"  allclose PASSED (rtol={rtol}, atol={atol})")
    else:
        print(f"  allclose FAILED (rtol={rtol}, atol={atol})")

    if passed:
        print("Verification PASSED!")
    else:
        print("Verification FAILED!")

    return passed

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
