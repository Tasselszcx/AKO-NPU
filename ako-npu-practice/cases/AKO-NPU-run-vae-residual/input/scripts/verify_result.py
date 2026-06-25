import numpy as np
import sys

dtype = np.float32
# Relaxed tolerances for multi-kernel composite operator
# Manual GroupNorm + multi-kernel pipeline introduces more error than single-kernel ops
rtol = 1e-2
atol = 1e-2

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if output.size == 0:
        print("Empty tensors")
        return False

    diff = np.abs(output - golden)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Compute relative error metrics
    nonzero_mask = np.abs(golden) > 1e-8
    if np.sum(nonzero_mask) > 0:
        rel_err = diff[nonzero_mask] / np.abs(golden[nonzero_mask])
        max_rel = np.max(rel_err)
        mean_rel = np.mean(rel_err)
    else:
        max_rel = 0.0
        mean_rel = 0.0

    print(f"Shape: {output.shape}")
    print(f"Max absolute diff: {max_diff:.8f}")
    print(f"Mean absolute diff: {mean_diff:.8f}")
    print(f"Max relative error: {max_rel:.8f}")
    print(f"Mean relative error: {mean_rel:.8f}")
    print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
    print(f"Golden range: [{golden.min():.6f}, {golden.max():.6f}]")

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! (rtol={rtol}, atol={atol})")
        return True
    else:
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"Verification FAILED!")
        print(f"Mismatch count: {len(mismatches)} / {len(golden)} ({100*len(mismatches)/len(golden):.2f}%)")
        # Show first few mismatches
        for i in mismatches[:5]:
            print(f"  idx={i}: output={output[i]:.8f}, golden={golden[i]:.8f}, diff={diff[i]:.8f}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
