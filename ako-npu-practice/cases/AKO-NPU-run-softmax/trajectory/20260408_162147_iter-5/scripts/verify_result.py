import numpy as np
import sys

dtype = np.float32
rtol = 1e-3
atol = 1e-5

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {np.max(np.abs(output - golden)):.6e}")
        return True
    else:
        diff = np.abs(output - golden)
        print(f"Verification FAILED!")
        print(f"Max diff: {np.max(diff):.6e}, Mean diff: {np.mean(diff):.6e}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
