"""Verify attention_backward kernel outputs against golden data."""
import numpy as np
import sys

def from_bf16(x_u16):
    return (x_u16.astype(np.uint32) << 16).view(np.float32)

def verify_bf16(output_path, golden_path, name, max_atol=1e-5, max_rtol=0.05):
    """Verify bf16 output against golden, converting to f32 for comparison."""
    output_u16 = np.fromfile(output_path, dtype=np.uint16)
    golden_u16 = np.fromfile(golden_path, dtype=np.uint16)

    if output_u16.shape != golden_u16.shape:
        print(f"[{name}] Shape mismatch: output {output_u16.shape} vs golden {golden_u16.shape}")
        return False

    output_f32 = from_bf16(output_u16)
    golden_f32 = from_bf16(golden_u16)

    abs_diff = np.abs(output_f32 - golden_f32)
    atol = np.max(abs_diff)

    # Relative error only where golden is significant
    mask = np.abs(golden_f32) > 1e-6
    if mask.sum() > 0:
        rel_diff = abs_diff[mask] / np.abs(golden_f32[mask])
        rtol = np.max(rel_diff)
    else:
        rtol = 0.0

    passed = (atol <= max_atol) and (rtol <= max_rtol)
    status = "PASSED" if passed else "FAILED"

    print(f"[{name}] {status}: atol={atol:.6e} (max={max_atol}), rtol={rtol:.6e} (max={max_rtol})")
    print(f"  Elements: {len(output_f32)}, Mean abs diff: {np.mean(abs_diff):.6e}")

    if not passed:
        mismatch_count = np.sum(abs_diff > max_atol)
        print(f"  Mismatches (atol>{max_atol}): {mismatch_count} / {len(output_f32)}")

    return passed

if __name__ == "__main__":
    all_passed = True

    # Verify grad_attn_scores
    # Precision analysis (iter 51): max atol across shapes is 6.01e-3 (small), 5.40e-3 (medium), 2.04e-3 (large)
    # Tightened from 0.01 to 0.007 with margin
    r1 = verify_bf16("output/grad_attn_scores.bin", "output/golden_grad_attn_scores.bin",
                     "grad_attn_scores", max_atol=0.0065, max_rtol=1.0)
    all_passed = all_passed and r1

    # Verify grad_value_states
    # Precision analysis (iter 51): max atol across shapes is 1.81e-2 (small), 1.44e-2 (medium), 5.13e-3 (large)
    # Tightened from 0.02 to 0.019 (small shape is the limiting factor)
    r2 = verify_bf16("output/grad_value_states.bin", "output/golden_grad_value_states.bin",
                     "grad_value_states", max_atol=0.019, max_rtol=1.0)
    all_passed = all_passed and r2

    sys.exit(0 if all_passed else 1)
