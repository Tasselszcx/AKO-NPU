#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate test data for lm_head_projection operator.
Usage: python gen_data.py [B] [S] [logits_to_keep]
Default: B=1, S=128, logits_to_keep=1 (W1: decode single token)
"""

import os
import sys
import struct
import numpy as np

# Constants
HIDDEN_SIZE = 2048
VOCAB_SIZE = 102400


def float32_to_bf16_bytes(arr_f32):
    """Convert float32 numpy array to bfloat16 binary (truncate lower 16 bits)."""
    # View as uint32, right-shift by 16, cast to uint16
    arr_u32 = arr_f32.view(np.uint32)
    arr_u16 = (arr_u32 >> 16).astype(np.uint16)
    return arr_u16.tobytes()


def bf16_bytes_to_float32(buf, count):
    """Convert bfloat16 binary back to float32 numpy array."""
    arr_u16 = np.frombuffer(buf, dtype=np.uint16, count=count)
    arr_u32 = arr_u16.astype(np.uint32) << 16
    arr_f32 = arr_u32.view(np.float32)
    return arr_f32


def gen_data(B, S, logits_to_keep):
    K = HIDDEN_SIZE
    N = VOCAB_SIZE

    print(f"Generating data: B={B}, S={S}, logits_to_keep={logits_to_keep}")
    print(f"  hidden_states: [{B}, {S}, {K}] bf16")
    print(f"  weight:        [{N}, {K}] bf16")
    print(f"  output:        [{B}, {logits_to_keep}, {N}] bf16")

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # Generate random data in float32, then convert to bf16
    np.random.seed(42)
    hidden_states_f32 = np.random.uniform(-0.5, 0.5, (B, S, K)).astype(np.float32)
    weight_f32 = np.random.uniform(-0.05, 0.05, (N, K)).astype(np.float32)

    # Truncate to bf16 precision (for exact match with NPU bf16 input)
    hs_bf16_buf = float32_to_bf16_bytes(hidden_states_f32)
    wt_bf16_buf = float32_to_bf16_bytes(weight_f32)

    # Write input files
    with open("input/hidden_states.bin", "wb") as f:
        f.write(hs_bf16_buf)
    with open("input/weight.bin", "wb") as f:
        f.write(wt_bf16_buf)

    # Compute golden in float32 (from bf16-precision inputs)
    hs_bf16_f32 = bf16_bytes_to_float32(hs_bf16_buf, B * S * K).reshape(B, S, K)
    wt_bf16_f32 = bf16_bytes_to_float32(wt_bf16_buf, N * K).reshape(N, K)

    # Slice: hidden_states[:, -logits_to_keep:, :]
    sliced = hs_bf16_f32[:, -logits_to_keep:, :]  # [B, logits_to_keep, K]

    # MatMul: sliced @ weight.T → [B, logits_to_keep, N]
    golden_f32 = np.matmul(sliced, wt_bf16_f32.T)  # float32 accumulation

    # Convert golden to bf16 (to match NPU output)
    golden_bf16_buf = float32_to_bf16_bytes(golden_f32.astype(np.float32))

    with open("output/golden.bin", "wb") as f:
        f.write(golden_bf16_buf)

    total_hs_bytes = B * S * K * 2
    total_wt_bytes = N * K * 2
    total_out_bytes = B * logits_to_keep * N * 2
    print(f"  hidden_states.bin: {total_hs_bytes} bytes ({total_hs_bytes / 1024 / 1024:.1f} MB)")
    print(f"  weight.bin:        {total_wt_bytes} bytes ({total_wt_bytes / 1024 / 1024:.1f} MB)")
    print(f"  golden.bin:        {total_out_bytes} bytes ({total_out_bytes / 1024 / 1024:.1f} MB)")
    print("Data generation complete.")


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    logits_to_keep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    gen_data(B, S, logits_to_keep)
