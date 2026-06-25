#!/usr/bin/python3
# Generate input data and golden output that matches NPU matmul behavior.
# The NPU accumulates A(fp16)*B(fp16) in fp32 in baseK-sized chunks,
# matching the tiling's K-axis split.
import os
import numpy as np

def gen_golden_data():
    m = 1024
    n = 640
    k = 256
    baseK = 32  # matches the auto-determined baseK from tiling

    input_a = np.random.randint(1, 10, [m, k]).astype(np.float16)
    input_b = np.random.randint(1, 10, [k, n]).astype(np.float16)
    input_bias = np.random.randint(1, 10, [n]).astype(np.float32)
    alpha = 0.001

    # Simulate NPU matmul: accumulate K in baseK chunks, converting to fp32 per chunk
    golden = np.zeros([m, n], dtype=np.float32)
    for k_start in range(0, k, baseK):
        k_end = min(k_start + baseK, k)
        a_chunk = input_a[:, k_start:k_end].astype(np.float32)
        b_chunk = input_b[k_start:k_end, :].astype(np.float32)
        golden += np.matmul(a_chunk, b_chunk)
    golden += input_bias
    golden = np.where(golden >= 0, golden, golden * alpha)

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    input_a.tofile("./input/x1_gm.bin")
    input_b.tofile("./input/x2_gm.bin")
    input_bias.tofile("./input/bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
