# ============================================================================
# Test data generation for matmul_leakyrelu
# output = LeakyReLU(A @ B + bias, alpha=0.001)
# A: [1024, 256] float16, B: [256, 640] float16
# bias: [640] float32, output: [1024, 640] float32
# ============================================================================

import numpy as np
import os

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Generate input data
np.random.seed(42)
A = np.random.randn(1024, 256).astype(np.float16)
B = np.random.randn(256, 640).astype(np.float16)
bias = np.random.randn(640).astype(np.float32)

A.tofile("input/x1_gm.bin")
B.tofile("input/x2_gm.bin")
bias.tofile("input/bias.bin")

# Compute golden result
# Use float32 for matmul to match NPU behavior (half * half -> float)
temp = A.astype(np.float32) @ B.astype(np.float32) + bias
alpha = np.float32(0.001)
golden = np.where(temp > 0, temp, alpha * temp).astype(np.float32)
golden.tofile("output/golden.bin")

print(f"Generated test data:")
print(f"  input/x1_gm.bin:  A  [{A.shape}] {A.dtype}")
print(f"  input/x2_gm.bin:  B  [{B.shape}] {B.dtype}")
print(f"  input/bias.bin:    bias [{bias.shape}] {bias.dtype}")
print(f"  output/golden.bin: golden [{golden.shape}] {golden.dtype}")
print(f"  Golden stats: min={golden.min():.4f}, max={golden.max():.4f}, mean={golden.mean():.4f}")
