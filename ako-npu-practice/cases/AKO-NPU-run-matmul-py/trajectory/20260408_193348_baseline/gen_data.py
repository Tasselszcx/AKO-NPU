# ============================================================================
# Test data generation for matmul_leakyrelu
# C = LeakyReLU(MatMul(A, B) + bias, alpha=0.001)
# A[1024,256] half x B[256,640] half + bias[640] float -> output[1024,640] float
# ============================================================================

import numpy as np
import os

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Matrix dimensions
M, K, N = 1024, 256, 640
ALPHA = 0.001

# Generate random input data
np.random.seed(42)  # Reproducible results
a = np.random.randn(M, K).astype(np.float16)    # A matrix: half
b = np.random.randn(K, N).astype(np.float16)    # B matrix: half
bias = np.random.randn(N).astype(np.float32)    # Bias: float32

# Save inputs as binary files
a.tofile("input/x1_gm.bin")
b.tofile("input/x2_gm.bin")
bias.tofile("input/bias.bin")

# Compute golden reference
# NPU Cube does half*half->float accumulation, which is equivalent to:
# casting each half to float, then accumulating in float.
# But the actual precision depends on the hardware accumulator.
# Use float32 matmul of float16 inputs for the golden reference.
output = np.matmul(a.astype(np.float32), b.astype(np.float32)) + bias
# LeakyReLU: y = x if x >= 0, alpha * x if x < 0
golden = np.where(output >= 0, output, ALPHA * output).astype(np.float32)

golden.tofile("output/golden.bin")

print(f"Generated matmul_leakyrelu test data:")
print(f"  A: input/x1_gm.bin  shape={a.shape}, dtype={a.dtype}")
print(f"  B: input/x2_gm.bin  shape={b.shape}, dtype={b.dtype}")
print(f"  bias: input/bias.bin  shape={bias.shape}, dtype={bias.dtype}")
print(f"  golden: output/golden.bin  shape={golden.shape}, dtype={golden.dtype}")
print(f"  Golden stats: min={golden.min():.4f}, max={golden.max():.4f}, mean={golden.mean():.4f}")
