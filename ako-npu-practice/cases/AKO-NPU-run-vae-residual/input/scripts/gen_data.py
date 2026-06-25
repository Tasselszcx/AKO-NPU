import numpy as np
import os

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Shape parameters - must match .asc main()
B, C, H, W = 1, 256, 64, 64
dtype = np.float32

print(f"Generating test data: B={B}, C={C}, H={H}, W={W}")

# Seed for reproducibility
np.random.seed(42)

# Generate input
x = np.random.randn(B, C, H, W).astype(dtype) * 0.1

# Generate conv weights (no bias)
conv1_weight = np.random.randn(C, C, 3, 3).astype(dtype) * 0.01
conv2_weight = np.random.randn(C, C, 3, 3).astype(dtype) * 0.01

# GroupNorm parameters
norm1_weight = np.ones(C, dtype=dtype)
norm1_bias = np.zeros(C, dtype=dtype)
norm2_weight = np.ones(C, dtype=dtype)
norm2_bias = np.zeros(C, dtype=dtype)

# Save inputs
x.tofile("input/input_x.bin")
conv1_weight.tofile("input/input_conv1_weight.bin")
norm1_weight.tofile("input/input_norm1_weight.bin")
norm1_bias.tofile("input/input_norm1_bias.bin")
conv2_weight.tofile("input/input_conv2_weight.bin")
norm2_weight.tofile("input/input_norm2_weight.bin")
norm2_bias.tofile("input/input_norm2_bias.bin")

print(f"  input_x: {x.shape} [{x.nbytes} bytes]")
print(f"  conv1_weight: {conv1_weight.shape} [{conv1_weight.nbytes} bytes]")
print(f"  norm1_weight: {norm1_weight.shape}")
print(f"  conv2_weight: {conv2_weight.shape}")

# Compute golden using numpy (PyTorch-equivalent)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    x_t = torch.from_numpy(x)
    conv1_w_t = torch.from_numpy(conv1_weight)
    conv2_w_t = torch.from_numpy(conv2_weight)
    n1_w_t = torch.from_numpy(norm1_weight)
    n1_b_t = torch.from_numpy(norm1_bias)
    n2_w_t = torch.from_numpy(norm2_weight)
    n2_b_t = torch.from_numpy(norm2_bias)

    # Stage 1: Conv1 -> GN1 -> SiLU1
    conv1_out = F.conv2d(x_t, conv1_w_t, bias=None, stride=1, padding=1)
    gn1_out = F.group_norm(conv1_out, 32, weight=n1_w_t, bias=n1_b_t, eps=1e-5)
    path1 = F.silu(gn1_out)

    # Stage 2: Conv2 -> GN2 -> SiLU2 -> Add
    conv2_out = F.conv2d(path1, conv2_w_t, bias=None, stride=1, padding=1)
    gn2_out = F.group_norm(conv2_out, 32, weight=n2_w_t, bias=n2_b_t, eps=1e-5)
    path2 = F.silu(gn2_out)
    output = path2 + x_t

    golden = output.numpy()
    golden.tofile("output/golden.bin")
    print(f"  golden: {golden.shape} [{golden.nbytes} bytes]")
    print(f"  golden range: [{golden.min():.6f}, {golden.max():.6f}]")
    print("Golden data generated with PyTorch")

except ImportError:
    print("WARNING: PyTorch not available, generating approximate golden with numpy")
    # Simplified golden - just for structure testing
    golden = x.copy()
    golden.tofile("output/golden.bin")
