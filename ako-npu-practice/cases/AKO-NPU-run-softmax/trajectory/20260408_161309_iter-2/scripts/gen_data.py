import numpy as np
import os

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

rows = 1024
cols = 4096
dtype = np.float32

x = np.random.randn(rows, cols).astype(dtype)
x.tofile("input/input_x.bin")

# Golden: softmax per row
x_max = x.max(axis=1, keepdims=True)
x_exp = np.exp(x - x_max)
golden = x_exp / x_exp.sum(axis=1, keepdims=True)
golden.tofile("output/golden.bin")

print(f"Generated test data: [{rows}, {cols}], dtype={dtype}")
print(f"  input/input_x.bin: {x.shape}")
print(f"  output/golden.bin: {golden.shape}")
