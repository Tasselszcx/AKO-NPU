import numpy as np
import os

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

N = 64
S_q = 4096
S_kv = 4096

# Generate inputs
grad_is = np.random.randn(S_q, S_kv).astype(np.float32)
score = np.abs(np.random.randn(N, S_q, S_kv).astype(np.float32))
relu_mask = (np.random.rand(N, S_q, S_kv) > 0.5).astype(np.uint8)
weights = np.random.randn(N, S_q).astype(np.float32) * 0.01

grad_is.tofile("input/grad_is.bin")
score.tofile("input/score.bin")
relu_mask.tofile("input/relu_mask.bin")
weights.tofile("input/weights.bin")

# Compute golden
grad_score = np.where(relu_mask, grad_is[None, :, :] * weights[:, :, None], 0.0).astype(np.float32)
grad_weights = (grad_is[None, :, :] * score).sum(axis=-1).astype(np.float32)

grad_score.tofile("output/golden_grad_score.bin")
grad_weights.tofile("output/golden_grad_weights.bin")

print(f"Generated: N={N}, S_q={S_q}, S_kv={S_kv}")
print(f"  grad_score: {grad_score.shape}, grad_weights: {grad_weights.shape}")
