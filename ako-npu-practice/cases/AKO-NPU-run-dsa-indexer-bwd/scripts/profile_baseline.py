#!/usr/bin/env python3
"""Profile baseline to understand time breakdown."""
import warnings
warnings.filterwarnings("ignore")

import torch
import torch_npu
import time

device = "npu:0"
B, S_q, S_kv = 1, 4096, 4096
n_heads, head_dim = 64, 128

# Generate inputs
grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.float32, device=device)
k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.float32, device=device) * 0.01
relu_mask = torch.rand(B, S_q, n_heads, S_kv, device=device) > 0.5

# Warmup
for _ in range(3):
    grad_weighted = grad_index_score.unsqueeze(2).expand(-1, -1, n_heads, -1)
    score = torch.einsum("bsnd,btd->bsnt", q, k)
    score = torch.relu(score)
    grad_score = grad_weighted * weights
    grad_weights = (grad_weighted * score).sum(dim=-1, keepdim=True)
    grad_score = grad_score * relu_mask.float()
    grad_q = torch.einsum("bsnt,btd->bsnd", grad_score, k)
    grad_k = torch.einsum("bsnd,bsnt->btd", q, grad_score)
    torch.npu.synchronize()

# Profile each step
def time_op(name, fn, n=5):
    torch.npu.synchronize()
    times = []
    for _ in range(n):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        result = fn()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_t = sum(times) / len(times)
    print(f"  {name}: {mean_t:.3f} ms (mean)")
    return result, mean_t

print("=== Step-by-step profiling (B=1, S_q=4096, S_kv=4096) ===\n")

# Step 4 bwd: unsqueeze + expand
_, t1 = time_op("expand grad_weighted",
    lambda: grad_index_score.unsqueeze(2).expand(-1, -1, n_heads, -1).contiguous())

grad_weighted = grad_index_score.unsqueeze(2).expand(-1, -1, n_heads, -1)

# Step: recompute score = q @ k (einsum)
_, t2 = time_op("einsum q@k (score)",
    lambda: torch.einsum("bsnd,btd->bsnt", q, k))

score = torch.einsum("bsnd,btd->bsnt", q, k)

# Step: relu
_, t3 = time_op("relu(score)", lambda: torch.relu(score))
score = torch.relu(score)

# Step 3 bwd: grad_score = grad_weighted * weights (broadcast)
_, t4 = time_op("grad_score = grad_weighted * weights",
    lambda: grad_weighted * weights)
grad_score_pre = grad_weighted * weights

# Step: grad_weights = (grad_weighted * score).sum(-1, keepdim=True)
_, t5 = time_op("grad_weights = (grad_weighted*score).sum(-1)",
    lambda: (grad_weighted * score).sum(dim=-1, keepdim=True))

# Step 2 bwd: relu mask
_, t6 = time_op("grad_score * relu_mask",
    lambda: grad_score_pre * relu_mask.float())
grad_score = grad_score_pre * relu_mask.float()

# Step 1 bwd: einsum grad_q
_, t7 = time_op("einsum grad_score@k -> grad_q",
    lambda: torch.einsum("bsnt,btd->bsnd", grad_score, k))

# Step 1 bwd: einsum grad_k
_, t8 = time_op("einsum q@grad_score -> grad_k",
    lambda: torch.einsum("bsnd,bsnt->btd", q, grad_score))

total = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
print(f"\n  TOTAL: {total:.3f} ms")
print(f"\n  Matmul ops: {t2 + t7 + t8:.3f} ms ({(t2+t7+t8)/total*100:.1f}%)")
print(f"  Element-wise ops: {t1 + t3 + t4 + t5 + t6:.3f} ms ({(t1+t3+t4+t5+t6)/total*100:.1f}%)")

# Also try bmm approach
print("\n=== Alternative: bmm approach ===\n")
# Reshape q: [B, S_q, n_heads, head_dim] -> [B*S_q, n_heads, head_dim]
q_reshaped = q.reshape(B * S_q, n_heads, head_dim)
k_expanded = k.squeeze(0).unsqueeze(0).expand(B * S_q, -1, -1)  # [B*S_q, S_kv, head_dim]

_, t_bmm1 = time_op("bmm q@k^T -> score",
    lambda: torch.bmm(q_reshaped, k_expanded.transpose(-1, -2)))

# grad_q = grad_score @ k
gs_reshaped = grad_score.reshape(B * S_q, n_heads, S_kv)
_, t_bmm2 = time_op("bmm grad_score@k -> grad_q",
    lambda: torch.bmm(gs_reshaped, k_expanded))

# grad_k = q^T @ grad_score (need reduction over S_q)
q_perm = q.permute(0, 2, 1, 3).reshape(B * n_heads, S_q, head_dim)
gs_perm = grad_score.permute(0, 2, 1, 3).reshape(B * n_heads, S_q, S_kv)
_, t_bmm3 = time_op("bmm q^T@grad_score -> grad_k_per_head",
    lambda: torch.bmm(q_perm.transpose(-1, -2), gs_perm))

print(f"\n  BMM total: {t_bmm1 + t_bmm2 + t_bmm3:.3f} ms")
