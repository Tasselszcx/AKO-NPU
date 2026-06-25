#!/usr/bin/env python3
"""Profile more optimization ideas."""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import torch_npu
import time

device = "npu:0"
B, S_q, S_kv = 1, 4096, 4096
n_heads, head_dim = 64, 128

q = torch.randn(B, n_heads, S_q, head_dim, dtype=torch.float32, device=device)
k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
weights = torch.randn(B, n_heads, S_q, 1, dtype=torch.float32, device=device) * 0.01
relu_mask = torch.rand(B, n_heads, S_q, S_kv, device=device) > 0.5
k_t = k.transpose(-1, -2).unsqueeze(1)
k_unsq = k.unsqueeze(1)

def time_op(name, fn, n=5, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(n):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    mean_t = sum(times) / len(times)
    print(f"  {name}: {mean_t:.3f} ms")
    return mean_t

# Can we combine score+relu into one pass?
print("=== Fused ops ideas ===\n")

# Idea 1: Use clamp instead of relu
score = torch.matmul(q, k_t)
time_op("relu_(score)", lambda: score.clone().relu_())
time_op("clamp_(min=0)", lambda: score.clone().clamp_(min=0))
time_op("F.relu(score)", lambda: F.relu(score))
time_op("torch.clamp(score, min=0)", lambda: torch.clamp(score, min=0))

# Idea 2: Can we skip materializing score and fuse with grad_weights?
# grad_weights = (grad_is * relu(q@k^T)).sum(-1, keepdim)
# This would need computing q@k^T and immediately multiplying and reducing
print("\n=== Memory layout optimizations ===\n")

# Try contiguous() before matmul
q_contig = q.contiguous()
time_op("matmul q_contig@k_t", lambda: torch.matmul(q_contig, k_t))
time_op("matmul q@k_t (may be non-contig)", lambda: torch.matmul(q, k_t))

# Idea 3: Can we use addmm for the grad_k reduction?
# grad_k = sum_h (q_h^T @ gs_h) can be done as a big matmul if we reshape
q_flat = q.reshape(B * n_heads, S_q, head_dim)
gs = torch.where(relu_mask, grad_index_score.unsqueeze(1) * weights, torch.tensor(0.0, device=device))
gs_flat = gs.reshape(B * n_heads, S_q, S_kv)

time_op("bmm grad_k (flat)", lambda: torch.bmm(q_flat.transpose(-1, -2), gs_flat))

# Idea 4: Try addmm for accumulation
print("\n=== grad_k sum optimization ===\n")
gk_per_head = torch.bmm(q_flat.transpose(-1, -2), gs_flat)  # [N, D, K]
time_op("reshape+sum", lambda: gk_per_head.reshape(B, n_heads, head_dim, S_kv).sum(1))

# Alternative: use matrix multiply to sum heads
# [1, D, N*K] where we reshape [N,D,K] -> [D, N*K] then multiply by ones
gk_reshaped = gk_per_head.reshape(n_heads, head_dim * S_kv)  # [N, D*K]
ones = torch.ones(1, n_heads, device=device)
time_op("matmul ones@gk (sum heads)", lambda: torch.matmul(ones, gk_reshaped).reshape(head_dim, S_kv))

# Idea 5: Skip relu_mask - directly compute score > 0 from score
print("\n=== Score-based masking ===\n")
score_full = torch.matmul(q, k_t)
time_op("score > 0 (compute mask)", lambda: score_full > 0)
time_op("score.relu_() (reuse as mask)", lambda: score_full.clone().relu_())

# The issue: we need relu_mask as input because it was saved from forward
# But score = q@k (recomputed) and relu_mask = (score > 0) from forward
# They should be identical! So we can just use score > 0 directly

# Idea 6: Pre-compute grad_is * weights to save memory
print("\n=== Reuse optimization ===\n")
grad_is = grad_index_score.unsqueeze(1)
gw = grad_is * weights  # [B, N, S_q, S_kv]
time_op("grad_is * weights", lambda: grad_is * weights)
time_op("gw * relu_mask", lambda: gw * relu_mask)
time_op("torch.where(relu_mask, gw, 0)", lambda: torch.where(relu_mask, gw, torch.tensor(0.0, device=device)))

# Can we avoid computing grad_is*score for grad_weights and instead
# compute it differently? grad_weights = sum_k(grad_is[k] * score[k])
# = grad_is . score (dot product over last dim)
print("\n=== grad_weights alternatives ===\n")
time_op("(grad_is * score).sum(-1, keepdim)", lambda: (grad_is * score_full.relu()).sum(-1, keepdim=True))
# torch.sum with a product is the same as a dot product / inner product
# Can use einsum for this
time_op("einsum for grad_weights", lambda: torch.einsum("bnsk,bnsk->bns", grad_is.expand_as(score_full), score_full.relu()).unsqueeze(-1))

# baddbmm?
gis_flat = grad_is.expand(B, n_heads, S_q, S_kv).reshape(B*n_heads, S_q, S_kv)
sc_flat = score_full.relu().reshape(B*n_heads, S_q, S_kv)
time_op("bmm grad_is^T @ score (for grad_w)", lambda: torch.bmm(gis_flat.unsqueeze(2).reshape(B*n_heads*S_q, 1, S_kv), sc_flat.reshape(B*n_heads*S_q, S_kv, 1)))
