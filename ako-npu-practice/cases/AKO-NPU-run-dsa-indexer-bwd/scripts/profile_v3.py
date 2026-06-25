#!/usr/bin/env python3
"""Profile v3 - investigate memory bandwidth bottleneck and torch.compile."""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TORCH_NPU_ALLOW_INTERNAL_FORMAT"] = "0"  # Disable internal format for compile

import torch
import torch_npu
import time

device = "npu:0"
B, S_q, S_kv = 1, 4096, 4096
n_heads, head_dim = 64, 128

# Memory analysis
print("=== Memory Analysis ===")
score_bytes = B * S_q * n_heads * S_kv * 4
print(f"score tensor: {score_bytes / 1024**3:.2f} GB ({B}x{S_q}x{n_heads}x{S_kv} fp32)")
grad_score_bytes = score_bytes
print(f"grad_score tensor: {grad_score_bytes / 1024**3:.2f} GB")
print(f"Total intermediate tensors: ~{(score_bytes*3) / 1024**3:.1f} GB")
print()

grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.float32, device=device)
k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.float32, device=device) * 0.01
relu_mask = torch.rand(B, S_q, n_heads, S_kv, device=device) > 0.5

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

# Try chunked processing - process n_heads in chunks
print("=== Chunked processing (reduce peak memory) ===\n")

k_t = k.transpose(-1, -2).unsqueeze(1)
k_unsq = k.unsqueeze(1)

def run_chunked(chunk_size):
    q_chunks = q.split(chunk_size, dim=2)  # split n_heads
    w_chunks = weights.split(chunk_size, dim=2)
    m_chunks = relu_mask.split(chunk_size, dim=2)

    grad_q_chunks = []
    grad_k_accum = torch.zeros(B, head_dim, S_kv, device=device)
    grad_w_chunks = []

    grad_is = grad_index_score.unsqueeze(2)

    for qc, wc, mc in zip(q_chunks, w_chunks, m_chunks):
        c = qc.shape[2]
        score_c = torch.matmul(qc, k_t)
        score_c.relu_()

        gs_c = grad_is * wc
        gw_c = (grad_is * score_c).sum(dim=-1, keepdim=True)
        gs_c *= mc

        gq_c = torch.matmul(gs_c, k_unsq)

        q_perm_c = qc.permute(0, 2, 1, 3)
        gs_perm_c = gs_c.permute(0, 2, 1, 3)
        gk_c = torch.matmul(q_perm_c.transpose(-1, -2), gs_perm_c).sum(dim=1)
        grad_k_accum += gk_c

        grad_q_chunks.append(gq_c)
        grad_w_chunks.append(gw_c)

    grad_q = torch.cat(grad_q_chunks, dim=2)
    grad_weights = torch.cat(grad_w_chunks, dim=2)
    grad_k = grad_k_accum.transpose(-1, -2)
    return grad_q, grad_k, grad_weights

for cs in [64, 32, 16, 8, 4]:
    time_op(f"chunked heads (chunk={cs})", lambda cs=cs: run_chunked(cs))

# Try different memory layouts
print("\n=== Permuted q for better matmul layout ===\n")
q_nhs = q.permute(0, 2, 1, 3).contiguous()  # [B, n_heads, S_q, head_dim]
time_op("matmul q[n,s,d] @ k_t", lambda: torch.matmul(q_nhs, k.transpose(-1, -2).unsqueeze(0)))

# Try torch.compile if available
print("\n=== torch.compile test ===\n")
try:
    @torch.compile(backend="npu")
    def fused_backward(grad_is, q, k_t, k_unsq, weights, relu_mask):
        score = torch.matmul(q, k_t)
        score.relu_()
        grad_score = grad_is * weights
        grad_weights = (grad_is * score).sum(dim=-1, keepdim=True)
        grad_score *= relu_mask
        grad_q = torch.matmul(grad_score, k_unsq)
        return grad_q, grad_score, grad_weights

    grad_is = grad_index_score.unsqueeze(2)
    time_op("torch.compile fused", lambda: fused_backward(grad_is, q, k_t, k_unsq, weights, relu_mask))
except Exception as e:
    print(f"  torch.compile failed: {e}")

# Try contiguous grad_score before permute
print("\n=== Avoid permute overhead ===\n")
score = torch.matmul(q, k_t)
score.relu_()
grad_is = grad_index_score.unsqueeze(2)
grad_score = (grad_is * weights) * relu_mask

# Direct approach: q^T @ gs without permute
# q: [B,S,N,D], gs: [B,S,N,K] -> need [B,N,D,S]@[B,N,S,K] = [B,N,D,K]
# Instead of permute, use reshape
time_op("permute then matmul", lambda: torch.matmul(q.permute(0,2,1,3).transpose(-1,-2), grad_score.permute(0,2,1,3)))
time_op("reshape to 3d bmm", lambda: torch.bmm(
    q.permute(0,2,1,3).reshape(n_heads, S_q, head_dim).transpose(-1,-2),
    grad_score.permute(0,2,1,3).reshape(n_heads, S_q, S_kv)))
