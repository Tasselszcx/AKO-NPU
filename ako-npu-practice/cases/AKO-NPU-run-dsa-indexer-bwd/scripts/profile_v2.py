#!/usr/bin/env python3
"""Profile v2 (matmul version) to find new bottlenecks."""
import warnings
warnings.filterwarnings("ignore")

import torch
import torch_npu
import time

device = "npu:0"
B, S_q, S_kv = 1, 4096, 4096
n_heads, head_dim = 64, 128

grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.float32, device=device)
k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.float32, device=device) * 0.01
relu_mask = torch.rand(B, S_q, n_heads, S_kv, device=device) > 0.5

# Warmup
for _ in range(3):
    k_t = k.transpose(-1, -2).unsqueeze(1)
    score = torch.matmul(q, k_t)
    score = torch.relu(score)
    torch.npu.synchronize()

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
    print(f"  {name}: {mean_t:.3f} ms")
    return result, mean_t

print("=== V2 profiling (matmul, B=1, S_q=4096, S_kv=4096) ===\n")

_, t1 = time_op("expand grad_weighted", lambda: grad_index_score.unsqueeze(2).expand(-1, -1, n_heads, -1))
grad_weighted = grad_index_score.unsqueeze(2).expand(-1, -1, n_heads, -1)

k_t = k.transpose(-1, -2).unsqueeze(1)
_, t2 = time_op("matmul q@k_t (score)", lambda: torch.matmul(q, k_t))
score = torch.matmul(q, k_t)

_, t3 = time_op("relu(score)", lambda: torch.relu(score))
score = torch.relu(score)

_, t4 = time_op("grad_score = grad_weighted * weights", lambda: grad_weighted * weights)
grad_score_pre = grad_weighted * weights

_, t5 = time_op("grad_weights = (gw*score).sum(-1)", lambda: (grad_weighted * score).sum(dim=-1, keepdim=True))

_, t6 = time_op("grad_score * relu_mask", lambda: grad_score_pre * relu_mask.float())
grad_score = grad_score_pre * relu_mask.float()

k_unsq = k.unsqueeze(1)
_, t7 = time_op("matmul grad_score@k (grad_q)", lambda: torch.matmul(grad_score, k_unsq))

q_perm = q.permute(0, 2, 1, 3)
gs_perm = grad_score.permute(0, 2, 1, 3)
_, t8 = time_op("matmul q^T@gs (grad_k)", lambda: torch.matmul(q_perm.transpose(-1, -2), gs_perm))

_, t9 = time_op("sum heads + transpose", lambda: torch.matmul(q_perm.transpose(-1, -2), gs_perm).sum(dim=1).transpose(-1, -2))

total = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8
print(f"\n  TOTAL (no sum/transpose): {total:.3f} ms")
print(f"  Matmul ops: {t2 + t7 + t8:.3f} ms ({(t2+t7+t8)/total*100:.1f}%)")
print(f"  Element-wise: {t1+t3+t4+t5+t6:.3f} ms ({(t1+t3+t4+t5+t6)/total*100:.1f}%)")

# Try float16 approach
print("\n=== Float16 approach ===\n")
q_h = q.half()
k_h = k.half()
gs_h = grad_score.half()
relu_mask_h = relu_mask

_, t_mm_h = time_op("matmul q_h@k_h_t (fp16)", lambda: torch.matmul(q_h, k_h.transpose(-1, -2).unsqueeze(1)))
_, t_gq_h = time_op("matmul gs_h@k_h (fp16 grad_q)", lambda: torch.matmul(gs_h, k_h.unsqueeze(1)))
q_h_perm = q_h.permute(0, 2, 1, 3)
gs_h_perm = gs_h.permute(0, 2, 1, 3)
_, t_gk_h = time_op("matmul q_h^T@gs_h (fp16 grad_k)", lambda: torch.matmul(q_h_perm.transpose(-1, -2), gs_h_perm))

print(f"\n  FP16 matmul total: {t_mm_h + t_gq_h + t_gk_h:.3f} ms")

# Try in-place relu
print("\n=== In-place ops ===\n")
score2 = torch.matmul(q, k_t)
_, t_relu_inplace = time_op("relu_(score) in-place", lambda: score2.relu_())

# Try torch.where instead of * float()
_, t_where = time_op("where(relu_mask, gs, 0)", lambda: torch.where(relu_mask, grad_score_pre, torch.zeros_like(grad_score_pre)))
_, t_mul_bool = time_op("gs * relu_mask (no .float())", lambda: grad_score_pre * relu_mask)
