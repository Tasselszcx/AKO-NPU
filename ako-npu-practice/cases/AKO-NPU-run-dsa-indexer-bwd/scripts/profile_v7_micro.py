#!/usr/bin/env python3
"""Micro-benchmarks for remaining optimization ideas."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'
B, N, S_q, S_kv, D = 1, 64, 4096, 4096, 128

q = torch.randn(B, N, S_q, D, device=device)
k = torch.randn(B, S_kv, D, device=device)
grad_is = torch.randn(B, 1, S_q, S_kv, device=device)
weights = torch.randn(B, N, S_q, 1, device=device) * 0.01
relu_mask = torch.rand(B, N, S_q, S_kv, device=device) > 0.5
k_t = k.transpose(-1, -2).unsqueeze(1)
k_unsq = k.unsqueeze(1)

def time_op(name, fn, n=10, warmup=5):
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
    min_t = min(times)
    print(f"  {name}: {mean_t:.3f} ms (min: {min_t:.3f})")
    return mean_t

print("=== Current best kernel ===\n")
def run_best():
    score = F.relu(torch.matmul(q, k_t))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    _zero = torch.tensor(0.0, device=device)
    gs = torch.where(relu_mask, grad_is * weights, _zero)
    gq = torch.matmul(gs, k_unsq)
    gk = torch.matmul(q.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw

time_op("current best", run_best)

# Can we eliminate the score tensor for grad_weights?
# grad_weights[n,s] = sum_k(grad_is[s,k] * relu(q[n,s,:] @ k[k,:]))
# = grad_is[s,:] @ diag(q[n,s,:] @ K^T > 0) @ (q[n,s,:] @ K^T)
# Hmm, this doesn't simplify...

# What about changing the order? Compute grad_score first, then reuse for matmuls
print("\n=== Reordering ===\n")
def run_reordered():
    # Compute grad_score first (doesn't need score)
    _zero = torch.tensor(0.0, device=device)
    gs = torch.where(relu_mask, grad_is * weights, _zero)

    # Then matmuls with grad_score
    gq = torch.matmul(gs, k_unsq)
    gk = torch.matmul(q.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    # Score recomputation only for grad_weights
    score = F.relu(torch.matmul(q, k_t))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)

    return gq, gk, gw

time_op("reordered (gs first)", run_reordered)

# What about computing matmuls while score is being computed?
# Can we overlap the score matmul with the where operation?
print("\n=== Overlapped computation ===\n")
def run_overlapped():
    # Start grad_score (doesn't need score)
    _zero = torch.tensor(0.0, device=device)
    gs = torch.where(relu_mask, grad_is * weights, _zero)

    # Start matmuls immediately
    gq = torch.matmul(gs, k_unsq)

    # Score for grad_weights (can overlap with gq matmul on NPU?)
    score = F.relu(torch.matmul(q, k_t))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)

    gk = torch.matmul(q.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq, gk, gw

time_op("overlapped (gs->gq while computing score)", run_overlapped)

# What about pre-allocating output tensors?
print("\n=== Pre-allocated outputs ===\n")
gq_buf = torch.empty(B, N, S_q, D, device=device)
gk_buf = torch.empty(B, D, S_kv, device=device)
gw_buf = torch.empty(B, N, S_q, 1, device=device)
gs_buf = torch.empty(B, N, S_q, S_kv, device=device)
score_buf = torch.empty(B, N, S_q, S_kv, device=device)

def run_preallocated():
    torch.matmul(q, k_t, out=score_buf)
    score = F.relu(score_buf)
    torch.sum(grad_is * score, dim=-1, keepdim=True, out=gw_buf)
    _zero = torch.tensor(0.0, device=device)
    torch.where(relu_mask, grad_is * weights, _zero, out=gs_buf)
    torch.matmul(gs_buf, k_unsq, out=gq_buf)
    gk_per = torch.matmul(q.transpose(-1, -2), gs_buf)
    gk_per.sum(dim=1, out=gk_buf)
    return gq_buf, gk_buf.transpose(-1, -2), gw_buf

time_op("pre-allocated", run_preallocated)

# What about using contiguous tensors?
print("\n=== Contiguous optimization ===\n")
q_c = q.contiguous()
q_t_c = q.transpose(-1, -2).contiguous()

def run_contiguous():
    score = F.relu(torch.matmul(q_c, k_t))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    _zero = torch.tensor(0.0, device=device)
    gs = torch.where(relu_mask, grad_is * weights, _zero)
    gq = torch.matmul(gs, k_unsq)
    gk = torch.matmul(q_t_c, gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw

time_op("with contiguous q^T", run_contiguous)

# Can we avoid score for grad_weights by using a different formulation?
# grad_weights[b,n,s] = sum_k(grad_is[b,s,k] * relu(sum_d(q[b,n,s,d] * k[b,k,d])))
# = sum_k(grad_is[b,s,k] * max(0, sum_d(q[b,n,s,d] * k[b,k,d])))
# We need the relu of individual elements, so we can't avoid computing score...
# Unless we can express this differently.

# Alternative: use q@k directly in masked form
# score_masked = score * relu_mask  (same as relu since relu_mask was score > 0)
# Hmm, no - relu_mask is from the *original* forward computation, not from our recomputation
