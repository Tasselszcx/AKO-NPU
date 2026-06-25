#!/usr/bin/env python3
"""Rapid experimentation: try many small variations quickly."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'
B, N, S_q, S_kv, D = 1, 64, 4096, 4096, 128

q_orig = torch.randn(B, S_q, N, D, device=device)
k = torch.randn(B, S_kv, D, device=device)
grad_is_full = torch.randn(B, S_q, S_kv, device=device)
weights_orig = torch.randn(B, S_q, N, 1, device=device) * 0.01
relu_mask_orig = torch.rand(B, S_q, N, S_kv, device=device) > 0.5

k_2d = k.reshape(S_kv, D)
k_t = k_2d.T.contiguous()
_zero = torch.tensor(0.0, device=device)

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
        times.append((t1-t0)*1000)
    mean_t = sum(times)/len(times)
    min_t = min(times)
    print("  %-50s %7.1f ms (min: %7.1f)" % (name, mean_t, min_t))
    return mean_t

# Current best (reference)
def current_best():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

print("=== Current best ===")
time_op("current_best", current_best)

# Experiment 1: Pre-compute contiguous relu_mask_nh to avoid transpose inside where
print("\n=== Experiment 1: Contiguous relu_mask ===")
def exp_contig_mask():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3).contiguous()
    grad_is = grad_is_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_op("contiguous relu_mask", exp_contig_mask)

# Experiment 2: Combine where and mul into single expression
print("\n=== Experiment 2: Fused where*weights ===")
def exp_fused_where_mul():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    del score

    # Fuse: where(mask, grad_is * weights, 0)
    gs = torch.where(m_nh, grad_is * w_nh, _zero)

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_op("fused where(mask, gi*w, 0)", exp_fused_where_mul)

# Experiment 3: Use matmul instead of npu_linear for score
print("\n=== Experiment 3: matmul vs npu_linear for score ===")
def exp_matmul_score():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    k_t_4d = k.transpose(-1, -2).unsqueeze(1)
    score = F.relu(torch.matmul(q_nh, k_t_4d))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_op("matmul for score", exp_matmul_score)

# Experiment 4: Skip del score (see if memory pressure matters)
print("\n=== Experiment 4: No del score ===")
def exp_no_del():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    # No del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_op("no del score", exp_no_del)

# Experiment 5: Different reduction for grad_weights
print("\n=== Experiment 5: Different grad_weights reduction ===")
def exp_gw_einsum():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))

    # Use mul then reduce in a different order
    gw = (score * grad_is).sum(-1, keepdim=True)  # same but swap order
    del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_op("swap mul order for gw", exp_gw_einsum)

# Experiment 6: Use npu_linear for grad_k via 2D mm
print("\n=== Experiment 6: npu_linear for grad_k ===")
def exp_npu_linear_gk():
    q_nh = q_orig.permute(0, 2, 1, 3).contiguous()
    w_nh = weights_orig.permute(0, 2, 1, 3)
    m_nh = relu_mask_orig.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)

    # grad_k via npu_linear: q_flat^T @ gs_flat = [D, N*S] @ [N*S, K]
    # npu_linear(q_flat, gs_flat.T) ≠ what we want
    # Use mm instead
    gk = torch.mm(q_flat.T, gs_flat).unsqueeze(0).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_op("mm for grad_k + early contiguous", exp_npu_linear_gk)
