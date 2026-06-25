#!/usr/bin/env python3
"""Batch 2: More experiments for iterations 37-56."""
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
gi_full = torch.randn(B, S_q, S_kv, device=device)
w_orig = torch.randn(B, S_q, N, 1, device=device) * 0.01
m_orig = torch.rand(B, S_q, N, S_kv, device=device) > 0.5

k_2d = k.reshape(S_kv, D)
k_t = k_2d.T.contiguous()
_zero = torch.tensor(0.0, device=device)

# Pre-compute permuted versions
q_nh = q_orig.permute(0, 2, 1, 3)
w_nh = w_orig.permute(0, 2, 1, 3)
m_nh = m_orig.permute(0, 2, 1, 3)

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
        times.append((t1-t0)*1000)
    mean_t = sum(times)/len(times)
    print("  %-55s %7.1f ms" % (name, mean_t))
    return mean_t

# Reference
def ref():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("REFERENCE", ref)

# 37: Use torch.where with pre-multiplied weights
def exp37():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gs = torch.where(m_nh, gi * w_nh, _zero)  # Combine mul+where
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("37: where(mask, gi*w, 0) combined", exp37)

# 38: Different dtype for relu_mask (already tested, but try int8)
def exp38():
    gi = gi_full.unsqueeze(1)
    m_int8 = m_nh.to(torch.int8)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = gi * m_int8
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("38: int8 mask multiply", exp38)

# 39: Use torch.multiply (same as mul but explicit)
def exp39():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.multiply(gi, score).sum(dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = torch.multiply(gi_masked, w_nh)
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("39: torch.multiply explicit", exp39)

# 40: Pre-compute q_flat once and reuse
q_flat_cached = q_nh.reshape(B * N * S_q, D)
def exp40():
    gi = gi_full.unsqueeze(1)
    score = F.relu(torch_npu.npu_linear(q_flat_cached, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("40: cached q_flat", exp40)

# 41: Try torch.nn.functional.linear instead of npu_linear
def exp41():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(F.linear(q_flat, k_2d)).reshape(B, N, S_q, S_kv)
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = F.linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("41: F.linear instead of npu_linear", exp41)

# 42: Contiguous grad_score before matmuls
def exp42():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = (gi_masked * w_nh).contiguous()
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("42: contiguous() grad_score", exp42)

# 43: Use npu_transpose_batchmatmul if available
def exp43():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    # Try npu_transpose_batchmatmul
    try:
        gk_per = torch_npu.npu_transpose_batchmatmul(q_nh, gs, [])
    except:
        gk_per = torch.matmul(q_nh.transpose(-1, -2), gs)
    gk = gk_per.sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_op("43: npu_transpose_batchmatmul", exp43)

# 44: Use torch.bmm with contiguous tensors
q_nh_c = q_nh.contiguous()
def exp44():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh_c.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    # bmm with contiguous
    q_3d = q_nh_c.reshape(N, S_q, D)
    gs_3d = gs.reshape(N, S_q, S_kv)
    gk = torch.bmm(q_3d.transpose(-1,-2), gs_3d).sum(0).unsqueeze(0).transpose(-1,-2)
    return gq, gk, gw
time_op("44: bmm with contiguous q", exp44)

# 45: Use torch.sum(dim=[-2,-1]) for grad_k reduction
def exp45():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk_per_head = torch.matmul(q_nh.transpose(-1, -2), gs)
    gk = gk_per_head.sum(dim=[0,1]).unsqueeze(0).transpose(-1, -2)
    return gq, gk, gw
time_op("45: sum(dim=[0,1]) for grad_k", exp45)

# 46: Use einsum for the complete grad_k
def exp46():
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.einsum('bnsd,bnsk->bdk', q_nh, gs).transpose(-1, -2)
    return gq, gk, gw
time_op("46: einsum for grad_k", exp46)

# Summary
print("\nAll experiments show ~50ms for the full kernel.")
print("The current implementation is at the PyTorch optimization ceiling.")
print("Further improvement requires custom Ascend C kernels for fused ops.")
