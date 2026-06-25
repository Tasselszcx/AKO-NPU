#!/usr/bin/env python3
"""Batch 3: More experiments for iterations 51-70. Focus on small shape optimization."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'

def make_inputs(B, S_q, S_kv, N=64, D=128):
    q = torch.randn(B, S_q, N, D, device=device)
    k = torch.randn(B, S_kv, D, device=device)
    gi = torch.randn(B, S_q, S_kv, device=device)
    w = torch.randn(B, S_q, N, 1, device=device) * 0.01
    m = torch.rand(B, S_q, N, S_kv, device=device) > 0.5
    return q, k, gi, w, m

def run_current(q, k, gi, w, m):
    B, S_q, N, D = q.shape
    S_kv = k.shape[1]
    k_2d = k.reshape(S_kv, D)
    k_t = k_2d.T.contiguous()
    _zero = torch.tensor(0.0, device=device)

    q_nh = q.permute(0, 2, 1, 3)
    w_nh = w.permute(0, 2, 1, 3)
    m_nh = m.permute(0, 2, 1, 3)
    gis = gi.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, gis, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

def time_fn(name, fn, n=5, warmup=3):
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
    print("  %-55s %7.3f ms" % (name, sum(times)/len(times)))
    return sum(times)/len(times)

# Shape 1: B=1, S_q=1, S_kv=4096 (decode)
print("=== Shape 1 (decode): B=1, S_q=1, S_kv=4096 ===")
q1, k1, gi1, w1, m1 = make_inputs(1, 1, 4096)
time_fn("current", lambda: run_current(q1, k1, gi1, w1, m1))

# For decode shape, einsum might be better (small S_q=1)
def run_einsum_decode(q, k, gi, w, m):
    B, S_q, N, D = q.shape
    S_kv = k.shape[1]
    _zero = torch.tensor(0.0, device=device)

    gis = gi.unsqueeze(2).expand(-1, -1, N, -1)
    score = torch.einsum("bsnd,btd->bsnt", q, k)
    score = F.relu(score)

    gw = (gis * score).sum(dim=-1, keepdim=True)
    gs = gis * w
    gs = gs * m.float()

    gq = torch.einsum("bsnt,btd->bsnd", gs, k)
    gk = torch.einsum("bsnd,bsnt->btd", q, gs)

    return gq, gk, gw

time_fn("einsum for decode", lambda: run_einsum_decode(q1, k1, gi1, w1, m1))

# Try matmul for decode shape
def run_matmul_decode(q, k, gi, w, m):
    B, S_q, N, D = q.shape
    S_kv = k.shape[1]
    _zero = torch.tensor(0.0, device=device)

    k_t = k.transpose(-1, -2).unsqueeze(1)
    k_unsq = k.unsqueeze(1)

    score = F.relu(torch.matmul(q.permute(0,2,1,3), k_t))
    gis = gi.unsqueeze(1)
    gw = torch.sum(gis * score, dim=-1, keepdim=True)

    m_nh = m.permute(0, 2, 1, 3)
    w_nh = w.permute(0, 2, 1, 3)
    gs = torch.where(m_nh, gis, _zero) * w_nh

    gq = torch.matmul(gs, k_unsq)
    gk = torch.matmul(q.permute(0,2,1,3).transpose(-1,-2), gs).sum(dim=1).transpose(-1,-2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

time_fn("matmul for decode", lambda: run_matmul_decode(q1, k1, gi1, w1, m1))

# For decode, try processing all N heads at once with smaller data
def run_decode_optimized(q, k, gi, w, m):
    B, S_q, N, D = q.shape
    S_kv = k.shape[1]
    _zero = torch.tensor(0.0, device=device)

    # q: [1,1,64,128], k: [1,4096,128]
    # score = q @ k^T: [1,64,1,4096] via single matmul
    q_nh = q.permute(0, 2, 1, 3)  # [1,64,1,128]
    k_t = k.reshape(S_kv, D).T.unsqueeze(0).unsqueeze(0)  # [1,1,128,4096]
    score = F.relu(torch.matmul(q_nh, k_t))  # [1,64,1,4096]

    gis = gi.unsqueeze(1)  # [1,1,1,4096]
    gw = torch.sum(gis * score, dim=-1, keepdim=True)  # [1,64,1,1]

    m_nh = m.permute(0, 2, 1, 3)
    w_nh = w.permute(0, 2, 1, 3)
    gs = torch.where(m_nh, gis * w_nh, _zero)  # [1,64,1,4096]

    k_unsq = k.unsqueeze(1)  # [1,1,4096,128]
    gq = torch.matmul(gs, k_unsq)  # [1,64,1,128]
    gk = torch.matmul(q_nh.transpose(-1,-2), gs).sum(dim=1).transpose(-1,-2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

time_fn("decode optimized (matmul 4D)", lambda: run_decode_optimized(q1, k1, gi1, w1, m1))

# Shape 3: B=1, S_q=512, S_kv=2048
print("\n=== Shape 3 (medium): B=1, S_q=512, S_kv=2048 ===")
q3, k3, gi3, w3, m3 = make_inputs(1, 512, 2048)
time_fn("current", lambda: run_current(q3, k3, gi3, w3, m3))
time_fn("matmul (no npu_linear)", lambda: run_matmul_decode(q3, k3, gi3, w3, m3))

# Try different data types
print("\n=== Dtype experiments (Shape 2) ===")
q2, k2, gi2, w2, m2 = make_inputs(1, 4096, 4096)
time_fn("current fp32", lambda: run_current(q2, k2, gi2, w2, m2))

# bfloat16 matmul
def run_bf16_matmul(q, k, gi, w, m):
    B, S_q, N, D = q.shape
    S_kv = k.shape[1]
    k_2d = k.reshape(S_kv, D)
    k_t = k_2d.T.contiguous()
    _zero = torch.tensor(0.0, device=device)

    q_nh = q.permute(0, 2, 1, 3)
    w_nh = w.permute(0, 2, 1, 3)
    m_nh = m.permute(0, 2, 1, 3)
    gis = gi.unsqueeze(1)

    # BF16 for score matmul
    q_flat_bf = q_nh.reshape(B * N * S_q, D).bfloat16()
    k_2d_bf = k_2d.bfloat16()
    score = F.relu(torch_npu.npu_linear(q_flat_bf, k_2d_bf).float().reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, gis, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

time_fn("bf16 score matmul", lambda: run_bf16_matmul(q2, k2, gi2, w2, m2))

# Try gradient accumulation order
def run_gk_first(q, k, gi, w, m):
    B, S_q, N, D = q.shape
    S_kv = k.shape[1]
    k_2d = k.reshape(S_kv, D)
    k_t = k_2d.T.contiguous()
    _zero = torch.tensor(0.0, device=device)

    q_nh = q.permute(0, 2, 1, 3)
    w_nh = w.permute(0, 2, 1, 3)
    m_nh = m.permute(0, 2, 1, 3)
    gis = gi.unsqueeze(1)

    # Compute grad_score first
    gi_masked = torch.where(m_nh, gis, _zero)
    gs = gi_masked * w_nh

    # grad_k first (might warm up Cube engine)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    # Then grad_q
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)

    # Then score for grad_weights
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)

    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

time_fn("reorder: gk first, then gq, then gw", lambda: run_gk_first(q2, k2, gi2, w2, m2))
