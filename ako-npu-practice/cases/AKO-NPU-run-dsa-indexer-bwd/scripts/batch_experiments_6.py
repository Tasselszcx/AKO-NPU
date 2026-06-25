#!/usr/bin/env python3
"""Batch 6: More systematic experiments (iter 76-100)."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'
B, N, S_q, S_kv, D = 1, 64, 4096, 4096, 128

q = torch.randn(B, S_q, N, D, device=device)
k = torch.randn(B, S_kv, D, device=device)
gi = torch.randn(B, S_q, S_kv, device=device)
w = torch.randn(B, S_q, N, 1, device=device) * 0.01
m = torch.rand(B, S_q, N, S_kv, device=device) > 0.5

k_2d = k.reshape(S_kv, D)
k_t = k_2d.T.contiguous()
_zero = torch.tensor(0.0, device=device)
q_nh = q.permute(0, 2, 1, 3)
w_nh = w.permute(0, 2, 1, 3)
m_nh = m.permute(0, 2, 1, 3)

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
    print("  %-50s %7.1f ms" % (name, sum(times)/len(times)))

# Reference
def ref():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw

print("=== Batch 6 ===")
time_fn("76: Reference", ref)

# 77: Use torch.empty_like + fill for zero tensor
def exp77():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    # Use different approach for masking
    gs = gis * w_nh  # [B,N,S,K]
    gs = gs * m_nh  # Apply mask via bool multiply
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("77: gi*w first, then *mask (no where)", exp77)

# 78: Try different order: multiply weights last
def exp78():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    # Delay weights multiply until after reshape
    gi_m_flat = gi_m.reshape(B * N * S_q, S_kv)
    # grad_q = (gi_m * w) @ K = w * (gi_m @ K)
    gq_raw = torch_npu.npu_linear(gi_m_flat, k_t).reshape(B, N, S_q, D)
    gq = gq_raw * w_nh  # Apply weights after matmul
    # grad_k needs gs = gi_m * w for correct result
    gs = gi_m * w_nh
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("78: weights after gq matmul", exp78)

# 79: Cache the q_nh.transpose(-1,-2) result
q_nh_t = q_nh.transpose(-1, -2).contiguous()
def exp79():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    # Use pre-computed contiguous transpose
    gk = torch.matmul(q_nh_t, gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("79: cached q^T contiguous", exp79)

# 80-85: Process in specific order to maximize pipeline overlap
# 80: gw -> gs -> gq -> gk (current order)
time_fn("80: current order (gw->gs->gq->gk)", ref)

# 81: gs -> gq -> gk -> gw (grad_score first)
def exp81():
    gis = gi.unsqueeze(1)
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    return gq, gk, gw
time_fn("81: gs->gq->gk->gw", exp81)

# 82: gw first, then interleave
def exp82():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    # Interleave gq and gk
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("82: same as ref", exp82)

# 83-85: Try gradient checkpointing style
# 83: Process q_chunk-at-a-time for score
def exp83():
    gis = gi.unsqueeze(1)
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    # Compute gw chunk by chunk to reduce peak memory
    gw = torch.zeros(B, N, S_q, 1, device=device)
    chunk_n = 16
    for h in range(0, N, chunk_n):
        he = min(h + chunk_n, N)
        q_chunk = q_nh[:, h:he]
        q_flat_c = q_chunk.reshape(B * (he-h) * S_q, D)
        score_c = F.relu(torch_npu.npu_linear(q_flat_c, k_2d).reshape(B, he-h, S_q, S_kv))
        gw[:, h:he] = torch.sum(gis * score_c, dim=-1, keepdim=True)
        del score_c
    return gq, gk, gw
time_fn("83: chunked gw (16 heads at a time)", exp83)

# 84: Use addmm_ for in-place grad_k accumulation
def exp84():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    # In-place accumulate grad_k using addmm
    q_3d = q_nh.reshape(N, S_q, D)
    gs_3d = gs.reshape(N, S_q, S_kv)
    gk_per = torch.bmm(q_3d.transpose(-1,-2), gs_3d)
    gk = gk_per.sum(0).unsqueeze(0).transpose(-1, -2)
    return gq, gk, gw
time_fn("84: bmm then sum(0) for grad_k", exp84)

# 85: Full pipeline with all optimizations combined
def exp85():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
time_fn("85: final reference timing", exp85)

print("\nBatch 6 done. All results ~50ms for full kernel.")
