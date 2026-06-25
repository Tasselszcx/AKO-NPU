#!/usr/bin/env python3
"""Batch of 20+ experiments for rapid iteration counting."""
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

results = []

# Reference
def ref():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = w_orig.permute(0, 2, 1, 3)
    m_nh = m_orig.permute(0, 2, 1, 3)
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
    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

r = time_op("REFERENCE", ref)
results.append(("reference", r))

# Exp 29: Use torch.addcmul for fused multiply-add
print("\n=== Experiments 29-48 ===")

# 29: addcmul for gi*score
def exp29():
    q_nh = q_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    # Use addcmul: out = zeros + gi * score
    out = torch.zeros(B, N, S_q, S_kv, device=device)
    torch.addcmul(out, gi, score, value=1.0, out=out)
    gw = out.sum(-1, keepdim=True)
    return gw
results.append(("29: addcmul for gi*score", time_op("29: addcmul for gi*score", exp29)))

# 30: torch.tensordot for grad_k
def exp30():
    q_nh = q_orig.permute(0, 2, 1, 3)
    m_nh = m_orig.permute(0, 2, 1, 3)
    w_nh = w_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    gk = torch.tensordot(q_nh, gs, dims=[[0,1,2],[0,1,2]])
    return gk
results.append(("30: tensordot for grad_k", time_op("30: tensordot for grad_k", exp30)))

# 31-35: Different S_kv tile sizes in the score computation
for tile in [512, 1024, 2048]:
    def exp_tile(tile=tile):
        q_nh = q_orig.permute(0, 2, 1, 3)
        w_nh = w_orig.permute(0, 2, 1, 3)
        m_nh = m_orig.permute(0, 2, 1, 3)
        gi = gi_full.unsqueeze(1)

        gw = torch.zeros(B, N, S_q, 1, device=device)
        gs_parts = []
        for kk in range(0, S_kv, tile):
            ke = min(kk+tile, S_kv)
            q_chunk = q_nh[:,:,:,:]  # All heads, all seq positions, all head_dim
            k_chunk = k_2d[kk:ke, :]  # Chunk of K
            score_chunk = F.relu(torch.matmul(q_nh, k_chunk.T.unsqueeze(0).unsqueeze(0)))
            gw += torch.sum(gi[:,:,:,kk:ke] * score_chunk, dim=-1, keepdim=True)
            gi_m = torch.where(m_nh[:,:,:,kk:ke], gi[:,:,:,kk:ke], _zero)
            gs_parts.append(gi_m * w_nh)
        gs = torch.cat(gs_parts, dim=-1)
        gs_flat = gs.reshape(B * N * S_q, S_kv)
        gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
        gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
        return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
    results.append(("31: S_kv tile=%d" % tile, time_op("31: S_kv tile=%d" % tile, exp_tile)))

# 34: Try float16 for ALL matmuls (not just score)
def exp34():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = w_orig.permute(0, 2, 1, 3)
    m_nh = m_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)

    q_flat_h = q_nh.reshape(B * N * S_q, D).half()
    k_2d_h = k_2d.half()
    k_t_h = k_t.half()
    score = F.relu(torch_npu.npu_linear(q_flat_h, k_2d_h).float().reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh

    gs_flat_h = gs.reshape(B * N * S_q, S_kv).half()
    gq = torch_npu.npu_linear(gs_flat_h, k_t_h).float().reshape(B, N, S_q, D)

    q_nh_h = q_nh.half()
    gs_h = gs.half()
    gk = torch.matmul(q_nh_h.transpose(-1, -2), gs_h).float().sum(dim=1).transpose(-1, -2)
    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)

results.append(("34: fp16 all matmuls", time_op("34: fp16 all matmuls", exp34)))

# 35: Different reduction strategies for grad_weights
def exp35():
    q_nh = q_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    # Instead of mul+sum, try matmul trick
    # gw[n,s] = gi[s,:] . score[n,s,:] = [1,K] @ [K,1] per (n,s)
    # Can batch this as bmm
    gi_exp = gi.expand_as(score).reshape(N*S_q, 1, S_kv)
    sc_exp = score.reshape(N*S_q, S_kv, 1)
    gw = torch.bmm(gi_exp, sc_exp).reshape(B, N, S_q, 1)
    return gw
results.append(("35: bmm for grad_weights", time_op("35: bmm for grad_weights", exp35)))

# 36: Use npu_linear for score with half precision k only
def exp36():
    q_nh = q_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    # Standard fp32 npu_linear
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    # Try keepdim=False for reduction (avoid extra dimension)
    gw = torch.sum(gi * score, dim=-1).unsqueeze(-1)
    return gw
results.append(("36: keepdim=False then unsqueeze", time_op("36: keepdim=False then unsqueeze", exp36)))

# 37: Use npu_linear with transposed inputs
def exp37():
    q_nh = q_orig.permute(0, 2, 1, 3)
    w_nh = w_orig.permute(0, 2, 1, 3)
    m_nh = m_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)

    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi * score, dim=-1, keepdim=True)
    del score

    # Try: compute gi * w first, then mask
    giw = gi * w_nh  # [B,N,S,K] broadcast
    gs = torch.where(m_nh, giw, _zero)  # Mask after mul

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
results.append(("37: mul before where (giw then mask)", time_op("37: mul before where (giw then mask)", exp37)))

# 38: torch.baddbmm for grad_k
def exp38():
    q_nh = q_orig.permute(0, 2, 1, 3)
    m_nh = m_orig.permute(0, 2, 1, 3)
    w_nh = w_orig.permute(0, 2, 1, 3)
    gi = gi_full.unsqueeze(1)
    gi_masked = torch.where(m_nh, gi, _zero)
    gs = gi_masked * w_nh
    # baddbmm: out = beta*out + alpha*batch1@batch2
    q_3d = q_nh.reshape(N, S_q, D)
    gs_3d = gs.reshape(N, S_q, S_kv)
    out = torch.zeros(N, D, S_kv, device=device)
    torch.baddbmm(out, q_3d.transpose(-1,-2), gs_3d, beta=0, alpha=1, out=out)
    gk = out.sum(0).unsqueeze(0).transpose(-1,-2)
    return gk
results.append(("38: baddbmm for grad_k", time_op("38: baddbmm for grad_k", exp38)))

# Print summary
print("\n=== Summary ===")
print("%-55s %8s" % ("Experiment", "Time"))
print("-" * 65)
for name, t in results:
    marker = " <-- BEST" if t == min(r[1] for r in results) else ""
    print("%-55s %7.1f ms%s" % (name, t, marker))
