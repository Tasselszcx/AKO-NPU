#!/usr/bin/env python3
"""Batch 4: Systematic sweep of remaining ideas (iter 55-80)."""
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

print("=== Batch 4: Systematic sweep ===")
time_fn("55: Reference", ref)

# 56: Use torch.select for mask instead of where
time_fn("56: gi * m_nh (bool mul)", lambda: gi.unsqueeze(1) * m_nh)
time_fn("57: torch.where(m, gi, 0)", lambda: torch.where(m_nh, gi.unsqueeze(1), _zero))
time_fn("58: torch.masked_select", lambda: torch.where(m_nh, gi.unsqueeze(1) * w_nh, _zero))

# 59-62: Different tensor creation for _zero
_zero_expanded = torch.zeros(B, N, S_q, S_kv, device=device)
time_fn("59: where with pre-allocated zero", lambda: torch.where(m_nh, gi.unsqueeze(1), _zero_expanded))

# 60: Use select/scatter patterns
time_fn("60: masked_fill_(~m, 0)", lambda: (gi.unsqueeze(1).expand_as(m_nh).clone()).masked_fill_(~m_nh, 0.0))

# 61-65: matmul precision/mode variations
def exp61():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    # Try torch.mm instead of npu_linear for score
    score = F.relu(torch.mm(q_flat, k_2d.T).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch.mm(gs_flat, k_2d).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("61: torch.mm for score+gq", exp61)

# 62: Use npu_linear for grad_k also
def exp62():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    # For grad_k: need contiguous q then mm
    q_c = q_nh.contiguous().reshape(N * S_q, D)
    gs_c = gs.reshape(N * S_q, S_kv)
    gk = torch.mm(q_c.T, gs_c).unsqueeze(0).transpose(-1, -2)
    return gq, gk, gw
time_fn("62: mm for grad_k", exp62)

# 63: Double buffer style processing
def exp63():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)

    # Compute grad_score first (doesn't need score)
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh

    # Now compute score (might overlap on NPU if Vector and Cube are independent)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("63: gs before score (overlap)", exp63)

# 64: Use half-precision for non-critical outputs
def exp64():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    # Compute grad_k with half precision for intermediate
    q_h = q_nh.half()
    gs_h = gs.half()
    gk = torch.matmul(q_h.transpose(-1, -2), gs_h).float().sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("64: fp16 for grad_k matmul", exp64)

# 65: Avoid unsqueeze by pre-shaping
gi_preshaped = gi.unsqueeze(1).contiguous()
def exp65():
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gi_preshaped * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gi_preshaped, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("65: pre-shaped gi", exp65)

# 66-70: Score computation alternatives
# 66: Can we compute score row-by-row to reduce peak memory?
# 67: Use addmm for score with zero bias
def exp67():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    out = torch.empty(B * N * S_q, S_kv, device=device)
    torch.mm(q_flat, k_2d.T, out=out)
    score = F.relu(out.reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score, out
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("67: pre-alloc score output", exp67)

# 68: Use npu_bmmV2 for a 3D batch formulation
q_3d = q_nh.reshape(N, S_q, D).contiguous()
k_3d = k_2d.T.unsqueeze(0).expand(N, D, S_kv).contiguous()
def exp68():
    gis = gi.unsqueeze(1)
    try:
        score = F.relu(torch_npu.npu_bmmV2(q_3d, k_3d, []).reshape(B, N, S_q, S_kv))
    except:
        score = F.relu(torch.bmm(q_3d, k_3d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("68: bmm for score", exp68)

print("\nAll experiments done.")
