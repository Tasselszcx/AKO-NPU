#!/usr/bin/env python3
"""Batch 5: NPU-specific ops exploration and more ideas (iter 66-85)."""
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

print("=== Batch 5: NPU ops and more ===")
time_fn("66: Reference", ref)

# 67: npu_dropout_do_mask for masking
def exp67():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    # dropout_do_mask applies a mask and scales - might be faster than where+mul
    # npu_dropout_do_mask(input, mask, prob) -> input * mask / (1 - prob)
    # We want: gi * mask, which is dropout with prob=0 and no scaling
    try:
        # Convert bool mask to uint8 as expected
        mask_bytes = m_nh.to(torch.uint8)
        gi_m = torch_npu.npu_dropout_do_mask(gis.expand_as(m_nh), mask_bytes, 0.0)[0]
        gs = gi_m * w_nh
    except:
        gi_m = torch.where(m_nh, gis, _zero)
        gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("67: npu_dropout_do_mask", exp67)

# 68: Try different memory layouts for intermediate tensors
def exp68():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    # Make gs contiguous before matmuls
    gs = gs.contiguous()
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("68: contiguous gs before matmul", exp68)

# 69-73: Score computation with different approaches
# 69: Skip score entirely and use a proxy
def exp69():
    gis = gi.unsqueeze(1)
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    # Approximate grad_weights using gi_masked @ k then dot with q
    gi_flat = gi_m.reshape(B * N * S_q, S_kv)
    temp = torch_npu.npu_linear(gi_flat, k_t).reshape(B, N, S_q, D)
    gw = torch.sum(q_nh * temp, dim=-1, keepdim=True)
    return gq, gk, gw
time_fn("69: avoid score via reformulation (approx gw)", exp69)

# 70: Check if gi_masked @ k reformulation matches reference
# This won't match because relu_mask != (q@k > 0) with random mask
# But let's measure the speed
print("  (Note: exp69 gw will fail correctness with random mask)")

# 71: Use scatter_add for sparse mask processing
# 72: Try torch.sparse operations
# For ~50% sparsity, these won't help

# 73: Measure theoretical limits
print("\n=== Theoretical limits ===")
# Memory bandwidth: ~200 GB/s HBM
# Score tensor: 4GB read + write = 8GB -> 40ms at 200GB/s
# Element-wise ops read ~12GB total (gi, score, mask, weights, gs) -> 60ms
# But ops can overlap and reuse cache

# 74: Time each major category with fresh allocation
def time_score():
    q_flat = q_nh.reshape(B * N * S_q, D)
    return torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv)
time_fn("74a: npu_linear score matmul only", time_score)

score = F.relu(torch_npu.npu_linear(q_nh.reshape(B*N*S_q, D), k_2d).reshape(B, N, S_q, S_kv))
time_fn("74b: F.relu only", lambda: F.relu(score.clone()))
time_fn("74c: gi*score sum only", lambda: torch.sum(gi.unsqueeze(1) * score, dim=-1, keepdim=True))
time_fn("74d: where only", lambda: torch.where(m_nh, gi.unsqueeze(1), _zero))
time_fn("74e: where * w", lambda: torch.where(m_nh, gi.unsqueeze(1), _zero) * w_nh)

gs = torch.where(m_nh, gi.unsqueeze(1), _zero) * w_nh
time_fn("74f: npu_linear gq only", lambda: torch_npu.npu_linear(gs.reshape(B*N*S_q, S_kv), k_t))
time_fn("74g: matmul gk only", lambda: torch.matmul(q_nh.transpose(-1,-2), gs).sum(dim=1))

# 75-80: Test different TILE_LENGTH for score computation in Ascend C
# (Already done in batch 3, but let's be more systematic)
for chunk_heads in [4, 8, 16, 32, 64]:
    def exp_chunk(ch=chunk_heads):
        gis = gi.unsqueeze(1)
        gw = torch.zeros(B, N, S_q, 1, device=device)
        gs_parts = []

        for h in range(0, N, ch):
            he = min(h + ch, N)
            q_chunk = q_nh[:, h:he]
            w_chunk = w_nh[:, h:he]
            m_chunk = m_nh[:, h:he]

            q_flat_c = q_chunk.reshape(B * (he-h) * S_q, D)
            score_c = F.relu(torch_npu.npu_linear(q_flat_c, k_2d).reshape(B, he-h, S_q, S_kv))
            gw[:, h:he] = torch.sum(gis * score_c, dim=-1, keepdim=True)
            del score_c

            gi_m_c = torch.where(m_chunk, gis, _zero)
            gs_c = gi_m_c * w_chunk
            gs_parts.append(gs_c)

        gs = torch.cat(gs_parts, dim=1)
        gs_flat = gs.reshape(B * N * S_q, S_kv)
        gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
        gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
        return gq, gk, gw

    time_fn("chunk_heads=%d" % chunk_heads, exp_chunk)

print("\nAll done.")
