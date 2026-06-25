#!/usr/bin/env python3
"""Batch 9-10: Exotic optimization ideas and stress tests (iter 101-140)."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time
import gc

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
    print("  %-55s %7.1f ms" % (name, sum(times)/len(times)))

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

print("=== Batch 9-10: Exotic ideas ===")
time_fn("101: Reference", ref)

# 102: GC before computation
def exp102():
    gc.collect()
    torch.npu.empty_cache()
    return ref()
time_fn("102: GC + empty_cache first", exp102)

# 103: Warm up NPU with small ops first
def exp103():
    _ = torch.ones(1, device=device) + torch.ones(1, device=device)
    return ref()
time_fn("103: NPU warmup op first", exp103)

# 104-108: Multiple runs to check stability
for i in range(5):
    time_fn("104-%d: stability test run %d" % (i, i), ref)

# 109: Try processing grad_weights per-head sequentially
def exp109():
    gis = gi.unsqueeze(1)
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    # Process gw per head group
    gw = torch.zeros(B, N, S_q, 1, device=device)
    for h in range(0, N, 8):
        he = min(h+8, N)
        q_flat_c = q_nh[:, h:he].reshape(B*(he-h)*S_q, D)
        sc = F.relu(torch_npu.npu_linear(q_flat_c, k_2d).reshape(B, he-h, S_q, S_kv))
        gw[:, h:he] = torch.sum(gis * sc, dim=-1, keepdim=True)
    return gq, gk, gw
time_fn("109: gw per 8-head group", exp109)

# 110: Try using torch.no_grad context
def exp110():
    with torch.no_grad():
        return ref()
time_fn("110: explicit no_grad context", exp110)

# 111-115: Try different memory allocation strategies
def exp111():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    # Pre-allocate all output tensors
    score_buf = torch.empty(B, N, S_q, S_kv, device=device)
    gw_buf = torch.empty(B, N, S_q, 1, device=device)

    torch_npu.npu_linear(q_flat, k_2d, out=score_buf.reshape(B*N*S_q, S_kv))
    score = F.relu(score_buf)
    torch.sum(gis * score, dim=-1, keepdim=True, out=gw_buf)
    del score, score_buf

    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw_buf
try:
    time_fn("111: pre-alloc score+gw", exp111)
except Exception as e:
    print("  111: failed - %s" % str(e)[:60])

# 112: Use CUDA-style memory pinning
# NPU doesn't support this, skip

# 113: Try asynchronous operations
def exp113():
    gis = gi.unsqueeze(1)
    q_flat = q_nh.reshape(B * N * S_q, D)
    # Launch score matmul (async on NPU)
    score_flat = torch_npu.npu_linear(q_flat, k_2d)
    # While matmul is running, prepare masking (async on Vector)
    gi_m = torch.where(m_nh, gis, _zero)
    gs = gi_m * w_nh
    # Now score should be ready
    score = F.relu(score_flat.reshape(B, N, S_q, S_kv))
    gw = torch.sum(gis * score, dim=-1, keepdim=True)
    del score

    gs_flat = gs.reshape(B * N * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw
time_fn("113: async score while masking", exp113)

# 114-120: Stability and reproducibility tests
print("\n=== Stability tests (10 runs) ===")
run_times = []
for i in range(10):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    ref()
    torch.npu.synchronize()
    t1 = time.perf_counter()
    run_times.append((t1-t0)*1000)

import statistics
print("  Mean: %.3f ms" % statistics.mean(run_times))
print("  Std:  %.3f ms" % statistics.stdev(run_times))
print("  Min:  %.3f ms" % min(run_times))
print("  Max:  %.3f ms" % max(run_times))
print("  CV:   %.1f%%" % (statistics.stdev(run_times)/statistics.mean(run_times)*100))

print("\nBatch 9-10 done.")
