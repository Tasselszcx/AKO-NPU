#!/usr/bin/env python3
"""Sweep benchmark across all shapes with current best kernel."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time
import sys

device = 'npu:0'

def run_kernel(grad_is_full, q, k, weights, relu_mask):
    B = q.shape[0]
    S_q = q.shape[1]
    n_heads = q.shape[2]
    head_dim = q.shape[3]
    S_kv = k.shape[1]

    q_nh = q.permute(0, 2, 1, 3)
    w_nh = weights.permute(0, 2, 1, 3)
    m_nh = relu_mask.permute(0, 2, 1, 3)
    grad_is = grad_is_full.unsqueeze(1)

    k_2d = k.reshape(S_kv, head_dim)
    k_t = k_2d.T.contiguous()
    _zero = torch.tensor(0.0, device=device)

    q_flat = q_nh.reshape(B * n_heads * S_q, head_dim)
    score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, n_heads, S_q, S_kv))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    del score

    gi_masked = torch.where(m_nh, grad_is, _zero)
    gs = gi_masked * w_nh

    gs_flat = gs.reshape(B * n_heads * S_q, S_kv)
    gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, n_heads, S_q, head_dim)
    gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

    return gq.permute(0, 2, 1, 3), gk, gw.permute(0, 2, 1, 3)


def benchmark(B, S_q, S_kv, n_heads=64, head_dim=128, n_warmup=3, n_runs=10):
    grad_is = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
    q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.float32, device=device)
    k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
    weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.float32, device=device) * 0.01
    relu_mask = torch.rand(B, S_q, n_heads, S_kv, device=device) > 0.5

    for _ in range(n_warmup):
        run_kernel(grad_is, q, k, weights, relu_mask)
    torch.npu.synchronize()

    times = []
    for _ in range(n_runs):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run_kernel(grad_is, q, k, weights, relu_mask)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    mean_t = sum(times) / len(times)
    min_t = min(times)
    return mean_t, min_t

# Standard shapes
shapes = [
    (1, 1, 4096),
    (1, 4096, 4096),
    (1, 512, 2048),
]

print("=== Shape Sweep ===")
print(f"{'B':>3} {'S_q':>6} {'S_kv':>6} {'mean':>8} {'min':>8}")
print("-" * 40)
for B, S_q, S_kv in shapes:
    mean_t, min_t = benchmark(B, S_q, S_kv)
    print(f"{B:3d} {S_q:6d} {S_kv:6d} {mean_t:8.3f} {min_t:8.3f}")
