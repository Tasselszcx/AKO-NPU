#!/usr/bin/env python3
"""Parameter sweep: try different configurations systematically."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time
import itertools

device = 'npu:0'

def time_kernel(B, S_q, S_kv, N=64, D=128, n_runs=5, warmup=3):
    q = torch.randn(B, S_q, N, D, device=device)
    k = torch.randn(B, S_kv, D, device=device)
    gi = torch.randn(B, S_q, S_kv, device=device)
    w = torch.randn(B, S_q, N, 1, device=device) * 0.01
    mask = torch.rand(B, S_q, N, S_kv, device=device) > 0.5

    k_2d = k.reshape(S_kv, D)
    k_t = k_2d.T.contiguous()
    _zero = torch.tensor(0.0, device=device)

    def run():
        q_nh = q.permute(0, 2, 1, 3)
        w_nh = w.permute(0, 2, 1, 3)
        m_nh = mask.permute(0, 2, 1, 3)
        grad_is = gi.unsqueeze(1)

        q_flat = q_nh.reshape(B * N * S_q, D)
        score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
        gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
        del score

        gi_masked = torch.where(m_nh, grad_is, _zero)
        gs = gi_masked * w_nh

        gs_flat = gs.reshape(B * N * S_q, S_kv)
        gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
        gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)

        return gq, gk, gw

    for _ in range(warmup):
        run()
    torch.npu.synchronize()

    times = []
    for _ in range(n_runs):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        times.append((t1-t0)*1000)

    return sum(times)/len(times), min(times)

# Sweep shapes
print("=== Shape Sweep ===")
shapes = [
    (1, 1, 4096),
    (1, 4096, 4096),
    (1, 512, 2048),
    (1, 256, 4096),
    (1, 1024, 4096),
    (1, 2048, 4096),
    (1, 128, 1024),
    (1, 64, 512),
]

print("%-5s %-6s %-6s %8s %8s" % ("B", "S_q", "S_kv", "mean", "min"))
print("-" * 40)
for B, S_q, S_kv in shapes:
    try:
        mean_t, min_t = time_kernel(B, S_q, S_kv)
        print("%-5d %-6d %-6d %8.3f %8.3f" % (B, S_q, S_kv, mean_t, min_t))
    except Exception as e:
        print("%-5d %-6d %-6d %s" % (B, S_q, S_kv, str(e)[:50]))

# Sweep n_heads
print("\n=== N_heads Sweep (S_q=4096, S_kv=4096) ===")
for N in [16, 32, 64, 128]:
    try:
        mean_t, min_t = time_kernel(1, 4096, 4096, N=N)
        print("N=%-4d %8.3f ms (min: %8.3f)" % (N, mean_t, min_t))
    except Exception as e:
        print("N=%-4d %s" % (N, str(e)[:50]))

# Sweep head_dim
print("\n=== Head_dim Sweep (S_q=4096, S_kv=4096) ===")
for D in [64, 128, 256]:
    try:
        mean_t, min_t = time_kernel(1, 4096, 4096, D=D)
        print("D=%-4d %8.3f ms (min: %8.3f)" % (D, mean_t, min_t))
    except Exception as e:
        print("D=%-4d %s" % (D, str(e)[:50]))
