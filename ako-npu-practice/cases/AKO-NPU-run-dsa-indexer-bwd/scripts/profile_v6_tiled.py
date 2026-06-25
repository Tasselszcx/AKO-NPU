#!/usr/bin/env python3
"""Profile tiled approach along S_q dimension."""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import torch_npu
import time

device = "npu:0"
B, S_q, S_kv = 1, 4096, 4096
n_heads, head_dim = 64, 128

q = torch.randn(B, n_heads, S_q, head_dim, dtype=torch.float32, device=device)
k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
weights = torch.randn(B, n_heads, S_q, 1, dtype=torch.float32, device=device) * 0.01
relu_mask = torch.rand(B, n_heads, S_q, S_kv, device=device) > 0.5

k_t = k.transpose(-1, -2).unsqueeze(1)
k_unsq = k.unsqueeze(1)
grad_is = grad_index_score.unsqueeze(1)

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
        times.append((t1 - t0) * 1000)
    mean_t = sum(times) / len(times)
    print(f"  {name}: {mean_t:.3f} ms")
    return mean_t

# Full version (current best)
def run_full():
    score = F.relu(torch.matmul(q, k_t))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    _zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    gs = torch.where(relu_mask, grad_is * weights, _zero)
    gq = torch.matmul(gs, k_unsq)
    gk = torch.matmul(q.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw

time_op("full version", run_full)

# Tiled along S_q - each tile only materializes [B, N, tile_sq, S_kv] score
def run_tiled(tile_sq):
    grad_q = torch.empty_like(q)
    grad_weights = torch.empty(B, n_heads, S_q, 1, device=device)
    grad_k_accum = torch.zeros(B, head_dim, S_kv, device=device)
    _zero = torch.tensor(0.0, device=device, dtype=torch.float32)

    for s in range(0, S_q, tile_sq):
        e = min(s + tile_sq, S_q)
        q_tile = q[:, :, s:e, :]  # [B, N, tile, D]
        grad_is_tile = grad_is[:, :, s:e, :]
        w_tile = weights[:, :, s:e, :]
        mask_tile = relu_mask[:, :, s:e, :]

        # Score tile
        score_tile = F.relu(torch.matmul(q_tile, k_t))  # [B, N, tile, S_kv]

        # grad_weights tile
        grad_weights[:, :, s:e, :] = torch.sum(grad_is_tile * score_tile, dim=-1, keepdim=True)

        # grad_score tile
        gs_tile = torch.where(mask_tile, grad_is_tile * w_tile, _zero)

        # grad_q tile
        grad_q[:, :, s:e, :] = torch.matmul(gs_tile, k_unsq)

        # grad_k accumulation
        gk_tile = torch.matmul(q_tile.transpose(-1, -2), gs_tile)  # [B, N, D, S_kv]
        grad_k_accum += gk_tile.sum(dim=1)

    grad_k = grad_k_accum.transpose(-1, -2)
    return grad_q, grad_k, grad_weights

for ts in [4096, 2048, 1024, 512, 256, 128]:
    time_op(f"tiled S_q (tile={ts})", lambda ts=ts: run_tiled(ts))

# Tiled along n_heads
def run_tiled_heads(tile_nh):
    grad_q_parts = []
    grad_weights_parts = []
    grad_k_accum = torch.zeros(B, head_dim, S_kv, device=device)
    _zero = torch.tensor(0.0, device=device, dtype=torch.float32)

    for h in range(0, n_heads, tile_nh):
        he = min(h + tile_nh, n_heads)
        q_tile = q[:, h:he, :, :]
        w_tile = weights[:, h:he, :, :]
        mask_tile = relu_mask[:, h:he, :, :]

        score_tile = F.relu(torch.matmul(q_tile, k_t))
        gw_tile = torch.sum(grad_is * score_tile, dim=-1, keepdim=True)
        gs_tile = torch.where(mask_tile, grad_is * w_tile, _zero)
        gq_tile = torch.matmul(gs_tile, k_unsq)
        gk_tile = torch.matmul(q_tile.transpose(-1, -2), gs_tile).sum(dim=1)
        grad_k_accum += gk_tile

        grad_q_parts.append(gq_tile)
        grad_weights_parts.append(gw_tile)

    return torch.cat(grad_q_parts, dim=1), grad_k_accum.transpose(-1, -2), torch.cat(grad_weights_parts, dim=1)

for th in [64, 32, 16, 8]:
    time_op(f"tiled heads (tile={th})", lambda th=th: run_tiled_heads(th))
