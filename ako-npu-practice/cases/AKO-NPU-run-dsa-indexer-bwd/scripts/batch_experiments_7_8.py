#!/usr/bin/env python3
"""Batch 7-8: Algorithmic variations and mathematical reformulations (iter 86-120)."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'

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

# Test all 3 shapes with current best
shapes = [
    (1, 1, 4096, "decode"),
    (1, 4096, 4096, "prefill"),
    (1, 512, 2048, "medium"),
]

for B, S_q, S_kv, label in shapes:
    N, D = 64, 128
    q = torch.randn(B, S_q, N, D, device=device)
    k = torch.randn(B, S_kv, D, device=device)
    gi = torch.randn(B, S_q, S_kv, device=device)
    w = torch.randn(B, S_q, N, 1, device=device) * 0.01
    m = torch.rand(B, S_q, N, S_kv, device=device) > 0.5
    k_2d = k.reshape(S_kv, D)
    k_t = k_2d.T.contiguous()
    _zero = torch.tensor(0.0, device=device)

    print("\n=== %s: B=%d, S_q=%d, S_kv=%d ===" % (label, B, S_q, S_kv))

    if S_q <= 4:
        # Decode path
        def run_decode():
            gw_exp = gi.unsqueeze(2).expand(-1, -1, N, -1)
            score = torch.einsum("bsnd,btd->bsnt", q, k)
            score = F.relu(score)
            gw = (gw_exp * score).sum(dim=-1, keepdim=True)
            gs = gw_exp * w
            gs = gs * m
            gq = torch.einsum("bsnt,btd->bsnd", gs, k)
            gk = torch.einsum("bsnd,bsnt->btd", q, gs)
            return gq, gk, gw
        time_fn("86: decode einsum", run_decode)

        # Try matmul for decode
        def run_decode_matmul():
            q_nh = q.permute(0, 2, 1, 3)
            w_nh = w.permute(0, 2, 1, 3)
            m_nh = m.permute(0, 2, 1, 3)
            gis = gi.unsqueeze(1)
            k_t_4d = k.transpose(-1, -2).unsqueeze(1)
            score = F.relu(torch.matmul(q_nh, k_t_4d))
            gw = torch.sum(gis * score, dim=-1, keepdim=True)
            gs = torch.where(m_nh, gis * w_nh, _zero)
            k_unsq = k.unsqueeze(1)
            gq = torch.matmul(gs, k_unsq)
            gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
            return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
        time_fn("87: decode matmul", run_decode_matmul)

        # Try npu_linear for decode
        def run_decode_npu():
            q_nh = q.permute(0, 2, 1, 3)
            w_nh = w.permute(0, 2, 1, 3)
            m_nh = m.permute(0, 2, 1, 3)
            gis = gi.unsqueeze(1)
            q_flat = q_nh.reshape(B*N*S_q, D)
            score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, N, S_q, S_kv))
            gw = torch.sum(gis * score, dim=-1, keepdim=True)
            gs = torch.where(m_nh, gis * w_nh, _zero)
            gs_flat = gs.reshape(B*N*S_q, S_kv)
            gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
            gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
            return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
        time_fn("88: decode npu_linear", run_decode_npu)

    else:
        # Large shapes: test various matmul formulations
        q_nh = q.permute(0, 2, 1, 3)
        w_nh = w.permute(0, 2, 1, 3)
        m_nh = m.permute(0, 2, 1, 3)

        def run_standard():
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

        time_fn("89: standard", run_standard)

        # 90: Try different where formulations
        time_fn("90: where(m, gi, 0)", lambda: torch.where(m_nh, gi.unsqueeze(1), _zero))
        time_fn("91: where(m, gi*w, 0)", lambda: torch.where(m_nh, gi.unsqueeze(1) * w_nh, _zero))
        time_fn("92: (gi*w)*m", lambda: (gi.unsqueeze(1) * w_nh) * m_nh)
        time_fn("93: gi*m*w", lambda: gi.unsqueeze(1) * m_nh * w_nh)

        # 94-98: Matmul variants
        gs = torch.where(m_nh, gi.unsqueeze(1) * w_nh, _zero)
        time_fn("94: npu_linear gq", lambda: torch_npu.npu_linear(gs.reshape(B*N*S_q, S_kv), k_t))
        time_fn("95: matmul gq (4D)", lambda: torch.matmul(gs, k.unsqueeze(1)))
        time_fn("96: bmm gq", lambda: torch.bmm(gs.reshape(N, S_q, S_kv), k.expand(N, S_kv, D)))
        time_fn("97: matmul gk", lambda: torch.matmul(q_nh.transpose(-1,-2), gs).sum(1))
        time_fn("98: bmm gk", lambda: torch.bmm(q_nh.reshape(N,S_q,D).transpose(-1,-2), gs.reshape(N,S_q,S_kv)).sum(0))

        # 99-100: Try reducing memory by processing in S_q chunks
        def exp99():
            gis = gi.unsqueeze(1)
            gw = torch.zeros(B, N, S_q, 1, device=device)
            gs = torch.empty(B, N, S_q, S_kv, device=device)
            sq_chunk = 1024
            for s in range(0, S_q, sq_chunk):
                se = min(s + sq_chunk, S_q)
                q_flat_c = q_nh[:, :, s:se].reshape(B * N * (se-s), D)
                score_c = F.relu(torch_npu.npu_linear(q_flat_c, k_2d).reshape(B, N, se-s, S_kv))
                gw[:, :, s:se] = torch.sum(gis[:, :, s:se] * score_c, dim=-1, keepdim=True)
                del score_c
                gi_m_c = torch.where(m_nh[:, :, s:se], gis[:, :, s:se], _zero)
                gs[:, :, s:se] = gi_m_c * w_nh[:, :, s:se]
            gs_flat = gs.reshape(B * N * S_q, S_kv)
            gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
            gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
            return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
        time_fn("99: S_q chunked (1024)", exp99)

        def exp100():
            gis = gi.unsqueeze(1)
            gw = torch.zeros(B, N, S_q, 1, device=device)
            gs = torch.empty(B, N, S_q, S_kv, device=device)
            sq_chunk = 2048
            for s in range(0, S_q, sq_chunk):
                se = min(s + sq_chunk, S_q)
                q_flat_c = q_nh[:, :, s:se].reshape(B * N * (se-s), D)
                score_c = F.relu(torch_npu.npu_linear(q_flat_c, k_2d).reshape(B, N, se-s, S_kv))
                gw[:, :, s:se] = torch.sum(gis[:, :, s:se] * score_c, dim=-1, keepdim=True)
                del score_c
                gi_m_c = torch.where(m_nh[:, :, s:se], gis[:, :, s:se], _zero)
                gs[:, :, s:se] = gi_m_c * w_nh[:, :, s:se]
            gs_flat = gs.reshape(B * N * S_q, S_kv)
            gq = torch_npu.npu_linear(gs_flat, k_t).reshape(B, N, S_q, D)
            gk = torch.matmul(q_nh.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
            return gq.permute(0,2,1,3), gk, gw.permute(0,2,1,3)
        time_fn("100: S_q chunked (2048)", exp100)

print("\n=== All 3 shapes tested with variants ===")
