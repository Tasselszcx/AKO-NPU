#!/usr/bin/env python3
"""Test ideas for fusing element-wise operations."""
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'
B, N, S_q, S_kv, D = 1, 64, 4096, 4096, 128

q_nh = torch.randn(B, N, S_q, D, device=device)
k_2d = torch.randn(S_kv, D, device=device)
k_t = k_2d.T.contiguous()
grad_is = torch.randn(B, 1, S_q, S_kv, device=device)
weights_nh = torch.randn(B, N, S_q, 1, device=device) * 0.01
relu_mask_nh = torch.rand(B, N, S_q, S_kv, device=device) > 0.5
_zero = torch.tensor(0.0, device=device)

score = F.relu(torch_npu.npu_linear(q_nh.reshape(N*S_q, D), k_2d).reshape(B, N, S_q, S_kv))

def time_op(name, fn, n=10, warmup=5):
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
    print("  %-50s %7.1f ms (min: %7.1f)" % (name, sum(times)/len(times), min(times)))

print("=== Current element-wise ops (separate) ===")
time_op("gi*score sum (gw)", lambda: torch.sum(grad_is * score, dim=-1, keepdim=True))
time_op("where(mask, gi, 0)", lambda: torch.where(relu_mask_nh, grad_is, _zero))
time_op("gi_masked * weights", lambda: torch.where(relu_mask_nh, grad_is, _zero) * weights_nh)
time_op("total separate", lambda: (
    torch.sum(grad_is * score, dim=-1, keepdim=True),
    torch.where(relu_mask_nh, grad_is, _zero) * weights_nh
))

# Idea: Can we express both outputs as a single operation?
# grad_score = where(mask, gi, 0) * w = mask * gi * w
# grad_weights = (gi * score).sum(-1)
# These share the same grad_is input but use different operations

# Idea: Process all S_kv in parallel for each (n, s) position
# For each position (n, s):
#   gw[n,s] = sum_k(gi[s,k] * score[n,s,k])
#   gs[n,s,k] = mask[n,s,k] ? gi[s,k] * w[n,s] : 0

# Can we combine using a single matmul trick?
# grad_weights[n,s] = gi[s,:] . score[n,s,:] = diag(gi) @ score^T [per (n,s)]

# What if we process in chunks of S_kv?
print("\n=== Chunked S_kv ===")
def chunked_skv(chunk_size):
    gw_accum = torch.zeros(B, N, S_q, 1, device=device)
    gs_parts = []
    for k_start in range(0, S_kv, chunk_size):
        k_end = min(k_start + chunk_size, S_kv)
        gi_chunk = grad_is[:, :, :, k_start:k_end]
        sc_chunk = score[:, :, :, k_start:k_end]
        mask_chunk = relu_mask_nh[:, :, :, k_start:k_end]
        w = weights_nh

        gw_accum += torch.sum(gi_chunk * sc_chunk, dim=-1, keepdim=True)
        gs_chunk = torch.where(mask_chunk, gi_chunk, _zero) * w
        gs_parts.append(gs_chunk)

    return gw_accum, torch.cat(gs_parts, dim=-1)

for cs in [4096, 2048, 1024, 512]:
    time_op("chunked S_kv (chunk=%d)" % cs, lambda cs=cs: chunked_skv(cs))

# Try using torch_npu custom op
print("\n=== torch_npu specific ops ===")
# npu_masked_fill_range might help?
try:
    # gi * score where mask is true, 0 otherwise
    # = gi * score * mask.float()
    mask_float = relu_mask_nh.float()
    time_op("gi*score*mask_float (for gw)", lambda: torch.sum(grad_is * score * mask_float, dim=-1, keepdim=True))
    time_op("gi*mask_float*w (for gs)", lambda: grad_is * mask_float * weights_nh)
except Exception as e:
    print("  Error: %s" % e)

# What about using addcmul for fused multiply-add?
print("\n=== addcmul / addmm ===")
# gw = (gi * score).sum(-1) = sum of element-wise product
# Can we use torch.inner / torch.tensordot?
gi_2d = grad_is.reshape(S_q, S_kv)
sc_ns = score.reshape(N*S_q, S_kv)
# gw[n,s] = gi[s,:] . score[n,s,:]
# = torch.mm(score.reshape(N*S, K), gi.reshape(S, K).T) but this gives [N*S, S] not what we want
# We need element-wise dot product, not matrix product

# What about using the fact that gi broadcasts?
# gw[n,s] = sum_k(gi[s,k] * score[n,s,k])
# If we reshape score to [N, S*K] and gi to [1, S*K]:
# No, that doesn't preserve the per-(s,k) pairing after reshape

# The only practical fusion is via a custom kernel or torch.compile(npu)
print("\nConclusion: element-wise fusion requires custom kernel or working torch.compile(npu)")
