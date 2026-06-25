#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch_npu
import time

device = 'npu:0'
B, N, S_q, S_kv, D = 1, 64, 4096, 4096, 128

q = torch.randn(B, N, S_q, D, device=device)
k = torch.randn(B, S_kv, D, device=device)
grad_is = torch.randn(B, 1, S_q, S_kv, device=device)
weights = torch.randn(B, N, S_q, 1, device=device) * 0.01
relu_mask = torch.rand(B, N, S_q, S_kv, device=device) > 0.5
k_t = k.transpose(-1, -2).unsqueeze(1)
k_unsq = k.unsqueeze(1)

def time_op(name, fn, n=5, warmup=5):
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
    print(f'  {name}: {sum(times)/len(times):.3f} ms')

@torch.jit.script
def fused_kernel(grad_is: torch.Tensor, q: torch.Tensor, k_t: torch.Tensor,
                 k_unsq: torch.Tensor, weights: torch.Tensor,
                 relu_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score = F.relu(torch.matmul(q, k_t))
    grad_weights = torch.sum(grad_is * score, dim=-1, keepdim=True)
    _zero = torch.tensor(0.0, device=grad_is.device, dtype=grad_is.dtype)
    grad_score = torch.where(relu_mask, grad_is * weights, _zero)
    grad_q = torch.matmul(grad_score, k_unsq)
    grad_k_per_head = torch.matmul(q.transpose(-1, -2), grad_score)
    grad_k = grad_k_per_head.sum(dim=1).transpose(-1, -2)
    return grad_q, grad_k, grad_weights

time_op('jit full kernel', lambda: fused_kernel(grad_is, q, k_t, k_unsq, weights, relu_mask))

def nojit_kernel():
    score = F.relu(torch.matmul(q, k_t))
    gw = torch.sum(grad_is * score, dim=-1, keepdim=True)
    _zero = torch.tensor(0.0, device=device, dtype=torch.float32)
    gs = torch.where(relu_mask, grad_is * weights, _zero)
    gq = torch.matmul(gs, k_unsq)
    gk = torch.matmul(q.transpose(-1, -2), gs).sum(dim=1).transpose(-1, -2)
    return gq, gk, gw

time_op('non-jit kernel', nojit_kernel)
