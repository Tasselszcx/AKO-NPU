#!/usr/bin/env python3
"""Profile v4 - find remaining bottlenecks."""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch_npu
import time

device = "npu:0"
B, S_q, S_kv = 1, 4096, 4096
n_heads, head_dim = 64, 128

q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.float32, device=device)
k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.float32, device=device) * 0.01
relu_mask = torch.rand(B, S_q, n_heads, S_kv, device=device) > 0.5

q_nh = q.permute(0, 2, 1, 3).contiguous()
weights_nh = weights.permute(0, 2, 1, 3).contiguous()
relu_mask_nh = relu_mask.permute(0, 2, 1, 3).contiguous()
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

print("=== V4 step-by-step profiling ===\n")

t1 = time_op("permute q", lambda: q.permute(0, 2, 1, 3))
t2 = time_op("permute weights", lambda: weights.permute(0, 2, 1, 3))
t3 = time_op("permute relu_mask", lambda: relu_mask.permute(0, 2, 1, 3))

t4 = time_op("matmul score=q@k_t", lambda: torch.matmul(q_nh, k_t))
score = torch.matmul(q_nh, k_t)

t5 = time_op("relu_(score)", lambda: score.clone().relu_())
score.relu_()

t6 = time_op("grad_score=grad_is*weights", lambda: grad_is * weights_nh)
t7 = time_op("(grad_is*score).sum(-1,keepdim)", lambda: (grad_is * score).sum(dim=-1, keepdim=True))

grad_score = (grad_is * weights_nh) * relu_mask_nh
t8 = time_op("grad_score*=relu_mask", lambda: (grad_is * weights_nh).__imul__(relu_mask_nh))

t9 = time_op("matmul grad_q=gs@k", lambda: torch.matmul(grad_score, k_unsq))
t10 = time_op("matmul grad_k=q^T@gs", lambda: torch.matmul(q_nh.transpose(-1,-2), grad_score))
t11 = time_op("sum+transpose grad_k", lambda: torch.matmul(q_nh.transpose(-1,-2), grad_score).sum(dim=1).transpose(-1,-2))

t12 = time_op("permute grad_q back", lambda: torch.matmul(grad_score, k_unsq).permute(0,2,1,3))
t13 = time_op("permute grad_w back", lambda: (grad_is * score).sum(dim=-1, keepdim=True).permute(0,2,1,3))

total = t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11+t12+t13
print(f"\nEstimated total: {total:.3f} ms")

# Try contiguous permute
print("\n=== Contiguous permute vs view ===")
time_op("q.permute().contiguous()", lambda: q.permute(0,2,1,3).contiguous())
time_op("q.permute() (view only)", lambda: q.permute(0,2,1,3))

# Try addmm / baddbmm
print("\n=== Alternative matmul approaches ===")
# grad_k accumulation: can we use a single matmul instead of matmul+sum?
# Reshape: q[B,N,S,D] -> [B*N, S, D], gs[B,N,S,K] -> [B*N, S, K]
q_flat = q_nh.reshape(B*n_heads, S_q, head_dim)
gs_flat = grad_score.reshape(B*n_heads, S_q, S_kv)
time_op("bmm q_flat^T@gs_flat", lambda: torch.bmm(q_flat.transpose(-1,-2), gs_flat))
time_op("reshape+sum for grad_k", lambda: torch.bmm(q_flat.transpose(-1,-2), gs_flat).reshape(B, n_heads, head_dim, S_kv).sum(1).transpose(-1,-2))

# Element-wise fusion: can we compute grad_score and apply mask in one go?
print("\n=== Fused element-wise ===")
time_op("grad_is*weights*relu_mask (fused mul chain)", lambda: grad_is.mul(weights_nh).mul_(relu_mask_nh))
time_op("torch.where(mask, grad_is*w, 0)", lambda: torch.where(relu_mask_nh, grad_is * weights_nh, torch.tensor(0.0, device=device)))

# Skip score recomputation: what if we fuse score+relu into one?
print("\n=== matmul+relu fusion ===")
time_op("matmul+relu separate", lambda: torch.matmul(q_nh, k_t).relu_())
# torch.nn.functional doesn't have fused matmul+relu, but check if NPU optimizes it automatically
