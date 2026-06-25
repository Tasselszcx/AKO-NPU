"""
DeepSeek Sparse Attention (DSA) — Indexer Backward

DSA Indexer forward 的反向传播 kernel。给定上游梯度 grad_index_score，
对 q、k、weights 三个输入求梯度。

Forward 回顾:
    1. score = einsum('bsnd,btd->bsnt', q, k)     [B, S_q, n_heads, S_kv]
    2. score = ReLU(score)
    3. weighted = score * weights                    [B, S_q, n_heads, S_kv]
    4. index_score = weighted.sum(dim=2)             [B, S_q, S_kv]
    5. causal mask + topk（不可导，topk 的梯度通过 scatter 传递）

Backward 推导:
    4→ grad_weighted = grad_index_score.unsqueeze(2).expand_as(weighted)
    3→ grad_score = grad_weighted * weights
       grad_weights = (grad_weighted * score).sum(dim=-1, keepdim=True)
    2→ grad_score = grad_score * (score > 0).float()           (ReLU mask)
    1→ grad_q = einsum('bsnt,btd->bsnd', grad_score, k)
       grad_k = einsum('bsnd,bsnt->btd', q, grad_score)      (sum over heads)

输入:
    grad_index_score: [B, S_q, S_kv] float32         — 上游梯度
    q: [B, S_q, n_heads, head_dim] float32            — forward 时保存的 q
    k: [B, S_kv, head_dim] float32                    — forward 时保存的 k
    weights: [B, S_q, n_heads, 1] float32             — forward 时保存的 weights
    relu_mask: [B, S_q, n_heads, S_kv] bool           — forward 时 score > 0 的 mask

输出:
    grad_q: [B, S_q, n_heads, head_dim] float32
    grad_k: [B, S_kv, head_dim] float32
    grad_weights: [B, S_q, n_heads, 1] float32

参数:
    n_heads: 64
    head_dim: 128

测试 workload shapes:
    B=1, S_q=1,    S_kv=4096    (decode backward)
    B=1, S_q=4096, S_kv=4096    (prefill backward)
    B=1, S_q=512,  S_kv=2048    (中等)

精度要求: max_atol=1e-4, max_rtol=0.01

优化机会:
    - 两个 einsum（grad_q 和 grad_k）是 batched matmul，用 Cube 单元
    - ReLU mask 是 element-wise，用 Vector 单元
    - grad_weights 是 reduction（sum over S_kv），用 ReduceSum
    - forward 保存的中间结果（score, relu_mask）需要从 GM 读取
"""

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars, device):
    B = axes_and_scalars["batch_size"]
    S_q = axes_and_scalars["seq_len_q"]
    S_kv = axes_and_scalars["seq_len_kv"]
    n_heads = 64
    head_dim = 128

    grad_index_score = torch.randn(B, S_q, S_kv, dtype=torch.float32, device=device)
    q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.float32, device=device)
    k = torch.randn(B, S_kv, head_dim, dtype=torch.float32, device=device)
    weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.float32, device=device) * 0.01

    # Forward 中间结果：score > 0 的 mask（ReLU 的反向需要）
    # 模拟：约 50% 的 score 为正
    relu_mask = torch.rand(B, S_q, n_heads, S_kv, device=device) > 0.5

    return {
        "grad_index_score": grad_index_score,
        "q": q,
        "k": k,
        "weights": weights,
        "relu_mask": relu_mask,
    }


@torch.no_grad()
def run(grad_index_score, q, k, weights, relu_mask):
    """
    DSA Indexer backward:
    反向传播 grad_index_score → grad_q, grad_k, grad_weights
    """
    # Step 4 backward: sum(dim=2) 的反向是 unsqueeze + expand
    grad_weighted = grad_index_score.unsqueeze(2).expand(
        -1, -1, q.shape[2], -1
    )  # [B, S_q, n_heads, S_kv]

    # Step 3 backward: element-wise mul
    # 需要 forward 的 score（但我们只有 relu_mask，用 q@k 重新计算）
    score = torch.einsum("bsnd,btd->bsnt", q, k)
    score = F.relu(score)

    grad_score = grad_weighted * weights        # [B, S_q, n_heads, S_kv]
    grad_weights = (grad_weighted * score).sum(dim=-1, keepdim=True)  # [B, S_q, n_heads, 1]

    # Step 2 backward: ReLU mask
    grad_score = grad_score * relu_mask.float()

    # Step 1 backward: einsum
    grad_q = torch.einsum("bsnt,btd->bsnd", grad_score, k)         # [B, S_q, n_heads, head_dim]
    grad_k = torch.einsum("bsnd,bsnt->btd", q, grad_score)         # [B, S_kv, head_dim]

    return grad_q, grad_k, grad_weights


# 测试 workload shapes
TEST_SHAPES = [
    {"batch_size": 1, "seq_len_q": 1,    "seq_len_kv": 4096},
    {"batch_size": 1, "seq_len_q": 4096, "seq_len_kv": 4096},
    {"batch_size": 1, "seq_len_q": 512,  "seq_len_kv": 2048},
]

# 固定参数
N_HEADS = 64
HEAD_DIM = 128

# 精度要求
MAX_ATOL = 1e-4
MAX_RTOL = 0.01
