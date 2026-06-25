"""
DeepSeek Sparse Attention (DSA) — Indexer 核心计算

来自 deepseek-sparse-attention-pytorch (https://github.com/ZhengKai91/deepseek-sparse-attention-pytorch)
DSA 的核心思想：用一个轻量级 Indexer 预测每个 query 需要关注的 top-k key 位置，
然后只在这些位置上做标准 attention，从而将 O(n²) 复杂度降到 O(n·k)。

本文件提取了 Indexer 的前向计算核心——给定已经投影+RoPE 后的 q、k、weights，
计算 index_score 并选出 topk 位置。这是 DSA 中计算最密集的部分。

数学定义:
    1. index_score = einsum('bsnd,btd->bsnt', q, k)   # [B, S_q, n_heads, S_kv]
    2. index_score = ReLU(index_score)
    3. weighted = index_score * weights                  # 逐头加权
    4. index_score = weighted.sum(dim=2)                 # [B, S_q, S_kv] 合并头
    5. 加 causal mask (上三角 -inf)
    6. topk_indices = topk(index_score, k=index_topk)   # [B, S_q, topk]

输入:
    q: [B, S_q, n_heads, head_dim] bfloat16       — Indexer 的 query（已经过 q_proj + RoPE）
    k: [B, S_kv, head_dim] bfloat16               — Indexer 的 key（已经过 k_proj + RoPE，单头）
    weights: [B, S_q, n_heads, 1] bfloat16         — 每头权重（已经过 w_proj + 缩放）

输出:
    topk_indices: [B, S_q, topk] int64             — 每个 query 选中的 key 位置
    index_score: [B, S_q, S_kv] bfloat16           — 完整的 index 分数（用于训练时的 index loss）

参数:
    n_heads: 64             — Indexer 的注意力头数
    head_dim: 128           — 每头维度
    index_topk: 2048        — 每个 query 选择的 key 数量

测试 workload shapes:
    B=1, S_q=1,    S_kv=4096   (decode: 单 token 查询全 KV cache)
    B=1, S_q=4096, S_kv=4096   (prefill: 全序列自注意力)
    B=1, S_q=1,    S_kv=32768  (长序列 decode)

精度要求:
    topk_indices 必须与 PyTorch 参考完全一致（整数比较）
    index_score 允许 bf16 精度误差：MERE < 2^-7, MARE < 10 * 2^-7

说明:
    这是 DSA 系统中的 Indexer 组件，不是完整的 attention。
    完整的 DSA = Indexer(选位置) + Sparse Attention(在选中位置做 QKV attention)。
    本任务只优化 Indexer 部分。
"""

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars, device):
    B = axes_and_scalars["batch_size"]
    S_q = axes_and_scalars["seq_len_q"]
    S_kv = axes_and_scalars["seq_len_kv"]
    n_heads = 64
    head_dim = 128

    q = torch.randn(B, S_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, S_kv, head_dim, dtype=torch.bfloat16, device=device)
    weights = torch.randn(B, S_q, n_heads, 1, dtype=torch.bfloat16, device=device) * (n_heads ** -0.5) * (head_dim ** -0.5)

    return {"q": q, "k": k, "weights": weights}


@torch.no_grad()
def run(q, k, weights):
    """
    DSA Indexer forward:
    1. einsum: q[B,S_q,n,d] x k[B,S_kv,d] -> score[B,S_q,n,S_kv]
    2. ReLU
    3. weighted sum over heads -> [B,S_q,S_kv]
    4. causal mask
    5. topk
    """
    # Step 1: Compute index scores via einsum
    index_score = torch.einsum("bsnd,btd->bsnt", q.float(), k.float())  # [B, S_q, n_heads, S_kv]

    # Step 2: ReLU activation
    index_score = F.relu(index_score)

    # Step 3: Weighted sum over heads
    weighted = index_score * weights.float()  # [B, S_q, n_heads, S_kv]
    index_score = weighted.sum(dim=2)          # [B, S_q, S_kv]

    # Step 4: Causal mask (上三角置 -inf)
    S_q = index_score.shape[1]
    S_kv = index_score.shape[2]
    if S_q > 1:
        mask = torch.full((S_q, S_kv), float("-inf"), device=index_score.device).triu_(1)
        index_score = index_score + mask

    # Step 5: Top-k selection
    index_topk = 2048
    topk_k = min(index_topk, S_kv)
    topk_indices = index_score.topk(topk_k, dim=-1)[1]  # [B, S_q, topk]

    return topk_indices, index_score.to(torch.bfloat16)


# 测试 workload shapes
TEST_SHAPES = [
    {"batch_size": 1, "seq_len_q": 1,    "seq_len_kv": 4096},    # decode
    {"batch_size": 1, "seq_len_q": 4096, "seq_len_kv": 4096},    # prefill
    {"batch_size": 1, "seq_len_q": 1,    "seq_len_kv": 32768},   # 长序列 decode
]

# 固定参数
N_HEADS = 64
HEAD_DIM = 128
INDEX_TOPK = 2048
