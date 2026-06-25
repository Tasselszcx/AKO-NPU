"""
LM Head Projection with Logit Slicing

来自 EXAONE-4.0-1.2B，hidden 2048 → vocab 102400。
自回归解码时只算最后 logits_to_keep 个位置。

数学定义:
    output = hidden_states @ weight.T
    即: [B, S, 2048] @ [2048, 102400] = [B, S, 102400]
    实际 benchmark 时框架会做末尾切片 [:, -logits_to_keep:, :]
    kernel 实现可以直接只算需要的行，避免冗余计算。

输入:
    hidden_states: [B, S, 2048] bfloat16
    weight: [102400, 2048] bfloat16

输出:
    output: [B, logits_to_keep, 102400] bfloat16

测试 workload shapes:
    B=1, S=128,  logits_to_keep=1     (decode 单 token)
    B=1, S=512,  logits_to_keep=1     (长序列 decode)
    B=1, S=1024, logits_to_keep=128   (prefill 多 token)
    B=4, S=256,  logits_to_keep=1     (batch decode)

精度要求: max_atol=1e-3, max_rtol=0.05

优化机会:
    - logits_to_keep=1 时只需计算最后一行，计算量从 S*V 降到 1*V
    - 大 vocab (102400) 的 matmul，N 维度很大，适合 Cube 单元
    - bfloat16 输入可利用 Cube 的 bf16 算力
    - weight 矩阵可常驻 L1/L2 cache
"""

import torch


def get_inputs(axes_and_scalars, device):
    batch_size = axes_and_scalars["batch_size"]
    seq_len = axes_and_scalars["seq_len"]
    hidden_size = axes_and_scalars["hidden_size"]    # 2048
    vocab_size = axes_and_scalars["vocab_size"]       # 102400

    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.bfloat16, device=device)
    std = 1.0 / (hidden_size ** 0.5)
    weight = torch.randn(
        vocab_size, hidden_size,
        dtype=torch.bfloat16, device=device) * std

    return {"hidden_states": hidden_states, "weight": weight}


@torch.no_grad()
def run(hidden_states, weight):
    # [B, S, 2048] @ [2048, 102400] = [B, S, 102400]
    return torch.matmul(hidden_states, weight.t())


# 测试 workload shapes
TEST_SHAPES = [
    {"batch_size": 1, "seq_len": 128,  "hidden_size": 2048, "vocab_size": 102400},
    {"batch_size": 1, "seq_len": 512,  "hidden_size": 2048, "vocab_size": 102400},
    {"batch_size": 1, "seq_len": 1024, "hidden_size": 2048, "vocab_size": 102400},
    {"batch_size": 4, "seq_len": 256,  "hidden_size": 2048, "vocab_size": 102400},
]

# logits_to_keep: benchmark 框架做末尾切片，kernel 可选择只算这些行
LOGITS_TO_KEEP_MAP = {
    128: 1,      # decode 单 token
    512: 1,      # 长序列 decode
    1024: 128,   # prefill 多 token
    256: 1,      # batch decode
}

# 固定参数
HIDDEN_SIZE = 2048
VOCAB_SIZE = 102400

# 精度要求
MAX_ATOL = 1e-3
MAX_RTOL = 0.05
