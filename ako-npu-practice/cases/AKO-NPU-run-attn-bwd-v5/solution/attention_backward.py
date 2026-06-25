"""
Attention Softmax+Dropout+Value Matmul Backward

反向传播路径：
1. transpose grad
2. grad w.r.t. value_states（matmul backward）
3. grad through dropout
4. grad through softmax
5. GQA head aggregation

输入:
    grad_attn_output: [B, seq_q, 80, 128] bfloat16
    attn_weights: [B, 80, seq_q, seq_kv] bfloat16
    attn_weights_dropped: [B, 80, seq_q, seq_kv] bfloat16
    value_states: [B, 8, seq_kv, 128] bfloat16
    dropout_mask: [B, 80, seq_q, seq_kv] bool
    attention_dropout: 0.1

输出:
    grad_attn_scores: [B, 80, seq_q, seq_kv] bfloat16
    grad_value_states: [B, 8, seq_kv, 128] bfloat16

测试 workload shapes:
    batch=4,  seq_q=256,  seq_kv=256   (小 shape)
    batch=8,  seq_q=373,  seq_kv=449   (非对齐 shape)
    batch=4,  seq_q=1024, seq_kv=2048  (长序列)

精度要求: max_atol=1e-5, max_rtol=0.05
"""

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars, device):
    batch_size = axes_and_scalars["batch_size"]
    seq_len_q = axes_and_scalars["seq_len_q"]
    seq_len_kv = axes_and_scalars["seq_len_kv"]
    num_attention_heads = 80
    num_key_value_heads = 8
    head_dim = 128
    attention_dropout = 0.1

    grad_attn_output = torch.randn(
        batch_size, seq_len_q, num_attention_heads, head_dim,
        dtype=torch.bfloat16, device=device)

    attn_scores_raw = torch.randn(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv,
        dtype=torch.float32, device=device)
    attn_weights = torch.softmax(attn_scores_raw, dim=-1).to(torch.bfloat16)

    dropout_mask = torch.rand(
        batch_size, num_attention_heads, seq_len_q, seq_len_kv,
        device=device) > attention_dropout

    attn_weights_dropped = (
        attn_weights.float() * dropout_mask / (1.0 - attention_dropout)
    ).to(torch.bfloat16)

    value_states = torch.randn(
        batch_size, num_key_value_heads, seq_len_kv, head_dim,
        dtype=torch.bfloat16, device=device)

    return {
        "grad_attn_output": grad_attn_output,
        "attn_weights": attn_weights,
        "attn_weights_dropped": attn_weights_dropped,
        "value_states": value_states,
        "dropout_mask": dropout_mask,
        "attention_dropout": attention_dropout,
    }


@torch.no_grad()
def run(grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout):
    """
    反向传播路径：
    1. transpose grad
    2. grad w.r.t. value_states（matmul backward）
    3. grad through dropout
    4. grad through softmax
    5. GQA head aggregation
    """
    num_attention_heads = 80
    num_key_value_heads = 8
    num_key_value_groups = num_attention_heads // num_key_value_heads  # = 10

    batch_size = grad_attn_output.shape[0]
    seq_len_q = grad_attn_output.shape[1]
    seq_len_kv = value_states.shape[2]
    head_dim = value_states.shape[3]

    # GQA expand value states: [B, 8, kv, d] -> [B, 80, kv, d]
    value_states_expanded = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
    ).reshape(batch_size, num_attention_heads, seq_len_kv, head_dim)

    # 1. Transpose: (B, seq_q, heads, d) -> (B, heads, seq_q, d)
    grad_out_t = grad_attn_output.transpose(1, 2).to(torch.float32)

    # 2. grad_attn_weights_dropped = grad_out @ V^T
    grad_attn_weights_dropped = torch.matmul(
        grad_out_t,
        value_states_expanded.to(torch.float32).transpose(-2, -1)
    )

    # 3. grad through dropout: multiply by mask / (1 - p)
    grad_attn_weights = (
        grad_attn_weights_dropped * dropout_mask / (1.0 - attention_dropout)
    )

    # 4. grad through softmax:
    # grad_input = softmax * (grad_output - sum(grad_output * softmax))
    attn_w_f32 = attn_weights.to(torch.float32)
    sum_term = (grad_attn_weights * attn_w_f32).sum(dim=-1, keepdim=True)
    grad_attn_scores = attn_w_f32 * (grad_attn_weights - sum_term)
    grad_attn_scores = grad_attn_scores.to(torch.bfloat16)

    # 5. grad w.r.t. value_states_expanded = attn_dropped^T @ grad_out
    grad_value_expanded = torch.matmul(
        attn_weights_dropped.to(torch.float32).transpose(-2, -1),
        grad_out_t
    )

    # GQA aggregate: [B, 8, 10, kv, d] -> sum over groups -> [B, 8, kv, d]
    grad_value_states = grad_value_expanded.reshape(
        batch_size, num_key_value_heads, num_key_value_groups, seq_len_kv, head_dim
    ).sum(dim=2).to(torch.bfloat16)

    return grad_attn_scores, grad_value_states


# 测试 workload shapes
TEST_SHAPES = [
    {"batch_size": 4, "seq_len_q": 256,  "seq_len_kv": 256},   # 小 shape
    {"batch_size": 8, "seq_len_q": 373,  "seq_len_kv": 449},   # 非对齐 shape
    {"batch_size": 4, "seq_len_q": 1024, "seq_len_kv": 2048},  # 长序列
]

# 精度要求
MAX_ATOL = 1e-5
MAX_RTOL = 0.05

# 固定参数
NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
ATTENTION_DROPOUT = 0.1
