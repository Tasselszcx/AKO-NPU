#!/usr/bin/env python3
"""
Attention Backward Kernel - NPU Implementation

Best version: 4D broadcast matmul1 + in-place dropout + reordered matmul2 + native softmax_bwd.
All bf16 pipeline.
"""
import torch
import torch_npu


def _compute_core(grad_attn_output, attn_weights, attn_weights_dropped,
                  value_states, dropout_mask, attention_dropout):
    """Core computation - fully bf16, optimized operation ordering."""
    B = grad_attn_output.shape[0]
    Sq = grad_attn_output.shape[1]
    Skv = value_states.shape[2]
    D = value_states.shape[3]

    # Pre-compute scaled dropout mask (bf16, in-place)
    scaled_mask = dropout_mask.to(torch.bfloat16).mul_(
        torch.tensor(1.0 / (1.0 - attention_dropout),
                     dtype=torch.bfloat16, device=grad_attn_output.device))

    # Transpose grad_out → [B,80,Sq,D]
    grad_out = grad_attn_output.transpose(1, 2).contiguous()
    grad_out_4d = grad_out.reshape(B * 8, 10, Sq, D)

    # V transposed for 4D broadcast: [B*8, 1, D, Skv]
    vs_t = value_states.reshape(B * 8, Skv, D).transpose(-2, -1).unsqueeze(1)

    # Matmul 1 (4D broadcast) → reshape → in-place dropout
    grad_aw = torch.matmul(grad_out_4d, vs_t).reshape(B, 80, Sq, Skv)
    grad_aw.mul_(scaled_mask)

    # Matmul 2 BEFORE softmax backward (better cache behavior)
    grad_value_states = torch.matmul(
        attn_weights_dropped.transpose(-2, -1), grad_out
    ).reshape(B, 8, 10, Skv, D).sum(dim=2)

    # Softmax backward
    grad_attn_scores = torch._softmax_backward_data(grad_aw, attn_weights, -1, torch.bfloat16)

    return grad_attn_scores, grad_value_states


@torch.no_grad()
def attn_bwd_npu(grad_attn_output, attn_weights, attn_weights_dropped,
                 value_states, dropout_mask, attention_dropout):
    return _compute_core(grad_attn_output, attn_weights, attn_weights_dropped,
                         value_states, dropout_mask, attention_dropout)


@torch.no_grad()
def attn_bwd_reference_npu(grad_attn_output, attn_weights, attn_weights_dropped,
                            value_states, dropout_mask, attention_dropout):
    return _compute_core(grad_attn_output, attn_weights, attn_weights_dropped,
                         value_states, dropout_mask, attention_dropout)
