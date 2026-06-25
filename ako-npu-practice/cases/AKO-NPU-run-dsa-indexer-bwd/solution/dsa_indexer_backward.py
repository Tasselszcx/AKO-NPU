"""
DSA Indexer Backward - NPU Implementation

Iter 51: Hybrid approach - use einsum for small S_q (decode), npu_linear for large.
"""

import torch
import torch.nn.functional as F
import torch_npu


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
    B = q.shape[0]
    S_q = q.shape[1]
    n_heads = q.shape[2]
    head_dim = q.shape[3]
    S_kv = k.shape[1]

    _zero = torch.tensor(0.0, device=q.device, dtype=q.dtype)

    if S_q <= 4:
        # Small S_q (decode): use einsum which is faster for small batch
        grad_weighted = grad_index_score.unsqueeze(2).expand(-1, -1, n_heads, -1)
        score = torch.einsum("bsnd,btd->bsnt", q, k)
        score = F.relu(score)
        grad_weights = (grad_weighted * score).sum(dim=-1, keepdim=True)
        grad_score = grad_weighted * weights
        grad_score = grad_score * relu_mask
        grad_q = torch.einsum("bsnt,btd->bsnd", grad_score, k)
        grad_k = torch.einsum("bsnd,bsnt->btd", q, grad_score)
        return grad_q, grad_k, grad_weights
    else:
        # Large S_q: optimized matmul path
        q_nh = q.permute(0, 2, 1, 3)
        weights_nh = weights.permute(0, 2, 1, 3)
        relu_mask_nh = relu_mask.permute(0, 2, 1, 3)
        grad_is = grad_index_score.unsqueeze(1)

        k_2d = k.reshape(S_kv, head_dim)
        k_t_for_gq = k_2d.T.contiguous()

        q_flat = q_nh.reshape(B * n_heads * S_q, head_dim)
        score = F.relu(torch_npu.npu_linear(q_flat, k_2d).reshape(B, n_heads, S_q, S_kv))
        grad_weights_nh = torch.sum(grad_is * score, dim=-1, keepdim=True)
        del score

        gi_masked = torch.where(relu_mask_nh, grad_is, _zero)
        grad_score = gi_masked * weights_nh

        gs_flat = grad_score.reshape(B * n_heads * S_q, S_kv)
        grad_q_nh = torch_npu.npu_linear(gs_flat, k_t_for_gq).reshape(B, n_heads, S_q, head_dim)

        grad_k_per_head = torch.matmul(q_nh.transpose(-1, -2), grad_score)
        grad_k = grad_k_per_head.sum(dim=1).transpose(-1, -2)

        return grad_q_nh.permute(0, 2, 1, 3), grad_k, grad_weights_nh.permute(0, 2, 1, 3)


TEST_SHAPES = [
    {"batch_size": 1, "seq_len_q": 1,    "seq_len_kv": 4096},
    {"batch_size": 1, "seq_len_q": 4096, "seq_len_kv": 4096},
    {"batch_size": 1, "seq_len_q": 512,  "seq_len_kv": 2048},
]

N_HEADS = 64
HEAD_DIM = 128
MAX_ATOL = 1e-4
MAX_RTOL = 0.01
