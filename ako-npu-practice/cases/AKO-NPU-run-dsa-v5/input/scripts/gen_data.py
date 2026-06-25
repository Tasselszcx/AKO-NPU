#!/usr/bin/env python3
"""Generate test data and golden output for DSA Indexer."""

import numpy as np
import os
import sys

# Fixed parameters
N_HEADS = 64
HEAD_DIM = 128
INDEX_TOPK = 2048

def generate_data(B, S_q, S_kv, seed=42):
    np.random.seed(seed)

    # Generate inputs in bf16 (use float32 then convert)
    q_fp32 = np.random.randn(B, S_q, N_HEADS, HEAD_DIM).astype(np.float32) * 0.1
    k_fp32 = np.random.randn(B, S_kv, HEAD_DIM).astype(np.float32) * 0.1
    w_fp32 = np.random.randn(B, S_q, N_HEADS, 1).astype(np.float32) * 0.5

    # Simulate bf16 precision by round-tripping through bfloat16
    # numpy doesn't have native bf16, so we do manual conversion via uint16
    def to_bf16_bytes(arr):
        """Convert float32 array to bfloat16 bytes (truncate lower 16 bits)."""
        raw = arr.view(np.uint32)
        bf16 = ((raw >> 16) & 0xFFFF).astype(np.uint16)
        return bf16

    def bf16_to_float(bf16_uint16):
        """Convert bfloat16 uint16 array back to float32."""
        raw32 = bf16_uint16.astype(np.uint32) << 16
        return raw32.view(np.float32)

    # Convert to bf16 and back to float for reference computation
    q_bf16 = to_bf16_bytes(q_fp32)
    k_bf16 = to_bf16_bytes(k_fp32)
    w_bf16 = to_bf16_bytes(w_fp32)

    q_ref = bf16_to_float(q_bf16)
    k_ref = bf16_to_float(k_bf16)
    w_ref = bf16_to_float(w_bf16)

    return q_bf16, k_bf16, w_bf16, q_ref, k_ref, w_ref


def compute_golden(q_ref, k_ref, w_ref, B, S_q, S_kv):
    """Compute golden output using reference implementation."""
    topk_k = min(INDEX_TOPK, S_kv)

    # Allocate outputs
    all_topk_indices = np.zeros((B, S_q, topk_k), dtype=np.int64)
    all_index_score = np.zeros((B, S_q, S_kv), dtype=np.float32)

    for b in range(B):
        for s in range(S_q):
            # Step 1: einsum('nd,td->nt', q[b,s], k[b])
            # q[b,s]: [N_HEADS, HEAD_DIM], k[b]: [S_kv, HEAD_DIM]
            # score = q[b,s] @ k[b].T -> [N_HEADS, S_kv]
            score = q_ref[b, s] @ k_ref[b].T  # [N_HEADS, S_kv]

            # Step 2: ReLU
            score = np.maximum(score, 0.0)

            # Step 3: Weighted sum over heads
            # w[b,s]: [N_HEADS, 1], broadcast multiply then sum
            weighted = score * w_ref[b, s]  # [N_HEADS, S_kv]
            index_score = weighted.sum(axis=0)  # [S_kv]

            # Step 4: Causal mask (only for prefill S_q > 1)
            if S_q > 1:
                # Query position s only sees key positions [0, s]
                index_score[s+1:] = float('-inf')

            # Step 5: TopK
            topk_idx = np.argsort(index_score)[::-1][:topk_k]  # descending order
            all_topk_indices[b, s] = topk_idx.astype(np.int64)
            all_index_score[b, s] = index_score

    # Convert index_score to bf16 for output comparison
    raw32 = all_index_score.view(np.uint32)
    bf16 = ((raw32 >> 16) & 0xFFFF).astype(np.uint16)
    index_score_bf16 = bf16

    return all_topk_indices, index_score_bf16, all_index_score


def main():
    B = 1
    S_q = 1
    S_kv = 4096

    if len(sys.argv) >= 4:
        B = int(sys.argv[1])
        S_q = int(sys.argv[2])
        S_kv = int(sys.argv[3])

    print(f"Generating data: B={B}, S_q={S_q}, S_kv={S_kv}")

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    q_bf16, k_bf16, w_bf16, q_ref, k_ref, w_ref = generate_data(B, S_q, S_kv)

    # Save inputs as binary
    q_bf16.tofile("input/input_q.bin")
    k_bf16.tofile("input/input_k.bin")
    w_bf16.tofile("input/input_weights.bin")

    print(f"  input_q.bin: {q_bf16.shape} uint16 ({q_bf16.nbytes} bytes)")
    print(f"  input_k.bin: {k_bf16.shape} uint16 ({k_bf16.nbytes} bytes)")
    print(f"  input_weights.bin: {w_bf16.shape} uint16 ({w_bf16.nbytes} bytes)")

    # Compute golden
    topk_indices, index_score_bf16, index_score_f32 = compute_golden(
        q_ref, k_ref, w_ref, B, S_q, S_kv)

    topk_k = min(INDEX_TOPK, S_kv)

    # Save golden outputs
    topk_indices.tofile("output/golden_topk_indices.bin")
    index_score_bf16.tofile("output/golden_index_score.bin")
    # Also save float32 version for debugging
    index_score_f32.astype(np.float32).tofile("output/golden_index_score_f32.bin")

    print(f"  golden_topk_indices.bin: {topk_indices.shape} int64 ({topk_indices.nbytes} bytes)")
    print(f"  golden_index_score.bin: {index_score_bf16.shape} uint16 ({index_score_bf16.nbytes} bytes)")

    # Print some stats
    print(f"\n  index_score range: [{index_score_f32[index_score_f32 != float('-inf')].min():.4f}, "
          f"{index_score_f32[index_score_f32 != float('-inf')].max():.4f}]")
    print(f"  top-5 indices: {topk_indices[0, 0, :5]}")
    print(f"  top-5 values: {index_score_f32[0, 0, topk_indices[0, 0, :5]]}")


if __name__ == "__main__":
    main()
