#!/usr/bin/env python3
"""Generate test data for attention_backward kernel."""

import os
import sys
import struct
import numpy as np

# Parse command-line args
batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4
seq_q = int(sys.argv[2]) if len(sys.argv) > 2 else 256
seq_kv = int(sys.argv[3]) if len(sys.argv) > 3 else 256

print(f"Generating data: batch={batch}, seq_q={seq_q}, seq_kv={seq_kv}")

NUM_HEADS = 80
NUM_KV_HEADS = 8
NUM_GROUPS = 10
HEAD_DIM = 128
DROPOUT_PROB = 0.1

np.random.seed(42)

# Generate inputs in bfloat16
def to_bf16(arr):
    """Convert float32 numpy array to bfloat16 bytes."""
    # numpy doesn't have bfloat16, use struct packing
    f32 = arr.astype(np.float32).flatten()
    # bfloat16 is upper 16 bits of float32
    f32_bytes = f32.tobytes()
    bf16_bytes = bytearray()
    for i in range(0, len(f32_bytes), 4):
        bf16_bytes.extend(f32_bytes[i+2:i+4])  # upper 16 bits
    return bytes(bf16_bytes)

def from_bf16(data, shape):
    """Convert bfloat16 bytes to float32 numpy array."""
    n_elements = 1
    for s in shape:
        n_elements *= s
    f32_bytes = bytearray()
    for i in range(0, len(data), 2):
        f32_bytes.extend(b'\x00\x00')  # lower 16 bits = 0
        f32_bytes.extend(data[i:i+2])  # upper 16 bits from bf16
    return np.frombuffer(bytes(f32_bytes), dtype=np.float32).reshape(shape)

# grad_attn_output: [B, seq_q, 80, 128] bf16
grad_attn_output = np.random.randn(batch, seq_q, NUM_HEADS, HEAD_DIM).astype(np.float32) * 0.01

# attn_weights: [B, 80, seq_q, seq_kv] bf16 (softmax output, rows sum to ~1)
attn_scores_raw = np.random.randn(batch, NUM_HEADS, seq_q, seq_kv).astype(np.float32)
# Apply softmax along last dimension
attn_scores_max = attn_scores_raw.max(axis=-1, keepdims=True)
attn_scores_exp = np.exp(attn_scores_raw - attn_scores_max)
attn_weights = attn_scores_exp / attn_scores_exp.sum(axis=-1, keepdims=True)

# dropout_mask: [B, 80, seq_q, seq_kv] bool (uint8)
dropout_mask = (np.random.rand(batch, NUM_HEADS, seq_q, seq_kv) > DROPOUT_PROB).astype(np.uint8)

# attn_weights_dropped: [B, 80, seq_q, seq_kv] bf16
attn_weights_dropped_f32 = attn_weights * dropout_mask / (1.0 - DROPOUT_PROB)

# value_states: [B, 8, seq_kv, 128] bf16
value_states = np.random.randn(batch, NUM_KV_HEADS, seq_kv, HEAD_DIM).astype(np.float32) * 0.1

# Convert to bf16 and back to get exact bf16 values for golden computation
grad_attn_output_bf16 = from_bf16(to_bf16(grad_attn_output), grad_attn_output.shape)
attn_weights_bf16 = from_bf16(to_bf16(attn_weights), attn_weights.shape)
attn_weights_dropped_bf16 = from_bf16(to_bf16(attn_weights_dropped_f32), attn_weights_dropped_f32.shape)
value_states_bf16 = from_bf16(to_bf16(value_states), value_states.shape)

# ============ Compute golden outputs ============
# 1. Transpose: [B, seq_q, 80, 128] -> [B, 80, seq_q, 128]
grad_out_t = grad_attn_output_bf16.transpose(0, 2, 1, 3).astype(np.float32)

# 2. GQA expand value_states: [B, 8, skv, 128] -> [B, 80, skv, 128]
value_expanded = np.repeat(value_states_bf16, NUM_GROUPS, axis=1).astype(np.float32)

# 3. grad_attn_weights_dropped = grad_out_t @ V^T  [B, 80, sq, skv]
# Use float32 throughout to match kernel's matmul precision
grad_attn_weights_dropped = np.matmul(
    grad_out_t.astype(np.float32),
    value_expanded.astype(np.float32).transpose(0, 1, 3, 2)
).astype(np.float32)

# 4. Dropout backward (float32)
grad_attn_weights = (grad_attn_weights_dropped * dropout_mask.astype(np.float32) / np.float32(1.0 - DROPOUT_PROB)).astype(np.float32)

# 5. Softmax backward (float32)
attn_w_f32 = attn_weights_bf16.astype(np.float32)
product = (grad_attn_weights * attn_w_f32).astype(np.float32)
sum_term = product.sum(axis=-1, keepdims=True).astype(np.float32)
grad_attn_scores = (attn_w_f32 * (grad_attn_weights - sum_term)).astype(np.float32)
# Cast to bf16
grad_attn_scores_bf16 = from_bf16(to_bf16(grad_attn_scores), grad_attn_scores.shape)

# 6. grad_value_expanded = attn_dropped^T @ grad_out_t  [B, 80, skv, 128]
grad_value_expanded = np.matmul(
    attn_weights_dropped_bf16.astype(np.float32).transpose(0, 1, 3, 2),
    grad_out_t.astype(np.float32)
).astype(np.float32)

# 7. GQA aggregate: [B, 80, skv, 128] -> [B, 8, skv, 128]
grad_value_states = grad_value_expanded.reshape(
    batch, NUM_KV_HEADS, NUM_GROUPS, seq_kv, HEAD_DIM
).sum(axis=2)
grad_value_states_bf16 = from_bf16(to_bf16(grad_value_states), grad_value_states.shape)

# ============ Save matmul results for kernel input ============
# mm_result1 = grad_out_t @ V_expanded^T  [B, 80, sq, skv] float32
mm_result1 = grad_attn_weights_dropped  # already computed above
# mm_result2 = attn_dropped^T @ grad_out_t  [B, 80, skv, 128] float32
mm_result2 = grad_value_expanded  # already computed above

# ============ Save files ============
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Save inputs (bf16 binary)
with open("input/grad_attn_output.bin", "wb") as f:
    f.write(to_bf16(grad_attn_output))
with open("input/attn_weights.bin", "wb") as f:
    f.write(to_bf16(attn_weights))
with open("input/attn_weights_dropped.bin", "wb") as f:
    f.write(to_bf16(attn_weights_dropped_f32))
with open("input/value_states.bin", "wb") as f:
    f.write(to_bf16(value_states))
# Pack dropout mask as bitmask (8 elements per byte), row-by-row
# Each row has seq_kv bits, padded to ceil(seq_kv/8) bytes
mask_bytes_per_row = (seq_kv + 7) // 8
total_rows = batch * NUM_HEADS * seq_q
packed_mask = np.zeros(total_rows * mask_bytes_per_row, dtype=np.uint8)
mask_flat = dropout_mask.reshape(total_rows, seq_kv).astype(np.uint8)
for row in range(total_rows):
    for col in range(seq_kv):
        if mask_flat[row, col]:
            byte_idx = row * mask_bytes_per_row + col // 8
            packed_mask[byte_idx] |= (1 << (col % 8))
with open("input/dropout_mask.bin", "wb") as f:
    f.write(packed_mask.tobytes())
print(f"  dropout_mask packed: {dropout_mask.shape} -> {packed_mask.shape[0]} bytes ({mask_bytes_per_row} bytes/row)")
# Save mm_result1 as bf16 to reduce file size (80MB -> 40MB)
with open("input/mm_result1.bin", "wb") as f:
    f.write(to_bf16(mm_result1))
# Save mm_result2 as bf16 to reduce file size (40MB -> 20MB)
with open("input/mm_result2.bin", "wb") as f:
    f.write(to_bf16(mm_result2))

# Save golden outputs (bf16 binary)
with open("output/golden_grad_attn_scores.bin", "wb") as f:
    f.write(to_bf16(grad_attn_scores))
with open("output/golden_grad_value_states.bin", "wb") as f:
    f.write(to_bf16(grad_value_states))

# Also save float32 golden for precise comparison
with open("output/golden_grad_attn_scores_f32.bin", "wb") as f:
    f.write(grad_attn_scores.astype(np.float32).tobytes())
with open("output/golden_grad_value_states_f32.bin", "wb") as f:
    f.write(grad_value_states.astype(np.float32).tobytes())

print(f"Input sizes:")
print(f"  grad_attn_output: {grad_attn_output_bf16.shape} -> {os.path.getsize('input/grad_attn_output.bin')} bytes")
print(f"  attn_weights: {attn_weights_bf16.shape} -> {os.path.getsize('input/attn_weights.bin')} bytes")
print(f"  attn_weights_dropped: {attn_weights_dropped_bf16.shape} -> {os.path.getsize('input/attn_weights_dropped.bin')} bytes")
print(f"  value_states: {value_states_bf16.shape} -> {os.path.getsize('input/value_states.bin')} bytes")
print(f"  dropout_mask: {dropout_mask.shape} -> {os.path.getsize('input/dropout_mask.bin')} bytes")
print(f"Golden output sizes:")
print(f"  grad_attn_scores: {grad_attn_scores.shape}")
print(f"  grad_value_states: {grad_value_states.shape}")
print("Data generation complete.")
