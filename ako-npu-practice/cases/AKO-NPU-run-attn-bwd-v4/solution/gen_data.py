"""Generate test data for attention_backward kernel."""
import numpy as np
import os
import sys

BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) >= 4 else 4
SEQ_LEN_Q = int(sys.argv[2]) if len(sys.argv) >= 4 else 256
SEQ_LEN_KV = int(sys.argv[3]) if len(sys.argv) >= 4 else 256

NUM_ATTENTION_HEADS = 80
NUM_KEY_VALUE_HEADS = 8
NUM_KEY_VALUE_GROUPS = 10
HEAD_DIM = 128
ATTENTION_DROPOUT = 0.1

print(f"Generating data: B={BATCH_SIZE}, sq={SEQ_LEN_Q}, skv={SEQ_LEN_KV}")

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

np.random.seed(42)

def to_bf16(x):
    return (x.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

def from_bf16(x_u16):
    return (x_u16.astype(np.uint32) << 16).view(np.float32)

# Generate inputs
grad_attn_output_f32 = np.random.randn(BATCH_SIZE, SEQ_LEN_Q, NUM_ATTENTION_HEADS, HEAD_DIM).astype(np.float32) * 0.01
grad_attn_output_bf16 = to_bf16(grad_attn_output_f32)

attn_scores_raw = np.random.randn(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN_Q, SEQ_LEN_KV).astype(np.float32)
attn_scores_exp = np.exp(attn_scores_raw - attn_scores_raw.max(axis=-1, keepdims=True))
attn_weights_f32 = attn_scores_exp / attn_scores_exp.sum(axis=-1, keepdims=True)
attn_weights_bf16 = to_bf16(attn_weights_f32)

dropout_mask = (np.random.rand(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN_Q, SEQ_LEN_KV) > ATTENTION_DROPOUT).astype(np.uint8)

attn_weights_dropped_f32 = from_bf16(attn_weights_bf16) * dropout_mask.astype(np.float32) / (1.0 - ATTENTION_DROPOUT)
attn_weights_dropped_bf16 = to_bf16(attn_weights_dropped_f32)

value_states_f32 = np.random.randn(BATCH_SIZE, NUM_KEY_VALUE_HEADS, SEQ_LEN_KV, HEAD_DIM).astype(np.float32) * 0.1
value_states_bf16 = to_bf16(value_states_f32)

# Save inputs
grad_attn_output_bf16.tofile("input/grad_attn_output.bin")
attn_weights_bf16.tofile("input/attn_weights.bin")
attn_weights_dropped_bf16.tofile("input/attn_weights_dropped.bin")
value_states_bf16.tofile("input/value_states.bin")
dropout_mask.tofile("input/dropout_mask.bin")

# Compute golden
grad_out_f32 = from_bf16(grad_attn_output_bf16).reshape(BATCH_SIZE, SEQ_LEN_Q, NUM_ATTENTION_HEADS, HEAD_DIM)
attn_w_f32 = from_bf16(attn_weights_bf16).reshape(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN_Q, SEQ_LEN_KV)
attn_wd_f32 = from_bf16(attn_weights_dropped_bf16).reshape(BATCH_SIZE, NUM_ATTENTION_HEADS, SEQ_LEN_Q, SEQ_LEN_KV)
value_f32 = from_bf16(value_states_bf16).reshape(BATCH_SIZE, NUM_KEY_VALUE_HEADS, SEQ_LEN_KV, HEAD_DIM)

value_expanded = np.repeat(value_f32, NUM_KEY_VALUE_GROUPS, axis=1)
grad_out_t = grad_out_f32.transpose(0, 2, 1, 3).astype(np.float32)

# Matmul-1: use bf16 inputs (matches NPU cube computation)
# The NPU does bf16 * bf16 with f32 accumulation
# Our golden: convert to f32 then matmul (this gives f32 precision multiply, which differs)
# For better match: keep f32 computation (NPU accumulates in f32)
grad_aw_dropped = np.matmul(grad_out_t, value_expanded.transpose(0, 1, 3, 2))

# Dropout backward
grad_aw = grad_aw_dropped * dropout_mask.astype(np.float32) / (1.0 - ATTENTION_DROPOUT)

# Softmax backward
sum_term = (grad_aw * attn_w_f32).sum(axis=-1, keepdims=True)
grad_scores_f32 = attn_w_f32 * (grad_aw - sum_term)

# Cast to bf16 for output
grad_scores_bf16 = to_bf16(grad_scores_f32)

# Matmul-2 + GQA
grad_v_expanded = np.matmul(attn_wd_f32.transpose(0, 1, 3, 2), grad_out_t)
grad_v_reshaped = grad_v_expanded.reshape(BATCH_SIZE, NUM_KEY_VALUE_HEADS, NUM_KEY_VALUE_GROUPS, SEQ_LEN_KV, HEAD_DIM)
grad_v_f32 = grad_v_reshaped.sum(axis=2)
grad_v_bf16 = to_bf16(grad_v_f32)

# Save golden
grad_scores_bf16.tofile("output/golden_grad_attn_scores.bin")
grad_v_bf16.tofile("output/golden_grad_value_states.bin")

print("Generated input and golden data successfully")
