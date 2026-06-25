#!/usr/bin/env python3
"""Generate input data and golden reference for attention backward kernel."""
import os
import sys
import struct
import numpy as np

# Add parent dir to path for input module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from input.attention_backward import get_inputs, run, TEST_SHAPES, MAX_ATOL, MAX_RTOL

import torch

def save_bin(path, tensor):
    """Save tensor as binary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = tensor.contiguous().cpu().numpy().tobytes()
    with open(path, 'wb') as f:
        f.write(data)

def main():
    # Use first test shape for baseline
    shape_idx = int(os.environ.get('SHAPE_IDX', '0'))
    shape = TEST_SHAPES[shape_idx]
    print(f"Generating data for shape: {shape}")

    torch.manual_seed(42)
    device = 'cpu'  # Generate on CPU for golden reference

    inputs = get_inputs(shape, device)

    # Save inputs
    data_dir = os.environ.get('DATA_DIR', './data')
    save_bin(f'{data_dir}/input/grad_attn_output.bin', inputs['grad_attn_output'])
    save_bin(f'{data_dir}/input/attn_weights.bin', inputs['attn_weights'])
    save_bin(f'{data_dir}/input/attn_weights_dropped.bin', inputs['attn_weights_dropped'])
    save_bin(f'{data_dir}/input/value_states.bin', inputs['value_states'])
    save_bin(f'{data_dir}/input/dropout_mask.bin', inputs['dropout_mask'])

    # Save shape info
    with open(f'{data_dir}/shape_info.txt', 'w') as f:
        f.write(f"batch_size={shape['batch_size']}\n")
        f.write(f"seq_len_q={shape['seq_len_q']}\n")
        f.write(f"seq_len_kv={shape['seq_len_kv']}\n")
        f.write(f"num_attention_heads=80\n")
        f.write(f"num_key_value_heads=8\n")
        f.write(f"head_dim=128\n")
        f.write(f"attention_dropout=0.1\n")

    # Compute golden reference
    grad_attn_scores, grad_value_states = run(**inputs)

    save_bin(f'{data_dir}/golden/grad_attn_scores.bin', grad_attn_scores)
    save_bin(f'{data_dir}/golden/grad_value_states.bin', grad_value_states)

    print(f"Data saved to {data_dir}/")
    print(f"  grad_attn_output: {inputs['grad_attn_output'].shape} {inputs['grad_attn_output'].dtype}")
    print(f"  attn_weights: {inputs['attn_weights'].shape} {inputs['attn_weights'].dtype}")
    print(f"  attn_weights_dropped: {inputs['attn_weights_dropped'].shape} {inputs['attn_weights_dropped'].dtype}")
    print(f"  value_states: {inputs['value_states'].shape} {inputs['value_states'].dtype}")
    print(f"  dropout_mask: {inputs['dropout_mask'].shape} {inputs['dropout_mask'].dtype}")
    print(f"  grad_attn_scores (golden): {grad_attn_scores.shape} {grad_attn_scores.dtype}")
    print(f"  grad_value_states (golden): {grad_value_states.shape} {grad_value_states.dtype}")

if __name__ == '__main__':
    main()
