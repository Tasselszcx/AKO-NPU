#!/usr/bin/env python3
"""Generate test data for attention_backward kernel using the reference implementation."""
import sys
import os
import numpy as np

# Add input directory to path for importing the reference implementation
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try multiple possible paths to find input/attention_backward.py
_candidates = [
    os.path.join(SCRIPT_DIR, '..', '..', 'input'),               # solution/scripts -> project_root/input
    os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..', 'input'),  # original ops path
    os.path.join(SCRIPT_DIR, '..', '..', '..', '..', '..', '..', 'input'),
]
INPUT_DIR = None
for c in _candidates:
    p = os.path.abspath(c)
    if os.path.exists(os.path.join(p, 'attention_backward.py')):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    raise FileNotFoundError("Cannot find input/attention_backward.py")
sys.path.insert(0, INPUT_DIR)

import torch
from attention_backward import get_inputs, run

def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seq_q = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    seq_kv = int(sys.argv[3]) if len(sys.argv) > 3 else 256

    print(f"Generating data: batch={batch_size}, seq_q={seq_q}, seq_kv={seq_kv}")

    device = "cpu"
    axes_and_scalars = {
        "batch_size": batch_size,
        "seq_len_q": seq_q,
        "seq_len_kv": seq_kv,
    }

    # Generate inputs
    inputs = get_inputs(axes_and_scalars, device)

    # Run reference
    grad_attn_scores, grad_value_states = run(**inputs)

    # Create directories
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    def save_bf16(tensor, path):
        """Save bf16 tensor as binary file (via uint16 view)."""
        tensor.contiguous().view(torch.uint16).numpy().tofile(path)

    # Save inputs as binary files
    save_bf16(inputs["grad_attn_output"], "input/grad_attn_output.bin")
    save_bf16(inputs["attn_weights"], "input/attn_weights.bin")
    save_bf16(inputs["attn_weights_dropped"], "input/attn_weights_dropped.bin")
    save_bf16(inputs["value_states"], "input/value_states.bin")
    # dropout_mask: [B, 80, seq_q, seq_kv] bool -> uint8
    inputs["dropout_mask"].to(torch.uint8).contiguous().numpy().tofile("input/dropout_mask.bin")

    # Save golden outputs
    save_bf16(grad_attn_scores, "output/golden_grad_attn_scores.bin")
    save_bf16(grad_value_states, "output/golden_grad_value_states.bin")

    print(f"Input shapes:")
    print(f"  grad_attn_output: {list(inputs['grad_attn_output'].shape)}")
    print(f"  attn_weights: {list(inputs['attn_weights'].shape)}")
    print(f"  attn_weights_dropped: {list(inputs['attn_weights_dropped'].shape)}")
    print(f"  value_states: {list(inputs['value_states'].shape)}")
    print(f"  dropout_mask: {list(inputs['dropout_mask'].shape)}")
    print(f"Output shapes:")
    print(f"  grad_attn_scores: {list(grad_attn_scores.shape)}")
    print(f"  grad_value_states: {list(grad_value_states.shape)}")
    print("Data generation complete.")

if __name__ == "__main__":
    main()
