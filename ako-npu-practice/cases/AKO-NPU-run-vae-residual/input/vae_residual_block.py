"""
VAE Conv3x3 + GroupNorm + SiLU Residual Fused Block

来自 Sana-Sprint-1.6B (NVIDIA/MIT) 的 VAE encoder/decoder 残差块。

数学定义:
    path1 = SiLU(GroupNorm(Conv3x3(x)))
    path2 = SiLU(GroupNorm(Conv3x3(path1)))
    output = path2 + x  (residual connection)

输入:
    x: [B, 256, H, W] float32
    conv1_weight: [256, 256, 3, 3] float32
    norm1_weight: [256] float32
    norm1_bias: [256] float32
    conv2_weight: [256, 256, 3, 3] float32
    norm2_weight: [256] float32
    norm2_bias: [256] float32
    eps: float (1e-5)

输出:
    output: [B, 256, H, W] float32

测试 workload shapes:
    B=1, H=64,  W=64    (小 shape)
    B=1, H=128, W=128   (中 shape)
    B=2, H=64,  W=64    (batch)

精度要求: max_atol=1e-5, max_rtol=1e-3

融合机会:
    - Conv3x3 + GroupNorm + SiLU 可融合为一个 kernel
    - 两个 path 的中间结果不需要写回 GM
    - Residual add 可融合到第二个 path 的输出
    - GroupNorm 需要 reduction (mean/var)，涉及 Cube + Vector 混合
"""

import torch
import torch.nn.functional as F


def get_inputs(axes_and_scalars, device):
    B = axes_and_scalars["batch_size"]
    H = axes_and_scalars["height"]
    W = axes_and_scalars["width"]
    C = 256
    eps = 1e-5

    x = torch.randn(B, C, H, W, dtype=torch.float32, device=device)
    conv1_weight = torch.randn(C, C, 3, 3, dtype=torch.float32, device=device) * 0.02
    norm1_weight = torch.ones(C, dtype=torch.float32, device=device)
    norm1_bias = torch.zeros(C, dtype=torch.float32, device=device)
    conv2_weight = torch.randn(C, C, 3, 3, dtype=torch.float32, device=device) * 0.02
    norm2_weight = torch.ones(C, dtype=torch.float32, device=device)
    norm2_bias = torch.zeros(C, dtype=torch.float32, device=device)

    return {
        "x": x,
        "conv1_weight": conv1_weight,
        "norm1_weight": norm1_weight,
        "norm1_bias": norm1_bias,
        "conv2_weight": conv2_weight,
        "norm2_weight": norm2_weight,
        "norm2_bias": norm2_bias,
        "eps": eps,
    }


@torch.no_grad()
def run(x, conv1_weight, norm1_weight, norm1_bias,
        conv2_weight, norm2_weight, norm2_bias, eps):
    num_groups = 32
    residual = x

    # Path 1
    out = F.conv2d(x, conv1_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm1_weight, bias=norm1_bias, eps=eps)
    out = F.silu(out)

    # Path 2
    out = F.conv2d(out, conv2_weight, bias=None, stride=1, padding=1)
    out = F.group_norm(out, num_groups, weight=norm2_weight, bias=norm2_bias, eps=eps)
    out = F.silu(out)

    return out + residual


# 测试 workload shapes
TEST_SHAPES = [
    {"batch_size": 1, "height": 64,  "width": 64},
    {"batch_size": 1, "height": 128, "width": 128},
    {"batch_size": 2, "height": 64,  "width": 64},
]

# 固定参数
CHANNELS = 256
NUM_GROUPS = 32
KERNEL_SIZE = 3
EPS = 1e-5

# 精度要求
MAX_ATOL = 1e-5
MAX_RTOL = 1e-3
