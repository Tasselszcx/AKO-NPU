"""
MatMul + LeakyReLU 融合算子

数学定义:
    output = LeakyReLU(MatMul(A, B) + bias, alpha=0.001)

输入:
    A: [1024, 256] float16
    B: [256, 640]  float16
    bias: [640]    float32

输出:
    output: [1024, 640] float32
"""

import torch
import torch.nn.functional as F

def matmul_leakyrelu(a, b, bias, alpha=0.001):
    output = torch.matmul(a.float(), b.float()) + bias
    return F.leaky_relu(output, negative_slope=alpha)

M, K, N = 1024, 256, 640
ALPHA = 0.001
