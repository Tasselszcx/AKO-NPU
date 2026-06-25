# 全部优化轨迹图

## V5

### DSA Indexer — 9.33x
![DSA-v5](plots/dsa-v5.png)

### DSA Indexer Backward — 3.31x
![DSA-indexer-bwd](plots/dsa-indexer-bwd.png)

### MatMul+LeakyReLU（从 Ascend C 出发）— 6.15x
![MatMul-asc-v5](plots/matmul-asc-v5.png)

### MatMul+LeakyReLU（从 PyTorch 出发）— 4.84x
![MatMul-py-v5](plots/matmul-py-v5.png)

### Attention Backward（torch_npu）— 2.6-3.6x
![Attn-bwd-v5](plots/attn-bwd-v5.png)

## V4

### MatMul+LeakyReLU（从 PyTorch 出发）— 24.6x
![MatMul-py-v4](plots/matmul-py-v4.png)

### Attention Backward — 16.4x
![Attn-bwd-v4](plots/attn-bwd-v4.png)

### MatMul+LeakyReLU（从 Ascend C 出发）— 3.01x
![MatMul-asc-v4](plots/matmul-asc-v4.png)

## V2

### MatMul+LeakyReLU（从 PyTorch 出发）— 9.0x
![MatMul-py](plots/matmul-py.png)

### Attention Backward — 14.7x
![Attn-bwd](plots/attn-bwd.png)

### MatMul+LeakyReLU（从 Ascend C 出发）— 2.1x
![MatMul-asc](plots/matmul-asc.png)

## V3

### LM Head Projection
![LM-head](plots/lm-head.png)

## V1

### Softmax — 1.16x
![Softmax](plots/softmax.png)
