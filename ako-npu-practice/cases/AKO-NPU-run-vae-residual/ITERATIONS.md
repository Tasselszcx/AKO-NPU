# Optimization Iterations

## Baseline

| Kernel | Duration (us) | BlockDim | Pct |
|--------|-------------|----------|------|
| im2col_kernel | 13,513 | 48 | 88.7% |
| weight_transpose_kernel | 639 | 32 | 4.2% |
| groupnorm_kernel | 477 | 1 | 3.1% |
| transpose_nhwc2nchw_kernel | 445 | 48 | 2.9% |
| matmul_kernel | 107 | 24 | 0.7% |
| silu_kernel | ~30 | 48 | ~0.2% |
| residual_add_kernel | ~30 | 48 | ~0.2% |

**Total single stage est**: ~15,241 us
**Total full pipeline est**: ~30,300 us (2 stages + residual_add)

**Key bottleneck**: im2col_kernel (88.7%) - transfer granularity ~120B/call, MTE2 bandwidth utilization very low

**Precision**: max_atol=0.0114, PASSED (rtol=0.01, atol=0.01)

---

## Iterations
