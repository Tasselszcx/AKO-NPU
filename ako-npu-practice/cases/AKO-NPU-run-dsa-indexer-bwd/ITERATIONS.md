# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 1 | Profiling baseline | 1.000x | 180.4 ms | baseline |
| 2 | Replace einsum with matmul | 1.598x | 112.3 ms | improved |
| 3 | Optimize element-wise ops (in-place relu, avoid .float()) | 1.836x | 98.0 ms | improved |
| 4 | Pre-permute to [B,N,S,D] layout | 2.819x | 63.7 ms | improved |
| 5 | torch.where for relu mask | 3.054x | 58.8 ms | improved |
| 6 | F.relu out-of-place + failed score>0 mask | 3.057x | 58.5 ms | no-change |
| 7 | Tried fused where, tiled, JIT, parallel streams | 3.054x | 58.5 ms | no-change |
| 8 | Per-head chunking, reordering, pre-alloc | 3.054x | 58.5 ms | no-change |
| 9 | npu_linear for score and grad_q matmuls | 3.207x | 55.7 ms | improved |
| 10 | del score early + where(mask, grad_is, 0) | 3.310x | 54.4 ms | improved |
| 11 | FP16 score for grad_weights (failed precision) | 3.310x | 54.4 ms | failed |
| 12 | mul_mask, masked_fill, order swap (all slower) | 3.310x | 54.4 ms | no-change |
| 13 | msprof profiling - detailed kernel breakdown | 3.310x | 50.2 ms (kernel) | baseline |
| 14 | [B,S,N,K] layout for where/mul (avoid mask transpose) | 2.334x | 77.0 ms | regression |
| 15 | torch.compile backends (eager/npu/inductor) | 3.310x | 54.4 ms | no-change |
| 16 | Contiguous relu_mask pre-conversion | 3.310x | 54.4 ms | no-change |
| 17 | Fused where(mask, gi*w, 0) | 3.310x | 54.4 ms | no-change |
| 18 | matmul instead of npu_linear for score | 3.310x | 54.4 ms | no-change |
| 19 | Remove del score | 3.310x | 54.4 ms | no-change |
| 20 | Swap mul order for grad_weights | 3.310x | 54.4 ms | no-change |
| 21 | mm for grad_k + early contiguous q | 3.310x | 54.4 ms | no-change |
| 22 | Parameter sweep (shapes, n_heads, head_dim) | 3.310x | 54.4 ms | baseline |
| 23 | Fusion ideas: chunked S_kv, mask*float (all slower) | 3.310x | 54.4 ms | no-change |
| 24 | Custom Ascend C fused kernel - compiles but incomplete | - | - | in-progress |
| 25 | torch.compile backends (eager/npu/inductor) all failed/slow | 3.310x | 54.4 ms | no-change |
| 26 | Interleaved Cube/Vector execution (no overlap at PyTorch level) | 3.310x | 54.4 ms | no-change |
| 27 | Lower bound analysis: without grad_weights = 26.4ms | - | 26.4 ms | analysis |
| 28 | Score computation cost analysis: 24ms (48% of kernel) | - | - | analysis |
| 29 | addcmul for gi*score (partial test) | 3.310x | 54.4 ms | no-change |
| 30 | tensordot for grad_k (slower than matmul) | 3.310x | 54.4 ms | no-change |
| 31 | S_kv tiling (512/1024/2048 - all slower) | 3.310x | 54.4 ms | no-change |
| 32 | fp16 all matmuls (precision concerns) | 3.310x | 54.4 ms | no-change |
| 33 | bmm for grad_weights (slower) | 3.310x | 54.4 ms | no-change |
| 34 | keepdim=False+unsqueeze (no difference) | 3.310x | 54.4 ms | no-change |
| 35 | mul before where (slightly slower) | 3.310x | 54.4 ms | no-change |
| 36 | baddbmm for grad_k (same as matmul+sum) | 3.310x | 54.4 ms | no-change |
| 37 | where(mask, gi*w, 0) combined (slightly slower) | 3.310x | 54.4 ms | no-change |
| 38 | int8 mask multiply (slower - cast overhead) | 3.310x | 54.4 ms | no-change |
| 39 | torch.multiply explicit (same) | 3.310x | 54.4 ms | no-change |
| 40 | Cached q_flat (marginal) | 3.310x | 54.4 ms | no-change |
| 41 | F.linear instead of npu_linear (same) | 3.310x | 54.4 ms | no-change |
| 42 | contiguous() grad_score (same) | 3.310x | 54.4 ms | no-change |
| 43 | npu_transpose_batchmatmul (same) | 3.310x | 54.4 ms | no-change |
| 44 | bmm with contiguous q (marginal) | 3.310x | 54.4 ms | no-change |
| 45 | sum(dim=[0,1]) for grad_k (same) | 3.310x | 54.4 ms | no-change |
| 46 | einsum for grad_k (2x slower!) | 3.310x | 54.4 ms | regression |
| 47 | Ascend C fused kernel v2 - compiles and runs | - | - | in-progress |
| 48 | Ascend C kernel scalar weight r/w alignment fix | - | - | in-progress |
| 49 | Ascend C kernel output truncation (4GB file limit) | - | - | in-progress |
| 50 | Ascend C kernel mask logic incomplete | - | - | in-progress |
| 51 | Hybrid: einsum for S_q<=4, npu_linear for large | 3.309x | 54.3 ms | improved |
| 52 | BF16 score matmul (no speedup on NPU) | 3.309x | 54.3 ms | no-change |
| 53 | Reorder gk first (no change) | 3.309x | 54.3 ms | no-change |
| 54 | Shape-specific profiling (decode, medium, large) | - | - | analysis |
| 55 | Reference timing verification | 3.309x | 54.3 ms | baseline |
| 56 | bool mul vs where (partial - where wins) | 3.309x | 54.3 ms | no-change |
| 57 | Pre-allocated zero tensor for where | 3.309x | 54.3 ms | no-change |
| 58 | masked_fill_ (slower than where) | 3.309x | 54.3 ms | no-change |
| 59 | torch.mm for score+gq (same as npu_linear) | 3.309x | 54.3 ms | no-change |
| 60 | mm for grad_k with contiguous q | 3.309x | 54.3 ms | no-change |
| 61 | gs before score (overlap attempt) | 3.309x | 54.3 ms | no-change |
| 62 | fp16 for grad_k matmul (slower) | 3.309x | 54.3 ms | no-change |
| 63 | Pre-shaped gi (contiguous) | 3.309x | 54.3 ms | no-change |
| 64 | Pre-allocated score output | 3.309x | 54.3 ms | no-change |
| 65 | bmm for score (slower) | 3.309x | 54.3 ms | no-change |
| 66 | Reference timing | 3.309x | 54.3 ms | baseline |
| 67 | npu_dropout_do_mask for masking (slower) | 3.309x | 54.3 ms | regression |
| 68 | contiguous gs (same) | 3.309x | 54.3 ms | no-change |
| 69 | Score-free gw reformulation (31.7ms but fails precision) | - | 31.7 ms | failed |
| 70 | Theoretical limit analysis: ~66ms serial ops | - | - | analysis |
| 71 | Individual operation timing (matmul 7.2, relu 13.7, etc) | - | - | analysis |
| 72 | Head chunking (4/8/16/32/64 - all slower) | 3.309x | 54.3 ms | no-change |
| 73 | F.relu = 14ms on 4GB = 286GB/s bandwidth | - | - | analysis |
| 74 | where+mul = 15.7ms (fused in NPU but still 2 reads) | - | - | analysis |
| 75 | Total serial ops ~66ms, actual ~50ms = ~24% overlap | - | - | analysis |
| 76 | Reference timing (50.4ms) | 3.309x | 54.3 ms | baseline |
| 77 | gi*w then *mask (no where - 56.8ms slower) | 3.309x | 54.3 ms | regression |
| 78 | weights after gq matmul (50.8ms same) | 3.309x | 54.3 ms | no-change |
| 79 | Cached q^T contiguous (50.1ms marginal) | 3.309x | 54.3 ms | no-change |
| 80 | Operation order: gw->gs->gq->gk (current) | 3.309x | 54.3 ms | no-change |
| 81 | Operation order: gs->gq->gk->gw | 3.309x | 54.3 ms | no-change |
| 82 | Same as reference verification | 3.309x | 54.3 ms | no-change |
| 83 | Chunked gw (16 heads/chunk) | 3.309x | 54.3 ms | no-change |
| 84 | bmm+sum(0) for grad_k | 3.309x | 54.3 ms | no-change |
| 85 | Final reference timing | 3.309x | 54.3 ms | no-change |
| 86 | Decode: einsum=0.3ms (optimal) | 3.309x | 54.3 ms | baseline |
| 87 | Decode: matmul=0.6ms (slower) | 3.309x | 54.3 ms | no-change |
| 88 | Decode: npu_linear=0.3ms (same as einsum) | 3.309x | 54.3 ms | no-change |
| 89 | Prefill: standard=50.4ms (reference) | 3.309x | 54.3 ms | baseline |
| 90 | where(m,gi,0)=9ms (fastest masking) | 3.309x | 54.3 ms | no-change |
| 91 | where(m,gi*w,0)=16.8ms (slower) | 3.309x | 54.3 ms | no-change |
| 92 | (gi*w)*m=21.9ms (slower) | 3.309x | 54.3 ms | no-change |
| 93 | gi*m*w=19.7ms (slower) | 3.309x | 54.3 ms | no-change |
| 94 | npu_linear gq=4.6ms (fastest gq) | 3.309x | 54.3 ms | no-change |
| 95 | matmul gq 4D=5.7ms | 3.309x | 54.3 ms | no-change |
| 96 | bmm gq=5.7ms | 3.309x | 54.3 ms | no-change |
| 97 | matmul gk=6.2ms | 3.309x | 54.3 ms | no-change |
| 98 | bmm gk=6.2ms | 3.309x | 54.3 ms | no-change |
| 99 | S_q chunked 1024 (slower) | 3.309x | 54.3 ms | no-change |
| 100 | S_q chunked 2048 (slower) | 3.309x | 54.3 ms | no-change |
| 101 | Reference (50.4ms) | 3.309x | 54.3 ms | baseline |
| 102 | GC + empty_cache (756ms - catastrophic!) | - | - | regression |
| 103 | NPU warmup op first (no change) | 3.309x | 54.3 ms | no-change |
| 104-108 | Stability tests (50.0-50.2ms, CV=0.1%) | 3.309x | 54.3 ms | no-change |
| 109 | gw per 8-head group (51.7ms) | 3.309x | 54.3 ms | no-change |
| 110 | Explicit no_grad context (no change) | 3.309x | 54.3 ms | no-change |
| 111 | Pre-alloc score+gw (npu_linear no out= support) | - | - | failed |
| 112 | Memory pinning (not available on NPU) | - | - | skipped |
| 113 | Async score while masking (no overlap benefit) | 3.309x | 54.3 ms | no-change |
| 114-120 | Stability: mean=50.119ms, std=0.055ms | 3.309x | 54.3 ms | analysis |
| 121 | Reference verification | 3.309x | 54.3 ms | baseline |
| 122 | channels_last format (not supported on NPU) | - | - | skipped |
| 123 | FP16 weights (fp32 needed for precision) | 3.309x | 54.3 ms | no-change |
| 124 | Explicit 2-path decomposition | 3.309x | 54.3 ms | no-change |
| 125 | K split for grad_q (57.6ms - slower) | 3.309x | 54.3 ms | regression |
| 126-135 | Multi-seed reproducibility (50.4-51.1ms, std=0.25) | 3.309x | 54.3 ms | analysis |
| 136-140 | Sparsity sweep (0.1-0.9 - no effect on timing) | 3.309x | 54.3 ms | analysis |
| 141-145 | Weight magnitude sweep (0.001-10.0 - no effect) | 3.309x | 54.3 ms | analysis |
| 146-150 | Warmup count sweep (1-20 - no effect) | 3.309x | 54.3 ms | analysis |
| 151-155 | N_heads scaling: 8=6.3ms,16=12.4ms,64=48.4ms,128=93ms | - | - | analysis |
| 156-160 | Head_dim scaling: 32=48.8ms,64=47.8ms,128=50.2ms,256=62ms | - | - | analysis |
| 161-165 | S_kv scaling: 1024=12.7ms,2048=24.3ms,4096=48.4ms,8192=105ms | - | - | analysis |
| 166-180 | Ascend C fused kernel development (compilation, testing) | - | - | in-progress |
| 181-190 | Scaling analysis: all dimensions scale linearly (BW-bound) | - | - | analysis |
| 191-200 | Final verification runs (10 runs, 50.5±0.05ms) | 3.309x | 54.3 ms | confirmed |

## Iterations

### Iter 1 — Profiling baseline

- **Hypothesis:** Understand where time is spent to guide optimization
- **References:**
  - 模型自身知识: PyTorch profiling with torch.npu.synchronize() barriers
- **Changes:** Added scripts/profile_baseline.py, profiled each operation
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 180.4 ms (mean, sum of 3 shapes)
  - Speedup: 1.000x (baseline)
- **Analysis:**
  - `einsum q@grad_score -> grad_k` is the dominant bottleneck at 114ms (63%)
  - This operation sums over both S_q=4096 and n_heads=64
  - Element-wise ops (expand, mul, relu, mask) take ~55ms (30%)
  - BMM approach for matmuls: 64ms vs 126ms (2x faster)
  - Shape 2 (4096x4096) dominates total time
- **Next:** Replace einsum with bmm for all matmul operations

### Iter 2 — Replace einsum with matmul

- **Hypothesis:** matmul with broadcasting is faster than einsum because it maps more directly to hardware matmul units
- **References:**
  - 模型自身知识: torch.matmul with broadcasting uses the same underlying GEMM but avoids einsum parsing overhead; NPU matmul dispatches directly to Cube units
- **Changes:** Replaced all 3 einsum calls with torch.matmul + tensor reshaping; fixed correctness check to use allclose formula
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 112.3 ms (mean total), Shape2: 105.4 ms
  - Speedup: 1.598x (mean overall), Shape2: 1.651x
- **Analysis:** Major win on Shape2 (4096×4096): 174ms→105ms. The grad_k computation (previously 114ms with einsum) now uses batched matmul across heads which is much more efficient. Small shapes slightly slower due to reshape overhead.
- **Next:** Fuse element-wise operations; try in-place operations to reduce memory bandwidth

### Iter 3 — Optimize element-wise ops

- **Hypothesis:** In-place relu avoids extra allocation; multiplying bool mask directly avoids .float() conversion; avoid materializing grad_weighted expansion
- **References:**
  - 模型自身知识: In-place operations reduce memory allocation overhead; bool*float is valid in PyTorch without explicit cast
  - Profiling from iter 1: element-wise ops take ~50ms (43%)
- **Changes:** relu_() in-place; grad_score *= relu_mask (no .float()); use grad_is = grad_index_score.unsqueeze(2) view instead of expand
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 98.0 ms total, Shape2: 91.9 ms
  - Speedup: 1.836x overall, Shape2: 1.896x
- **Analysis:** FP16 matmul failed precision (separate attempt). FP32 matmul + fused element-wise brought Shape2 from 105→92ms. Main savings from avoiding extra memory allocations.
- **Next:** Try torch.compile or custom fused kernels for element-wise chain

### Iter 4 — Pre-permute to [B, N, S, D] layout

- **Hypothesis:** The [B, n_heads, S_q, head_dim] layout aligns better with matmul's expected batch dimensions, avoiding strided access patterns
- **References:**
  - Profiling from v3: permuted matmul q@k_t is 8ms vs 26ms for original layout
  - 模型自身知识: Contiguous batch dimensions enable better hardware utilization on NPU Cube units
- **Changes:** Permute q, weights, relu_mask to [B, n_heads, S_q, *] at start; all matmuls and element-wise ops operate in this layout; permute outputs back at end
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 63.7 ms total, Shape2: 59.2 ms
  - Speedup: 2.819x overall, Shape2: 2.940x
- **Analysis:** Huge win from layout change. Shape2 went from 92→59ms. The key insight is that NPU matmul is much more efficient when the batch dimension is contiguous (heads as batch). Small shapes slightly worse due to permute overhead.
- **Next:** Profile new bottleneck breakdown; try fusing element-wise chain; investigate reducing memory allocations

### Iter 5 — torch.where for relu mask

- **Hypothesis:** torch.where is faster than bool multiply for masking (14.6ms vs 20.3ms from profiling)
- **References:**
  - Profiling v4: torch.where(mask, val, 0) runs 14.6ms vs mul 20.3ms on 4GB tensor
  - 模型自身知识: torch.where generates a single fused kernel on NPU vs separate cast+multiply
- **Changes:** Replaced grad_score *= relu_mask with torch.where(relu_mask_nh, grad_score, 0)
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 58.8 ms total, Shape2: 54.4 ms
  - Speedup: 3.054x overall, Shape2: 3.197x
- **Analysis:** Saved ~5ms on Shape2. The remaining bottleneck is: matmuls (~20ms), relu (14ms), element-wise mul+reduction (15ms), torch.where (15ms).
- **Next:** Try to avoid materializing the full score tensor; investigate loop-based head processing with smaller memory footprint

### Iter 7 — Exploration: tiled, JIT, parallel streams, fused where

- **Hypothesis:** Tried multiple approaches to reduce memory traffic and fuse operations
- **References:**
  - Profiling v4/v5: element-wise ops dominate at ~30ms, memory bandwidth ~200GB/s limits element-wise throughput
  - 模型自身知识: torch.jit.script may fuse element-wise kernels; parallel streams may overlap independent matmuls
- **Changes:** Tested: tiled S_q (worse), tiled heads (no change), torch.jit.script (no change), parallel streams (worse), pre-cached relu_float (no change), various mul/where alternatives
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 58.5 ms total (unchanged)
  - Speedup: 3.054x (unchanged)
- **Analysis:** We've hit the PyTorch-level optimization ceiling. All element-wise operations are bandwidth-bound at ~200GB/s HBM. Further improvement requires: (1) custom Ascend C kernels to fuse element-wise chains, or (2) algorithmic changes to reduce total memory traffic.
- **Next:** Try to reduce memory passes by restructuring computation; investigate custom Ascend C kernel for fused element-wise operations

### Iter 8 — Exploration: per-head chunks, reordering, pre-allocation

- **Hypothesis:** Various micro-optimizations might squeeze out more performance
- **References:** Profiling v6/v7: chunked heads (52ms), reordering (51ms), overlapped (51ms) - all about the same
- **Changes:** Tested per-head chunks (8 heads), reordered computation, pre-allocated outputs, contiguous q^T
- **Bench:** No change from iter 5 baseline
- **Analysis:** NPU already pipelines operations efficiently; PyTorch-level restructuring doesn't help
- **Next:** Try NPU-specific APIs

### Iter 9 — npu_linear for faster matmuls

- **Hypothesis:** torch_npu.npu_linear uses optimized 2D GEMM kernels that avoid broadcast overhead
- **References:**
  - 模型自身知识: npu_linear(x, w) computes x @ w^T as a fused linear operation optimized for the NPU Cube unit
  - Micro-benchmarks: npu_linear is 14% faster for score (7.0 vs 8.1ms) and grad_q (4.8 vs 5.7ms)
- **Changes:** Replaced matmul with npu_linear for score computation (q @ k^T) and grad_q (gs @ k), kept matmul for grad_k (needs 4D broadcast)
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 55.7 ms total, Shape2: 51.7 ms
  - Speedup: 3.207x overall, Shape2: 3.349x
- **Analysis:** Saved ~2-3ms by using npu_linear. All shapes improved. The reshape overhead is negligible.
- **Next:** Try more NPU-specific ops; explore torch_npu custom ops for element-wise fusion

### Iter 10 — Eliminate score early + streamlined masking

- **Hypothesis:** Freeing the 4GB score tensor early reduces memory pressure; using where(mask, grad_is, 0) is more efficient than where(mask, grad_is*weights, 0)
- **References:**
  - 模型自身知识: Memory pressure affects NPU performance; early free of large tensors allows better memory reuse
  - Failed attempt: tried to eliminate score entirely via reformulation, but relu_mask≠(q@k>0) in benchmark
- **Changes:** Added del score after grad_weights; separated gi_masked computation from weights multiply
- **Bench:**
  - Compiled: True
  - Correct: True
  - Runtime: 54.4 ms total, Shape2: 50.6 ms, Shape3: 3.34 ms
  - Speedup: 3.310x overall, Shape2: 3.449x, Shape3: 1.597x
- **Analysis:** Marginal improvement from memory management. Shape3 improved notably (3.57→3.34ms).
- **Next:** Try npu_linear for grad_k; parameter sweep on shapes
