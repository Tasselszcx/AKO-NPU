# Iteration Log

## Benchmark Shape
batch=4, seq_q=256, seq_kv=256

## Summary

| Iter | Title | Duration(us) | Speedup | Status |
|------|-------|-------------|---------|--------|
| baseline | Pure Vector implementation | 162,841 | 1.00x | baseline |
| 1 | Batch Cast + merge DMA + reduce barriers | 146,270 | 1.11x | improved |
| 2 | Attempted Matmul API (Cube) — reverted to v6 | 146,329 | 1.11x | no change (exploration) |
| 3 | __mix__ mode with Matmul API for MM1+MM2 | 1,723 | 94.51x | **major improvement** |
| 4 | Batch vector ops in softmax bwd + reduce buffers | 1,390 | 117.17x | improved |
| 5 | Pattern ReduceSum AR for batch row sums | 1,121 | 145.36x | improved |

## Baseline Analysis
- **Task Duration: 162,841 us (162.84 ms)**
- Block Dim: 24 (B*kv_heads=32 tasks, 8 cores idle)
- Primary bottleneck: SCALAR Bound (avg 68.58%)
- VEC ratio: avg 41.47%
- vec_wait: 12.25% (~45 PipeBarrier<PIPE_ALL>)
- GM→UB BW usage: 0.70%
- L2 Cache hit: 93.9%

### Key bottleneck sources:
1. **SCALAR 68.58%**: Row-by-row dot product for matmul → massive scalar loop control overhead
2. **Core imbalance**: 32 tasks / 24 cores → 8 cores do 2 tasks, 16 cores do 1 task, rest idle
3. **PipeBarrier<PIPE_ALL>**: ~45 barriers killing pipeline parallelism
4. **Low BW utilization**: GM BW < 1% → not bandwidth-bound

## Iterations

### Iter 1 — Batch Cast + merge DMA + reduce PipeBarriers

- **Hypothesis:** Reducing per-row PipeBarrier calls and batching Cast operations will lower scalar overhead
- **Changes:**
  1. Batch Cast in MM1 Phase 2 (entire B tile at once instead of per-row)
  2. Merged 3 DMA loads in Phase 3 into parallel group with single barrier
  3. Merged DMA loads in MM2 Phase 4
  4. Removed ~500 per-row PipeBarrier<PIPE_ALL> from softmax backward
- **Bench:**
  - Compiled: True
  - Correct: True (allclose=100%, max_rtol_nz=0.0081)
  - Runtime: 146,270 us
  - Speedup: 1.11x
- **Analysis:** 10.2% improvement. SCALAR ratio still dominant (72%). Core issue remains: row-by-row dot product requires SetValue/GetValue scalar ops (65,536 per head per matmul).
- **Next:** Try Matmul high-level API or ReduceSum AR pattern with proper TQue buffers to eliminate scalar matmul loops.

### Iter 2 — Attempted Matmul API (Cube) Integration

- **Hypothesis:** Replacing Vector dot-product matmul with Cube Matmul API would eliminate ~65,536 scalar SetValue/GetValue ops per head and dramatically reduce SCALAR bottleneck.
- **Attempted strategies:**
  1. **Single `__aicore__` kernel with Matmul API** — Failed: `__aicore__` runs on AIV only, Matmul API needs AIC (Cube core). Got AICORE_EXCEPTION (error 507015).
  2. **Single `__mix__` kernel** — Not attempted due to UB buffer sharing complexity: REGIST_MATMUL_OBJ reserves UB for Matmul, pipe.InitBuffer reserves UB for Vector, and they conflict when used in the same kernel.
  3. **Two-kernel approach (`__cube__` + `__aicore__`)** — Compiled but hung at runtime. Issues: (a) workspace argument needs `__kfc_workspace__` annotation for GetSysWorkSpacePtr() to work, (b) workspace mapping between cube and vector kernels is complex (different core assignments, different task granularity), (c) sequential dependency between phases means all 10 GQA heads share the same workspace slot.
- **Outcome:** Reverted to v6 Vector implementation. Performance unchanged (146,329 us ≈ 146,270 us).
- **Bench:**
  - Compiled: True
  - Correct: True (allclose=100%, max_rtol_nz=0.0145)
  - Runtime: 146,329 us
  - Speedup: 1.11x (same as iter-1)
- **Analysis:** SCALAR ratio remains 71.68%. The core issue (scalar-bound Vector matmul) persists. Matmul API integration requires deeper platform knowledge about:
  - `__mix__` mode UB partitioning between Matmul and user Vector buffers
  - System workspace management with `__kfc_workspace__`
  - Re-Init safety limits (docs say max 4 Matmul objects, but Init calls are separate)
- **Next directions:**
  1. Try `__mix__` mode with explicit UB partitioning via SetBufferSpace (limit Matmul UB, leave rest for Vector)
  2. Consider pure-Cube multi-launch approach with per-head workspace isolation
  3. Alternative: Optimize Vector matmul further — reduce scalar ops by pre-computing results in larger vector tiles

### Iter 3 — __mix__ mode with Matmul API for MM1+MM2

- **Hypothesis:** Using `__mix__(1, 1)` mode with Matmul high-level API for both MM1 and MM2 would eliminate the scalar-bound Vector matmul entirely, replacing it with hardware Cube engine execution.
- **Changes:**
  1. Changed kernel to `__mix__(1, 1)` mode (simultaneous AIC Cube + AIV Vector)
  2. Added `-DHAVE_WORKSPACE -DHAVE_TILING` to CMakeLists.txt for proper workspace management
  3. Registered 2 Matmul objects (mm1 and mm2) with `REGIST_MATMUL_OBJ` and separate tilings
  4. MM1: bf16 A[sq,128] @ B[skv,128]^T → float C[sq,skv] via `IterateAll` to GM workspace
  5. MM2: bf16 A[sq,skv]^T @ B[sq,128] → float C[skv,128] via `IterateAll` to GM workspace
  6. Vector phases (dropout bwd, softmax bwd, GQA accumulation) use 64KB UB buffer (Matmul takes the rest)
  7. Host generates matmul tilings with `MatmulApiTiling` (single-core per-head computation)
  8. Phase 1 (grad_out transpose) kept as-is
  9. Eliminated scalar dot-product loops (~65,536 SetValue/GetValue per head) — now handled by Cube engine
- **Key learnings:**
  - `__kfc_workspace__` annotation not available in CANN 8.3.RC1 — use `-DHAVE_WORKSPACE -DHAVE_TILING` instead
  - `__mix__(1, 1)` allows 24 blocks (one per AI Core); `__mix__(1, 2)` would limit to 12 blocks
  - `REGIST_MATMUL_OBJ` must be before `pipe.InitBuffer` (docs confirm)
  - TCubeTiling must be copied from GM to stack for kernel-side use
  - Matmul IterateAll + End can be called repeatedly in a loop with different tensor pointers
- **Bench:**
  - Compiled: True
  - Correct: True (allclose=100%, max_rtol_nz=0.0182)
  - Runtime: 1,722.91 us
  - Speedup: **94.51x** vs baseline, **84.91x** vs iter-1
- **Analysis:** Massive 94.5x improvement! Task duration dropped from 146ms to 1.7ms.
  - SCALAR ratio: 32.38% avg (down from 72%)
  - Cube ratio: 1.03% avg — Cube is being used but at low utilization
  - vec_fp32: 9.2%, vec_misc: 6.6%
  - cube_fops: 436,907/core — Cube is active
  - L2 Cache hit: 42.1% (lower than before — new memory access pattern)
  - Core imbalance: 48 total blocks (24 AIC + 24 AIV), but aic_time varies from 0 to 1720us
  - Head overhead: 2.63us (0.2%) — negligible
- **Next directions:**
  1. Optimize Vector phases (dropout/softmax bwd still uses scalar ReduceSum with GetValue)
  2. Tune Matmul tiling parameters (baseM, baseN)
  3. Consider per-head task split (320 tasks) vs per-kv_head (32 tasks) for better load balance
  4. Pipeline Matmul and Vector phases for overlap

### Iter 4 — Batch vector ops in softmax bwd + reduce buffer count

- **Hypothesis:** Batching element-wise vector operations (Mul, Muls, DataCopy) across all rows within a tile instead of per-row loops will reduce scalar loop overhead and improve vector unit utilization. Additionally, reducing the number of UB buffers (removing separate maskH and outBf16 buffers) increases tileSq from 13 to 17 rows, reducing tile iterations from 20 to 15.
- **Changes:**
  1. Replaced 5 per-row loop bodies (DataCopy, Cast, Cast, Mul, Muls) with batched operations:
     - Batched `DataCopy(gradW, mm1In, totalElems)` for all rows at once
     - Per-row Cast (uint8->half->float) still needed due to type conversion constraints
     - Batched `Mul(gradW, gradW, attnWf32, totalElems)` across all rows
     - Batched `Muls(gradW, gradW, DROPOUT_SCALE, totalElems)` across all rows
  2. Separated DMA loads: load mm1In + maskIn first, process dropout backward, then load attnWBf16 separately (avoids double-load of bf16 weights)
  3. Reused mm1In buffer as half temp during mask conversion (after DataCopy to gradW, mm1In is free)
  4. Removed separate outBf16 buffer — reuse attnWBf16 for output
  5. Batched final `Mul(gradW, attnWf32, gradW, totalElems)` across all rows
  6. Increased tileSq from 13 to 17 by reducing per-row buffer footprint from 4864 to 3844 bytes
- **Bench:**
  - Compiled: True
  - Correct: True (allclose=100%, max_rtol_nz=0.031)
  - Runtime: 1,389.72 us
  - Speedup: **117.17x** vs baseline, **1.24x** vs iter-3
- **Analysis:** 19.3% improvement over iter 3.
  - MTE2 instructions: 28,432 (down 15% from 33,552) — fewer DMA ops from larger tiles
  - Avg MTE2 transfer: 8.50KB (up from 7.20KB) — more efficient transfers
  - UB read BW: 16.9 GB/s (up from 13.7) — better vector throughput
  - UB write BW: 11.7 GB/s (up from 9.4) — improved write path
  - AIC scalar_ratio still 32.39% (unchanged — Matmul API internal overhead)
  - Core imbalance persists: 16 cores active (~1387us), 8 idle (32 tasks / 24 cores = 2 tasks/core for first 16 only)
  - Cube ratio: 1.28% (up slightly from 1.03%) — marginally better Cube utilization
  - cube_wait: 3.52% (up from 2.83%) — more time waiting for Cube
- **Next directions:**
  1. Task granularity: split per-head (B*numHeads=320 tasks) to use all 24 cores
  2. Pipeline overlap between Matmul and Vector phases
  3. Tune Matmul tiling (baseM/baseN) for this workload size
  4. Consider increasing vecBufSize if Matmul can work with less UB

### Iter 5 — Pattern ReduceSum AR for batch row sums

- **Hypothesis:** Replacing per-row Level 2 ReduceSum calls with a single Pattern::Reduce::AR ReduceSum call per tile will eliminate the per-row reduction overhead (each call internally iterates over 256 elements), reducing AIV scalar ratio and improving vector throughput.
- **Changes:**
  1. Replaced per-row `ReduceSum<float>(sumBuf, mm1In[row], redTmp, skv)` loop (16 calls/tile) with single `ReduceSum<float, Pattern::Reduce::AR, true>(sumBuf, mm1In, patternRedTmp, srcShape, true)` call per tile
  2. Pattern ReduceSum AR computes all row sums at once, writing results to sumBuf[0..cSq-1]
  3. Kept per-row `Adds(gradW[r], gradW[r], -sumBuf.GetValue(r), skv)` loop since each row needs its own sum value
  4. Changed redTmp (Level 2 float type) to patternRedTmp (uint8_t type as required by Pattern form 1)
  5. Allocated 2048 bytes for Pattern ReduceSum tmp buffer; tileSq remains 16 (was 17) due to larger fixed overhead, but ceil(256/16)=16 tiles = same as ceil(256/17)=16
  6. Used `isReuseSource=true` to allow source tensor (mm1In) to be overwritten — safe since mm1In is not used after the reduce
- **Bench:**
  - Compiled: True
  - Correct: True (allclose=100%, max_rtol_nz=0.0353)
  - Runtime: 1,120.52 us
  - Speedup: **145.36x** vs baseline, **1.24x** vs iter-4
- **Analysis:** 19.4% improvement over iter-4.
  - UB read BW: 23.1 GB/s (up from 16.9) — +37% better vector throughput
  - UB write BW: 17.1 GB/s (up from 11.7) — +46% better write throughput
  - vec_fops: 5,783,957/core (up slightly from 5,725,931)
  - AIV scalar_ratio max: ~31% (similar), but AIV time on active cores dropped from ~1370us to ~1100us
  - AIC scalar_ratio: ~98% (unchanged — Matmul API internal)
  - Core imbalance persists: 16 active cores (~1118us), 8 nearly idle (~15us)
  - cube_wait: 4.51% (up from 3.52%)
  - MTE2 instructions unchanged (28,432) — same number of DMA operations
  - GM→UB BW: 4.28 GB/s (up from 3.45) — faster overall due to shorter kernel time
- **Next directions:**
  1. Task granularity: split per-head (B*numHeads=320 tasks) to use all 24 cores
  2. Pipeline overlap between Matmul and Vector phases
  3. Reduce per-row Cast loop overhead (mask cast, attnW cast, output cast)
  4. Tune Matmul tiling or increase vecBufSize for larger tiles

