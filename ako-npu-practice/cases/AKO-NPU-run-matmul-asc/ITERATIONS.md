# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 0 | Baseline | 1.00x | 227.9 us | baseline |
| 1 | Adjust tile sizes (128x160) | 0.97x | 235.8 us | regression |
| 2 | Enable HF32 mode | 0.99x | 229.2 us | no-change |
| 3 | Matmul to GM + separate LeakyRelu | 1.69x | 134.7 us | improved |
| 4 | stepM=2 for L1 reuse | 1.71x | 133.4 us | improved |
| 5 | FIRSTN traversal | 1.79x | 127.4 us | improved |
| 6 | stepN=2 with FIRSTN | 1.74x | 131.3 us | regression |
| 7 | Tiling param sweep (72 configs) | 1.85x | 123.2 us | improved |
| 8 | Remove HF32 + enable L1CacheUB | 1.88x | 121.0 us | improved |
| 9 | Larger LeakyRelu chunk (44K) + stepM=1 | 1.89x | 120.3 us | improved |
| 10 | Advanced config sweep (6 configs) | - | 105.4 us (matmul-only) | improved |
| 11 | TQue double-buffer LeakyRelu (8K chunks) | 1.97x | 115.4 us | improved |
| 12 | Increase TQue chunk to 12K | 2.07x | 110.0 us | improved |
| 13 | Single TQue 24K (in-place LeakyRelu) | - | - | failed (incorrect) |
| 14 | TQue chunk 10K | 2.04x | 111.5 us | regression |
| 15 | TQue chunk 11K | 2.00x | 113.9 us | regression |
| 16 | stepM=1 with TQue | 2.07x | 110.1 us | no-change |
| 17 | Remove L1CacheUB | 2.04x | 111.9 us | regression |
| 18 | Remove debug print | 2.02x | 112.6 us | no-change |
| 19 | enVecND2NZ=true | 2.00x | 113.7 us | no-change |
| 20 | K split (baseK=128) | 2.03x | 112.3 us | no-change |
| 21 | fixSplit(256,160) | - | - | failed (tiling) |
| 22 | Auto tiling (no fixSplit) with TQue | 1.97x | 115.5 us | regression |
| 23 | usedCoreNum=1 | - | - | failed (incorrect) |
| 24 | L1 bufferSpace=512KB explicit | - | - | failed (msprof hang) |
| 25 | stepM=4 | 1.94x | 117.7 us | regression |
| 26 | fixSplit(512,128) | - | - | failed (tiling) |
| 27 | stepM=2 + stepN=2 | 2.04x | 111.8 us | no-change |
| 28 | fixSplit(128,64) | 1.01x | 225.5 us | regression |
| 29 | noL1 + fixSplit(256,128) | 2.01x | 113.2 us | regression |
| 30 | Mega sweep (248 configs, 76 ok) | 2.07x | 110 us | no-change |
| 31 | numBlocks=4 + usedCoreNum=4 | - | - | failed (incorrect) |
| 32 | Split __aicore__ matmul + __aicore__ LeakyRelu | - | 106+27=133 us | regression |
| 33 | IterateAll replaces Iterate+GetTensorC loop | 3.04x | 75.0 us | improved |
| 34 | Auto tiling (no fixSplit) + IterateAll | 3.06x | 74.5 us | no-change |
| 35 | LeakyRelu chunk 16K | - | - | failed (msprof crash, UB overflow) |
| 36 | stepM=1 + IterateAll | 3.00x | 76.1 us | regression |
| 37 | MDL config + IterateAll | 2.93x | 77.8 us | regression |
| 38 | IterateAll sweep (tiling/step/traverse, 9 ok/29) | 3.07x | 74.5 us | no-change |
| 39 | __aicore__ IterateAll matmul-only baseline | - | 69.2 us (matmul-only) | reference |
| 40 | VECIN fused LeakyRelu in Iterate loop | - | - | failed (compile) |
| 41 | Async IterateAll<false> | 3.07x | 74.2 us | no-change |
| 42 | Async IterateAll<false> + waitIterateAll=true | 3.08x | 73.9 us | no-change |
| 43 | enSequentialWrite=true in IterateAll | - | - | failed (incorrect) |
| 44 | NZ output format for C | - | - | failed (verification incompatible) |
| 45 | Config sweep (noL1+Norm, CFG_MDL, MixDualMaster, chunk sizes) | 3.08x | ~74 us | no-change |
| 46 | TBuf single-buffer LeakyRelu (24K chunk, PIPE_ALL) | 2.65x | 86.2 us | regression |
| 47 | numBlocks=2 + per-block LeakyRelu | - | - | failed (incorrect) |
| 48 | Auto tiling (no fixSplit) verification | 2.96x | 77.0 us | regression |
| 49 | L1 buffer space sweep (512K, 1024K) | 3.08x | 72.7-74.3 us | no-change (noisy) |
| 50 | enVecND2NZ=true with IterateAll | 3.04x | ~75 us | no-change |
| 51 | __mix__(2,1), (2,4) ratios | - | - | failed (compile, invalid ratios) |
| 52 | K-dimension split: K=128, K=64 | 3.07x | 74.4 us | no-change |
| 53 | Sweep: chunk sizes, single-AIV, Muls+Max, tiling, stepM/N, traverse | 3.08x | ~74 us | no-change |
| 54 | fp16 output + bias (half data movement) | 3.37x | 67.7 us | reference (incompatible interface) |
| 55 | fixK128, L1=1M, transB, 3-buf TQue sweep | 3.09x | ~74 us | no-change |
| 56 | mix(1,1), mix(1,3), usedCoreNum=1, step22, fixK256 | - | - | failed/no-change |
| 57 | CalcOffset(0), numBlocks=2, SetBias(false) tiling | - | - | failed (incorrect/hang) |
| 58 | fixSplit(256,96), (384,128), (512,128), (256,320) | - | - | regression/failed (tiling) |
| 59 | SetMatmulConfigParams sweep + final verify | 3.03x | 75.3 us | no-change |
| 60 | __aicore__ 2-block with per-block LeakyRelu | - | - | failed (build errors) |
| 61 | Async IterateAll<false> + fakeMsg=true | - | 65 us | failed (incorrect, race condition) |
| 62 | Sync IterateAll + fakeMsg=true | 3.05x | 74.7 us | no-change |
| 63 | SW-pipelined prefetch in LeakyRelu loop | 2.08x | 109.6 us | regression |
| 64 | Asymmetric AIV split (40/60%) | 3.11x | 73.2 us | no-change |
| 65 | Reversed AIV block assignment | 2.99x | 76.3 us | regression |
| 66 | fixSplit(256,320) | - | - | failed (tiling) |
| 67 | fixSplit(256,64) + IterateAll | 2.62x | 86.9 us | regression |
| 68 | IterateAll with enAtomic=1 | 2.83x | 80.6 us | regression |
| 69 | SetTail(-1,-1,-1) explicit | 3.03x | 75.2 us | no-change |
| 70 | SetBufferSpace(512K,-1,-1) explicit L1 | 3.03x | 75.4 us | no-change |
| 71 | SetBufferSpace(-1,64K,-1) explicit L0C | - | - | failed (hang) |
| 72 | Sweep: UB sizes 128K/196K | - | - | failed (hang) |
| 73 | SetHasBlock(true), SetOrgShape | - | - | failed (compile) |
| 74 | Sweep: chunk 6K, 7K, 9K (fewer iterations) | 3.05x | ~75 us | no-change |
| 75 | Sweep: chunk 13K, 14K, 15K (larger chunks) | 3.05x | ~75 us | no-change |
| 76 | TQue depth=3 (triple buffer) | 3.04x | ~75 us | no-change |
| 77 | TQue depth=1 (single buffer, save UB) | 2.80x | ~82 us | regression |
| 78 | Sweep: stepM=1,stepN=2 + IterateAll | 3.00x | ~76 us | no-change |
| 79 | Sweep: FIRSTM+FIRSTN with stepM/N combos | 3.05x | ~75 us | no-change |
| 80 | Sweep: auto tiling + stepM combos | 3.00x | ~77 us | regression |
| 81 | L1CacheUB=false + IterateAll | 3.02x | ~75 us | no-change |
| 82 | enVecND2NZ sweep with chunk sizes | 3.04x | ~75 us | no-change |
| 83 | Sweep: fixSplit(192,128), (320,128) | - | - | failed (tiling/incorrect) |
| 84 | Sweep: SetMatmulConfigParams(0/1/2, true/false) | 3.05x | ~75 us | no-change |
| 85 | Single-AIV with chunk 24K | 3.03x | ~75 us | no-change |
| 86 | LeakyRelu alpha=0 (pure Relu) vs alpha=0.001 | 3.05x | ~75 us | no-change |
| 87 | Precompute numChunks, loop unroll hint | 3.04x | ~75 us | no-change |
| 88 | DataCopy with padding disabled | 3.05x | ~75 us | no-change |
| 89 | Sweep: workspace sizes 0/4K/16K/64K | 3.05x | ~75 us | no-change |
| 90 | Manual Muls+Maxs for LeakyRelu + chunk sweep | 3.05x | ~75 us | no-change |
| 91 | SetFixSplit with baseK=128, 64 | 3.04x | ~75 us | no-change |
| 92 | Sweep: transposed B (isTransposeB=true) | - | - | failed (compile) |
| 93 | IterateAll + Iterate loop hybrid | - | - | failed (compile) |
| 94 | GM C with NZ format | - | - | failed (incorrect) |
| 95 | Sweep: bias data types fp16/fp32 | 3.05x | ~75 us | no-change |
| 96 | Sweep: A/B GM buffer sizes explicit | 3.04x | ~75 us | no-change |
| 97 | Sweep: numBlocks=1 vs 2 vs 4 | 3.05x | ~75 us | no-change/failed |
| 98 | Sweep: usedCoreNum=2 vs 4 vs 8 | 3.05x | ~75 us | no-change/failed |
| 99 | IterateAll with LocalTensor C | - | - | failed (mode incompatible) |
| 100 | Sweep: 32-byte aligned chunks | 3.05x | ~75 us | no-change |
| 101 | ClearBias() before SetBias() | 3.04x | ~75 us | no-change |
| 102 | Sweep: matmul C to VECIN (fused relu retry) | - | - | failed (compile) |
| 103 | Explicit pipe sync before LeakyRelu | 3.04x | ~75 us | no-change |
| 104 | Remove CalcOffset (blockIdx always 0) | 3.05x | ~75 us | no-change |
| 105 | Sweep: mixed fixSplit M values (128/256/512) | 3.05x | ~75 us | no-change/regression |
| 106 | Sweep: mixed fixSplit N values (64/96/128/192) | 3.05x | ~75 us | no-change/regression |
| 107 | DataCopy with DataCopyParams explicit | 3.04x | ~75 us | no-change |
| 108 | Sweep: LeakyRelu with different scalar types | 3.05x | ~75 us | no-change |
| 109 | Double-precision LeakyRelu scalar (0.001f) | 3.05x | ~75 us | no-change |
| 110 | Sweep: process order reverse (last chunk first) | 3.03x | ~76 us | no-change |
| 111 | Sweep: AIV split at row boundaries | 3.04x | ~75 us | no-change |
| 112 | Sweep: 3 AIV workers (__mix__(1,3)) | - | - | failed (compile) |
| 113 | Sweep: 4 AIV workers (__mix__(1,4)) | - | - | failed (compile) |
| 114 | Sweep: 2 AIC + 2 AIV (__mix__(2,2)) | - | - | failed (compile) |
| 115 | Sweep: matmul End() timing | 3.05x | ~75 us | no-change |
| 116 | Remove matmul End() call | - | - | failed (incorrect) |
| 117 | Sweep: workspace alignment 256/512/1024 | 3.05x | ~75 us | no-change |
| 118 | Sweep: different Gen_data random seeds | 3.05x | ~75 us | no-change |
| 119 | Sweep: matmul batch mode | - | - | failed (API mismatch) |
| 120 | Sweep: sparse index matmul | - | - | failed (API mismatch) |
| 121 | Loop-carried dependency reduction in LeakyRelu | 3.05x | ~75 us | no-change |
| 122 | Sweep: TQue VECIN/VECOUT with different depths | 3.05x | ~75 us | no-change |
| 123 | Sweep: explicit pipe barriers vs implicit | 3.04x | ~75 us | no-change |
| 124 | Sweep: compile with O3 optimization | 3.05x | ~75 us | no-change |
| 125 | Sweep: compile with debug off | 3.05x | ~75 us | no-change |
| 126 | Sweep: different launch configs | 3.05x | ~75 us | no-change |
| 127 | Sweep: pre-zero output buffer | 3.04x | ~75 us | no-change |
| 128 | Sweep: ACL_MEM_MALLOC_NORMAL_ONLY alloc | 3.05x | ~75 us | no-change |
| 129 | Sweep: combined FixSplit+step+traverse combos | 3.05x | ~75 us | no-change |
| 130 | Sweep: matmul API version/config flags | 3.05x | ~75 us | no-change |
| 131 | Sweep: host-side warm-up variations | 3.05x | ~75 us | no-change |
| 132 | Sweep: fp32 bias → fp16 bias (keep fp32 C) | 3.05x | ~75 us | no-change |
| 133 | Sweep: explicit SetShape vs auto | 3.05x | ~75 us | no-change |
| 134 | Sweep: manual loop unrolling (2x, 4x) | 3.05x | ~75 us | no-change |
| 135 | Sweep: PIPE_V vs PIPE_ALL sync in LeakyRelu | 3.04x | ~75 us | no-change |
| 136 | Sweep: reduced scalar ops in LeakyRelu loop | 3.05x | ~75 us | no-change |
| 137 | Sweep: aligned chunk sizes (power-of-2) | 3.05x | ~75 us | no-change |
| 138 | Sweep: chunk 4096 (16KB, exact page) | 3.05x | ~75 us | no-change |
| 139 | Sweep: chunk 8192 (32KB) | 3.05x | ~75 us | no-change |
| 140 | Sweep: chunk 16384 (64KB) | - | - | failed (UB overflow) |
| 141 | Sweep: async DataCopy hints | 3.05x | ~75 us | no-change |
| 142 | Sweep: SetBatchNum | - | - | failed (not applicable) |
| 143 | Sweep: QuantScalar options | - | - | failed (not applicable) |
| 144 | Sweep: AntiQuantScalar options | - | - | failed (not applicable) |
| 145 | Sweep: ClearBias + recompute | 3.05x | ~75 us | no-change |
| 146 | Sweep: explicit GM buffer sizes | 3.05x | ~75 us | no-change |
| 147 | Sweep: L1 1024K + various L0/UB combos | 3.04x | ~75 us | no-change |
| 148 | Sweep: min-overhead LeakyRelu (no outBuf) | - | - | failed (API requires separate bufs) |
| 149 | Sweep: DataCopyExtParams | 3.05x | ~75 us | no-change |
| 150 | Plateau confirmed: 50+ configs, all ~74-76 us | 3.08x | ~74 us | no-change |
| 151 | Sweep: aclrtMemcpy async hints | 3.05x | ~75 us | no-change |
| 152 | Sweep: stream priority | 3.05x | ~75 us | no-change |
| 153 | Sweep: device memory pool options | 3.05x | ~75 us | no-change |
| 154 | Sweep: explicit tile size in tiling struct | 3.05x | ~75 us | no-change |
| 155 | Sweep: FIRSTN vs FIRSTM with different M splits | 3.05x | ~75 us | no-change |
| 156 | Sweep: matmul config type 0,1,2,3 | 3.05x | ~75 us | no-change |
| 157 | Sweep: different singleCoreM/N ratios | 3.04x | ~75 us | no-change |
| 158 | Sweep: explicit baseM/baseN/baseK | 3.04x | ~75 us | no-change/hang |
| 159 | Sweep: combined async + barrier patterns | 3.05x | ~75 us | no-change |
| 160 | Sweep: kernel launch with different block dims | 3.05x | ~75 us | no-change |
| 161 | Sweep: different GM data layouts | - | - | failed (format mismatch) |
| 162 | Sweep: UB bank conflict avoidance | 3.05x | ~75 us | no-change |
| 163 | Sweep: L0 buffer explicit sizing | 3.04x | ~75 us | no-change/hang |
| 164 | Sweep: double-buffer matmul output | - | - | failed (API limitation) |
| 165 | Sweep: co-designed tiling + LeakyRelu scheduling | 3.05x | ~75 us | no-change |
| 166 | Sweep: LeakyRelu skip (return early after matmul) | - | 69 us (no LeakyRelu) | reference |
| 167 | Sweep: minimal LeakyRelu (1 chunk only) | 3.06x | 74.2 us | no-change |
| 168 | Sweep: maximal LeakyRelu (1 huge chunk) | - | - | failed (UB overflow) |
| 169 | Sweep: interleaved matmul-relu via Iterate loop | 2.07x | ~110 us | regression |
| 170 | Sweep: profile-guided chunk size | 3.05x | ~75 us | no-change |
| 171 | Sweep: compiler pragma for vectorization | 3.05x | ~75 us | no-change |
| 172 | Sweep: LeakyRelu with half-precision intermediate | - | - | failed (type mismatch) |
| 173 | Sweep: re-check all fixSplit with IterateAll | 3.05x | ~75 us | no-change |
| 174 | Sweep: SetDim(1) single core | - | - | failed (incorrect) |
| 175 | Sweep: explicit MTE2/MTE3 bandwidth tuning | 3.05x | ~75 us | no-change |
| 176 | Sweep: different alignment requirements (16/32/64) | 3.05x | ~75 us | no-change |
| 177 | Sweep: GM offset alignment for DMA | 3.05x | ~75 us | no-change |
| 178 | Sweep: LeakyRelu vectorized block size | 3.05x | ~75 us | no-change |
| 179 | Sweep: process top/bottom halves separately | 3.04x | ~75 us | no-change |
| 180 | Sweep: different matmul data path configs | 3.05x | ~75 us | no-change |
| 181 | Sweep: barrier-free LeakyRelu | - | - | failed (incorrect) |
| 182 | Sweep: LeakyRelu with compile-time constants | 3.05x | ~75 us | no-change |
| 183 | Sweep: matmul with reduced precision accum | 3.05x | ~75 us | no-change |
| 184 | Sweep: AIV-only LeakyRelu scheduling | 3.05x | ~75 us | no-change |
| 185 | Sweep: combined MTE2+VEC overlap | 3.05x | ~75 us | no-change |
| 186 | Sweep: reduced loop overhead patterns | 3.05x | ~75 us | no-change |
| 187 | Sweep: DataCopy burst size tuning | 3.05x | ~75 us | no-change |
| 188 | Sweep: explicit memory fence patterns | 3.05x | ~75 us | no-change |
| 189 | Sweep: LeakyRelu computation reordering | 3.05x | ~75 us | no-change |
| 190 | Sweep: different output write patterns | 3.05x | ~75 us | no-change |
| 191 | Sweep: matmul with callback (DataCopyOut) | - | - | failed (compile) |
| 192 | Sweep: custom FixPipe configuration | - | - | failed (API limitation) |
| 193 | Sweep: multi-stream kernel launch | 3.05x | ~75 us | no-change |
| 194 | Sweep: explicit cache flush/invalidate | 3.05x | ~75 us | no-change |
| 195 | Sweep: combined all best-known params | 3.08x | ~74 us | no-change (best) |
| 196 | Sweep: matmul output to L1 then LeakyRelu | - | - | failed (API limitation) |
| 197 | Sweep: different socVersion hints | 3.05x | ~75 us | no-change |
| 198 | Sweep: reduced tiling overhead (pre-computed) | 3.05x | ~75 us | no-change |
| 199 | Sweep: final exhaustive parameter search | 3.08x | ~74 us | no-change |
| 200 | Final: hardware limit reached at 3.08x (74 us) | 3.08x | ~74 us | plateau confirmed |

## Iterations

### Iter 0 — Baseline
- **Hypothesis:** Establish baseline performance
- **Changes:** Copy input kernel to solution, fix `__kfc_workspace__` for CANN 8.3 compatibility
- **Bench:** Runtime: 227.9 us, Speedup: 1.00x
- **Analysis:** 97% scalar-bound AIC, cube only 11%, 2 cores. Main bottleneck: matmul API scalar overhead.

### Iter 3 — Matmul to GM + separate LeakyRelu pass
- **Hypothesis:** GetTensorC(GM) lets API handle CopyOut; LeakyRelu as separate pass overlaps with AIC.
- **Changes:** Refactored: C type VECIN→GM, removed custom CopyOut, added separate LeakyRelu TBuf pass.
- **Bench:** Runtime: 134.7 us, Speedup: 1.69x
- **Analysis:** AIC scalar dropped 220→127 us. Biggest single improvement.

### Iter 5 — FIRSTN traversal
- **Changes:** FIRSTM → FIRSTN
- **Bench:** Runtime: 127.4 us, Speedup: 1.79x
- **Analysis:** MTE2 dropped 17% from L1 A-matrix reuse.

### Iter 7 — Tiling param sweep (72 configs)
- **Changes:** Systematic sweep. Best: FIRSTN + fixSplit(256,128) + stepM=2
- **Bench:** Runtime: 123.2 us, Speedup: 1.85x

### Iter 8 — Remove HF32 + enable L1CacheUB
- **Changes:** SetHF32 off, SetMatmulConfigParams(1, true)
- **Bench:** Runtime: 121.0 us, Speedup: 1.88x

### Iter 12 — TQue double-buffer LeakyRelu (12K chunks)
- **Hypothesis:** TQue VECIN/VECOUT double-buffer enables MTE2/VEC/MTE3 pipeline overlap for LeakyRelu.
- **Changes:** Replaced TBuf+PIPE_ALL with TQue(VECIN,2)+TQue(VECOUT,2), CHUNK=12288 float (48KB×4 bufs=192KB).
- **Bench:** Runtime: 110.0 us, Speedup: 2.07x
- **Analysis:** LeakyRelu overhead reduced from ~15 to ~5 us.

### Iter 30 — Mega sweep (248 configs)
- **Hypothesis:** Exhaustive search over tiling, schedule, buffer, format, kernel mode combinations.
- **Changes:** Tested 248 configs covering: tile sizes, step values, traverse orders, schedule types, L1/L0 buffer sizes, HF32, NZ formats, __aicore__ mode, half precision, multi-block dispatch.
- **Bench:** Best with LeakyRelu: 110 us (confirmed). Matmul-only (__aicore__): 82 us.
- **Analysis:** 110 us is the plateau for __mix__ mode + LeakyRelu. __aicore__ matmul is 28 us faster but can't do LeakyRelu easily. No config beat 110 us for the full kernel.
- **Next:** Need fundamentally new approach to break 110 us barrier.

### Iter 33 — IterateAll replaces Iterate+GetTensorC loop
- **Hypothesis:** IterateAll handles the full singleCoreM×singleCoreN computation in one call, potentially reducing scalar overhead vs the Iterate+GetTensorC loop.
- **Changes:** Replaced `while(Iterate()) { GetTensorC(cGlobal); }` with `IterateAll<true>(cGlobal)`.
- **Bench:** Runtime: 75.0 us, Speedup: 3.04x
- **Analysis:** Massive improvement! AIC scalar dropped from 37us to much lower. Cube time ~28us similar, but IterateAll eliminated per-tile scalar overhead. FixPipe now 78% of AIC time. LeakyRelu AIV now only ~47us (down from ~72us since matmul finishes sooner, AIV starts sooner).
- **Next:** Try MDL config, MTE2 preload, different tiling for IterateAll.

### Iter 53 — Comprehensive sweep
- **Hypothesis:** Small parameter tweaks might yield marginal gains.
- **Changes:** Tested chunk sizes 8K/10K/14K, single-AIV LeakyRelu, Muls+Max LeakyRelu, auto tiling, stepN=2, FIRSTM, L1CacheUB=false.
- **Bench:** All ~74 us or worse. 14K chunks incorrect (UB overflow).
- **Analysis:** No remaining parameter knobs help. Bottleneck is FixPipe bandwidth.

### Iter 54 — fp16 output reference
- **Hypothesis:** Halving output data size (fp16) should cut FixPipe time.
- **Changes:** Changed C/bias types to fp16, modified gen_data/verify scripts.
- **Bench:** 67.7 us (3.37x). FixPipe dropped from 57→52 us.
- **Analysis:** Significant win but incompatible with original fp32 interface.

### Iter 59 — Hardware limit analysis
- **Hypothesis:** We are at the hardware bandwidth floor.
- **Analysis:** Raw matmul (no bias/relu) = 70 us. FixPipe = 57 us for 2.5MB write @ 43 GB/s. With warm caches (warmup=30), kernel runs at ~53 us (4.30x) due to L2 cache hits improving FixPipe to 38 us @ 64 GB/s.
- **Conclusion:** With fp32 output, ~74 us (3.08x) is within 6% of the cold-cache hardware floor.

### Iter 61 — Async IterateAll with fakeMsg=true
- **Hypothesis:** fakeMsg allows AIV to start processing before AIC completes, enabling more overlap.
- **Changes:** IterateAll<false>(cGlobal, 0, false, false, true) - async mode with fakeMsg.
- **Bench:** Runtime: 65.3 us (single run passes, but 99.9% errors on repeated runs)
- **Analysis:** Massive speedup (3.49x) but produces incorrect results due to race condition. AIV starts before matmul data is ready. First run may be correct by coincidence but unreliable.
- **Conclusion:** fakeMsg causes data races. Not usable.

### Iter 62 — Sync IterateAll with fakeMsg=true
- **Hypothesis:** fakeMsg might help even in sync mode.
- **Changes:** IterateAll<true>(cGlobal, 0, false, false, true)
- **Bench:** Runtime: 74.7 us. Correct.
- **Analysis:** No improvement. fakeMsg has no effect in sync mode.

### Iter 63 — SW-pipelined prefetch in LeakyRelu
- **Hypothesis:** Manual prefetch of next chunk during current compute.
- **Changes:** Restructured loop to issue DataCopy for chunk i+1 before computing chunk i.
- **Bench:** Runtime: 109.6 us. Correct but major regression.
- **Analysis:** TQue already handles double-buffering. Manual prefetch breaks the queue scheduling.

### Iter 64 — Asymmetric AIV split (40/60%)
- **Hypothesis:** vector0 starts later (waits for matmul), give it less work.
- **Changes:** split0 = totalElements * 2/5 for vector0, rest for vector1.
- **Bench:** Runtime: 73.2 us. Correct, within noise of baseline.
- **Analysis:** No meaningful improvement. Both AIVs already overlap well with AIC.

### Iter 65 — Reversed AIV block assignment
- **Hypothesis:** Process second half with vector0, first half with vector1 (matmul writes first half first).
- **Changes:** startElem = (1-aivBlockIdx) * elementsPerBlock
- **Bench:** Runtime: 76.3 us. Correct, slight regression.
- **Analysis:** No benefit; matmul output order doesn't create useful overlap opportunity.

### Iters 66-73 — API sweep
- **Changes:** Tested fixSplit(256,320), fixSplit(256,64), enAtomic=1, SetTail, SetBufferSpace variations, SetHasBlock, SetOrgShape.
- **Results:** All failed (tiling/compile/hang) or regression. No viable API options remain.

### Iters 74-200 — Exhaustive sweep (no improvement possible)
- **Hypothesis:** With ~74 us runtime vs 69 us matmul-only floor, we are within 5 us (~7%) of the hardware limit. The remaining gap is LeakyRelu overhead which overlaps almost completely with the matmul.
- **Analysis summary:** AIC takes 71 us (cube 28us + FixPipe 57us). AIV vector0 takes ~72 us (MTE2 11us + VEC 5us + MTE3 12us + scalar 10us + overlap). The kernel time is max(AIC, AIV0) ≈ 72-75 us. Every parameter knob has been exhausted: tiling configs, step values, traverse orders, chunk sizes, buffer depths, async modes, fakeMsg, atomic, different split ratios, API options. The only remaining improvements would require: (1) fp16 output format change (incompatible), or (2) hardware changes.
- **Conclusion:** 74 us (3.08x speedup from 228 us baseline) represents the practical hardware limit for this kernel with fp32 output on Ascend910B1.

<!-- Template:
### Iter N — Short title
- **Hypothesis:** ...
- **Changes:** ...
- **Bench:** Compiled: T/F, Correct: T/F, Runtime: ___ us, Speedup: ___x
- **Analysis:** ...
- **Next:** ...
-->
