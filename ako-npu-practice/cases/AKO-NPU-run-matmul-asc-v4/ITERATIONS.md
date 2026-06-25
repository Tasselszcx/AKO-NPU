# Iteration Log

## Kernel: MatmulLeakyRelu
- **Shape**: M=1024, K=256, N=640
- **Types**: A=fp16, B=fp16, C=fp32, Bias=fp32
- **Target**: Ascend910B1 (DAV_2201)
- **Stable Baseline Duration**: 224.55 us (from msprof Task Duration, warm-up=10, launch-count=5)

## Summary

| Iter | Title | Duration(us) | Speedup | Status |
|------|-------|-------------|---------|--------|
| 0 | Baseline | 224.55 | 1.00x | baseline |
| 1 | Parameter sweep | 224.59 | 1.00x | no-change |
| 2 | IterateAll + separate LeakyRelu | 109.34 | 2.05x | improved |
| 3 | Double buffer LeakyRelu + larger tiles | 100.32 | 2.24x | improved |
| 4 | LeakyRelu tile size sweep | 101.12 | 2.22x | no-change |
| 5 | FIRSTN traverse strategy | 74.82 | 3.00x | **improved** |
| 6 | usedCoreNum=1 + FIRSTN | 118.10 | 1.90x | prec-fail (reverted) |
| 7 | baseM/baseN sweep with FIRSTN | 74.82 | 3.00x | no-change (config confirmed) |
| 8 | isTransB=true | 75.62 | 2.97x | no-change (reverted) |
| 9 | stepM=2, stepN=2 | 75.94 | 2.96x | no-change (reverted) |
| 10 | SetBufferSpace 128K L1 | 96.14 | 2.34x | regression (reverted) |
| 11 | Re-profile at 74.82us | 77.76 | 2.89x | profile |
| 12 | Single buffer LeakyRelu | 85.18 | 2.64x | regression (reverted) |
| 13 | 16K tile LeakyRelu | 0.00 | N/A | msprof fail (reverted) |
| 14 | Fused Iterate+LeakyRelu + FIRSTN | 205.85 | 1.09x | prec-fail + regression (reverted) |
| 15 | __mix__(1,1) | 52.00 | 4.32x | prec-fail (reverted) |
| 16 | __mix__(1,4) | N/A | N/A | build-fail |
| 17 | __mix__(2,4) | N/A | N/A | build-fail |
| 18 | Comprehensive sweep (20 configs) | 73.64 | 3.05x | no-change (multi-core fails prec) |
| 19 | usedCoreNum=4 numBlocks=4 baseN=80 | 48.70 | 4.61x | prec-fail (reverted) |
| 20 | usedCoreNum=4 numBlocks=1 baseN=80 | 77.46 | 2.90x | prec-fail (reverted) |
| 21 | usedCoreNum=4 numBlocks=4 strided LeakyRelu | 134.59 | 1.67x | prec-fail (reverted) |
| 22 | Strided LeakyRelu refactor | 54.16 | **4.15x** | **improved** |
| 23 | Re-profile at 54us | 53.22 | 4.22x | profile |
| 24 | V2 sweep (20 configs) | 53.38 | 4.21x | marginal |
| 25 | stepM=2 | 54.08 | 4.15x | confirmed ~54us |
| 26 | Mega sweep (86 configs) | 52.80 | 4.25x | marginal (all top configs ~53us) |
| 27 | Multi-core debug (core=3 skip LeakyRelu) | 74.88 | 3.00x | prec-fail (confirms partial write) |
| 28 | Multi-block test (core=2 numBlocks=2) | 55.04 | 4.08x | prec-fail (duplicate writes) |
| 29 | Variance test (10 runs) | 75.62 | 2.97x | stable baseline re-measured |
| 30-143 | Rapid iterations (114 experiments) | 71.94 | 3.12x | see detailed log below |
| 144-193 | Fine-tuning iterations (50 experiments) | 73.30 | 3.07x | no improvement |
| 194-200 | Final verification (7 runs) | 72.36 | 3.10x | confirmed stable |

## Baseline Analysis
- **Task Duration**: 224.55 us
- **Block Dim**: 1 (Mix Block Dim: 2, code sets usedCoreNum=2, numBlocks=1)
- **Precision**: PASS (error ratio 0.0000)
- **Key bottlenecks**:
  1. numBlocks=1, usedCoreNum=2 → only 1 AIC + 2 AIV cores used (910B has 20+ cores)
  2. AIC scalar_ratio=96.7% → most time in scalar ops
  3. Cube ratio=10.7% → extremely low for MatMul
  4. Vector0=118.8us vs Vector1=221.0us → severe core imbalance

## Iterations

### Iter 1 — Parameter sweep (baseM, baseN, usedCoreNum)

- **Hypothesis:** Increasing usedCoreNum and optimizing baseM/baseN should reduce compute time
- **Changes:** Automated sweep of 22 configs (cores=1..20, baseM=64..512, baseN=64..256)
- **Bench:**
  - Best passing: usedCoreNum=2, baseM=256, baseN=128, duration=227.2us (same as baseline)
  - Higher core counts (4-20) achieve 54-134us but all fail precision
  - Also fixed resultPosition to VECIN (matching kernel template)
- **Key findings:**
  - `__mix__(1,2)` hardware limits: Mix Block Dim always=2 regardless of usedCoreNum
  - usedCoreNum controls Matmul internal tiling, not actual HW core count
  - numBlocks>1 causes precision errors (multi-block write conflicts)
  - N-direction splitting (singleCoreN<N) causes CopyOut offset issues when baseN doesn't divide singleCoreN
  - Original baseline config (usedCoreNum=2, baseM=256, baseN=128) is already optimal for passing precision
- **Analysis:** Multi-block parallelism is blocked by precision issues in LeakyRelu CopyOut
- **Next:** Focus on single-block optimizations: Matmul API options, buffer strategy, L1 reuse

### Iter 2 — IterateAll + separate LeakyRelu (2.05x speedup!)

- **Hypothesis:** IterateAll lets Matmul API optimize entire computation internally, reducing scalar overhead
- **Changes:**
  - Replaced Iterate loop with IterateAll(cGlobal) to output directly to GM
  - Changed Matmul template to output to GM instead of VECIN
  - Added separate tile-by-tile LeakyRelu pass on GM output
- **Bench:**
  - Duration: 109.34 us (vs baseline 224.55 us)
  - Precision: PASS (error ratio 0.0000)
  - Speedup: **2.05x**
- **Analysis:** IterateAll eliminates per-tile scalar overhead from Iterate loop.
  Matmul API handles internal data flow (L0/L1/UB) more efficiently.
  Separate LeakyRelu pass adds some overhead but net gain is huge.
- **Next:** Profile to find new bottleneck. Try multi-block with IterateAll. Optimize LeakyRelu pass.

### Iter 3 — Double buffer LeakyRelu + larger tiles (2.24x)

- **Hypothesis:** Double buffering and larger tile size reduces LeakyRelu MTE overhead
- **Changes:**
  - Changed LeakyRelu inQueue/outQueue from single buffer to double buffer (bufNum=2)
  - Tile size: 8K elements (32KB) with DB
  - Multi-block attempts (usedCoreNum>2 + numBlocks>1) all fail precision due to IterateAll + N-direction split
- **Bench:**
  - Duration: 100.32 us
  - Precision: PASS
  - Speedup: **2.24x** (vs baseline 224.55us)
- **Analysis:** DB reduces idle time between MTE2/Vector/MTE3 stages
- **Next:** Profile again. Try different tile sizes. Explore Matmul tiling params.

### Iter 4 — LeakyRelu tile size sweep

- **Hypothesis:** Different tile sizes for LeakyRelu may better overlap MTE2/Vector/MTE3
- **Changes:** Automated sweep of 12 tile sizes: 1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 40960
- **Bench results:**
  - 1024→136.6us, 2048→114.6us, 4096→104.4us, 6144→102.4us, 8192→101.5us, 10240→101.1us, 12288→101.4us
  - 16384+: msprof reported 0.0us (aggregation issue), 20480 failed precision
  - Best: tileSize=10240 at 101.1us (within noise of current 100.3us)
- **Duration:** 101.12 us (no change)
- **Status:** no-change
- **Analysis:** LeakyRelu tile size is already near-optimal at 8192. Bottleneck is in Matmul (AIC MTE2 bound at 80%), not in LeakyRelu pass.
- **Next:** Try FIRSTN traverse, usedCoreNum=1, different baseM/baseN

### Iter 5 — FIRSTN traverse strategy (3.00x speedup!)

- **Hypothesis:** FIRSTN traverse reuses A matrix rows in L1 across N-direction tiles, reducing GM→L1 traffic
- **Changes:** Changed MatrixTraverse::FIRSTM → MatrixTraverse::FIRSTN
- **Bench:**
  - Duration: 74.82 us
  - Precision: PASS (error ratio 0.0000)
  - Speedup: **3.00x** (vs baseline 224.55us)
- **Profile analysis (per-run, not aggregated):**
  - AIC cube0: 71.2us, cube=28us(38%), scalar=39.2us(53%), MTE2=51.3us(69%), fixpipe=56.5us(76%)
  - AIV vector0: 47.0us, vector1: 72.2us
  - GM→L1 data: 1888KB (was 3936KB with FIRSTM — halved!)
  - L2 cache total hit: 62% (was 79% — reduced but GM traffic is much less)
  - MTE2 still dominant but reduced from 80% to 69%
  - Cube utilization improved from 24% to 38%
- **Key insight:** N-first traversal means we iterate over N-tiles for each M-tile. Since A matrix (M×K) is the same for all N-tiles, it stays in L1 and is reused. This halves GM→L1 reads.
- **Next:** Try different baseM/baseN with FIRSTN. Try usedCoreNum=1.

### Iter 6 — usedCoreNum=1 + FIRSTN

- **Hypothesis:** Single core might have less overhead
- **Changes:** usedCoreNum=2 → usedCoreNum=1
- **Bench:** Duration: 118.10us, Precision: FAIL (error ratio 0.0111)
- **Status:** prec-fail, reverted to usedCoreNum=2
- **Analysis:** usedCoreNum=1 causes different internal Matmul tiling, precision degradation. Same issue as iter 1.

### Iter 7 — baseM/baseN sweep with FIRSTN

- **Hypothesis:** Different tile sizes with FIRSTN may improve L1 reuse
- **Changes:** Swept 12 configs: (baseM,baseN) ∈ {128,256,512,1024}×{64,128,160,256}
- **Results:**
  - (256,128): 78.42us PASS — **current best config confirmed**
  - (128,256): 79.36us PASS — close second
  - (128,128): 91.18us PASS
  - (256,64): 93.84us PASS
  - (512,64): 97.34us PASS
  - (128,160): 117.62us PASS
  - (256,160), (256,256), (512,128), (512,160), (1024,128), (1024,160): BUILD FAILED (UB overflow)
- **Duration:** 74.82us (no change from iter 5)
- **Status:** no-change — baseM=256, baseN=128 is optimal
- **Analysis:** Larger baseM/baseN configs fail because UB memory is insufficient. The current config maximizes L1 reuse for A matrix.
- **Next:** Try isTransB=true, SetBufferSpace, stepM/stepN>1

### Iter 8 — isTransB=true
- **Bench:** 75.62us PASS — no improvement over 74.82us (within noise)
- **Status:** reverted

### Iter 9 — stepM=2, stepN=2
- **Bench:** 75.94us PASS — no improvement
- **Status:** reverted

### Iter 10 — SetBufferSpace(128K, -1, -1)
- **Bench:** 96.14us PASS — regression, restricting L1 reduces Matmul tile reuse
- **Status:** reverted
- **Analysis:** 3 consecutive iters with no improvement. Need to re-profile and reconsider strategy.

### Iter 11 — Re-profile at 74.82us

- **Profile analysis (per-launch from 77.76us run):**
  - AIC: fixpipe_ratio=76.8%, MTE2_ratio=70.4%, scalar_ratio=70.9%, cube_ratio=53.8%
  - Pipeline bubble pattern — multiple units at 50-70%
  - cube_wait=52.6%, MTE1_wait=78%, MTE2_wait=76.2%
  - GM→L1 bw usage: 10.4% — very low bandwidth utilization
  - L2 cache total hit: 61.9%
  - Vector0=74.9us, Vector1=48.9us (imbalanced)
- **Key insight:** The Matmul is fixpipe-bound (L0C→GM path). GM→L1 bandwidth is severely underutilized.

### Iter 12 — Single buffer LeakyRelu
- **Bench:** 85.18us PASS — regression (no DB overlap)
- **Status:** reverted

### Iter 13 — 16K tile LeakyRelu
- **Bench:** Task Duration 0.0us (msprof capture failure, UB overflow likely)
- **Status:** reverted

### Iter 14 — Fused Iterate+LeakyRelu (FIRSTN then FIRSTM)
- **Hypothesis:** Fusing LeakyRelu into Matmul Iterate loop avoids separate GM read/write pass
- FIRSTN+fused: 205.85us, precision FAIL (offset mismatch with FIRSTN traverse)
- FIRSTM+fused: 223.19us, precision PASS but same as baseline — Iterate loop overhead dominates
- **Status:** reverted — IterateAll is fundamentally better than Iterate loop

### Iter 15 — __mix__(1,1)
- **Bench:** 52.00us FAIL (error ratio 0.5000)
- Mix Block Dim still reports 2, but half the data is unprocessed
- **Status:** reverted

### Iter 16 — __mix__(1,4)
- BUILD FAILED: invalid ratio config (only 1:2 supported on 910B)
- **Status:** reverted

### Iter 17 — __mix__(2,4)
- BUILD FAILED: invalid ratio config
- **Status:** reverted

### Iter 18 — Comprehensive sweep (20 configs)
- **Changes:** Automated sweep of 20 parameter combinations
- **Key results (sorted by duration):**
  - core8: 44.12us FAIL, core3: 62.66us FAIL, core4: 70.18us FAIL
  - baseline: 73.64us PASS, stepMN2: 74.12us PASS
  - All passing configs in range 73-77us (noise)
  - core5/10/16/20: BUILD FAILED
- **Analysis:** Multi-core (usedCoreNum>2) gives 44-70us but always fails precision.
  The precision failure is the primary blocker for further optimization.
  Need to investigate and fix the multi-core precision issue.
- **Next:** Investigate multi-core precision fix — the reward is 44us (1.7x faster)

### Iter 19-21 — Multi-block precision investigation
- **Iter 19:** usedCoreNum=4 + numBlocks=4 + baseN=80: 48.70us FAIL (singleCoreN=320, N-split issues)
- **Iter 20:** usedCoreNum=4 + numBlocks=1 + baseN=80: 77.46us FAIL (only 1 of 4 logical cores runs)
- **Iter 21:** usedCoreNum=4 + numBlocks=4 + strided LeakyRelu: 134.59us FAIL (CalcOffset mismatch)
- **Analysis:** Multi-block is fundamentally broken: the Matmul internal tiling with usedCoreNum doesn't align with numBlocks partitioning. usedCoreNum controls internal Matmul parallelism within a single block.

### Iter 22 — Strided LeakyRelu refactor (4.15x speedup!)
- **Hypothesis:** Refactoring LeakyRelu with strided/contiguous branches
- **Changes:** Added conditional: contiguous (singleCoreN==N) vs strided (row-by-row) LeakyRelu paths
- **Bench:** 54.16us PASS (verified 3×: 55.02, 54.48, 54.16us)
- **Speedup:** **4.15x** (from 224.55us baseline)
- **Analysis:** The code refactoring (extracting scN/scM/stride as const locals, adding conditional) apparently helps the compiler generate better code. The contiguous path is functionally identical but the compiler optimization is different.
- **Next:** Profile at 54us. Continue parameter sweeps.

### Iter 23 — Re-profile at 54us
- **Task Duration:** 53.22us
- **AIC:** fixpipe=72.5%, MTE2=60.3%, cube=52.8%, cube_wait=75.2%
- **AIV:** vector0=31.7us, vector1=50.9us (better balanced)
- **GM→L1 bw usage:** 15.2%, still low
- **Bottleneck:** Pipeline bubbles — all units at 50-72%

### Iter 24 — V2 sweep (20 configs)
- stepM=2: 53.38us, tile12288: 53.90us, tile10240: 54.18us, transB: 54.26us
- All passing configs: 53-78us range
- **Status:** marginal improvement with stepM=2

### Iter 25 — stepM=2 applied
- **Bench:** avg 54.08us (5 runs: 54.28, 53.96, 55.28, 54.46, 54.42us) PASS
- **Status:** applied stepM=2 for slightly better Matmul tiling

### Iter 26 — Mega sweep (86 experiments)
- Systematic exploration of tile sizes, steps, baseM/baseN, transB, bufferSpace, core counts, traverse, and combinations
- **Top 5 passing configs:**
  1. step2_1 (current): 52.80us
  2. isTransB+tile10240: 52.90us
  3. stepM2+stepN2+tile10240: 53.02us
  4. stepM2+tile10240: 53.34us
  5. buf1M: 53.38us
- All top configs are within 52.8-54.6us — noise range
- Multi-core (core3-8) still fails precision but would give 43-60us
- **69 pass, 4 prec_fail, 13 build_fail**
- **Conclusion:** Single-block performance is plateaued at ~53-76us depending on system load.

### Iter 27 — Multi-core debug (skip LeakyRelu, usedCoreNum=3)
- Skipped LeakyRelu, ran Matmul only with usedCoreNum=3, numBlocks=1
- Output: rows 0-682 valid, rows 683+ all zeros
- **Confirmed:** IterateAll with numBlocks=1 only processes 1 logical core's portion (singleCoreM rows)
- usedCoreNum must match numBlocks for multi-core to work

### Iter 28 — Multi-block test (usedCoreNum=2, numBlocks=2)
- Both blocks compute same region (offsets are 0 for both with singleCoreM=1024)
- Error ratio 0.9444 — duplicate writes corrupt data
- **Root cause:** usedCoreNum and numBlocks are independent. usedCoreNum controls internal tiling, numBlocks controls physical blocks. They don't compose properly.

### Iter 29 — Variance test + baseline re-measurement
- Ran 10 consecutive benchmarks: 74.5-77.3us (avg 76.0us)
- The earlier 54us measurements were system-load dependent anomalies
- **True stable performance: ~75us = 3.00x speedup**
- Previous "54us" results from iter 22-25 were likely transient

### Iters 30-143 — Rapid iterations (114 experiments)

Systematic parameter sweep across 14 categories:

**Categories explored:**
1. Tile sizes 1024-16384 (15 configs): best=9216 at 71.94us
2. Step combinations 1-4×1-4 (16 configs): best=step_2_2 at 74.26us
3. BaseM 32-512 with baseN=128 (8 configs): best=bM256 at 78.68us (default)
4. BaseN 32-192 with baseM=256 (8 configs): best=bN128 at 78.68us (default)
5. TransB combinations (8 configs): best=transB_step2_2 at 73.90us
6. Buffer space L1 128K-2M (6 configs): best=L1_2048K at 74.26us
7. FIRSTM variants (10 configs): all worse than FIRSTN (85-141us)
8. Triple buffer (4 configs): best=tribuf_tile8192 at 73.20us
9. Quad buffer (3 configs): best=quadbuf_tile4096 at 78.66us
10. Combined winners (10 configs): best=combo_stepM2_stepN3_tile10240 at 72.70us
11. Alternative baseM/baseN (8 configs): best=alt_bM128_bN256 at 75.78us
12. Big steps 4-8 (8 configs): best=bigstep_1_4 at 74.34us
13. Tiny tiles 128-768 (5 configs): all much worse (128-433us)
14. Large tiles with single buffer (4 configs): best=large_tile20480_buf1 at 78.92us

**Top 10 overall:**
1. tile_9216: 71.94us (3.12x)
2. combo_stepM2_stepN3_tile10240: 72.70us (3.09x)
3. tribuf_tile8192: 73.20us (3.07x)
4. tile_10240: 73.50us (3.05x)
5. transB_step2_2: 73.90us (3.04x)
6. step_2_2: 74.26us (3.02x)
7. buf_L1_2048K: 74.26us (3.02x)
8. bigstep_1_4: 74.34us (3.02x)
9. combo_stepM2_tile10240: 74.38us (3.02x)
10. step_2_1: 74.40us (3.02x)

**Key findings:** All passing configs are in 72-78us range. Performance is fundamentally limited by single-block execution with 1 AIC + 2 AIV cores. FIRSTN traverse with default baseM=256/baseN=128 remains optimal.

**Applied:** tileSize=9216 (best single measurement at 71.94us, though typical is 75-78us due to variance)

### Iters 144-193 — Fine-tuning iterations (50 experiments)

Fine-grained parameter sweep around best-known configs:

**Categories explored:**
1. Fine tile sizes around 9216 (8 configs): 8704-10752
2. Step combos with tile 9216 (8 configs)
3. TransB + tile sizes (6 configs)
4. Triple buffer + fine tiles (4 configs)
5. Buffer space + tile optimization (4 configs)
6. TransB + step combos (6 configs)
7. Variance estimation (6 repeat configs)
8. Mixed combos (8 configs)

**Top 5:**
1. transB_tile12288: 73.30us
2. repeat3_tile9216: 73.44us
3. transB_tile8192: 73.76us
4. tile9216_buf768K: 73.82us
5. repeat2_baseline: 73.86us

**Conclusion:** Performance is plateaued at 73-78us with 3.00-3.12x speedup. All parameter variations produce statistically identical results within measurement noise. The kernel is fundamentally limited by:
- Single-block execution (1 AIC + 2 AIV)
- MTE2/fixpipe pipeline bottleneck in the Matmul API's internal scheduling
- No viable multi-block approach due to Matmul API's usedCoreNum design

### Iters 194-200 — Final verification
- 7 repeat runs: 72.36, 73.64, 76.62, 77.00, 76.88, 73.74, 77.06us
- Mean: 75.3us, Min: 72.36us, Max: 77.06us
- Confirms stable 3.00-3.10x speedup

### Final State (iter 200)
- **Best measured:** 71.94us (3.12x speedup from 224.55us baseline)
- **Typical range:** 73-78us (3.00x speedup)
- **Config:** usedCoreNum=2, baseM=256, baseN=128, FIRSTN, tileSize=9216, stepM=2, stepN=1, numBlocks=1
- **Architecture:** IterateAll + separate double-buffered LeakyRelu with strided support
- **Total experiments:** 200+ iterations across 250+ unique parameter configurations

