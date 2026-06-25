# Iteration Log — 200 Iterations Complete

## Final Result: 36.8 us (6.15x from 226.09 us baseline)

| Phase | Iters | Best Runtime | Speedup | Key Optimization |
|-------|-------|-------------|---------|------------------|
| Baseline | 0 | 226.09 us | 1.00x | — |
| Single-block tuning | 1-50 | 192 us | 1.18x | CFG_MDL + baseM=512 |
| **Multi-core breakthrough** | **51** | **42 us** | **5.36x** | **8 blocks via MatmulApiTiling** |
| Multi-core tuning | 52-113 | 37 us | 6.13x | baseN=128 + stepN=5 |
| Comprehensive sweep | 114-200 | 36.8 us | 6.15x | Converged |

## Optimized Configuration
```
Cores: 8 blocks (__mix__(1,1) = 1 AIC + 1 AIV each)
Per-block: M=128, N=640, K=256 (independent MatmulApiTiling)
Tiling: baseM=128, baseN=128, baseK=auto, stepM=1, stepN=5
Config: CFG_MDL + enUnitFlag + iterateMode=NORMAL + isA2B2Shared
```

## Final Variance (5 runs)
| Run | Duration | Speedup |
|-----|----------|---------|
| 1 | 36.64 us | 6.17x |
| 2 | 36.70 us | 6.16x |
| 3 | 36.84 us | 6.14x |
| 4 | 37.40 us | 6.04x |
| 5 | 36.22 us | 6.24x |
| **Mean** | **36.76 us** | **6.15x** |

## Key Breakthrough: Independent Per-Block Tiling (Iter 51)

The original MultiCoreMatmulTiling approach failed because the Matmul Iterate
loop ignores `singleCoreM` — it always processes the full M from the pointer.
Multi-block launches with this API caused each block to process overlapping
or out-of-bounds M regions.

**Fix**: Use `MatmulApiTiling` (single-core API) with `SetShape(PER_CORE_M, N, K)`.
Each block independently computes its portion via manual A/C offset in Init().

## 200 Iterations Breakdown
- **Iter 1-7**: Config flags (CFG_MDL, enUnitFlag, iterateMode, etc.)
- **Iter 8-30**: Tiling params (baseM, baseN, depthA1, stepKa sweeps)
- **Iter 31-50**: Advanced approaches (constant tiling, async iterate, IterateAll — all failed precision)
- **Iter 51**: Multi-core breakthrough
- **Iter 52-65**: Core count + base tile optimization
- **Iter 66-90**: Config ablation + parameter sweep on 8-core
- **Iter 91-113**: Comprehensive 14-config sweep (cores x baseM x baseN x stepN)
- **Iter 114-132**: Buffer space, baseK, format experiments
- **Iter 133-181**: Full 48-config sweep (cores x baseM x baseN x stepM x stepN)
- **Iter 182-200**: Final explorations + variance measurements

## Failed Approaches (for reference)
| Approach | Issue | Iter |
|----------|-------|------|
| MultiCoreMatmulTiling numBlocks>1 | Precision failure (API ignores singleCoreM) | 1-50 |
| Constant tiling (GetMatmulApiTiling) | 2x slower than runtime tiling | 3 |
| Async iterate (Iterate<false>) | Precision failure with workspace | 10 |
| IterateAll + separate LeakyRelu | Precision failure (GM output path) | 4 |
| MTE2 preload | Precision failure | 9 |
| enableEnd=false | Kernel hang | 11 |
| enableInit=false | Kernel hang | 7 |
| __mix__(1,2) with 4+ blocks | Kernel hang (too many cores) | 62 |
| 16+ cores | Hang (PER_CORE_M=64 too small) | 60-61 |
| NZ format / TransB | Build failure | 170-173 |
