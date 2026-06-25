# Iteration Log

规则：同一批 SWEEP（同一个脚本/循环里跑的多组参数）只算 1 轮迭代。

## Summary

| Iter | Title | Speedup | Runtime | Status |
|------|-------|---------|---------|--------|
| 1 | 128x128 tiles + uint32 tiling copy | 1.067x | 212.97 us | improved |
| 2 | baseN=160 | 1.060x | 214.39 us | regression |
| 3 | baseM=64 | 0.783x | 290.41 us | regression |
| 4 | Iterate<false> async + barrier experiments | N/A | N/A | failed |
| 5 | Two-pass IterateAll + separate LeakyReLU | N/A | N/A | failed |
| 6 | Tiling sweep (baseM/baseN/cores/traverse, 42 configs) | 1.136x | 201.03 us | improved |
| 7 | Buffer+config sweep (UB/L1/step/soc, 10 configs) | 1.144x | 199.57 us | improved |
| 8 | Confirm best: 256x128 + UB=96KB | 1.130x | 201.21 us | no-change |
| 9 | **⭐ __mix__(1,1) breakthrough** | **2.068x** | **109.94 us** | **improved** |
| 10 | mix(1,1) param sweep (UB/tiling/cores/bias/shape, 19 configs) | 2.268x | 100.68 us | improved |
| 11 | Combination optimization sweep (6 configs) | 2.224x | ~101 us | no-change |
| 12 | IBShareB | 2.257x | 101.18 us | no-change |
| 13 | **CFG_MDL pipeline** | **2.289x** | **99.78 us** | **improved** |
| 14 | **CFG_MDL + enUnitFlag** | **2.377x** | **~96 us** | **improved** |
| 15 | IBShareB on CFG_MDL+unitFlag | 2.389x | 95.58 us | no-change |
| 16 | CFG_NORM + unitFlag | 2.281x | 100.12 us | regression |
| 17 | auto split (no SetFixSplit) | N/A | N/A | failed |
| 18 | UB=auto | 2.354x | 97.00 us | no-change |
| 19 | UB=64KB | 2.351x | 97.12 us | no-change |
| 20 | L1=256KB | 2.319x | 98.48 us | no-change |
| 21 | no step override | 2.356x | 96.94 us | no-change |
| 22 | EnableBias | 2.38x | ~96 us | no-change |
| 23 | EnableBias + no step | 2.372x | 96.26 us | no-change |
| 24 | Sync-mode exhaustive sweep (DB/precompute/soc/tile/baseK/cores etc, 17 configs) | 2.397x | 95.28 us | no-change |
| 25 | **⭐ Async architecture (Iterate<false>+GetTensorC<false>)** | **3.213x** | **71.08 us** | **improved** |
| 26 | splitRowNums=2 | 3.315x | 68.88 us | improved |
| 27 | splitRowNums=1 | N/A | N/A | failed |
| 28 | splitRowNums=8 | 3.083x | 74.08 us | regression |
| 29 | UB=96KB async | 3.287x | 69.48 us | no-change |
| 30 | mix(1,1)+async | N/A | N/A | failed |
| 31 | SetShape(-1,-1,K) async | 3.360x | 67.96 us | improved |
| 32 | EnableBias async | 3.314x | 68.90 us | no-change |
| 33 | 128x128 async | 2.913x | 78.40 us | regression |
| 34 | auto split async | N/A | N/A | failed |
| 35 | 4 cores async | N/A | N/A | failed |
| 36 | no step async | 3.340x | 68.38 us | no-change |
| 37 | CFG_NORM async | 2.504x | 91.20 us | regression |
| 38 | no unitFlag async | 3.002x | 76.08 us | regression |
| 39 | IBShareB async | 3.295x | 69.30 us | no-change |
| 40 | L1=256 async | 3.262x | 70.00 us | no-change |
| 41 | 256x64 async | 1.902x | 120.04 us | regression |
| 42 | no traverse async | 3.263x | 69.98 us | no-change |
| 43 | split4+SetShape auto | 3.310x | 69.00 us | no-change |
| 44 | 1 core async | N/A | N/A | failed |
| 45 | preload=2 (GetMDLConfig) | 3.372x | 67.72 us | improved |
| 46 | preload=3 | 3.311x | 68.98 us | no-change |
| 47 | preload=4 | 3.444x | 66.30 us | no-change |
| 48 | preload=1 | 3.285x | 69.52 us | no-change |
| 49 | **enShuffle=true** | **3.465x** | **65.90 us** | **improved** |
| 50 | shuffle+preload4 | 3.343x | 68.32 us | no-change |
| 51 | shuffle+preload3 | 3.303x | 69.14 us | no-change |
| 52 | IBShareB+shuffle | 3.407x | 67.02 us | no-change |
| 53 | UB=96+shuffle | 3.375x | 67.66 us | no-change |
| 54 | L1=256+shuffle | 3.381x | 67.54 us | no-change |
| 55 | Shuffle tuning sweep (EB/split4/DB/noStep/baseK etc, 10 configs) | 3.491x | 65.42 us | no-change |
| 56 | Architecture variants sweep (CType_VECIN/L1+L0C/UB/split8-16/128x64/Muls, 9 configs) | 3.375x | 67.66 us | no-change |
| 57 | Fine-tune sweep (alpha/noTraverse/UB32/L1/BiasVECIN/preload5-8 etc, 10 configs) | 3.452x | 66.16 us | no-change |
| 58 | Advanced config sweep (preload0/noStep+EB/noUnit/L1+L0C+UB/split4+p4 etc, 10 configs) | 3.414x | 66.88 us | no-change |
| 59 | splitRowSize/baseK/Shape combos sweep (10 configs) | 3.414x | 66.88 us | no-change |
| 60 | DB/preload6-10/BiasVECIN/UB128/L1=1M/numBlocks sweep (10 configs) | 3.445x | 66.28 us | no-change |
| 61 | UB/L1/tile/IBShareAB/traverse variations sweep (10 configs) | 3.434x | 66.52 us | no-change |
| 62 | MDL config combos/cores/tile sweep (10 configs) | 3.452x | 66.16 us | no-change |
| 63 | Misc params sweep (alpha_f/inQ_depth/L0C/hardcode/L2Cache etc, 10 configs) | 3.501x | 65.22 us | no-change |
| 64 | **baseK=64 + preload=4** | **3.501x** | **65.22 us** | **improved** |
| 65-67 | Fine-tuning sweeps on async baseK64 (30 configs total) | ~3.4x | ~67 us | no-change |
| 68 | MatmulConfig doMTE2Preload=1 (M失败) / =2 (更差78us) | 2.900x | 78.74 us | regression |
| 69 | MatmulConfig fields sweep (SpecialMDL/IBShareNorm fail, intCheck noise) | 3.334x | 68.50 us | no-change |
| 70 | baseK sweep {32,48,64,96,128,256,auto} on async | 3.379x | 67.58 us | no-change |
| 71 | Buffer combo sweep (UB/L1) on async | 3.386x | 67.44 us | no-change |
| 72 | EnableBias/noStep/intCheck/noExternC/2x_ws/split4/outDB sweep | 3.431x | 66.56 us | no-change |
| 73 | tile/IBShare/cores/FIRSTN sweep | 3.349x | 68.18 us | no-change |
| 74 | UB/L1 fine sweep {80-160KB, 128KB-1MB} | 3.411x | 66.94 us | no-change |
| 75 | **⭐ Two-phase: IterateAll→GM + in-place LeakyReLU** | **4.155x** | **54.96 us** | **improved** |
| 76 | Balanced LeakyReLU across cores | 4.143x | 55.12 us | no-change |
| 77 | No traverse with two-phase | 4.170x | 54.76 us | no-change |
| 78 | **⭐ SetDim(4) with IterateAll** | **6.495x** | **35.16 us** | **improved** |
| 79 | **⭐ Core count sweep → SetDim(8) best** | **8.837x** | **25.84 us** | **improved** |
| 80 | 8-core tuning sweep (autoSplit/UB/EB/noUnit/NORM/tileSize) | 9.026x | 25.30 us | no-change |
| 81 | 8-core config sweep (preloadN/intCheck) | 8.797x | 25.96 us | no-change |
| 64 | **baseK=64 + preload=4** | **3.501x** | **65.22 us** | **improved** |
| 65 | baseK64+preload fine-tune sweep (p2-8/noShuf/UB/noStep/EB/split4/k128 etc, 10 configs) | 3.402x | 67.12 us | no-change |
| 66 | baseK64 buffer sweep (L1/L0C/preload3-6-12/baseK48/IBShareB etc, 10 configs) | 3.433x | 66.52 us | no-change |
| 67 | baseK64 final sweep (baseK80-96/256x64/split8/128x128/noUnit etc, 10 configs) | 3.366x | 67.84 us | no-change |

## Current Best

- **Async `__aicore__`**: `Iterate<false>` + `GetTensorC<false>`
- **`GetMDLConfig(true, true, 2)`**: MDL + unitFlag + shuffle + preload=2
- **`SetFixSplit(256, 128, 64)`**: baseK=64 for smaller K iterations
- **`splitRowNums=2`**, **`SetDim(2)`**, **`FIRSTM`**, **`SetShape(-1,-1,K)`**
- **Runtime: ~66-70 us median ~68 us (3.35x)**
- **Correctness: PASS (MERE 5.43e-07)**
