# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 0 | Baseline (single core) | 1.00x | 408.34 us | baseline |
| 1 | Multi-core (SetDim+SetSingleShape) | N/A | N/A | failed |
| 2 | Multi-core SetDim(20) explicit | N/A | N/A | failed |
| 3 | Multi-core async iterate (2 cores) | N/A | N/A | failed |
| 4 | Two-pass: cube matmul + vector LeakyReLU | 18.9x | 21.60 us (12.44+9.16) | improved |
| 5 | LeakyReLU param sweep (cores, chunks) | 18.9x | 21.44 us (best combo) | no-change |
| 6 | CFG_MDL mode for matmul | 17.8x | 22.88 us (13.12+9.76) | regression |
| 7 | MixDualMaster + IBShare | 19.8x | 20.60 us (11.86+8.74) | improved |
| 8 | Matmul core count sweep (4/8/16/22) | 19.8x | 20.60 us (22 cores best) | no-change |
| 9 | Remove inter-kernel sync | 19.8x | 20.60 us | no-change |
| 10 | L1 cache depth tuning (depth=1) | 18.9x | 21.58 us | regression |
| 11 | MDL + MixDualMaster combined | 18.8x | 21.78 us | regression |
| 12 | Fused MIX mode 4 cores | N/A | N/A | failed |
| final | Verify best config (iter-7) | **~19x** | **~21 us** (12.6+9.2 avg) | **best** |
| 13 | Re-profile current best | 19.5x | 20.86 us (11.78+9.08) | profiling |
| 14 | LeakyReLU double buffer depth=2 | 19.2x | 21.30 us (12.22+9.08) | no-change |
| 15 | LeakyReLU larger UB chunks (23552 floats) | 19.7x | 20.74 us (12.22+8.52) | improved |
| 16 | SetSingleShape(128,320,256) matmul | N/A | N/A | failed |
| 17 | LeakyReLU 10 cores | 18.4x | 22.10 us (12.20+9.90) | regression |
| 18 | LeakyReLU TBuf+SetFlag (no TQue) | N/A | N/A | failed |
| 19 | SetFixSplit(64,128,-1) | 18.9x | 21.60 us (12.28+9.32) | no-change |
| 20 | FIRSTN traverse + FIRSTM traverse | 19.5x | 20.80 us (11.54+9.26) | improved |
| 21 | FIRSTM traverse (verified) | 19.6x | 20.80 us (11.54+9.26) | improved |
| 22 | LeakyReLU 16 cores | 18.6x | 21.92 us (12.86+9.06) | regression |
| 23 | Verify FIRSTM + large UB | 19.1x | 21.38 us (12.52+8.86) | no-change |
| 24 | ConstantTiling + MatmulProblemShape | 19.3x | 21.38 us (11.50+9.88) | improved(matmul) |
| 25 | Verify ConstantTiling (repeat) | 19.1x | 21.44 us (11.56+9.88) | confirmed |
| 25b | ConstantTiling launch-count=10 avg | **20.4x** | **19.98 us** (10.74+9.24) | **new best** |
| 26 | ConstantTiling + CONFIG_MDL | 18.5x | 22.08 us (12.16+9.92) | regression |
| 27 | ConstantTiling + UnitFlag | 17.9x | 22.62 us (12.78+9.84) | regression |
| 28 | ConstantTiling baseM=128 | 19.6x | 20.82 us (11.08+9.74) | no-change |
| 29 | ConstantTiling singleCoreM=104 | 17.9x | 22.44 us (13.54+8.90) | regression |
| 30 | Verify best (lc=10 avg) | **20.1x** | **20.34 us** (11.48+8.86) | **best** |
| 31 | CONFIG_IBSHARE | N/A | N/A | failed |
| 32 | Manual Muls+Max LeakyReLU | 19.6x | 20.34 us (11.14+9.20) | no-change |
| 33 | In-place single buffer LeakyReLU | 19.0x | 21.42 us (11.94+9.48) | regression |
| 34 | L2 cache hint on LeakyReLU | 19.9x | 20.50 us (11.56+8.94) | no-change |
| 35 | L1 depth override (depth=4) | 19.7x | 20.72 us (11.56+9.16) | no-change |
| 36 | 8 cores + double buffer LeakyReLU | 18.3x | 22.22 us (11.96+10.26) | regression |
| 37 | Final verify best (lc=10 avg) | **20.6x** | **19.80 us** (11.04+8.76) | **best** |
| 38 | MDL+Preload+MixDualMaster | 18.9x | 21.44 us (12.30+9.14) | regression |
| 39 | LeakyReLU 32 cores | 17.2x | 23.74 us (10.94+12.80) | regression |
| 40 | Verify current best | 19.8x | 20.52 us (11.00+9.52) | confirmed |
| 41 | LeakyReLU 24 cores | 19.0x | 21.40 us (11.54+9.86) | regression |
| 42 | Final measurement | 18.9x | 21.54 us (11.78+9.76) | confirmed |
| 43 | 5×5 variability test | ~20x median | ~20.4 us median (11.16+9.22) | confirmed |
| 44 | Re-profile current best | 19.9x | 20.50 us (11.60+8.90) | profiling |
| 45 | CONFIG_NORM + depth/step L1 tuning | 20.3x | 20.10 us (11.24+8.86) | improved(marginal) |
| 46 | CONFIG_MDL + constant tiling + depth | 19.4x | 21.02 us (11.36+9.66) | regression |
| 47 | 16 cores baseM=128 + depth tuning | 19.7x | 20.74 us (11.54+9.20) | no-change |
| 48 | L1 depth/step parameter sweep | ~20x | ~20.0 us (best dA4,dB4,sKa2,sKb2) | improved(marginal) |
| 49 | UnitFlag + depth tuning | 19.5x | 20.92 us (11.66+9.26) | regression |
| 50 | enableStaticPadZeros | N/A | hung (ND2NZ format required) | failed |
| 51 | Double buffer depth=2 LeakyReLU | 19.5x | 21.00 us (11.70+9.30) | regression |
| 52 | LeakyReLU core count sweep | ~20x | best 14 cores: median 20.6us | no-change |
| 53 | SetSingleShape(96,320/640) | N/A | tiling forced 24 cores, precision fail | failed |
| 54 | IBShare removal test | 19.8x | 20.64 us (11.74+8.90) | regression (IBShare helps) |
| 55 | Hardcoded LeakyReLU tiling constants | ~20x | ~20 us (marginal improvement) | improved(marginal) |
| 56 | Muls+Max manual LeakyReLU | ~20x | ~20.4 us (similar to API) | no-change |
| 57 | doSpecialBasicBlock | N/A | compile error (incompatible with IBShare) | failed |
| 58 | Unrolled LeakyReLU loop | ~20x | 20.58 us (11.72+8.86) | no-change |
| 59 | UB_CHUNK=24576 (full UB) | 19.1x | 21.42 us (11.84+9.58) | regression |
| 60 | stepM/stepN removal test | ~20x | 20.40 us (similar, defaults are 1) | no-change |
| 61 | Deep profiling analysis | 19.7x | 20.76 us (10.90+9.86) | profiling |
| 62 | Fused MIX multi-core matmul+LeakyReLU | N/A | hung (constant tiling incompatible with MIX Iterate) | failed |
| 63 | Compiler -O2 optimization | 19.4x | 21.00 us (11.82+9.18) | regression |
| 64 | FIRSTN traverse mode | ~20x | 20.18 us (12.08+8.10) | no-change |
| 65 | LeakyReLU 10 cores | ~19x | 21.20 us (11.82+9.38) | regression |
| 66 | singleCoreN=640, baseN=320 alignment fix | N/A | precision fail (host/kernel tiling mismatch) | failed |
| 67 | M-only split 8 cores (avoid N-boundary misalignment) | 17.5x | 23.42 us (14.50+8.92) | regression |
| 68 | depth/step param sweep (53 combos) | ~20x | best dA4,dB2,sKa2,sKb2=18.92 sweep, ~20.9 verified | no-change |
| 69 | Double-buffer LeakyReLU (bufNum=2, 11776 chunk) | ~20x | 21.04 us (12.34+8.70) | no-change |
| 70 | MTE2 Preload (M and N direction) | ~20x | ~20.5 us (M-preload), ~21.4 us (N-preload) | no-change |
| 71 | enableUBReuse + enableL1BankConflictOptimise | N/A | UBReuse OK (~20.5us), L1BankConflict causes NaN | failed(partial) |
| 72 | MatmulConfig options sweep: enableKdimReorderLoad, enableDoubleCache | ~20x | ~20.5-21.5 us (within noise) | no-change |
| 73 | baseM/baseN sweep (25 combos) | ~20x | best bM96,bN192=19.52 sweep, ~20.6 verified | no-change |
| 74 | __cube__ LeakyReLU | N/A | 2.9 us but precision fail (no vector in cube mode) | failed |
| 74b | __mix__(1,1) LeakyReLU | ~20x | 20.27 us (12.02+8.25) | no-change |
| 75 | Stability measurement (10 runs) | **19.8x** | **median 20.66 us** (relu 9.02 + matmul 11.50) | confirmed |
| 76 | Single VECCALC buffer LeakyReLU (46080 chunk) | ~19.4x | 21.04 us (11.89+9.15) | regression |
| 77 | ITERATE_MODE_ALL + 10-run stability | ~19.3x | median 21.20 us (within noise of 20.66 baseline) | no-change |
| 78 | CONFIG_MDL with constant tiling | ~18.9x | 21.54 us (12.63+8.91) | regression |
| 79 | depthB1=2 (reduce L1 B footprint) | ~19.2x | 21.22 us (12.14+9.08) | regression |
| 80 | ACL_STREAM_FAST_LAUNCH + FAST_SYNC | ~18.6x | 21.94 us (13.18+8.76) | regression |
| 81 | Re-profile launch-count=10 | ~18.8x | 21.72 us (12.78+8.94) | profiling |
| 82 | LeakyReLU 10 blocks (match active blocks) | ~17.7x | 23.00 us (13.26+9.74) | regression |
| 83 | stepKa/stepKb sweep (9 combos) | ~20x | best sKa4,sKb4=19.88 (current=20.16) | no-change |
| 84 | depthA1/depthB1 sweep (dA4,dB8 best) | **~20.4x** | **median 20.02 us** (matmul ~11.30 + relu ~8.72) | **improved** |
| 85 | Step tuning with dA4,dB8 (sKa/sKb sweep) | ~20x | sKa2,sKb1=20.16, sKa1,sKb2=20.54, sKa2,sKb2=20.02 | no-change |
| 86 | LeakyReLU __vector__ mode (eliminate cube overhead) | **~23.1x** | **median 17.70 us** (matmul 11.88 + relu 5.82) | **improved** |
| 87 | LeakyReLU core count sweep (8-40 cores) | ~23x | 20 cores best (17.78); 40 cores=19.02 (worse) | no-change |
| 88 | UB chunk size sweep (16K-24K) | ~23x | 16K=17.92, 20K=17.06, 23K=17.70, 24K=17.22 | no-change |
| 89 | FIRSTN traverse | ~23x | median 17.58 (within noise of FIRSTM 17.70) | no-change |
| 90 | enUnitFlag=false | ~23x | median 17.68 (within noise) | no-change |
| 91 | Double buffer LeakyReLU in __vector__ mode | **~23.7x** | **median 17.24 us** (matmul ~11.88 + relu ~5.36) | **improved** |
| 92 | depthA1=2 with dB8 | ~23.5x | median 17.34 us (within noise) | no-change |
| 93 | DB chunk sweep (4K-12K) | **~24.3x** | **median 16.80 us** (8K optimal) | **improved** |
| 94 | baseM/baseN sweep (16 combos) | N/A | all precision fail (host/kernel tiling mismatch) | failed |
| 95 | OUTER_PRODUCT + ORDER_M matmul configs | N/A | OUTER_PRODUCT precision fail; ORDER_M within noise | failed/no-change |
| 96 | 20-run stability measurement | **~23.8x** | **median 17.16 us** (matmul 11.78 + relu 5.32) | confirmed |
| 97 | Triple buffer (bufNum=3) LeakyReLU | **~24.0x** | **median 17.00 us** (marginal improvement) | **improved** |
| 98 | enableKdimReorderLoad | ~24x | ~17.0 (within noise) | no-change |
| 99 | A_TYPE IBShare=false | ~24x | 17.00 (within noise) | no-change |
| 100 | stepKa=4,stepKb=4 with TB | ~23.8x | median 17.14 (within noise) | no-change |
| 101 | enableInit=false+enableGetTensorC=false+ITERATE_MODE_ALL | ~24x | median ~17.18 us (mm 11.96 + rl 5.40) | no-change |
| 102 | enableEnd=false (remove End() call) | ~21x | median ~19.16 us (mm 13.70 + rl 5.38) — regression | regression |
| 103 | enableSetOrgShape=false | N/A | precision fail (SetOrgShape still needed with constant tiling) | failed |
| 104 | LeakyReLU chunk sweep: 16384 single-buf=5.80, 7680 triple-buf=5.48, 5440 best | ~24x | 5440 triple-buf remains optimal at ~5.3-5.4 us | no-change |
| 105 | fp16 intermediate: matmul→half, LeakyReLU(half→cast→float) | **~24.8x** | **median 16.48 us** (mm 11.56 + rl 4.92) | **improved** |
| 106 | UB chunk sweep for half-input: 4096=4.86, 5440=4.48, 6144=4.78, 7168=4.72, 8192=5.04 | ~24.8x | 5440 remains optimal | no-change |
| 107 | Depth sweep for half output: dA2dB4=12.18, dA4dB4=11.22, dA8dB4=12.42med, dA4dB8=11.56med | ~24.8x | dA4dB8 remains best with half output | no-change |
| 108 | LeakyReLU in half then cast vs cast then LeakyReLU | ~24.8x | 4.94 us (same) — computation not bottleneck | no-change |
| 109 | Core count sweep for half-input: 10=5.86, 14=5.18, 20=4.96, 30=5.16, 40=5.96 | ~24.8x | 20 cores remains optimal | no-change |
| 110 | Double buffer (depth=2) for half-input LeakyReLU | ~23x | 5.52 us (triple-buffer better at 4.9) | regression |
| 111 | enableSetDefineData=false | ~24.8x | within noise (4.90+12.28=17.18) | no-change |
| 112 | MixDualMaster=false vs true for half output | ~24.8x | false:12.24med, true:11.38med — true still better | no-change |
| 113 | 5-run stability measurement | **~24.2x** | **median 16.86 us** (mm 11.88 + rl 4.98) | confirmed |
| 114 | isA2B2Shared=true | ~24x | ~17.0 us (within noise) | no-change |
| 115 | LeakyReLU on half first, then cast to float | ~24x | ~16.9 us (within noise) | no-change |
| 116 | Quad buffer (bufNum=4, chunk=4096) LeakyReLU | ~24x | ~16.6 us (within noise) | no-change |
| 117 | Remove calcBuf: Cast directly to outLocal + in-place LeakyReLU (6400 chunk) | **~24.8x** | **median 16.49 us** (mm ~11.8 + rl ~4.7) | **improved** |
| 118 | UB_CHUNK sweep (4096-9600) with no-calcBuf | ~24-25x | 6144=16.52, 6400=16.49 best | no-change |
| 119 | LeakyReLU core count sweep (14-30) with no-calcBuf | ~24x | 20 cores remains optimal | no-change |
| 120 | depthA1/depthB1 sweep (12 combos) with no-calcBuf | ~24x | dA4,dB8 remains optimal | no-change |
| 121 | stepKa/stepKb sweep (9 combos) | ~24x | all within noise (high variance) | no-change |
| 122 | Hardcode all matmul shapes (eliminate GM tiling read) | **~25.8x** | **median 15.84 us** (mm 11.24 + rl 4.66) | **improved** |
| 123 | Hardcode CalcOffset constants | ~25.8x | within noise | no-change |
| 124 | 16 cores clean tiling (no tail blocks) | ~24x | ~16.3 us (fewer cores hurts more) | regression |
| 125 | Asymmetric buffer depths (in=2/out=3, in=3/out=2) | ~25x | within noise | no-change |
| 126 | Profile analysis (matmul: MTE2 78%, fixpipe 81%, scalar 28%; LeakyReLU: scalar 40-59%) | — | — | profiling |
| 127 | UB_CHUNK=8192 (even 4 iters per core) | ~25x | relu 4.64, within noise | no-change |
| 128 | LeakyReLU on half then cast (vs cast then LeakyReLU) | ~25x | within noise | no-change |
| 129 | Eliminate shapes struct member entirely | ~25x | median 16.18 (within noise of 15.84) | no-change |
| 130 | NZ format matmul output | N/A | precision fail (NZ layout incompatible with ND LeakyReLU) | failed |
| 131 | Float32 intermediate (skip Cast in LeakyReLU) | ~23x | relu=5.5 (worse: 2x bandwidth) | regression |
| 132 | Single buffer max chunk (32768 elements) | ~24x | relu=5.1 (worse: no pipeline overlap) | regression |
| 133 | Muls+Max on half (manual LeakyReLU) + larger chunk 8192 | ~25x | relu=4.7 (within noise) | no-change |
| 134 | Comprehensive depth×chunk sweep (11 combos) | ~25x | D3 C6400 remains optimal | no-change |
| 135 | Remove enableInit/enableGetTensorC/enableQuantVector flags (defaults) | ~24x | mm=12.48 (worse: default adds overhead) | regression |
| 136 | OUTER_PRODUCT+ORDER_M+stepN=2 schedule | ~25x | 15.62-16.72 (within noise) | no-change |
| 137 | SpecialMDL mode | N/A | compile fail | failed |
| 138 | enableMixDualMaster=false | ~25x | 15.74 (single run, within noise) | no-change |
| 139 | 20-run definitive measurement | **25.2x** | **median 16.22 us** (mm 11.40 + rl 4.84) | confirmed |
| 140 | Remove IBShare from A/B types | ~25x | 15.86 (within noise) | no-change |
| 141 | depthA1=8 depthB1=4 | ~25x | 15.86 (within noise) | no-change |
| 142 | stepKa=4 stepKb=2 | ~25x | 15.52 (within noise) | no-change |
| 143 | stepKa=1 stepKb=4 | ~25x | 15.78 (within noise) | no-change |
| 144 | stepM=2 | ~25x | 16.70 (within noise) | no-change |
| 145 | stepN=2 | ~25x | 16.00 (within noise) | no-change |
| 146 | depthA1=2 depthB1=2 | ~24x | 16.86 | no-change |
| 147 | depthA1=6 depthB1=6 | ~25x | 16.42 | no-change |
| 148 | depthA1=4 depthB1=12 | ~25x | 16.08 | no-change |
| 149 | depthA1=8 depthB1=8 | ~25x | 16.26 | no-change |
| 150 | LeakyReLU 12 cores | ~24x | relu=5.92 (worse: too few cores) | regression |
| 151 | LeakyReLU 22 cores | ~25x | relu=4.72 median (within noise of 4.84) | no-change |
| 152 | UB_CHUNK=5120 | ~25x | 15.88 (within noise) | no-change |
| 153 | UB_CHUNK=7168 | ~25x | 16.12 (within noise) | no-change |
| 154 | depthA1=3 depthB1=8 | ~25x | 16.08 | no-change |
| 155 | depthA1=5 depthB1=8 | ~24x | 16.92 | no-change |
| 156 | depthA1=4 depthB1=6 | ~25x | 16.32 | no-change |
| 157 | depthA1=4 depthB1=10 | ~25x | 14.98 (outlier mm=10.36) | no-change |
| 158 | stepKa=3 stepKb=3 | ~25x | 15.54 | no-change |
| 159 | UB_CHUNK=6144 | ~23x | 18.02 (outlier mm=13.38) | no-change |
| 160 | UB_CHUNK=6656 | ~25x | 16.32 | no-change |
| 161 | LeakyReLU 18 cores | ~25x | 15.84 | no-change |
| 162 | FIRSTN traverse | ~25x | 15.66 | no-change |
| 163 | Verify final best config | **~26.2x** | **15.62 us** | confirmed |
| 164 | Re-profile current best (session start) | ~22.5x | 18.16 us (mm 13.36 + rl 4.80) | profiling |
| 165 | enableDoubleCache + enablePreload (compile fail) | ~25x | ~16.58 (DC within noise) | no-change |
| 166 | enableL1CacheUB | ~25x | ~16.10 median (within noise) | no-change |
| 167 | enableUBReuse | ~25x | ~15.88 median (higher max_abs_err) | no-change |
| 168 | M-only split (SCM=48, SCN=640, 22 cores) | ~24x | ~17.16 median (regression) | regression |
| 169 | ACL_MEM_MALLOC_NORMAL_ONLY | ~25x | ~16.48 median (within noise) | no-change |
| 170 | baseM/baseN/baseK sweep (partial, 5 combos) | ~25x | best bM48,bN256,bK64=15.76 (single run) | no-change |
| 171 | baseK=32 (more K-iters) | ~24x | ~17.36 median (regression) | regression |
| 172 | enableReuse=false | ~25x | ~16.74 median (within noise) | no-change |
| 173 | reversed depth (dA8,dB4,sKa4,sKb2) | ~25x | ~16.27 median (within noise) | no-change |
| 174 | Float intermediate (no half Cast) | ~25x | ~16.52 median (mm faster, relu slower, net zero) | no-change |
| 175 | baseK=128 precision fail; baseK=32 regression | ~24x | bK128 precision fail; bK32 ~17.36 | regression |
| 176 | OUTER_PRODUCT schedule type | ~25x | precision fail (faster mm but wrong) | failed |
| 177 | Quad buffer (bufNum=4, CHUNK=4096) LeakyReLU | ~25x | ~15.72 (within noise) | no-change |
| 178 | UB_CHUNK sweep (5120, 7680) | ~25x | all within noise | no-change |
| 179 | 20-core M-only split (SCM=52, SCN=640) | ~24x | ~17.08 median (regression) | regression |
| 180 | Comprehensive depth/step sweep (89 combos) | ~25x | best dA2,dB6,sKa2,sKb4=15.06 (single run, unstable) | no-change |
| 181 | Verify baseline after sweep cleanup | ~23x | 17.56 us (single run, within variance) | confirmed |
| 182 | 20-run stability measurement | **~26.0x** | **median 15.70 us** (11 valid/20 runs, mm ~11.0 + rl ~4.7) | **confirmed** |
| 183 | IBShare=false on A or B | N/A | both cause precision failure + matmul doesn't execute | failed |
| 184 | L2CacheHint NORMAL for input, DISABLE for output | ~24x | relu=5.56 (regression from L2 overhead) | regression |
| 185 | LeakyReLU 24 cores | ~25x | relu=5.00 (more cores = more overhead) | no-change |
| 186 | LeakyReLU 16 cores | ~24x | ~17.00 median (fewer cores hurts) | regression |
| 187 | dA4,dB4,sKa4,sKb4 | ~25x | ~16.04-17.14 (within noise) | no-change |
| 188 | enableMixDualMaster=false | ~25x | ~16.26 median (within noise) | no-change |
| 189 | Reduced workspace (50%) | ~25x | ~16.02 median (within noise) | no-change |
| 190 | UB_CHUNK=8192 (power of 2) | ~25x | ~16.18 (single valid run, noise) | no-change |
| 191 | depthA1=6 depthB1=8 | ~24x | ~17.02 median | no-change |
| 192 | FIRSTN traverse (host tiling) | ~25x | ~16.12 median (within noise) | no-change |
| 193 | EnableMultiCoreSplitK | N/A | precision fail (tiling changed to 16 cores, singleCoreM=1024) | failed |
| 194 | Verify dA6,dB12,sKa4,sKb1 + step tune sKa2,sKb2 | ~25x | dA6dB12sKa4sKb1: median ~16.10; sKa2,sKb2: median ~16.52 | no-change |
| 195 | MatmulCallBackFunc analysis | N/A | callbacks only customize data copy, not activation; cube-only has no vector | failed(design) |
| 196 | depthB1=16 | N/A | precision fail on all 5 runs | failed |
| 197 | Async IterateAll<false> | ~25x | median ~16.26 (within noise of sync version) | no-change |
| 198 | ACL_MEM_MALLOC_NORMAL_ONLY (all buffers) | ~23x | ~17.47 median (clear regression) | regression |
| 199 | enablePreload | N/A | compile fail (no such member in MatmulConfig) | failed |
| 200 | UB_CHUNK=8192 (triple buf) | ~25x | median ~16.06 (within noise of 6400) | no-change |
| 201 | 20-run stability measurement | **~25.3x** | **median 16.12 us** (mm 11.42 + rl 4.69) | **confirmed** |
| **best** | **dA6,dB12,sKa4,sKb1, CHUNK=6400, triple buf, __vector__ relu** | **~25.3x** | **median 16.12 us** (20-run, range 14.88-17.74) | **best** |

## Baseline

- **Task Duration**: 408.34 us (single core, Block Dim=1, Mix Block Dim=2)
- **Correctness**: PASS (max_abs_err=5.72e-05, cosine_sim=0.9999999)
- **Frequency**: 1800 MHz (rated)
- **Key Profile**:
  - AIC: cube_ratio=11.9%, scalar_ratio=97.8%, mte2_ratio=32.6%, fixpipe_ratio=58.7%
  - AIV vector0: vec_ratio=1.5%, mte2_ratio=3.3%, mte3_ratio=5.1%
  - AIV vector1: vec_ratio=1.6%, time=218.2 us (half of vector0)
- **Bottlenecks**:
  1. Single core execution (Block Dim=1) — biggest opportunity
  2. AIC scalar_ratio=97.8% — cube core mostly doing scalar ops
  3. fixpipe_ratio=58.7% — high fixpipe overhead
  4. Low cube_ratio=11.9% — cube underutilized

## Iterations

### Iter 1 — Enable multi-core (20 cores via SetDim + SetSingleShape)

- **Hypothesis:** Using all available Cube Cores (SetDim=GetCoreNumAic, SetSingleShape(256,128,256)) should distribute work across 20 cores, reducing latency ~20x from single core baseline.
- **Changes:** Changed SetDim(1) to SetDim(GetCoreNumAic()), added SetSingleShape(256, 128, 256).
- **Bench:**
  - Compiled: True
  - Correct: False (cosine_sim=0.49, 73.5% mismatches)
  - Runtime: N/A (failed correctness)
  - Speedup: N/A
- **Analysis:** Tiling API returned usedCoreNum=24 (not 20 as expected). The CopyOut offset calculation in kernel assumes FIRSTM traversal order with specific singleCoreM/singleCoreN values. With 24 cores, the auto-tiling changed the cut differently from what the CopyOut logic expects. Reverted.
- **Next:** Need to understand the Matmul library's multi-core behavior better. Try using IterateAll instead of manual Iterate+CopyOut, or fix CopyOut to work with auto-tiling results.

### Iter 2 — Multi-core SetDim(20) explicit

- **Hypothesis:** Explicitly setting SetDim(20) with SetSingleShape(256,128,256) to exactly match 4×5=20 core split might fix the correctness issue.
- **Changes:** Set usedCoreNum=20, added SetSingleShape(256,128,256).
- **Bench:**
  - Compiled: True
  - Correct: False (cosine_sim=0.558, 63.8% mismatches)
  - Runtime: N/A
  - Speedup: N/A
- **Analysis:** Even with exact 20 cores, correctness still fails. The CopyOut offset logic appears correct mathematically, so the issue is likely deeper — possibly related to MIX mode workspace management or Matmul library's internal multi-core synchronization when using VECIN output. The matmul library's Iterate+GetTensorC pattern in MIX mode may require specific synchronization that differs from single-core.
- **Next:** Try approach 1: Use IterateAll to output C directly to GM, then do LeakyReLU as a separate vector pass on GM data. Or approach 2: Study the matmul_splitm example more carefully for MIX mode multi-core + VECIN.

### Iter 3 — Multi-core async iterate (2 cores, aicore mode)

- **Hypothesis:** Following the official matmul_leaky_relu_async example pattern with SetDim(2), __aicore__ mode, async Iterate<false> and workspace should enable 2-core execution.
- **Changes:** Rewrote kernel to match official async example: __aicore__ mode, SetDim(2), workspace for async iterate, split LeakyRelu into 4 row chunks. Changed SetBias to SetBias (matching example API).
- **Bench:**
  - Compiled: True
  - Correct: False (max_abs_err=93.4, cosine_sim=0.39, all rows wrong)
  - Runtime: 75.40 us (but invalid due to precision)
  - Speedup: N/A
- **Analysis:** Despite following the official example closely, correctness failed completely. Investigation shows the output has correct magnitude range but wrong values everywhere — both cores are producing incorrect results. The CopyOut offset formula with FIRSTM traversal and splitRowNums=4 should be correct for singleCoreM=512, singleCoreN=640, but the data is scrambled. Likely issue: the tiling API's SetBias vs EnableBias API difference, or the workspace offset calculation `GetBlockIdx() * singleCoreM * singleCoreN` may conflict with system workspace. Reverted.
- **Next:** Try a fundamentally different approach: two-pass (cube matmul to GM + separate vector LeakyReLU) to completely decouple multi-core matmul from post-processing.

### Iter 4 — Two-pass: cube matmul (22 cores) + vector LeakyReLU (20 cores)

- **Hypothesis:** Decoupling matmul from LeakyReLU into two separate kernel launches avoids the multi-core VECIN CopyOut issue. Pass 1 uses __cube__ mode with IterateAll to output C→GM automatically. Pass 2 uses __aicore__ mode to apply LeakyReLU in-place with multi-core parallelism.
- **Changes:**
  - Kernel 1: __cube__ with IterateAll(cGlobal), SetDim(GetCoreNumAic()), SetTail for edge cores
  - Kernel 2: __aicore__ LeakyReLU with 20 cores, chunked UB processing (16K floats per chunk)
  - Two kernel launches in main()
  - #define ASCENDC_CUBE_ONLY for cube-only mode
- **Bench:**
  - Compiled: True
  - Correct: True (max_abs_err=5.72e-05, cosine_sim=0.9999999)
  - Matmul: 12.44 us (22 cube cores)
  - LeakyReLU: 9.16 us (20 vector cores)
  - Total: 21.60 us
  - Speedup: 18.9x
- **Analysis:** Massive improvement! The two-pass approach works perfectly. The tiling API auto-selected baseM=96, baseN=256, singleCoreM=94, singleCoreN=320, usedCoreNum=22. LeakyReLU is now the bottleneck at 9.16us (42% of total). The LeakyReLU kernel is a simple element-wise operation on 655K float32 values — it should be memory-bandwidth bound.
- **Next:** Optimize LeakyReLU kernel (reduce overhead, increase chunk size). Also consider if tiling params (baseM/baseN) can be tuned for better matmul performance.

### Iter 5 — LeakyReLU parameter sweep (cores and chunk sizes)

- **Hypothesis:** Tuning LeakyReLU core count and UB chunk size could reduce the 9.16us overhead.
- **Changes:** Tested combinations: 10/20/40 cores × different chunk sizes (8K/16K/20K floats) × single/double buffer.
- **Bench:**
  - SetFixSplit(128,128,-1): tiling FAILED (0 cores), aborted
  - 20 cores, double-buffer 8K: relu=9.48us, matmul=11.96us, total=21.44us
  - 10 cores, single 20K: relu=10.24us, matmul=12.20us, total=22.44us (worse)
  - 40 cores, single 20K: relu=14.00us, matmul=12.02us, total=26.02us (worse)
  - Best: 20 cores, single-buffer 16K floats (iter-4 config: 21.60us total)
- **Analysis:** The LeakyReLU kernel is near bandwidth-limited at 9.16us for 2.5MB data (read+write). 20 cores with 16K float chunks is the sweet spot. Fewer cores means more work per core (slower). More cores means too much launch overhead. Double buffering didn't help because the per-core workload is too small for overlapping.
- **Next:** Focus on reducing matmul time. Try MDL mode, L1 cache optimization, or different tiling strategies for the cube kernel.

### Iter 6 — CFG_MDL mode for matmul

- **Hypothesis:** Multi-Depth Loading (MDL) overlaps data loading with computation, potentially reducing matmul latency.
- **Changes:** Changed matmul template config from default CFG_NORM to CFG_MDL.
- **Bench:**
  - Compiled: True, Correct: True
  - Matmul: 13.12 us (regression from 12.44us), LeakyReLU: 9.76 us
  - Total: 22.88 us, Speedup: 17.8x
- **Analysis:** MDL mode is slower for this problem size (1024×640×256). The overhead of managing multiple depth levels for L1 outweighs benefits. MDL is designed for very large matrices (e.g., 8192×8192) where data reuse in L1 matters. Reverted.

### Iter 7 — MixDualMaster mode with IBShare

- **Hypothesis:** enableMixDualMaster allows both vector cores in a block to independently copy out results, doubling MTE3 bandwidth. IBShare enables input buffer sharing between AIC and AIV for A and B matrices.
- **Changes:** Added custom MatmulConfig with enableMixDualMaster=true. Added IBShare=true for A and B MatmulType parameters. Used matmul_mixdualmaster example as reference.
- **Bench:**
  - Compiled: True, Correct: True (max_abs_err=5.72e-05, cosine_sim=0.9999999)
  - Matmul: 11.86 us (improved from 12.44), LeakyReLU: 8.74 us (improved from 9.16)
  - Total: 20.60 us, Speedup: 19.8x
- **Analysis:** MixDualMaster improved matmul by ~5%. IBShare allows more efficient memory access. The LeakyReLU improvement is likely due to better L2 cache behavior. New best result.
- **Next:** Try further matmul optimizations: L1 cache parameters (depthA1, stepKa), or different core counts. Also consider fusing the two passes back into one kernel to eliminate the kernel launch overhead.

### Iter 8 — Matmul core count sweep

- **Hypothesis:** Different core counts may have different tiling efficiency.
- **Changes:** Tested SetDim(4), SetDim(8), SetDim(20), SetDim(GetCoreNumAic=20).
- **Bench:**
  - 4 cores: matmul=19.72us, total=28.88us
  - 8 cores: matmul=14.04us, total=23.66us
  - 16 cores (auto from SetDim=20): matmul=11.70us, total=20.78us
  - 22 cores (GetCoreNumAic): matmul=11.86us, total=20.60us ← best
- **Analysis:** More cores is generally better. The auto-tiling with GetCoreNumAic produces the best overall result. The sweet spot is maximum available cores (22 for cube on this platform).
- **Next:** Try to eliminate the kernel launch overhead between the two passes, or explore L1/L2 cache optimization.

### Iter 9 — Remove inter-kernel sync

- **Hypothesis:** Removing the aclrtSynchronizeStream between the two kernel launches allows back-to-back execution.
- **Changes:** Removed first aclrtSynchronizeStream between matmul and leakyrelu kernels.
- **Bench:** Matmul=11.74, LeakyReLU=9.46, Total=21.20 us (marginal, within noise)
- **Analysis:** The kernels on the same stream already execute in order; the sync is only needed for host-side correctness. Removing it doesn't change profiled kernel times.

### Iter 10 — L1 cache depth tuning

- **Hypothesis:** Setting depth parameters (depthA1=1, stepKa=1) could reduce L1 overhead for small K.
- **Changes:** Added set_depthA1(1), set_depthB1(1), set_stepM(1), set_stepN(1), set_stepKa(1), set_stepKb(1).
- **Bench:** Matmul=13.14 (regression!), LeakyReLU=8.44, Total=21.58 us
- **Analysis:** Reducing L1 depths hurt matmul performance. The auto-tuned depths were better for this problem size. Reverted.

### Iter 11 — MDL + MixDualMaster combined

- **Hypothesis:** Combining MDL with MixDualMaster might get the benefits of both.
- **Changes:** Changed CFG_NORM to CFG_MDL in the custom config.
- **Bench:** Matmul=12.76 (regression from 11.86), LeakyReLU=9.02, Total=21.78 us
- **Analysis:** MDL mode consistently hurts for this problem size. The matrix is too small for multi-depth loading to benefit. Reverted.

### Iter 12 — Fused MIX mode with 4 cores (attempt)

- **Hypothesis:** A single-pass fused MIX kernel with 4 cores could avoid the second kernel launch overhead.
- **Changes:** Reverted to MIX mode with 4 cores, SetSingleShape(256,640,256), SetFixSplit(256,128,-1).
- **Bench:**
  - Compiled: True
  - Correct: False — cores 0 and 3 correct, cores 1 and 2 completely wrong (cosine_sim~0.31)
- **Analysis:** MIX mode multi-core has a fundamental correctness issue with the matmul library for non-edge cores. Only __cube__ mode with IterateAll handles multi-core correctly. This confirms the two-pass approach is the correct architecture for this operator.
- **Conclusion:** Reverted to iter-7 (best) configuration.

### Iter 13 — Re-profile current best (detailed analysis)

- **Hypothesis:** Need detailed profiling to identify specific bottlenecks for next optimization.
- **Changes:** No code changes. Fresh profiling with msprof op.
- **Bench:**
  - Compiled: True, Correct: True
  - Matmul: 11.78 us (22 cube cores), LeakyReLU: 9.08 us (20 vector cores, Mix Block Dim=40)
  - Total: 20.86 us, Speedup: 19.5x
- **Analysis:**
  - **Matmul**: Core imbalance ~30% (cores 0-9: 7.7-8.6us, cores 10-21: 9.9-10.9us). MTE2 ratio 43-62%, fixpipe ratio 39-62%. Each core reads ~209KB A data + 118KB B/output. baseM=96, baseN=256 auto-tiling.
  - **LeakyReLU**: Massive load imbalance — cores 0-9 process ~128KB each (real work), cores 10-19 process nearly 0KB. GM_to_UB bandwidth utilization only ~7%. Scalar overhead dominates (scalar_ratio ~16-29%). Only 3 MTE2 instructions per vector (UB_CHUNK=16K floats=64KB per chunk, 2 chunks per core).
  - **Key bottleneck**: LeakyReLU bandwidth utilization is far below theoretical. With 2.5MB total data, theoretical MTE2 time at 1.8TB/s = ~1.4us. Actual 9.08us means 6.5x overhead from kernel launch, scalar setup, and sync.
- **Next:** Optimize LeakyReLU: (1) Use larger UB chunks (40K floats = 160KB, fitting in 192KB UB), (2) Use double-buffer TQue depth=2 to overlap MTE2/MTE3 with Vector, (3) Consider reducing to 10 cores since half are idle anyway, (4) Try in-place single-buffer approach with TBuf.

### Iter 14 — LeakyReLU double buffer depth=2

- **Hypothesis:** Double buffer (depth=2) for TQue should overlap MTE2/MTE3 with Vector compute, hiding transfer latency.
- **Changes:** Changed inQueue/outQueue from depth=1 to depth=2, reduced UB_CHUNK to 11264 floats to fit 4 buffers in 192KB UB.
- **Bench:** Compiled: True, Correct: True. LeakyReLU: 9.08us (no change), Matmul: 12.22us. Total: 21.30us.
- **Analysis:** Double buffer didn't help because per-core workload is too small (only 2-3 chunks). The pipeline benefit is negated by smaller chunk size (44KB vs 64KB). Reverted to single buffer.

### Iter 15 — LeakyReLU larger UB chunks (23552 floats = 92KB)

- **Hypothesis:** Larger UB chunks reduce the number of MTE2/MTE3 transactions per core, reducing per-transaction overhead.
- **Changes:** Increased UB_CHUNK from 16384 (64KB) to 23552 (92KB) per buffer, total 184KB < 192KB UB.
- **Bench:** Compiled: True, Correct: True. LeakyReLU: 8.52us (improved from 9.08us), Matmul: 12.22us. Total: 20.74us, Speedup: 19.7x.
- **Analysis:** 6% improvement on LeakyReLU. Fewer chunks per core (2 instead of 3) means less loop overhead and fewer MTE transactions. New best for LeakyReLU.

### Iter 16 — SetSingleShape(128, 320, 256) for matmul

- **Hypothesis:** Explicit singleCoreM=128 (power of 2) should give better tiling regularity than auto-tiled singleCoreM=94.
- **Changes:** Added tilingApi.SetSingleShape(128, 320, K) to force tiling shape.
- **Bench:** Compiled: True, Correct: False (cosine_sim=0.76, 34% mismatches). Tiling gave usedCoreNum=24 (not expected 16).
- **Analysis:** The tiling API adjusted the core count to 24 instead of the expected 8×2=16, causing CalcOffset mismatch. Reverted.

### Iter 17 — LeakyReLU 10 cores (more data per core)

- **Hypothesis:** Reducing from 20 to 10 cores doubles per-core data (256KB), improving bandwidth utilization per core.
- **Changes:** Changed reluTiling.usedCoreNum from 20 to 10.
- **Bench:** Compiled: True, Correct: True. LeakyReLU: 9.90us (regression), Matmul: 12.20us. Total: 22.10us.
- **Analysis:** Despite better per-core bandwidth utilization, 10 cores is too few — the total parallelism reduction outweighs the per-core improvement. The kernel launch overhead per core is not the dominant factor. Reverted to 20 cores.

### Iter 18 — LeakyReLU TBuf+SetFlag (bypass TQue)

- **Hypothesis:** Using TBuf (no queue overhead) with manual SetFlag/WaitFlag synchronization should eliminate TQue management overhead.
- **Changes:** Replaced TQue with TBuf<VECCALC> + manual MTE2_V/V_MTE3/MTE3_V synchronization.
- **Bench:** Compiled: True, Correct: False (cosine_sim=0.83, 24% mismatches).
- **Analysis:** Manual synchronization with TBuf failed correctness — likely because DataCopy from TBuf doesn't properly trigger MTE2/MTE3 hardware events. The TQue pattern is required for correct MTE synchronization. Reverted to TQue approach.

### Iter 19-22 — Various matmul and LeakyReLU tuning

- Iter 19: SetFixSplit(64,128,-1) — no change, auto-tiling still chose same singleCoreM/N
- Iter 20-21: SetTraverse(FIRSTM) — small improvement on matmul (11.54 vs 12.2us)
- Iter 22: LeakyReLU 16 cores — regression

### Iter 24-25 — ConstantTiling (major improvement)

- **Hypothesis:** Using MatmulApiStaticTiling (compile-time constant tiling) eliminates runtime tiling copy overhead. Pass only a tiny MatmulProblemShape (32 bytes) instead of full TCubeTiling (~4KB).
- **Changes:** Complete rewrite of matmul kernel to use constant tiling pattern: GetMatmulApiTiling<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mmCFG) at compile time, REGIST_MATMUL_OBJ with nullptr tiling, SetOrgShape at runtime.
- **Bench (launch-count=10 avg):** Matmul: 10.74us (from ~12.2us), LeakyReLU: 9.24us. Total: 19.98us, Speedup: 20.4x.
- **Analysis:** ~15% improvement on matmul! The tiny tiling struct (8 uint32_t vs ~1000 uint32_t) means much less GM-to-register copy at kernel start. The matmul library can also optimize better with compile-time known parameters.

### Optimization Summary (iter 13-43)

Starting from ~21.12us (matmul 12.74 + LeakyReLU 8.38), optimized to ~20.0us median (matmul ~11.2 + LeakyReLU ~9.2).

**Successful optimizations:**
1. **ConstantTiling** (iter 24): Replaced TCubeTiling (~4KB GM copy) with MatmulApiStaticTiling (compile-time) + MatmulProblemShape (32 bytes). Matmul improved ~15% (12.2→10.7us).
2. **Larger UB chunks** (iter 15): LeakyReLU UB_CHUNK increased from 16384 to 23552 floats. Reduced MTE transactions per core.
3. **FIRSTM traverse** (iter 21): Explicitly set M-first traversal order for slightly better memory access pattern.
4. **MixDualMaster + IBShare** (from iter 7): Both AIC vector cores independently copy results.

**Failed directions:**
- MDL mode (always worse for this small matrix)
- UnitFlag (regression)
- CONFIG_IBSHARE (correctness failure with constant tiling)
- TBuf with manual SetFlag (correctness failure)
- Double buffer for LeakyReLU (too little data per core)
- Core count changes for LeakyReLU (20 is sweet spot)
- Manual Muls+Max vs LeakyRelu API (API is faster)

### Iter 26-35 — Further tuning attempts (all no change or regression)

- Iter 26: CONFIG_MDL constant tiling — regression (12.16us matmul)
- Iter 27: UnitFlag — regression (12.78us matmul)
- Iter 28: baseM=128 — similar perf (11.08us matmul)
- Iter 29: singleCoreM=104 — regression (mismatch with auto-tiling)
- Iter 31: CONFIG_IBSHARE — correctness failed
- Iter 32: Manual Muls+Max LeakyReLU — slightly slower than API (9.20 vs 8.86us)
- Iter 33: In-place single buffer — worse due to manual sync overhead
- Iter 34: L2 cache hint — no change
- Iter 35: L1 depth override — no change

### Iter 45 — CONFIG_NORM + L1 depth/step tuning (dA=8, dB=8, sKa=4, sKb=4)

- **Hypothesis:** Following the matmul_high_performance example, setting L1 depth/step params in the constant tiling should reduce GM→L1 transfers. K=256 with baseK=64 means 4 K-iterations; loading all at once into L1 should improve data reuse.
- **Changes:** Added constantCFG.depthA1=8, depthB1=8, stepKa=4, stepKb=4 to the constant tiling config. Total L1 = 96KB(A) + 256KB(B) = 352KB ≤ 512KB.
- **Bench:** matmul=11.24us, LeakyReLU=8.86us, total=20.10us. Improvement ~3% on matmul vs 11.60us baseline.
- **Analysis:** Depth params help even with CONFIG_NORM. Need to sweep different values.

### Iter 46 — CONFIG_MDL + constant tiling + depth tuning

- **Hypothesis:** CONFIG_MDL with constant tiling and depth params might benefit from "big packet" transfers.
- **Changes:** Changed CONFIG_NORM to CONFIG_MDL.
- **Bench:** matmul=11.36us, LeakyReLU=9.66us, total=21.02us. Regression.
- **Analysis:** MDL adds overhead for small matrices. Reverted to CONFIG_NORM.

### Iter 47 — 16 cores with baseM=128 + depth tuning

- **Hypothesis:** SetDim(16) gives baseM=128 (recommended optimal), singleCoreM=128 (divides M=1024 evenly → no tail blocks), but fewer cores.
- **Changes:** SetDim(16), updated shapeParams to {128, 320, 256, 128, 256, 64}.
- **Bench:** matmul=11.54us, LeakyReLU=9.20us, total=20.74us. Fewer cores hurt more than better base blocks help.
- **Analysis:** Reverted to 22 cores. The auto-tiled baseM=96 with 22 cores is better than baseM=128 with 16 cores for this problem size.

### Iter 48 — L1 depth/step parameter sweep

- **Hypothesis:** Different depth/step combinations may have different optimal points for this small matrix.
- **Changes:** Automated sweep of 22 depth/step configurations with launch-count=5/10.
- **Key Results (launch-count=5):**
  - dA=4,dB=4,sKa=2,sKb=2: 10.74+9.02=19.76us
  - dA=2,dB=2,sKa=1,sKb=1: 11.28+8.22=19.50us
  - dA=3,dB=3,sKa=1,sKb=1: 11.04+8.38=19.42us
  - Default (no depth): 11.70+9.48=21.18us
- **Key Results (launch-count=10, more stable):**
  - dA=4,dB=4,sKa=2,sKb=2: 11.10+9.00=20.10us ← most consistently good
  - dA=8,dB=8,sKa=4,sKb=4: 12.04+8.54=20.58us
- **Analysis:** Depth tuning consistently improves over default (~1us total). dA=4,dB=4,sKa=2,sKb=2 is the sweet spot — moderate L1 usage (176KB) with 2 K-blocks per step. Larger depths (dA=8,dB=8) reduce LeakyReLU time (L2 cache effect) but increase matmul time. High variance (~1.5us) makes fine-tuning difficult.
- **Applied:** dA=4,dB=4,sKa=2,sKb=2 as new default.

### Iter 49 — UnitFlag + depth tuning

- **Changes:** Added mmCFG.enUnitFlag = true to constant tiling config.
- **Bench:** matmul=11.66us, relu=9.26us, total=20.92us. Regression.
- **Analysis:** UnitFlag overhead outweighs MMAD-FIXPIPE parallelization benefit for this small matrix. Reverted.

### Iter 50 — enableStaticPadZeros

- **Changes:** Added mmCFG.enableStaticPadZeros = true.
- **Bench:** Kernel hung during execution. The feature requires ND2NZ input format. Failed.

### Iter 51 — Double buffer depth=2 LeakyReLU

- **Changes:** Changed TQue depth from 1 to 2, reduced UB_CHUNK to 11520 to fit 4 buffers.
- **Bench:** matmul=11.70us, relu=9.30us, total=21.00us. Regression.
- **Analysis:** Smaller chunks (45KB vs 92KB) mean more MTE transactions per core. Double buffering overhead exceeds the overlap benefit for this small data.

### Iter 52 — LeakyReLU core count sweep

- **Changes:** Automated sweep of 2/4/6/8/10/12/14/16/20 cores.
- **Results (launch-count=5):** 2 cores=38.44us, 4=29.04, 6=24.32, 8=24.18, 10=20.76, 12=21.12, 14=19.10, 16=?, 20=baseline
- **Analysis:** 14 cores showed potential (19.10us) but with launch-count=10, 14-core median was 20.56us vs 20-core 20.82us. Within noise. Kept 20 cores.

### Iter 53 — SetSingleShape(96,320/640) for even tiling

- **Changes:** Tried SetSingleShape(96,320) and SetSingleShape(96,640).
- **Bench:** Both failed correctness — tiling API overrode to 24 cores (3 columns not 2), causing CalcOffset mismatch.

### Iter 54 — IBShare removal test

- **Changes:** Removed IBShare=true from A_TYPE and B_TYPE MatmulType definitions.
- **Bench:** matmul=11.74us (regression from ~11.1us), relu=8.90us, total=20.64us.
- **Analysis:** IBShare provides ~0.5us benefit on matmul. Restored.

### Iter 55 — Hardcoded LeakyReLU tiling constants

- **Changes:** Replaced runtime GM CopyTiling with compile-time constexpr values for totalElements, usedCoreNum, elementsPerCore.
- **Bench:** Marginal improvement (~0.1us) on relu. Kept since no downside.

### Iter 56 — Muls+Max manual LeakyReLU

- **Changes:** Replaced LeakyRelu(out, in, alpha, n) with Muls(out, in, alpha, n) + Max(out, in, out, n).
- **Bench:** Similar performance to API version. No improvement. Reverted to LeakyRelu API.

### Iter 57 — doSpecialBasicBlock

- **Changes:** Added mmCFG.doSpecialBasicBlock = true.
- **Bench:** Compile error — ambiguous template specializations when combined with IBShare. Failed.

### Iter 58 — Unrolled LeakyReLU loop

- **Changes:** Manually unrolled the 2-iteration while loop into explicit chunk 1 and chunk 2 blocks.
- **Bench:** 20.58us, no improvement. Reverted.

### Iter 59 — UB_CHUNK=24576 (full 192KB UB)

- **Changes:** Increased UB_CHUNK from 23552 to 24576 (96KB each buffer, total 192KB = full UB).
- **Bench:** relu=9.58us, regression. Using full UB may cause internal management overhead. Reverted.

### Iter 44 — Detailed profiling of current best

- **Hypothesis:** Need detailed per-core profiling to identify specific bottlenecks for next optimization phase.
- **Changes:** No code changes. Fresh profiling with msprof op (warm-up=10, launch-count=5).
- **Bench:**
  - Matmul: 11.60 us (22 cube cores), LeakyReLU: 8.90 us (20 vector cores, Mix Block Dim=40)
  - Total: 20.50 us, Speedup: 19.9x
- **Analysis:**
  - **Matmul core imbalance ~30%**: Cores 0-10 run 7.2-8.1us, cores 11-21 run 9.9-10.9us. MTE2 dominates at 68-81% ratio. baseM=96, baseN=256 auto-tiling. Each core reads ~208KB from GM (A) + ~118KB (C output). Cube ratio only ~19% — very MTE2-bound.
  - **LeakyReLU massive load imbalance**: Cores 0-9 vector sub-blocks each process 128KB (32K floats). Cores 10-19 process **zero data** (0 KB GM_to_UB). Only scalar overhead (~2-4us) on idle cores. Yet Task Duration = 8.9us. This means 50% of cores are wasted.
  - **Root cause**: With 20 blocks, elementsPerCore=32768 (128KB float32). Cores 0-9 use both vector0 and vector1 to process their 128KB chunk. But cores 10-19 somehow get zero data — possibly the MixDualMaster scheduling assigns work across vector sub-blocks rather than blocks, so only 10 blocks worth of vector sub-blocks get actual data.
  - **Key insight**: Theoretical MTE2 time for 2.5MB at 1.8TB/s = 1.4us. Actual 8.9us means ~6x overhead from kernel launch, scalar setup, and idle cores.
- **Next:** Try (1) different base block sizes (128,256,64) per high-perf example, (2) constant tiling depth/step params, (3) SetFixSplit for matmul, (4) reduce LeakyReLU idle core overhead.

### Iter 62 — Fused MIX multi-core matmul+LeakyReLU with constant tiling

- **Hypothesis:** Eliminate separate LeakyReLU kernel (~10us) by fusing it into the matmul kernel using __mix__(1,2) mode with VECIN output. Constant tiling should work with the fused approach.
- **Changes:** Rewrote kernel as single __mix__(1,2) function with matmul outputting to VECIN, then LeakyReLU + CopyOut per tile.
- **Bench:**
  - Compiled: First attempt failed (enableMixDualMaster incompatible with Iterate). Second attempt compiled but hung during execution.
  - Correct: N/A
  - Runtime: N/A (hung)
- **Analysis:** The constant tiling config designed for __cube__ (output to GM via IterateAll) is fundamentally incompatible with __mix__ (Iterate + VECIN). enableMixDualMaster must be off for Iterate, but the multi-core constant tiling was built for cube-only mode. The kernel launched (22 blocks) but deadlocked, likely because the constant tiling params (singleCoreM=94, singleCoreN=320) don't match what the MIX mode iterate expects.
- **Next:** Instead of fused MIX, try: (1) FIRSTN traverse for matmul, (2) compiler optimization flags, (3) half-precision LeakyReLU output to reduce bandwidth, (4) fewer cores for matmul to reduce launch overhead.

### Iter 61 — Deep profiling analysis of current best

- **Hypothesis:** Detailed per-core profiling to find remaining optimization opportunities.
- **Changes:** No code changes; fresh msprof run with detailed CSV analysis.
- **Bench:**
  - Matmul: 10.90 us (22 cores, cube-only)
  - LeakyReLU: 9.86 us (20 cores, mix type)
  - Total: 20.76 us
  - Speedup: 19.7x
- **Analysis:**
  - **Matmul (10.9 us)**:
    - Dominated by MTE2 (66-85%) and fixpipe (70-86%) — data loading and output writeback
    - cube_ratio only 11-17% — severely underutilized
    - scalar_ratio 25-50% — high tiling overhead
    - Core 10 has 57% L2 hit rate (vs 85-99% for others) — suggests data locality issue
    - Cores 0-10 run ~8.2-8.9 us; cores 11-21 run ~9.1-10.1 us — ~20% imbalance
    - Memory per core: ~256KB read from GM to L1, ~120KB GM_to_UB — consistent with matmul data movement
  - **LeakyReLU (9.86 us)**:
    - Launched as mix type with Mix Block Dim=40 (20 blocks × 2 sub-blocks)
    - **Critical finding**: Only blocks 0-9 do actual vector work (128KB each, MTE2+VEC active)
    - Blocks 10-19: aiv_vec_time≈0, GM_to_UB=0KB, only scalar overhead (1.8-5.2 us)
    - For active blocks 0-9: MTE2 is bottleneck (~2.3us), vec_wait_ratio=30-45%
    - BW utilization: 52-77 GB/s MTE2 active BW (theoretical peak ~1.8 TB/s) — very low
    - Zero bank conflicts across all blocks
    - **Idle core overhead**: blocks 10-19 spend 1.8-5.2 us doing nothing but scalar init
  - **Key bottlenecks**:
    1. LeakyReLU has ~50% idle cores (10 out of 20) — unclear why data mapping fails
    2. Matmul MTE2 dominates (data loading), not cube compute
    3. Two kernel launches = two kernel launch overheads (~2-3 us each)
    4. Matmul core imbalance: last 11 cores are ~15% slower
- **Next:** Try (1) fp16 output to halve LeakyReLU data movement, (2) fixpipe LeakyReLU in matmul (CUBE fusion), (3) investigate LeakyReLU idle core issue, (4) reduce matmul core count to improve per-core utilization.

### Iter 79 — depthB1=2 (reduce L1 B footprint)

- **Hypothesis:** Reducing depthB1 from 4 to 2 halves B's L1 footprint (128KB→64KB, total 112KB). Smaller L1 usage may reduce conflict with other data paths.
- **Changes:** Changed constantCFG.depthB1 from 4 to 2.
- **Bench:**
  - Compiled: True, Correct: True
  - Matmul: 12.14 us (regression from 11.50), LeakyReLU: 9.08 us
  - Total: 21.22 us, Speedup: ~19.2x
- **Analysis:** depthB1=2 degrades matmul by ~0.6us. The smaller L1 buffer means more frequent GM→L1 reloads for B data. The iter-68 sweep had shown dA4,dB2 as best in sweep (18.92us) but that was launch-count=1; with proper warmup, dA4,dB4 is better. Reverted.
- **Next:** Try ACL_STREAM_FAST_LAUNCH for reduced kernel launch overhead.

### Iter 80 — ACL_STREAM_FAST_LAUNCH + FAST_SYNC

- **Hypothesis:** Using aclrtCreateStreamWithConfig with ACL_STREAM_FAST_LAUNCH|ACL_STREAM_FAST_SYNC flags should reduce kernel dispatch latency between the two kernels.
- **Changes:** Replaced aclrtCreateStream with aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC).
- **Bench:**
  - Compiled: True, Correct: True
  - Matmul: 13.18 us (severe regression), LeakyReLU: 8.76 us
  - Total: 21.94 us, Speedup: ~18.6x
- **Analysis:** FAST_LAUNCH stream degraded matmul by ~1.7us. This flag changes scheduling behavior that hurts cube kernel performance. Possibly reduces inter-core coordination efficiency. Reverted.
- **Next:** Try single-core MIX matmul+LeakyReLU fusion (SetDim=1) to eliminate 2nd kernel entirely.

### Iter 81 — Re-profile with launch-count=10

- **Hypothesis:** Need stable measurements to guide further optimization.
- **Changes:** No code changes. msprof op with launch-count=10.
- **Bench:**
  - Matmul: 12.78 us (22 cores), LeakyReLU: 8.94 us (20 blocks, 40 mix sub-blocks)
  - Total: 21.72 us (higher than 20.66 median due to measurement variance)
- **Analysis:**
  - **Matmul**: Core imbalance remains ~30% (min 8.83us, max 12.09us). MTE2 dominates (7.1-7.6us, ~75%). Cube only 1.4us (~14%). fixpipe 7.5-7.9us (high). The kernel is MTE2-bound.
  - **LeakyReLU**: Blocks 0-9 active (128KB/v0 + 128KB/v1 = 256KB data each). Blocks 10-19 completely idle (0KB GM_to_UB). Scalar overhead dominates active blocks (1.9-3.6us). MTE2 1.75-3.23us. Vec only 0.3us.
  - **Key insight**: LeakyReLU bottleneck is scalar overhead (kernel launch/init per block), not data movement. 50% of blocks are idle despite all 20 being allocated work in the code.
- **Next:** Try launching LeakyReLU with 10 blocks instead of 20 to eliminate idle core overhead, since profiling shows only 10 blocks do actual work.

### Iter 82 — LeakyReLU 10 blocks (match actual active blocks)

- **Hypothesis:** Since profiling shows only 10 of 20 blocks do actual data movement, launching with 10 blocks should eliminate idle block overhead and give each block 256KB (more per-core bandwidth utilization).
- **Changes:** Changed usedCoreNum from 20 to 10 in both kernel and host.
- **Bench:**
  - Compiled: True, Correct: True
  - Matmul: 13.26 us, LeakyReLU: 9.74 us (regression from ~8.94)
  - Total: 23.00 us, Speedup: ~17.7x
- **Analysis:** 10 blocks is worse than 20 blocks. Despite 10 blocks being idle in profiling, they still contribute to the MixDualMaster parallelism. With fewer blocks, less vector parallelism is available. The 20-block configuration is optimal. Reverted.
- **Next:** Try fundamentally different approach: use half16 output for matmul to halve C data and reduce LeakyReLU bandwidth.

### Iter 83 — stepKa/stepKb sweep (9 combos)

- **Hypothesis:** Different step combinations may improve MTE2 pipelining for the matmul kernel.
- **Changes:** Automated sweep of 9 stepKa×stepKb combinations (1,2,4 × 1,2,4), launch-count=5.
- **Bench (best combos):**
  - sKa=4,sKb=4: 11.06+8.82=19.88us
  - sKa=4,sKb=2: 11.66+8.24=19.90us
  - sKa=2,sKb=4: 10.90+9.24=20.14us
  - sKa=2,sKb=2: 11.32+8.84=20.16us (current config)
- **Analysis:** All within noise (~0.3us). Current config (sKa=2,sKb=2) is near-optimal. The sweep confirms limited optimization headroom in step parameters.
- **Next:** Try combining matmul output as half16 + LeakyReLU on half16 to reduce bandwidth.

### Iter 84 — depthA1/depthB1 sweep

- **Hypothesis:** Larger L1 depth for B (depthB1=8) keeps all B data in L1 across K-iterations, reducing GM→L1 reloads.
- **Changes:** Swept dA×dB: (4,4), (8,4), (4,8), (8,8). 5 runs each with launch-count=5.
- **Bench:**
  - dA4,dB4 (baseline): median ~20.9us
  - dA8,dB8: median ~20.2us
  - dA8,dB4: 21.04us
  - **dA4,dB8: median 20.02us** (best: 19.66us)
- **Analysis:** depthB1=8 lets all B data (4 K-blocks × 256×64×2B = 128KB) reside in L1. With K=256 and baseK=64, there are 4 K-iterations; depthB1=8 > 4 means the library pre-loads all B blocks into L1 before compute starts, enabling full B reuse across M-iterations. This is especially beneficial because each core reads the same B data multiple times (once per baseM row-block). Applied dA4,dB8 as new best.
- **Next:** Try further optimizations: iterateOrder=ORDER_N, or additional depth tuning with stepKa/stepKb adjusted.

### Iter 85 — Step tuning with dA4,dB8

- **Hypothesis:** With the new dA4,dB8 config, different step values might be optimal.
- **Changes:** Swept 9 combinations of sKa×sKb (1,2,4)×(1,2,4) with launch-count=5. Verified top candidates with 5-run stability tests.
- **Bench (sweep, launch-count=5):**
  - sKa=2,sKb=1: 19.18us (but 5-run median=20.16)
  - sKa=1,sKb=2: 19.34us (but 5-run median=20.54)
  - sKa=2,sKb=2: 21.26us (but 5-run median=20.02, more stable)
- **Analysis:** High variance makes single-shot sweep results unreliable. 5-run stability tests show sKa=2,sKb=2 is the most consistent. Kept sKa=2,sKb=2.
- **Next:** Try LeakyReLU with half16 intermediate for bandwidth reduction.

### Iter 86 — LeakyReLU __vector__ mode (major improvement!)

- **Hypothesis:** Changing LeakyReLU from `__aicore__` to `__vector__` mode eliminates cube sub-block allocation. In __aicore__ mode, each block gets cube0+vector0+vector1 sub-blocks (Mix Block Dim=40 for 20 blocks). The cube0 sub-blocks are completely idle (pure scalar overhead 4-5us), wasting hardware resources. In __vector__ mode, only vector sub-blocks are allocated.
- **Changes:** Changed `__global__ __aicore__` to `__global__ __vector__` for the leakyrelu_custom kernel.
- **Bench (10-run median):**
  - Compiled: True, Correct: True
  - Matmul: median 11.88 us (22 cube cores)
  - **LeakyReLU: median 5.82 us** (20 vector blocks, no cube overhead!)
  - **Total: median 17.70 us, Speedup: 23.1x**
- **Analysis:** This is the biggest single improvement since the two-pass architecture was introduced. The __vector__ mode avoids allocating cube sub-blocks for pure vector work. In __aicore__/mix mode, each block had 3 sub-blocks (cube0, vector0, vector1) = 60 sub-blocks for 20 blocks. The cube0 sub-blocks did zero useful work but still consumed ~4-5us each for scalar init. In __vector__ mode, only vector sub-blocks are allocated, eliminating this overhead completely. LeakyReLU improved from 9.02us to 5.82us (35% reduction).
- **Next:** Continue optimizing: try different vector core counts, UB chunk sizes, or matmul improvements.

### Iter 87 — LeakyReLU core count sweep (8-40 cores, __vector__ mode)

- **Hypothesis:** With __vector__ mode, optimal core count might differ from the __aicore__ configuration.
- **Changes:** Swept 8/10/14/16/20/24/32/40 cores for LeakyReLU. Verified 40 cores with 10-run median.
- **Bench (sweep, launch-count=5):**
  - 8 cores: 20.14us, 10: 21.46, 14: 18.54, 16: 19.08, **20: 17.78**, 24: 19.78, 32: 18.90, 40: 17.62
  - 40 cores 10-run median: 19.02us (relu=6.50, worse due to per-block overhead with small data)
- **Analysis:** Single-launch sweep showed 40 cores slightly better, but 10-run median confirmed 20 cores is more stable (17.70 vs 19.02). With 40 blocks, each handles only 64KB (vs 128KB for 20), making per-block overhead more significant relative to useful work.
- **Next:** Try UB chunk size optimization for __vector__ mode, or matmul improvements.

### Iter 88 — UB chunk size sweep in __vector__ mode

- **Hypothesis:** Smaller or larger UB chunks might be optimal in __vector__ mode.
- **Changes:** Swept 16384/20480/23552/24000/24576 floats per UB buffer. Verified 16384 with 10-run median.
- **Bench:**
  - 16384: single=16.86, 10-run median=17.92 (relu=5.66, matmul=12.26)
  - 20480: 17.06 (single)
  - 23552: 17.14 (single), 10-run median=17.70 (current)
  - 24000: 19.06 (single, outlier)
  - 24576: 17.22 (single)
- **Analysis:** All within noise. 23552 (current) gives the most stable median. Kept 23552.
- **Next:** Try matmul improvements: different baseM/baseN or traverse order.

### Iter 89 — FIRSTN traverse

- **Hypothesis:** FIRSTN traverse might improve B data reuse.
- **Changes:** Changed SetTraverse from FIRSTM to FIRSTN.
- **Bench:** 3-run median 17.58us (FIRSTM baseline 17.70us). Within noise.
- **Analysis:** No significant difference. FIRSTM and FIRSTN produce similar results for this shape. Reverted.

### Iter 90 — enUnitFlag=false

- **Hypothesis:** Disabling UnitFlag might reduce overhead for small matrices.
- **Changes:** Added mmCFG.enUnitFlag = false.
- **Bench:** 5-run median 17.68us. Within noise.
- **Analysis:** UnitFlag has negligible effect with constant tiling. Reverted.
- **Next:** Try L2 cache optimization approaches for the matmul kernel.

### Iter 91 — Double buffer LeakyReLU in __vector__ mode

- **Hypothesis:** In __vector__ mode, double buffering (bufNum=2) can overlap MTE2/VEC/MTE3 pipeline stages. Previously failed in __aicore__ mode because per-core data was too small (1.4 chunks). With __vector__ mode and reduced overhead, the pipeline benefit may materialize with more chunks per core.
- **Changes:** Changed inQueue/outQueue from depth=1 to depth=2. UB_CHUNK reduced from 23552 to 11776 (46KB each, 4 buffers = 184KB). Each core now processes ~3 chunks (32768/11776) instead of ~1.4.
- **Bench (15-run median):**
  - LeakyReLU: ~5.36 us (from 5.82, ~8% improvement)
  - Matmul: ~11.88 us (unchanged)
  - **Total: median 17.24 us, Speedup: 23.7x**
- **Analysis:** Double buffering works in __vector__ mode! With 3 chunks per core, there's enough iteration to amortize the pipeline setup. The 0.46us relu improvement comes from overlapping MTE2 data copy with VEC computation on the previous chunk. New best configuration.
- **Next:** Continue optimizing matmul or try more LeakyReLU improvements.

### Iter 92 — depthA1=2 with dB8

- **Hypothesis:** Reducing A depth from 4 to 2 frees L1 space, potentially reducing contention.
- **Changes:** depthA1=4→2.
- **Bench:** 15-run median 17.34us (vs 17.24 baseline). Within noise.
- **Analysis:** No significant difference. Reverted to dA4.
- **Next:** Try matmul L2 cache optimization or different singleCore shapes.

### Iter 93 — DB chunk size sweep

- **Hypothesis:** Different double-buffer chunk sizes balance pipeline overlap vs per-chunk overhead.
- **Changes:** Swept 4096/6144/8192/11776 floats per DB chunk (10 runs each).
- **Bench:**
  - 4096: median 17.42 (8 chunks/core, too much overhead)
  - 6144: median 17.22 (5.3 chunks/core)
  - **8192: median 16.80** (4 chunks/core, optimal pipeline)
  - 11776: median 17.24 (2.8 chunks/core, too few for pipeline)
- **Analysis:** 8192 (32KB per buffer, 4 buffers = 128KB) gives 4 chunks per core, which is the sweet spot for MTE2/VEC/MTE3 pipeline overlap. Fewer chunks means less pipeline benefit; more chunks means too much per-chunk overhead. Applied UB_CHUNK=8192 as new best.
- **Next:** Continue matmul optimizations or try new LeakyReLU approaches.

### Iter 94 — baseM/baseN sweep

- **Hypothesis:** Different base block sizes might improve cube utilization or reduce fixpipe overhead.
- **Changes:** Swept 16 combinations of baseM (64,96,128,192) × baseN (128,192,256,320) in the constant tiling shapeParams.
- **Bench:** All 16 combinations failed — either precision failure or compile failure. Changing baseM/baseN in the constant tiling without also updating the host-side tiling (singleCoreM, singleCoreN) creates a fundamental mismatch.
- **Analysis:** To properly sweep baseM/baseN, both the constant tiling config AND the host-side tiling generation must be updated consistently. This requires more complex code changes.
- **Next:** Try other matmul config options or investigate matmul with different shapeParams including singleCore updates.

### Iter 95 — OUTER_PRODUCT and ORDER_M matmul configs

- **Hypothesis:** Different schedule types or iterate orders might improve MTE2 pipelining.
- **Changes:** (a) OUTER_PRODUCT schedule type; (b) ORDER_M iterate order.
- **Bench:** OUTER_PRODUCT caused precision failure. ORDER_M compiled and passed precision but matmul=12.90us (within noise).
- **Analysis:** Neither config improves performance for this shape. The default INNER_PRODUCT + UNDEF iterate order is optimal.
- **Next:** Try more aggressive matmul optimizations or system-level improvements.

### Iter 96 — 20-run definitive measurement

- **Hypothesis:** Establish reliable baseline for the current best configuration.
- **Changes:** No code changes. 20 independent msprof runs with launch-count=5.
- **Bench:**
  - Runs: 20
  - LeakyReLU: median 5.32us, range [5.10, 5.74]
  - Matmul: median 11.78us, range [10.88, 13.98]
  - **Total: median 17.16us, mean 17.25us, range [16.10, 19.12]**
  - **Speedup: 23.8x (median), 23.7x (mean)**
- **Analysis:** Configuration is stable with reasonable variance. The matmul has higher variance (3.1us range) vs relu (0.64us range). The best runs achieve 16.1us (25.4x), suggesting there's ~1us of inter-run variability from system factors.

### Iter 97 — Triple buffer (bufNum=3) LeakyReLU

- **Hypothesis:** bufNum=3 adds one more pipeline stage for deeper overlap of MTE2/VEC/MTE3.
- **Changes:** Changed bufNum from 2 to 3. Reduced UB_CHUNK from 8192 to 5440 (6 buffers × 5440 × 4B = 130KB < 192KB). Each core now processes ~6 chunks.
- **Bench (20-run):**
  - **Median: 17.00us** (24.0x), mean: 17.11us
  - vs double buffer: 17.16us (23.8x)
- **Analysis:** Marginal improvement (~0.16us). Triple buffer provides slightly deeper pipeline overlap with 6 chunks/core vs 4 chunks/core, but the improvement is small because the bottleneck has shifted from pipeline depth to MTE2 bandwidth. Kept as new best.
- **Next:** Continue with remaining matmul or system-level optimizations.

### Iter 98 — enableKdimReorderLoad

- **Changes:** Added mmCFG.enableKdimReorderLoad = true.
- **Bench:** ~17.0us, within noise. Reverted.

### Iter 99 — A_TYPE IBShare=false

- **Changes:** Disabled IBShare for A matrix only (kept for B).
- **Bench:** 17.00us, no change. Reverted.

### Iter 100 — stepKa=4,stepKb=4 with triple buffer

- **Changes:** Increased steps from 2,2 to 4,4.
- **Bench:** 10-run median 17.14us (vs 17.00 baseline). Within noise. Reverted.
- **Next:** Try more creative approaches or accept current performance level.

<!-- Template — copy for each new iteration:

### Iter N — Short title

- **Hypothesis:** Why this change is expected to help
- **Changes:** What was modified
- **Bench:**
  - Compiled: True/False
  - Correct: True/False
  - Runtime: ___ ms (mean), ___ ~ ___ ms (min ~ max)
  - Speedup: ___x (mean), ___ ~ ___x (min ~ max)
- **Analysis:** Why it worked or failed
- **Next:** What to try next
-->

### Iter 164-192 — Session 2 Optimization Summary

- **Starting point:** ~15.62 us (26.2x) single run, ~16.22 us (25.2x) 20-run median
- **Ending point:** ~15.70 us (26.0x) 11-run median
- **Net change:** No significant improvement; performance plateau confirmed

**Directions explored (29 iterations):**

1. **MatmulConfig options** (iter 165-167, 172): enableDoubleCache, enableL1CacheUB, enableUBReuse, enableReuse=false — all within noise
2. **Core split variations** (iter 168, 179): M-only split with 22/20 cores — regression (less N-parallelism)
3. **Memory allocation** (iter 169): ACL_MEM_MALLOC_NORMAL_ONLY — no effect
4. **baseM/baseN/baseK sweep** (iter 170-171, 175): baseK=128 precision fail, baseK=32 regression, other combos within noise
5. **Depth/step comprehensive sweep** (iter 173, 180, 187, 191): 89 combinations tested; current dA4,dB8,sKa2,sKb2 remains optimal
6. **Float intermediate** (iter 174): Eliminates Cast in LeakyReLU but doubles read bandwidth — net zero
7. **Schedule type** (iter 176): OUTER_PRODUCT precision failure
8. **Buffer depth** (iter 177): Quad buffer — no improvement over triple buffer
9. **UB chunk sweep** (iter 178, 190): 5120, 7680, 8192 — 6400 remains sweet spot
10. **20-run stability** (iter 182): Established reliable median of 15.70 us
11. **IBShare removal** (iter 183): Required for correctness with constant tiling
12. **L2 cache hints** (iter 184): Adds overhead
13. **LeakyReLU core count** (iter 185-186): 16/24 cores tested, 20 remains optimal
14. **MixDualMaster toggle** (iter 188): Within noise
15. **Workspace reduction** (iter 189): No effect on kernel timing
16. **Host tiling traverse** (iter 192): No effect (host tiling doesn't influence kernel constant tiling)

**Conclusion:** The kernel is at the hardware performance limit for the two-pass cube+vector architecture on Ascend910B. The matmul is MTE2-bound (~11 us, ~5.5x theoretical minimum), and LeakyReLU is bandwidth-bound (~4.7 us, ~2.3x theoretical minimum). Further improvement would require:
1. A fundamentally different approach (single-pass fused kernel, which has proven incompatible with constant tiling)
2. Hardware with more bandwidth or lower kernel launch overhead
3. Reducing the problem size (fixed at 1024x256x640)

### Iter 193-202 — Session 3 Final Iterations

- **Starting point:** ~15.70 us (26.0x) 20-run median from iter-182
- **Ending point:** ~16.12 us (25.3x) 20-run median from iter-201
- **Net change:** No significant improvement; within run-to-run variance

**Directions explored (10 iterations):**

1. **EnableMultiCoreSplitK** (iter 193): K-direction split caused tiling to change to 16 cores with singleCoreM=1024, breaking the kernel's hardcoded 22-core M×N layout. Precision fail.
2. **MIX fusion / Step tuning** (iter 194): MIX fusion confirmed as incompatible (per iter 3/12/62). Tested step variants sKa2,sKb2 vs current sKa4,sKb1 — within noise.
3. **MatmulCallBackFunc** (iter 195): Analysis showed callbacks only customize data copy (GM↔L1, CO1→GM), not activation application. Cube-only mode has no vector compute for LeakyReLU. Not applicable.
4. **depthB1=16** (iter 196): All 5 runs failed precision. depthB1=12 is the maximum safe value.
5. **Async IterateAll<false>** (iter 197): Compiled and passed precision, but 6-run median 16.26 us — within noise of synchronous version.
6. **ACL_MEM_MALLOC_NORMAL_ONLY** (iter 198): All device allocations changed from HUGE_FIRST to NORMAL_ONLY. Clear regression (~17.47 median). Reverted.
7. **enablePreload** (iter 199): No such member in MatmulConfig for CANN 8.3.RC1. Compile fail.
8. **UB_CHUNK=8192 triple buffer** (iter 200): 15-run median 16.06 — within noise of 6400. 6400 retained.
9. **20-run stability measurement** (iter 201): Definitive measurement: median 16.12 us, mean 16.20 us, range 14.88-17.74 us. Matmul median 11.42 us, LeakyReLU median 4.69 us. Speedup: 25.3x median.
10. **Final summary** (iter 202): See below.

### Iter 202 — Final Summary

**Final Configuration:**
- Architecture: Two-pass (cube matmul + vector LeakyReLU)
- Matmul: 22 cube cores, constant tiling, enableMixDualMaster, IBShare on A/B
  - shapeParams: {94, 320, 256, 96, 256, 64}
  - Depth/step: depthA1=6, depthB1=12, stepKa=4, stepKb=1, stepM=1, stepN=1
  - Output: half (fp16) to reduce GM bandwidth
  - Config flags: enableInit=false, enableGetTensorC=false, enableQuantVector=false, enableSetDefineData=false
- LeakyReLU: 20 vector cores, triple buffer (bufNum=3), CHUNK=6400
  - Read half from GM → Cast to float → LeakyReLU(alpha=0.001) → Write float to GM
  - UB layout: 3×6400×2B (half in) + 3×6400×4B (float out) = 112.5KB < 192KB

**Final Performance (20-run stability, iter-201):**
- Median: 16.12 us (25.3x vs 408.34 us baseline)
- Mean: 16.20 us (25.2x)
- Min: 14.88 us (27.4x)
- Max: 17.74 us (23.0x)
- Matmul: median 11.42 us, range 10.12-12.96 us
- LeakyReLU: median 4.69 us, range 4.50-5.04 us

**Performance Breakdown:**
- Matmul accounts for ~71% of total time; MTE2-bound (data loading dominates)
- LeakyReLU accounts for ~29% of total time; bandwidth-bound
- Theoretical minimum: matmul ~2.0 us (cube compute), LeakyReLU ~2.1 us (bandwidth limit)
- Achieved ~5.7x of matmul theoretical, ~2.2x of relu theoretical

**Key Optimizations (cumulative impact):**
1. Two-pass architecture (iter 4): 408→21.6 us (18.9x) — multi-core parallelism
2. MixDualMaster + IBShare (iter 7): 21.6→20.6 us — dual vector output
3. Constant tiling (iter 24): 20.6→20.0 us — eliminate GM tiling copy
4. __vector__ mode LeakyReLU (iter 86): 20.0→17.7 us — eliminate idle cube blocks
5. Double/triple buffer LeakyReLU (iter 91/97): 17.7→17.0 us — pipeline overlap
6. UB chunk tuning (iter 93): 17.0→16.8 us — optimal chunk size
7. fp16 intermediate (iter 105): 16.8→16.5 us — halve C bandwidth
8. No-calcBuf in-place LeakyReLU (iter 117): 16.5→16.5 us — eliminate extra buffer
9. Hardcoded shapes (iter 122): 16.5→15.8 us — eliminate GM shape read
10. Depth tuning (iter 84+): 15.8→15.7 us — L1 cache optimization

**Plateau Analysis:**
The kernel has been at the performance plateau since approximately iter-122 (~15.8 us). Over 80 subsequent iterations explored every available knob (depth/step parameters, buffer configurations, core counts, memory allocation modes, traverse orders, L2 hints, compiler flags, etc.) with no statistically significant improvement. The performance is hardware-limited by:
1. MTE2 bandwidth for matmul data loading (~75% of matmul time)
2. Kernel launch overhead (~2-3 us per kernel, two kernels)
3. Core imbalance in matmul (~30% variation between fastest and slowest cores)
