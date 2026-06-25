# Iteration Log

## Summary

<!-- Append one row per iteration. Status: improved / no-change / regression / failed -->

| Iter | Title | Speedup(mean) | Runtime(mean) | Status |
|------|-------|---------|--------------|--------|
| 0 | Baseline | 1.0x | W1=289us W2=289us W3=404us W4=1126us | baseline |
| 1 | Merge B>1 batches into single matmul | W4: 3.83x | W1=288us W2=290us W3=408us W4=294us | improved |
| 2 | MDL config sweep (KdimReorder+UnitFlag+SetFixSplit) | W3: 1.02x | W1=289us W2=290us W3=399us W4=294us | no-change |
| 3 | Traverse order (FIRSTN) + further config tuning | 1.0x | W1=290us W2=289us W3=403us W4=295us | no-change |
| 4 | Core count sweep (1-24) | 1.0x | best@20-24 cores | no-change |
| 5 | SetFixSplit baseN sweep (128-512) | 1.0x | baseN=256 optimal | no-change |
| 6 | SetFixSplit baseM sweep for W3 (16-256) | 1.0x | baseM=128 optimal | no-change |
| 7 | Norm template (replace MDL) | 0.73x | W1=398us W2=395us W3=550us W4=398us | regression |
| 8 | OUTER_PRODUCT schedule + ORDER_M | N/A | all workloads failed | failed |

## Iterations

### Iter 0 — Baseline

- **Bench:**
  - Compiled: True
  - Correct: True (all 4 workloads PASS)
  - Runtime: W1=289.43us, W2=288.73us, W3=403.84us, W4=1126.09us
- **Analysis:**
  - All workloads MTE2-bound (98-99% for W1/W2/W4, 90-93% for W3)
  - GM→L1 bandwidth utilization ~26% (far below 60% target)
  - Cube utilization: W1/W2 ~11%, W3 ~40%, W4 ~12%
  - W4 is 4x serial matmul (each reads full 400MB weight)
  - Theoretical minimum: weight read = 400MB / 1.8TB/s ≈ 222us per matmul
- **Next:** Merge multi-batch case to avoid redundant weight reads

### Iter 1 — Merge B>1 batches into single matmul

- **Hypothesis:** W4 (B=4, ltk=1) executes 4 serial matmuls, each reading entire 400MB weight. Merging to single matmul with M=4 reads weight once.
- **Changes:** Replaced per-batch loop kernel with single matmul kernel. Host gathers sliced rows via D2D memcpy into contiguous buffer. Kernel simplified (no LmHeadTilingData, just TCubeTiling).
- **Bench:**
  - Compiled: True
  - Correct: True (all 4 workloads PASS)
  - Runtime: W1=288.05us, W2=289.67us, W3=408.04us, W4=293.89us
  - Speedup: W4=3.83x, W1/W2/W3≈1.0x
- **Analysis:** W4 improved from 1126us to 294us because weight is now read only once. W1/W2/W3 unchanged (same code path for B=1). D2D gather cost is negligible (4 rows of 2048 bf16 = 16KB total).
- **Next:** Profile to find remaining optimization opportunities. All workloads still MTE2-bound (~98%). Consider: better tiling params, L2 cache optimization, bandwidth utilization improvement.

### Iter 2 — MDL config parameter sweep

- **Hypothesis:** MDL config params (enableKdimReorderLoad, enUnitFlag, doMTE2Preload, SetFixSplit) should improve MTE2 efficiency.
- **Changes:** Tried multiple combinations:
  - doMTE2Preload=2: 2.5x speedup but ALL workloads fail precision (data corruption). Likely K not fully loaded condition not met.
  - enableKdimReorderLoad=true: no measurable change
  - enUnitFlag=true: W3 marginally improved (408→395us)
  - SetFixSplit(128/-, 256, -1): W3 marginally improved
  - Final combination: KdimReorder + UnitFlag + SetFixSplit
- **Bench:**
  - Compiled: True
  - Correct: True (all 4 PASS)
  - Runtime: W1=289.09us, W2=290.29us, W3=399.12us, W4=294.31us
  - Speedup: W3=1.02x, others≈1.0x
- **Analysis:** These tuning knobs have minimal impact because the kernel is fundamentally bandwidth-bound. The 400MB weight read dominates. doMTE2Preload=2 gives dramatic speedup but causes data corruption (need to investigate preconditions). The remaining gap to theoretical minimum (~222us) is ~30% overhead from Matmul API initialization, scalar computation, and MTE2 pipeline gaps.
- **Next:** Investigate doMTE2Preload preconditions (K full-load + N DoubleBuffer). Or explore completely different approaches: tiling constant optimization, reduced core count to reduce overhead.

### Iter 3 — Traverse order and further config tuning

- **Hypothesis:** FIRSTN traversal order may improve L2 cache utilization for B matrix chunks.
- **Changes:** Added SetMatmulConfigParams with FIRSTN traversal. Kept KdimReorder+UnitFlag+SetFixSplit.
- **Bench:**
  - Compiled: True
  - Correct: True (all 4 PASS)
  - Runtime: W1=290us, W2=289us, W3=403us, W4=295us
  - Speedup: ≈1.0x
- **Analysis:** No improvement. The kernel is fundamentally bandwidth-bound by 400MB weight read. All parameter tuning has negligible effect. The ~30% gap to theoretical minimum (222us) comes from Matmul API overhead (scalar computation, initialization, pipeline gaps). Only doMTE2Preload=2 showed dramatic improvement (2.5x to 117us) but fails precision due to K-full-load precondition not being met.
- **Conclusion:** Current optimization has reached practical limits for high-level API-based approach. Further optimization requires either (1) fixing doMTE2Preload prerequisites, or (2) hand-written Cube instructions to bypass API overhead.

### Iter 5 — SetFixSplit baseN sweep (128-512)

- **Hypothesis:** Different baseN values may improve N-dimension tiling efficiency for the large N=102400 output.
- **Changes:** Swept baseN ∈ {128, 192, 256, 320, 384, 512} via automated parameter sweep.
- **Results:**
  | baseN | W1 (us) | W2 (us) | W3 (us) | W4 (us) | W3 pass |
  |-------|---------|---------|---------|---------|---------|
  | 128   | 290.0   | 288.9   | 418.5   | 297.2   | ✓       |
  | 192   | 290.8   | 289.8   | 428.0   | 294.7   | ✓       |
  | 256   | 290.7   | 290.0   | 398.4   | 295.6   | ✓       |
  | 320   | 289.7   | 292.4   | N/A     | 306.1   | ✗       |
  | 384   | 289.7   | 289.9   | N/A     | 303.7   | ✗       |
  | 512   | 321.4   | 320.5   | N/A     | 320.3   | ✗       |
- **Analysis:** baseN=256 is optimal. Smaller baseN (128, 192) increases W3 latency due to more iterations. Larger baseN (320+) causes W3 precision failure (M=128 with large baseN likely exceeds tiling constraints). baseN=512 degrades all workloads. No improvement over current setting.
- **Next:** Try Norm template, OUTER_PRODUCT schedule, or doMTE2Preload=1 (M-direction).

### Iter 6 — SetFixSplit baseM sweep for W3 (16-256)

- **Hypothesis:** W3 has M=128; different baseM values may improve M-dimension tiling.
- **Changes:** Swept baseM ∈ {16, 32, 64, 128, 256} for the M≥128 case.
- **Results:**
  | baseM | W3 (us) | W3 pass | W1 (us) | W4 (us) |
  |-------|---------|---------|---------|---------|
  | 16    | 2484.0  | ✓       | 287.8   | 292.0   |
  | 32    | 1336.1  | ✓       | 290.8   | 292.9   |
  | 64    | 704.8   | ✓       | 287.6   | 295.7   |
  | 128   | 399.5   | ✓       | 288.1   | 293.2   |
  | 256   | N/A     | ✗       | 288.2   | 294.2   |
- **Analysis:** baseM=128 is optimal (matches M=128, no tail blocks). Smaller baseM introduces more iteration overhead. baseM=256 exceeds M and fails. W1/W2/W4 unaffected (they have M<128 so this code path isn't hit).
- **Next:** Try Norm template or OUTER_PRODUCT schedule.

### Iter 7 — Norm template (replace MDL)

- **Hypothesis:** Norm template streams basic blocks one at a time, enabling earlier MTE1 pipeline startup. May reduce MTE2 gaps.
- **Changes:** Replaced GetMDLConfig with GetNormalConfig (type=0). Updated both kernel-side config and host-side tiling config params.
- **Bench:**
  - Compiled: True
  - Correct: True (all 4 PASS)
  - Runtime: W1=397.9us, W2=395.3us, W3=549.7us, W4=397.7us
  - Speedup: 0.73x (regression)
- **Analysis:** Norm template is 35-40% slower than MDL for all workloads. MDL's bulk MTE2 transfer is better for this bandwidth-bound kernel with large B matrix. Norm's incremental loading adds overhead without benefit. Reverted to MDL.
- **Next:** Try OUTER_PRODUCT schedule type with MDL.

### Iter 8 — OUTER_PRODUCT schedule + ORDER_M

- **Hypothesis:** OUTER_PRODUCT with ORDER_M does N-direction MTE1 cycling, which could help since singleCoreN >> baseN for N=102400.
- **Changes:** Set scheduleType=OUTER_PRODUCT, iterateOrder=ORDER_M in both kernel config and tiling config.
- **Bench:**
  - Compiled: True (after fixing field name: iterateOrder not IterateOrder)
  - Correct: False — all workloads failed (msprof crash/no output)
- **Analysis:** The MDL docs state "singleCoreK>baseK时，不能使能ScheduleType::OUTER_PRODUCT". Our K=2048 and baseK is likely smaller, violating this constraint. Reverted.
- **Next:** Try doMTE2Preload=1 (M-direction preload).
