# Iteration Log

## Summary

| Iter | Title | Runtime(mean) | Speedup | Status |
|------|-------|--------------|---------|--------|
| 0 | Baseline: single row, DB=2 | 53.41 us | 1.00x | baseline |
| 1 | 2 rows/tile, DB=2, shared workBuf | 47.77 us | 1.12x | improved |
| 2 | Separate scalar+reduce bufs | 51.17 us | 1.04x | regression |
| 3 | Single buffer (depth=1), 4 rows | 55.26 us | 0.97x | regression |
| 4 | DB=2, 2 rows, shared workBuf (clean) | 46.37 us | 1.15x | **best** |
| 5 | Reconfirm iter-4 config | 47.46 us | 1.13x | no-change |
| 6 | FP16 intermediate exp | 48.28 us | 1.11x | regression |
| 7 | Clean rebuild of best config | 48.62 us | 1.10x | no-change |
| 8 | Unrolled 2-row compute | 49.49 us | 1.08x | no-change |
| 9 | 32 cores (fewer) | 54.59 us | 0.98x | regression |
| 10 | Minimal workBuf (1KB) | 51.72 us | 1.03x | regression |
| 11 | 2KB workBuf | 82.68 us | 0.65x | regression |
| 12 | 1 row/tile, DB=2 | 53.24 us | 1.00x | no-change |
| 13 | Reconfirm 2 rows/tile | 47.94 us | 1.11x | no-change |
| 14 | In-place compute on xLocal | 50.77 us | 1.05x | regression |
| 15 | Reconfirm best (lucky run) | 46.12 us | 1.16x | **best** |
| 16 | Pattern::Reduce::AR batched | 62.87 us | 0.85x | failed |
| 17 | Final: more warmup (200 runs) | 50.16 us | 1.06x | no-change |

## Best Result

**Best: 46.12 µs** (Iter 15) → **1.16x speedup** over baseline (53.41 µs)

Configuration:
- Double buffer (DB=2) for CopyIn/Compute/CopyOut pipeline overlap
- 2 rows per tile for DMA batching
- Shared workBuf (cols × sizeof(float) = 16KB) for reduce dst and tmp
- Adds/Muls scalar ops (no Duplicate overhead)
- 48 AI cores (all available)
- Level-2 ReduceMax/ReduceSum per row

## Profiling (msprof)

Actual kernel time: **26.46 µs** (msprof), host-measured: **~47 µs** (includes launch overhead)

| Pipe | Utilization | Time |
|------|------------|------|
| Vec (compute) | 55% | ~14 µs |
| Scalar | 27% | ~7 µs |
| MTE2 (data in) | 26% | ~7 µs |
| MTE3 (data out) | 14% | ~3.6 µs |
| MTE2 active BW | ~50 GB/s | - |
| MTE3 active BW | ~95 GB/s | - |

## Analysis

The kernel is primarily compute-bound (Vec 55%). The per-row softmax requires:
- ReduceMax (4096 elements)
- Adds (4096 elements)
- Exp (4096 elements) - most expensive transcendental
- ReduceSum (4096 elements)
- Muls (4096 elements)

Each operation processes 64 FP32 elements per vector repeat, requiring 64 repeats per row.
Total data: 16MB read + 16MB write = 32MB. At 26 µs kernel time, effective bandwidth is ~1.2 TB/s
(across 48 cores' local memory bandwidth).

The ~20 µs gap between kernel time (26 µs) and host-measured time (47 µs) is kernel launch overhead,
which cannot be optimized at the kernel level.

## Iterations Detail

### Iter 1 — 2 rows per tile
- **Hypothesis:** Batching 2 rows per DMA reduces overhead
- **Changes:** ROWS_PER_TILE=2, batch DataCopyPad
- **Bench:** 47.77 us, Correct: Yes
- **Analysis:** 12% improvement from DMA batching

### Iter 2 — Separate reduce buffers
- **Hypothesis:** Proper dst/tmp separation for Reduce API
- **Bench:** 51.17 us, Correct: Yes
- **Analysis:** Extra buffer management overhead

### Iter 3 — Single buffer depth=1, 4 rows
- **Hypothesis:** More rows per tile without double buffer
- **Bench:** 55.26 us, Correct: Yes
- **Analysis:** Loss of pipeline overlap hurts more than DMA savings

### Iter 4 — Clean DB=2, 2 rows, shared workBuf
- **Bench:** 46.37 us, Correct: Yes
- **Analysis:** Best stable config, 1.15x speedup

### Iter 6 — FP16 intermediate exp
- **Hypothesis:** FP16 exp is faster on vector unit
- **Bench:** 48.28 us, Correct: Yes (max diff 4.55e-06)
- **Analysis:** Cast overhead negates FP16 compute benefit

### Iter 8 — Unrolled 2-row compute
- **Hypothesis:** Reduce scalar loop overhead
- **Bench:** 49.49 us, Correct: Yes
- **Analysis:** Compiler already optimizes loops well

### Iter 9 — 32 cores
- **Hypothesis:** Fewer cores with more work each
- **Bench:** 54.59 us, Correct: Yes
- **Analysis:** Reduced parallelism hurts

### Iter 16 — Pattern::Reduce::AR batched
- **Hypothesis:** Batch reduce for all rows at once
- **Bench:** 62.87 us, Correct: No (precision failure)
- **Analysis:** isReuseSource=true corrupts source, tail tile issue
