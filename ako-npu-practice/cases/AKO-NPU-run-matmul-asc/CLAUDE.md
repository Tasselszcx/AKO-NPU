# 项目持久记忆

## 迭代计数规则

**一次 SWEEP（参数扫描）只算一次迭代**。不论 sweep 内部测试了多少个配置，在 ITERATIONS.md 中只占一行。不要把 sweep 的每个配置拆成单独的迭代编号，这是浪费。

## 当前优化状态

- **Baseline**: 227.9 us
- **当前最佳**: ~75 us (3.04-3.08x) [IterateAll + __mix__(1,2)] (warmup=10)
- **200 iterations completed**: hardware plateau confirmed, no further optimization possible
- **当前最佳 (warm cache)**: ~53 us (4.30x) (warmup=30, L2 cache hit)
- **Matmul-only 理论极限 (__mix__)**: ~70 us (IterateAll, warmup=10)
- **Matmul-only 理论极限 (warm cache)**: ~48 us (warmup=30)
- **fp16 output reference**: ~68 us (incompatible interface)
- **FixPipe bandwidth limit**: 57 us (cold) / 38 us (warm) for 2.5MB write
- **AIV LeakyRelu 时间**: ~45-71 us (cold) / ~30-50 us (warm)

## 已确认有效的优化

1. matmul C 输出到 GM（不用 VECIN 融合）+ 独立 LeakyRelu pass
2. FIRSTN 遍历（L1 A矩阵复用）
3. SetFixSplit(256, 128, -1)
4. L1CacheUB=true (SetMatmulConfigParams(1, true))
5. stepM=2, stepN=1
6. TQue 双缓冲 LeakyRelu（VECIN+VECOUT 各 2 buffer，chunk=12288 float）
7. numBlocks=1, usedCoreNum=2, __mix__(1,2)
8. IterateAll replaces Iterate+GetTensorC loop (huge improvement: 110→75 us)

## 已确认无效/有害的方向

- HF32: 略有害（在当前 scalar-bound 场景下）
- OUTER_PRODUCT / N_BUFFER_33 schedule: incorrect
- __mix__(2,2) / __mix__(1,4): 编译失败
- __mix__(1,1): incorrect
- usedCoreNum=1: incorrect
- fixSplit(384,128), (256,192), (192,192): incorrect（不整除 singleCoreM/N）
- fixSplit(512,128), (256,160), (256,256): tiling 失败（超出 UB/L0）
- fixSplit(128,64), (64,128): 大幅退步（scalar 开销）
- stepM=4: 退步
- 显式 baseK: hang
- 显式 L0A/L0B: incorrect
- FIRSTM: 退步（比 FIRSTN 差 ~20%）
- async Iterate<false>: hang
- finer sync (PIPE_MTE2/V/MTE3 替代 PIPE_ALL): incorrect
- async IterateAll<false>: no improvement vs sync
- async IterateAll<false> + waitIterateAll=true: no improvement
- MDL config (SetMatmulConfigParams(2, true)): slightly worse
- CFG_MDL template config: regression (81 us)
- enableMixDualMaster=true: incorrect (0.5 error ratio)
- TBuf single-buffer PIPE_ALL LeakyRelu: regression (86 us, no pipeline overlap)
- numBlocks=2: incorrect (wrong data distribution)
- reversed AIV block assignment: incorrect (data race)
- async IterateAll + enSequentialWrite: hang
- mmConfigType=0 (Norm) + noL1CacheUB: no reliable improvement
- auto tiling (no fixSplit): slightly worse (~77 vs ~75 us)
- LeakyRelu chunk sizes 8K-12K: no significant difference
- transposed A matrix: compile error
- stepM=1 with IterateAll: slightly worse
- VECIN C output for fused LeakyRelu: compile error
- Single AIV (aivBlockIdx!=0 return): no improvement (~74 us)
- Muls+Max instead of LeakyRelu intrinsic: no improvement (~75 us)
- fixSplit(128,128): regression (80 us)
- fixSplit(512,64): regression (117 us)
- fixSplit(256,256): compile error (invalid core ratio)
- stepM=3: regression (79 us)
- stepN=2: no improvement (~76 us)
- FIRSTM with IterateAll: regression (103 us)
- L1CacheUB=false: no improvement (~75 us)
- fp16 output: 67.7 us but incompatible interface (original requires fp32)
- __mix__(1,1): incorrect (50% error, tiling expects 2 AIVs)
- __mix__(1,3): compile error (invalid core ratio on 2-core chip)
- usedCoreNum=1 with __mix__(1,2): incorrect
- stepM=2 + stepN=2: no improvement (~74.6 us)
- fixSplit(256,128,256): tiling failed/hang
- fixSplit(512,128): tiling hang
- Raw matmul (no bias, no relu): 70 us = FixPipe hard floor
- CalcOffset(0) with numBlocks=1: incorrect (singleCoreM=512, only half computed)
- numBlocks=2 with linear LeakyRelu: incorrect (AIV offset mismatch)
- SetBias(false) tiling with SetBias in kernel: hang (tiling/kernel mismatch)
- warmup=20+ reveals ~53 us steady-state (L2 cache hit), but warmup=10 baseline is ~74 us
- async IterateAll<false> + fakeMsg=true: 65 us but INCORRECT (race condition, 99.9% errors)
- sync IterateAll + fakeMsg=true: no improvement (74.7 us)
- SW-pipelined prefetch in LeakyRelu: regression (109.6 us, breaks TQue scheduling)
- Asymmetric AIV split (40/60%): no improvement (73.2 us)
- Reversed AIV block assignment: slight regression (76.3 us)
- fixSplit(256,320): tiling failed
- fixSplit(256,64) + IterateAll: regression (86.9 us)
- enAtomic=1 in IterateAll: regression (80.6 us)
- SetTail(-1,-1,-1): no improvement
- SetBufferSpace explicit L1/L0C/UB: no improvement or hang
- SetHasBlock, SetOrgShape: compile errors
- VECIN C output + Iterate fused LeakyRelu: GetTensorC still returns GlobalTensor, not LocalTensor
- fixSplit(256,128,128) with baseK: no improvement (74.0 us)
- fixSplit(128,128) + stepM=4: regression (76.7 us)
- Relu instead of LeakyRelu (alpha=0): no improvement (75.9 us, passes tolerance)
- Reverse chunk processing order: no improvement (74.3 us)
- TQue depth=1: regression (79.8 us, less pipelining)
- Chunk 6K/9K: no improvement (74-77 us)

## NPU 环境

- Ascend910B1 (DAV_2201), 1800MHz
- CANN 8.3.RC1
- `__kfc_workspace__` 不可用，需要去掉
- npu-smi 不可用但 NPU 环境正常
- msprof 可用，需要 chmod 700 输出目录
- set_env.sh 在 /usr/local/Ascend/ascend-toolkit/set_env.sh
