# big-matmul：大 shape 计算密集 MatMul 实验

> 记录时间：2026-06-28 ｜ 主机：`set-hlsc-mlp-codelab-pc181` ｜ 卡：Ascend910B2C（单卡调优用 #15，整机用 #0–15）
> 目的：跳出「数据极小」的 baseline 案例，构造一个**真正吃算力**的大 shape MatMul，用来 (1) 量化单卡算力利用率、(2) 通过 tiling 调参拿到精度安全的单卡加速比、(3) 16 卡并行展示整机可用性。
> 基线算子：复用 `cases/AKO-NPU-run-matmul-py` 的 MatMul+LeakyReLU kernel（`__mix__(1,2)` 模式），仅把 shape / tiling / 核数做成编译期宏参数化，**不改算子逻辑**。

## 1. 结论速览

- **算子**：`C = LeakyReLU(A·B + bias, α=0.001)`，A `[4096,4096]` fp16 × B `[4096,4096]` fp16 + bias fp32 → out fp32。单次 **137.4 GFLOP**（2·M·N·K）。
- **单卡 tiling 调参拿到 1.50× 加速比，精度全程 PASS**：原始 tiling `baseM×baseN=256×128`（23096 µs）→ 最优 `128×256`（15428 µs）。Cube 利用率 `aic_cube_ratio` 从 **0.495 → 0.635（+28%）**。
- **16 卡并行**：48 次 kernel 启动（16 卡 × 3 次）全部精度 PASS，墙钟 60.6 s；串行等价 ≈349 s，整机并发吞吐 ≈**5.8×**。
- **诚实的天花板**：msprof 显示 `Block Dim=1 / Mix Block Dim=2`——**实际只用了 1 个 Cube 核**。8.91 TFLOPS 是单 Cube 的数字，远低于整卡 ~24 核峰值。真正放大单卡算力需要**多 Cube 算子改写**（手写 CalcOffset 无法正确按 launch block 切分输出，属于算子重写范畴，超出"参数调优"边界）。

## 2. 单卡 tiling sweep（4096³，SetDim=2，blocks=1）

| baseM×baseN | kernel Task Duration | TFLOPS | aic_cube_ratio | 精度 |
|-------------|----------------------|--------|----------------|------|
| 256×128（原始 tiling 基线） | 23096 µs | 5.95 | 0.495 | ✅ PASS |
| 128×128 | 19211 µs | 7.15 | 0.527 | ✅ PASS |
| **128×256（最优）** | **15428 µs** | **8.91** | **0.635** | ✅ PASS |
| 256×256 / 128×512 / 512×128 | — | — | — | ❌ FAIL |

- **加速比** = 23096 / 15428 = **1.497× ≈ 1.50×**（vs 原始 tiling，精度 PASS，shape 不变）。
- FAIL 三组原因一致：单块 tile 超出 L0C 容量，`GetTiling` 返回 0（`[TILING] M=0 N=0 …`），kernel 写出全零 → MERE≈1.0。这不是精度退化，是 tiling 非法，等价于"该配置不可用"。
- 计时口径：msprof `OpBasicInfo.csv` 的 **Task Duration**（纯 kernel 时延，--warm-up=5 --launch-count=10），符合 AKO 硬约束"baseline = 纯 kernel 时间"。

## 3. 单卡算力利用率（msprof）

最优配置 `128×256` 的 profiling：

| 指标 | 值 | 解读 |
|------|----|----|
| Task Duration | 15428 µs | 纯 kernel |
| Block Dim / Mix Block Dim | 1 / 2 | **只 1 个 AIC + 2 个 AIV 实际运行** |
| aic_cube_ratio | 0.635 | Cube 流水占比，调参后已较高 |
| 等效算力 | 8.91 TFLOPS | 单 Cube 口径 |

**关键观察**：`SetDim(usedCoreNum)` 只是告诉 tiling 按几个核切，**并不会真正拉起多个 Cube**；要真多核必须 `numBlocks>1` 的 launch grid。但本 kernel 的手写 `CalcOffset`/`CopyOut` 在 `blocks>1` 时只有一个 block 的输出区写对（实测 4096³ blocks=2 时前 2048 行全错、后 2048 行对），说明**多 Cube 需要算子级改写**，不在本次"参数化 + 调参"范围内。这恰好印证了 AKO 框架的分工：参数调优能拿到 1.5×，更大的算力释放要进入算子开发（CANNBot）阶段。

## 4. 16 卡并行（整机利用）

脚本 `par16_run.sh`：为每张卡建独立工作目录（软链最优二进制 + 各自 input/output，避免文件竞争），16 卡同时各跑 3 次最优配置，统计每卡精度与总墙钟。

```
card 0..15  launches_ok=3/3  precision=PASS      # 16 张卡全部 PASS
wall_clock = 60.6 s   total_launches = 48
```

- 单进程隔离运行 ≈ 7.27 s（**几乎全是 host 端 device-init + H2D/D2H**，NPU kernel 只占 ~15 ms）。
- 串行等价 48 × 7.27 ≈ 349 s，16 卡并发墙钟 60.6 s → 整机并发吞吐 ≈ **5.8×**。
- **为什么不是 16×**：单进程是 host-overhead-bound（设备初始化 ~7 s ≫ kernel 15 ms），16 路并发时 host CPU / PCIe 争用把单进程拉长到 ~20 s。这是"单发二进制"harness 的固有限制，不是 NPU 算力瓶颈——若把多次迭代放进**同一进程内复用 context**，并发比会接近核数。
- HBM 维度看不出占用：4096³ 三个矩阵 fp16/fp32 工作集约 200 MB，相对 61 G 卡 <1%，且 kernel 仅 15 ms，`npu_info.py` 采样基本采到进程间隙 → 全卡显示 ~0.6%。**这类算子算力密集、显存极小**，与 EXPERIMENTS.md 中 baseline 案例的结论一致。

## 5. 复现

```bash
cd ako-npu-practice/experiments/big-matmul
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 单配置：bash run_config.sh <tag> <M> <N> <K> <SetDim> <blocks> <baseM> <baseN>
ASCEND_RT_VISIBLE_DEVICES=15 bash run_config.sh best 4096 4096 4096 2 1 128 256

# tiling sweep（在 4096³ 上扫 baseM/baseN）
ASCEND_RT_VISIBLE_DEVICES=15 \
  for cfg in "128 128" "128 256" "256 128"; do set -- $cfg; \
  bash run_config.sh tk_$1x$2 4096 4096 4096 2 1 $1 $2; done

# 16 卡并行（先确保 build_tk_128x256/ 已生成）
REPS=3 bash par16_run.sh
```

- `PY` 默认 `/usr/local/conda/bin/python`（有 numpy；`/usr/bin/python3` 无 numpy）。
- `SKIP_PROF=1` 可跳过 msprof。
- ASC_DIR 用 `find ASCConfig.cmake` 发现的真实路径（`latest` 软链缺 `kernel_tiling` 头）。

### 踩坑清单

| 现象 | 原因 | 解法 |
|------|------|------|
| 大 tile（≥256×256 @4096³）精度 FAIL、输出全零 | tile 超 L0C，`GetTiling` 返回 0 | 用 ≤128×256 的合法 tiling |
| `blocks>1` 多块只有一块输出对 | 手写 CalcOffset 不能按 launch block 切分 | 多 Cube 需算子改写，调参阶段固定 blocks=1 |
| `matmul-asc` 基线编译 segfault | bisheng 不识别 `__kfc_workspace__` | 改用 `matmul-py`（同算子，`__mix__(1,2)`，无该宏） |
| `No module named numpy` | set_env 把无 numpy 的 `/usr/bin/python3` 推到 PATH | 用 conda python 跑 gen/verify |

## 6. 与 AKO 框架的关系

本实验严格遵守 AKO 硬约束：**shape 固定不变、精度失败即视为该配置不可用、计时用 msprof 纯 kernel 时间、一次 sweep 当一轮迭代**。结果验证了框架的核心判断——

- **参数调优（AKO 循环能做的）**：tiling sweep 在不碰算子逻辑、精度全 PASS 的前提下拿到 **1.50× 单卡加速**，Cube 利用率 +28%。
- **算子改写（需进入 CANNBot 开发阶段）**：单 Cube → 多 Cube 才能进一步逼近整卡峰值，这超出参数调优边界，正是 AKO Step 2 的职责。

## 7. 涉及文件

| 文件 | 说明 |
|------|------|
| `matmul_leakyrelu.asc` | 参数化基线 kernel（shape/tiling/核数 = 编译期宏，含 `[TILING]` 调试打印） |
| `CMakeLists.txt` | `MM_M/N/K/CORES/BLOCKS/BASEM/BASEN` 缓存变量 |
| `run_config.sh` | 单配置 build→run→verify→msprof 驱动 |
| `par16_run.sh` | 16 卡并行 demo（每卡独立工作目录） |
| `scripts/gen_data.py` / `verify_result.py` | golden 生成 / MERE-MARE 精度比对 |
| `build_tk_*/msprof_latest/` | 各 tiling 配置的 msprof 原始 csv |
