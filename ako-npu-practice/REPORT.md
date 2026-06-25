# AKO-NPU 实践记录

记录 AKO-NPU 框架在不同算子上的实践案例、框架版本演进、以及发现的问题。

## 版本演进

### V1：初始验证（子 agent 模式）

**框架状态**：有 init.sh、TASK.md 作为入口、skills 通过 symlink 放在根目录 `skills/`

**运行方式**：从父会话通过 Agent 工具 spawn 子 agent

**案例**：

| 案例 | 算子 | 输入类型 | 迭代数 | 最优结果 |
|------|------|---------|--------|---------|
| `AKO-NPU-run-add` | Add (32K float32) | Ascend C | 20 | 1.94 GB/s 吞吐（但改了 shapes） |
| `AKO-NPU-run-softmax` | Softmax [1024,4096] float32 | Ascend C（手写） | 17 | 46.12 us (1.16x) |

**发现的问题**：
1. **Skills 不被自动发现**：子 agent 调用 `/ascendc-env-check` 报 `Unknown skill`。因为 Claude Code 的 skill 自动发现只在独立 `claude` 进程启动时扫描 `.claude/skills/`，子 agent 不走这个流程
2. **Agent 手动 Read skill 文件代替**：虽然 Skill 工具调用失败，agent 退而求其次用 Read 工具读 SKILL.md，但只在开头读了一次
3. **Shapes 被 agent 修改**：Add 算子从 32K 改到 16M 来"提升吞吐"，导致 speedup 不可比
4. **init.sh 重复 CANNBot 的工作**：AKO-NPU 自己写了 init.sh，和 CANNBot 原版功能重叠

---

### V2：独立进程 + CANNBot 集成

**框架改动**：
- 删除 init.sh，改为纯规则项目
- TASK.md 重写为路由入口（Step 0 环境准备 → Step 1 路由 → Step 2 CANNBot 开发 → Step 3 AKO 优化）
- 新增 OPTIMIZE.md（迭代优化规则）、HINTS.md（停滞策略）
- Skills/agents 安装交给 CANNBot 原版 init.sh，AKO-NPU 的 TASK.md Step 0 负责提升 symlink 到父目录
- 解耦：去除所有硬编码 skill 名和 CANNBot 路径，改用描述性分类和配置变量

**运行方式**：直接在项目目录下 `claude --dangerously-skip-permissions`（独立进程）

**案例**：

| 案例 | 算子 | 输入类型 | 迭代数 | 最优结果 |
|------|------|---------|--------|---------|
| `AKO-NPU-run-matmul-asc` | MatMul+LeakyReLU [1024,256]×[256,640] | Ascend C（asc-devkit 样例） | 60 | 68.6 us（msprof 实测） |
| `AKO-NPU-run-matmul-py` | 同上 | PyTorch 描述 | 96 | 32.8 us / 6.97x（msprof 实测） |
| `AKO-NPU-run-attn-bwd` | Attention Backward [4,256,256] | PyTorch 描述 | 186 | 0.186 ms overlap / 0.264 ms 串行 |

**独立验证结果（2026-04-09 在卡7上）**：

| 案例 | 声称最优 | 实测性能 | 精度验证 |
|------|---------|---------|---------|
| softmax | 46.12 us (1.16x) | 49.91 us (1.07x) | ✅ PASS |
| matmul-asc | ~110 us (2.07x) | 79.5 us (msprof Task Duration) | ✅ error ratio=0.0000（当次通过） |
| matmul-py | 23.6 us (9.68x) | 24.3 us / 9.41x (msprof) | ✅ MERE=1.2e-6, MARE safe=1.2e-4, 用了 CANN 标准 |
| attn-bwd | 0.198 ms (14x) | 0.201 ms overlap / 0.268 ms 串行 | ❌ grad_attn_scores atol=3e-5（限1e-5），grad_value_states atol=1.2e-4（限1e-5） |
| lm-head | - | 编译失败（代码有 API 兼容问题） | - |

**精度问题详情**：
- **matmul-asc**：本次验证 error ratio=0.0000（通过），但之前验证 rdiff 13.4%，说明精度不稳定（可能跟随机输入数据有关）
- **matmul-py**：用了 CANN 官方 MERE/MARE 标准，精度合格（因为 CANNBot reviewer 建立了正确的验证脚本）
- **attn-bwd**：两个输出的 atol 都超标 3-12 倍，但 rtol≤0.05 通过。bench.sh 只检查 rtol 放过了问题。输出值范围很小（±0.006 和 ±0.018），绝对误差虽小但相对于值域已不可忽略

**发现的问题**：

1. **Skills 确认在独立进程中被自动发现**：4 个 skill 成功调用（ascendc-env-check、ascendc-npu-arch、ascendc-tiling-design、ascendc-api-best-practices），全部 `success=true`

2. **Skills 只在开头读一次**：三个项目模式一致——开头 1 小时内集中读完 skills，后面 15-19 小时的迭代中再也没读过。原因是 skill 内容加载到上下文后不再需要重新读，但 compact 后会丢失

3. **TASK.md 不是 CLAUDE.md，compact 后规则丢失**：TASK.md 被 Read 进来后是对话历史的一部分，compact 会压缩掉。只有 `./CLAUDE.md` 和 `.claude/CLAUDE.md` 会在 compact 后重新注入

4. **精度验证是最大问题**：
   - matmul-asc：精度完全不对（13% rdiff），但 agent 没注意
   - attn-bwd：atol 超标但 bench.sh 只检查 rtol，放过了
   - 三个项目**都没读过 ops-precision-standard skill**（只有 matmul-py 的 CANNBot reviewer 子 agent 读了）

5. **attn-bwd 的 CANNBot 流程执行不完整**：只有 Architect 用了正确的 `ascendc-kernel-architect` subagent_type，后面 Developer 和 Reviewer 全用了 default agent。Reviewer 根本没被调用 → 没有 REVIEW.md → 没有精度标准校验

6. **matmul-py vs attn-bwd 的对比揭示了问题**：
   - matmul-py：忠实执行 CANNBot 7 步流程 → reviewer 读了 ops-precision-standard → verify_result.py 用 MERE/MARE → 开发阶段精度正确
   - attn-bwd：跳过了 reviewer → 自己写了 atol/rtol 检查 → 标准松 → 精度不达标

7. **Sweep 占用大量迭代编号**：一次参数搜索跑几十个配置，每个都算单独迭代，迅速耗尽迭代预算

8. **D2H 在计时外**：attn-bwd 的 bench 把 Device→Host 拷贝放在计时外，overlap 模式的计时只看 stream 不看 stream2

---

### V3：规则持久化 + 迭代规范

**框架改动**：
- `TASK.md` → `CLAUDE.md`：放在项目根目录，Claude Code 启动时自动读取，compact 后重新注入，规则不再丢失
- `@OPTIMIZE.md` 和 `@HINTS.md` import：随 CLAUDE.md 一起注入，也不会丢
- Shapes 不可变：确定后整个优化过程中不得更改
- Sweep = 1 迭代：参数搜索整体算一次，不逐个算
- 每轮迭代必须重新调用 skill 刷新知识（防止 compact 后知识丢失）
- Baseline 定义：基于 msprof Task Duration，排除 host I/O
- 迭代计数规则：profile = 1 iter，失败/revert = 1 iter，禁止批量打包编号
- NPU 环境提示：npu-smi 不可用不代表 NPU 不可用

**案例与独立验证结果（2026-04-10 在卡7上）**：

| 案例 | 算子 | 迭代数 | Baseline | 最优（声称） | 实测性能 | 精度 |
|------|------|--------|----------|-------------|---------|------|
| `attn-bwd-v2` | Attention Backward [4,256,256] | 5 | 162,841 us | 1,723 us (94.5x) | 1,117 us (msprof) | ✅ allclose 100% |
| `vae-residual` | VAE Conv+GN+SiLU Residual [1,256,64,64] | 200+ | ~15,000 us | ~100 us (146x) | 27-42 us (msprof, 多kernel) | ⚠️ max_rel_error=2084（有异常值），mean=0.007 |
| `lm-head` | LM Head [1,128,2048]→[102400] | 138 | W1=289us W4=1126us | W1=285us W4=287us | W1=285us W3=327us W4=287us | ✅ error_ratio=0, atol/rtol 通过 |

**注意事项**：
- **vae-residual** 精度有隐患：max relative error = 2084（极端异常值），虽然 mean 和 allclose 通过了，但个别点误差极大
- **attn-bwd-v2** 只跑了 5 轮迭代（可能提前停止或遇到问题）
- **lm-head** 的 W1/W2/W4（logits_to_keep=1）优化空间极小，W3（logits_to_keep=128）从 404us→327us 有 19% 改善，W4 从 1126us→287us 有 3.9x 改善（batch merge）

---

### V4：精度硬约束

**框架改动**：
- 精度硬约束：精度失败必须立即 revert，不允许带着错误精度继续优化
- bench 必须同时检查 atol 和 rtol，任一超标即 FAIL

**案例与独立验证结果（2026-04-10 在卡7上）**：

| 案例 | 算子 | 迭代数 | Baseline | 最优（声称） | 实测性能 | 精度 |
|------|------|--------|----------|-------------|---------|------|
| `matmul-asc-v4` | MatMul+LeakyReLU (Ascend C) | 33 | 224.55 us | 75 us (2.97x) | 74.5 us (msprof) | ✅ error_ratio=0.0000 |
| `matmul-py-v4` | MatMul+LeakyReLU (PyTorch) | 204 | 408.34 us | 21 us (19x) | 4.8us+11.8us=16.6us (两阶段) | ✅ allclose PASS, mean_rel=1.77e-4 |
| `attn-bwd-v4` | Attention Backward [4,256,256] | 173 | 316.5 ms | 19.3 ms (16.4x) | 精度 PASS（所有3个shapes） | ✅ 三个 shape 全部 PASS |

**关键发现**：
- **matmul-asc-v4 vs V2 matmul-asc**：V4 精度通过 (error_ratio=0)，V2 曾出现 13.4% rdiff → 精度硬约束有效
- **matmul-py-v4 两阶段架构**：拆成 Matmul(4.8us) + LeakyReLU(11.8us) 两个 kernel，总计 16.6us，比 V2 的 24.3us 更快
- **attn-bwd-v4 vs V2 attn-bwd**：V4 精度在三个 shapes 上全部 PASS，V2 的 atol 超标问题不再出现

**V4 代码分析：为什么 matmul-asc-v4 (3x) 远差于 matmul-py-v4 (24.6x)**：

同一个算子（MatMul+LeakyReLU [1024,256]×[256,640]），两者架构完全不同：

| | matmul-asc-v4 (74.5us) | matmul-py-v4 (16.6us) |
|---|---|---|
| 架构 | 单 kernel `__mix__(1,2)` | 两阶段：`__cube__` + `__vector__` |
| Matmul 核数 | **1-2 个核** | **22 个 Cube 核** |
| LeakyReLU 核数 | 同 kernel 2 个 Vector 核 | **20 个 Vector 核** |

matmul-asc-v4 被困在 2 核的原因：agent 多次尝试多核（iter 6/19/20/21/27/28/56），但 `__mix__` 模式下多核写 GM 有竞争导致精度失败，V4 的精度硬约束强制 revert。最终只能用 2 核。

matmul-py-v4 绕过了这个问题：拆成两个独立 kernel（Pass 1 纯 Cube 做 Matmul，Pass 2 纯 Vector 做 LeakyReLU），每个 kernel 内部不存在 Cube/Vector 的 GM 写冲突，因此可以用满所有核。代价是多了一次 GM 读写（中间结果经过 GM），但 22 核 vs 2 核的并行度收益远大于额外 GM 开销。

**结论**：差异不是优化质量问题，而是架构选择的局限——`__mix__` 单 kernel 模式在 CANN 8.3.RC1 上多核精度有 bug，asc-v4 被精度约束卡住了。如果 asc-v4 也采用两阶段架构，理论上能达到类似性能。

**attn-bwd-v4 的精度阈值说明**：

verify_result.py 中阈值被放宽为 `max_atol=0.0065/0.019, max_rtol=1.0`（原始要求 atol=1e-5）。这是因为 bf16 计算的精度极限约为 2^-7 ≈ 0.0078，atol=1e-5 对 bf16 不可达。放宽 atol 是合理的，但 `max_rtol=1.0`（等于不检查）不够严格。

**V4 读 Skill 情况**：

| 项目 | Skill 工具调用 | Read 次数 | 持续读了多久 | 之后空白多久 |
|------|-------------|----------|-------------|------------|
| matmul-asc-v4 | 0 | 12 | 39 分钟 | 22 分钟 |
| matmul-py-v4 | 0 | 5（主进程） | 49 分钟 | 11 小时没读 |
| attn-bwd-v4 | 0 | 19 | 3 小时 | 之后到结束没读 |

三个 V4 实验**都没用 Skill 工具（0 次调用）**，全是 Read 手动读的。OPTIMIZE.md 里"每轮必须重新调用 skill"的规则**仍未被遵守**。attn-bwd-v4 略好（3 小时内读了 19 次），但后半程依然是空白。

---

---

### V5：References 字段引导 skill 查阅

**框架改动**：
- ITERATIONS.md 模板新增 `References` 必填字段，要求记录每轮参考的信息来源（skills 文件、asc-devkit 文档/示例、模型自身知识）
- 不强制每轮必须读 skill，但通过"填空题"引导 agent 去查阅

**案例与独立验证结果（2026-04-11 在卡7上）**：

| 案例 | 算子 | 迭代数 | Baseline | 实测性能 | 加速比 | 精度 |
|------|------|--------|----------|---------|--------|------|
| `dsa-v5` | DSA Indexer [1,1,4096] | 125 | ~40 us | 4.3 us (msprof) | **9.33x** | ✅ index_score+topk PASS |
| `matmul-asc-v5` | MatMul+LeakyReLU (Ascend C) | 200 | 226.09 us | 36.2 us (msprof) | **6.15x** | ✅ error_ratio=0 |
| `matmul-py-v5` | MatMul+LeakyReLU (PyTorch) | 105 | 98.62 us | 20.36 us (msprof) | 4.84x | ✅ error_ratio=0 |
| `attn-bwd-v5` | Attention Backward (3 shapes, torch_npu) | 25 | 1.40/11.42/53.26 ms | 0.53/3.77/14.90 ms | **2.6x-3.6x** | ✅ atol=0, rtol=0 |
| `dsa-indexer-bwd` | DSA Indexer Backward [1,4096,4096] | 200 | 180.4 ms | 54.3 ms | **3.31x** | ✅ |

**V5 读 Skill 情况与 References 字段效果**：

| 项目 | Skill 工具 | Read 次数 | References 字段 | 引用 skills/asc-devkit | 引用模型知识+micro-bench |
|------|-----------|----------|----------------|----------------------|------------------------|
| dsa-v5 | 2 | 7 | **40 处** | **~20 处**（MatmulConfig、matmul_high_performance 等） | 少量 |
| matmul-asc-v5 | 0 | 13 | 0（按 Phase 摘要格式，未逐轮记录） | 实际读了 13 次 skill 文件（session 记录） | — |
| matmul-py-v5 | 0 | 11 | **32 处** | **24 处**（ops-profiling、api-buffer、tiling-design、matmul 示例等） | 少量 |
| attn-bwd-v5 | 0 | 17 | **16 处** | 2 处（ascendc-npu-arch） | **大量**（每步有 micro-benchmark 数据支撑） |

**关键发现**：
- **References 字段在 3/4 个实验中有效**：dsa-v5、matmul-py-v5、attn-bwd-v5 都填写了 References
- **DSA-v5 大量引用了 asc-devkit 文档**：包括 MatmulConfig.md、SetFixSplit.md、matmul_high_performance 示例、L2Cache 示例等，说明 References 字段确实引导了 agent 去查阅具体文件
- **attn-bwd-v5 混合使用 skills 和模型知识**：References 里既有"Skills: ascendc-api-best-practices"也有"模型自身知识: broadcast matmul avoids materializing expanded tensor"
- **matmul-asc-v5 是唯一没填 References 的**：可能因为它用了不同的 ITERATIONS.md 格式（按 Phase 分组而非逐轮记录）

**matmul-asc V4→V5 对比**：

| | V4 (74.5 us, 3.01x) | V5 (36.2 us, 6.15x) |
|---|---|---|
| 核数 | 2（被精度 bug 卡住） | **8**（通过 MatmulApiTiling 绕过） |
| 关键突破 | 无，停在单核+LeakyReLU | iter 51 发现 MatmulApiTiling 多核 |
| 架构 | `__mix__(1,2)` 单 kernel | 同，但用了不同的 tiling API |

**attn-bwd-v5 分析**：跑了 25 轮，使用 torch_npu 算子优化（非 Ascend C kernel 级别）。通过 bf16 全流水线、native softmax backward、broadcast matmul 等 PyTorch/torch_npu 层面优化，实现了 **2.6x-3.6x 加速**（small: 1.40→0.53ms, medium: 11.42→3.77ms, large: 53.26→14.90ms）。精度验证三个 shape 全部 PASS（atol=0, rtol=0）。

**dsa-indexer-bwd 分析**：200 轮迭代，DSA Indexer 的反向传播 kernel（对 einsum+ReLU+加权求和求梯度）。优化路径：einsum→matmul(1.6x) → in-place relu(1.8x) → 预 permute layout(2.8x) → torch.where mask(3.05x) → npu_linear(3.2x) → 内存管理(3.31x)。后 100+ 轮尝试了 fused kernel、NPU ops、chunking、dtype 等方向，均未突破 50ms ceiling。使用 torch_npu 层面优化，非 Ascend C kernel。

---

### Baseline 与加速比的可比性分析

**重要：不同实验的加速比不宜直接横向对比。** 各实验的 baseline 定义和计时方式差异很大：

| 实验 | Baseline 含义 | 计时包含 |
|------|-------------|---------|
| V2 attn-bwd | 纯 kernel 时间（2.78ms） | 不含 host I/O，不含 REGIST |
| V3 attn-bwd-v2 | 纯 Vector 实现（162.8ms） | 不含 host I/O |
| V4 attn-bwd-v4 | 端到端（316.5ms） | 含 host I/O + REGIST 初始化开销 |

**attn-bwd-v2 的 148x "加速"详解：**

iter 3 引入 `__mix__(1,1)` 模式，从纯 Vector 标量循环做矩阵乘（每个 head 65536 次 SetValue/GetValue）切换到 Cube 硬件引擎（Matmul API + IterateAll）。这不是微优化，是**计算单元的架构切换**——相当于从"用 CPU 逐元素做矩阵乘"换成"用 Tensor Core 做矩阵乘"。因此 162.8ms → 1.7ms 的跳跃是合理的。

**attn-bwd-v4 为什么"只有" 16.4x 而 V2 有 14x：**

两者的实际 kernel 计算时间是差不多的（都约 0.2ms）。差异在于：
- V4 的 19.3ms 里约 17-19ms 是 `REGIST_MATMUL_OBJ` 的平台初始化开销（iter 40 测出空 kernel 只需 0.008ms），这是 CANN 8.3.RC1 的平台固定开销
- V2 的 0.2ms 把 REGIST 开销排除在了 warmup 之外

所以不是 V2 更快，是**计时标准不同**。

---

## 关键发现总结

### 框架层面

| 问题 | 严重性 | V1 | V2 | V3 | V4 | V5 |
|------|--------|----|----|-----|-----|-----|
| Skills 不被自动发现（子 agent） | 高 | ❌ | ✅ 改用独立进程 | ✅ | ✅ | ✅ |
| Skills 只读一次，compact 后丢失 | 高 | ❌ | ❌ | 规则要求重读 | 规则要求重读 | ✅ References 引导（3/4 有效） |
| 规则 compact 后丢失 | 高 | ❌ | ❌ | ✅ 改名 CLAUDE.md | ✅ | ✅ |
| 精度无硬约束 | 高 | ❌ | ❌ | ❌ | ✅ 失败必须 revert | ✅ |
| Shapes 可被 agent 修改 | 中 | ❌ | ✅ | ✅ | ✅ | ✅ |
| Sweep 浪费迭代编号 | 中 | ❌ | ❌ | ✅ 1 sweep = 1 iter | ✅ | ✅ |
| 硬编码 skill 名和路径 | 低 | ❌ | ✅ 解耦 | ✅ | ✅ | ✅ |
| 迭代有参考来源记录 | 中 | ❌ | ❌ | ❌ | ❌ | ✅ References 字段（3/4 填写） |

### 精度层面

| 问题 | 影响 |
|------|------|
| CANN 官方用 MERE/MARE，不是 atol/rtol | Agent 自己搞了更松的标准 |
| ops-precision-standard skill 从没被主动调用 | 只有 CANNBot reviewer 在开发阶段读了 |
| bench.sh 只检查 rtol 不检查 atol | 放过了超标的结果 |
| 优化阶段不用开发阶段的 verify 脚本 | 开发阶段建立的严格标准被绕过 |

### CANNBot 集成层面

| 问题 | 影响 |
|------|------|
| subagent_type 不一定被正确使用 | attn-bwd 除 architect 外全用了 default |
| Reviewer 可能被跳过 | attn-bwd 没有 REVIEW.md |
| 开发阶段和优化阶段精度标准脱节 | 开发用 MERE/MARE，优化用自己的 |

---

## 案例目录

```
cases/
├── AKO-NPU-run-add/            # V1: Add 算子（子 agent）
├── AKO-NPU-run-softmax/        # V1: Softmax 算子（独立进程验证）
├── AKO-NPU-run-matmul-asc/     # V2: MatMul+LeakyReLU（给 Ascend C）
├── AKO-NPU-run-matmul-py/      # V2: MatMul+LeakyReLU（给 PyTorch）
├── AKO-NPU-run-attn-bwd/       # V2: Attention Backward（PyTorch，186 轮）
├── AKO-NPU-run-attn-bwd-v2/    # V3: Attention Backward（规则持久化，已启动）
├── AKO-NPU-run-vae-residual/   # V3: VAE Residual（已启动）
└── AKO-NPU-run-lm-head/        # V3: LM Head Projection（已启动）
```

V4 案例待后续创建。
