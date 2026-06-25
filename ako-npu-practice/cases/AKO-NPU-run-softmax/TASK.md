# AKO-NPU

优化 solution/ 中的 NPU 内核，追求最大性能。优化后的内核必须产出与参考实现一致的正确结果。

你的目标是真正的延迟降低——不是最大化报告的加速比。不要使用在生产中没有价值的技巧：注入额外的流/线程来逃避计时、篡改计时函数或评测脚本、返回未初始化的结果、或任何其他形式的 reward hacking。

## 重要：本项目面向 NPU（华为昇腾），不是 GPU

你对 NPU 编程的知识可能有限。本项目已通过 Claude Code 自动发现机制加载了完整的 NPU 开发 skills 和 agents。

**在做任何 NPU 相关的技术决策前，先查阅相关 skills。不要凭直觉或 GPU 经验来编写 NPU 代码。**

### 可用的 Skills

以下 skills 已注册到 Claude Code，会在相关场景自动加载，你也可以主动调用：

| Skill | 用途 | 何时使用 |
|-------|------|---------|
| `ascendc-env-check` | 环境检查 | Setup 阶段检查 NPU 和 CANN 环境 |
| `ascendc-direct-invoke-template` | 直调工程模板 | 构建编译运行流程 |
| `ascendc-api-best-practices` | API 最佳实践 | 选择和使用 Ascend C API |
| `ascendc-tiling-design` | Tiling 设计方法论 | 设计切分策略、Buffer 规划 |
| `ascendc-npu-arch` | NPU 架构知识 | 了解硬件参数、核数、带宽 |
| `ops-profiling` | 性能采集与分析 | 采集 msprof 数据、定位瓶颈 |
| `ascendc-precision-debug` | 精度调试 | 正确性验证失败时 |
| `ascendc-runtime-debug` | 运行时调试 | 编译或运行报错时 |
| `ascendc-code-review` | 代码审查 | 优化后质量检查 |
| `ascendc-docs-search` | 文档搜索 | 查找 API 文档和示例 |

### 可用的 Agents

以下 subagents 已注册，可用于委派专业任务：

| Agent | 用途 |
|-------|------|
| `ascendc-kernel-architect` | 架构设计、Tiling 方案 |
| `ascendc-kernel-developer` | 代码实现、编译测试 |
| `ascendc-kernel-reviewer` | 代码审查、质量评估 |

## Setup

确保用户已运行 `bash init.sh <path-to-skills-repo>` 完成初始化。确认：
- `input/` — 内核文件（必须）
- `context/` — 参考资料（可选）
- `.claude/skills/` — NPU 开发 skills 已链接

然后：
1. **分析输入**：读取 input/、context/、HINTS.md。确认内核类型、输入 shapes、算子语义。如果无法确定 shapes，**停下来问用户**。
2. **检查环境**：使用 `ascendc-env-check` skill 确认 NPU 环境可用、Ascend C 编译器就绪。
3. **创建分支**：创建并切换到新分支（如 `opt/<kernel-name>`）。
4. **初始化 solution/**：创建 `solution/` 和 `scripts/` 目录。将 input/ 中的内核文件复制到 solution/。
5. **构建评测方法**：参考 `ascendc-direct-invoke-template` skill 了解编译运行流程，参考 `ops-profiling` skill 了解性能测评方法，为该内核建立编译、正确性验证和计时方案。将完整的 bench 命令（需要包含 `2>&1 | tee _bench_output.txt`）填入 bench-wrapper.sh 替换 `{{BENCH_COMMAND}}`，生成 `scripts/bench.sh`。
6. **验证 baseline**：运行 `bash scripts/bench.sh`。确认编译通过、结果正确。然后 `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`。

## Optimization

- 使用 `bash scripts/bench.sh` 测量性能。
- 使用 `ops-profiling` skill 分析瓶颈——不要盲目优化。
- 使用 `ascendc-tiling-design` skill 获取 Tiling 优化方向。
- 使用 `ascendc-api-best-practices` skill 选择最优 API。
- 使用 `ascendc-npu-arch` skill 了解硬件限制和理论峰值。
- 充分利用所有可用信息：`context/`、`HINTS.md`、之前的迭代结果。
- 遵循 HINTS.md 中的停滞规则。

### Iteration Protocol

每次修改 solution/ 代码 + 运行 `bash scripts/bench.sh` 为一次迭代——无论结果是改善、退步还是失败。迭代顺序编号（1, 2, 3, …）。

**完成以下所有步骤后才能开始下一次迭代：**

1. **运行 benchmark** — `bash scripts/bench.sh iter-N`（label 必须，格式为 `iter-N`）。
2. **更新 `ITERATIONS.md`**
3. **Git commit** — `[iter N] 简短描述优化方向`
