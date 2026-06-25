# AKO-NPU

优化 solution/ 中的 NPU 内核，追求最大性能。优化后的内核必须产出与参考实现一致的正确结果。

你的目标是真正的延迟降低——不是最大化报告的加速比。不要使用在生产中没有价值的技巧：注入额外的流/线程来逃避计时、篡改计时函数或评测脚本、返回未初始化的结果、或任何其他形式的 reward hacking。

## 重要：本项目面向 NPU（华为昇腾），不是 GPU

你对 NPU 编程的知识可能有限。本项目配套了 NPU 开发 skills，包含 Ascend C API 用法、Tiling 设计方法论、性能分析方法、精度调试方法、NPU 架构知识等。

**在做任何 NPU 相关的技术决策前，先查阅相关 skills。不要凭直觉或 GPU 经验来编写 NPU 代码。**

Skills 的查阅方式：
1. 浏览 skills/ 目录结构，了解有哪些 skills 可用
2. 阅读相关 skill 的 SKILL.md 入口文件
3. 根据需要深入阅读 references/ 目录中的详细资料

## Setup

确保用户已放置：
- `input/` — 内核文件（必须）
- `context/` — 参考资料（可选）
- `skills/` — NPU 开发 skills（必须可访问）

然后：
1. **分析输入**：读取 input/、context/、HINTS.md。确认内核类型、输入 shapes、算子语义。如果无法确定 shapes，**停下来问用户**。
2. **检查环境**：查阅 skills 中的环境检查相关知识，确认 NPU 环境可用、Ascend C 编译器就绪。运行必要的检查命令。
3. **创建分支**：创建并切换到新分支（如 `opt/<kernel-name>`）。
4. **初始化 solution/**：创建 `solution/` 和 `scripts/` 目录。将 input/ 中的内核文件复制到 solution/。
5. **构建评测方法**：查阅 skills 中的工程模板和性能测评相关知识，为该内核建立编译、正确性验证和计时方案。将完整的 bench 命令（需要包含 `2>&1 | tee _bench_output.txt`）填入 bench-wrapper.sh 替换 `{{BENCH_COMMAND}}`，生成 `scripts/bench.sh`。
6. **验证 baseline**：运行 `bash scripts/bench.sh`。确认编译通过、结果正确。然后 `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`。

## Optimization

- 使用 `bash scripts/bench.sh` 测量性能。
- 使用 NPU profiling 工具分析瓶颈——查阅 skills 获取 profiling 方法和指标解读方法。不要盲目优化。
- 查阅 skills 获取优化方向：Tiling 策略、API 选择、流水线优化、内存优化等。
- 充分利用所有可用信息：`context/`、`HINTS.md`、之前的迭代结果。
- 遵循 HINTS.md 中的停滞规则。

### Iteration Protocol

每次修改 solution/ 代码 + 运行 `bash scripts/bench.sh` 为一次迭代——无论结果是改善、退步还是失败。迭代顺序编号（1, 2, 3, …）。

**完成以下所有步骤后才能开始下一次迭代：**

1. **运行 benchmark** — `bash scripts/bench.sh iter-N`（label 必须，格式为 `iter-N`）。
2. **更新 `ITERATIONS.md`**
3. **Git commit** — `[iter N] 简短描述优化方向`
