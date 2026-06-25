# 迭代优化规则

优化 solution/ 中的 NPU 内核，追求最大性能。优化后的内核必须产出与参考实现一致的正确结果。

**在做任何 NPU 相关的技术决策前，先查阅相关 skills。不要凭直觉或 GPU 经验来编写 NPU 代码。**

## Setup

1. **分析输入**：读取 input/、context/、HINTS.md。确认内核类型、输入 shapes、算子语义。如果无法确定 shapes，**停下来问用户**。确定后的 shapes 在整个优化过程中**不得更改**——所有迭代必须在相同的输入规格下对比性能，否则 speedup 数字没有意义。
2. **检查环境**：查阅环境检查相关的 skill，确认 NPU 环境可用、Ascend C 编译器就绪。
3. **创建分支**：创建并切换到新分支（如 `opt/<kernel-name>`）。
4. **初始化 solution/**：创建 `solution/` 和 `scripts/` 目录。将 input/ 中的内核文件复制到 solution/。
5. **构建评测方法**：
   - 检查 `bench/` 目录。如果用户提供了自定义评测脚本，使用用户的方案。
   - 如果 `bench/` 为空或不存在，查阅工程模板类 skill 了解编译运行流程，查阅性能采集类 skill 了解 NPU profiling 方法，为该内核建立编译、正确性验证和计时方案。
   - 将完整的 bench 命令（需要包含 `2>&1 | tee _bench_output.txt`）填入 bench-wrapper.sh 替换 `{{BENCH_COMMAND}}`，生成 `scripts/bench.sh`。
6. **验证 baseline**：运行 `bash scripts/bench.sh`。确认编译通过、结果正确。然后 `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`。

## Optimization

- 使用 `bash scripts/bench.sh` 测量性能。
- 使用性能采集类 skill 分析瓶颈——不要盲目优化。
- 使用 Tiling 设计类 skill 获取切分和 Buffer 规划优化方向。
- 使用 API 最佳实践类 skill 选择最优 API。
- 使用 NPU 架构类 skill 了解硬件限制和理论峰值。
- 充分利用所有可用信息：`context/`、HINTS.md、之前的迭代结果。
- 遵循 HINTS.md 中的停滞规则。

## Iteration Protocol

每次修改 solution/ 代码 + 运行 `bash scripts/bench.sh` 为一次迭代——无论结果是改善、退步还是失败。迭代顺序编号（1, 2, 3, …）。

**完成以下所有步骤后才能开始下一次迭代：**

1. **运行 benchmark** — `bash scripts/bench.sh iter-N`（label 必须，格式为 `iter-N`）。
2. **更新 `ITERATIONS.md`**
3. **Git commit** — `[iter N] 简短描述优化方向`
