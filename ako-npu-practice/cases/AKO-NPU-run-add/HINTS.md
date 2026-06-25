# Hints

## NPU Skills 使用指引

本项目的 skills/ 目录包含 NPU 开发的专业知识。这些 skills 覆盖了环境配置、API 用法、Tiling 设计、性能分析、精度调试、NPU 架构等方面。

**使用原则**：
- 在做技术决策前，先浏览 skills/ 目录，找到相关的 skill 并查阅
- 不要假设你知道 NPU 的 API 用法或限制——查阅 skills 确认
- Skills 中的 SKILL.md 是入口，references/ 目录有详细资料
- 如果 skills 中没有覆盖的知识，标注出来并基于已有信息做最佳判断

## 停滞策略

- Iter 1 前，使用 NPU profiling 工具采集 baseline 性能数据（查阅 skills 中的 profiling 方法）。
- 连续 3 次迭代无改善：重新 profile，查阅 skills 中的瓶颈优化建议，review ITERATIONS.md 寻找规律。制定计划后再继续。
- 连续 5 次迭代无改善：重新评估整体策略——查阅 skills 中的 Tiling 设计方法论，考虑是否需要换架构方向。

<!-- 用户可在下方添加额外指令，例如：
- 优化约束或重点方向
- 要尝试或要避免的策略
- Agent 行为控制
- 依赖策略（如"不要安装任何包"） -->
