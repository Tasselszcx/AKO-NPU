<h1 align="center">AKO-NPU</h1>
<p align="center"><b>Agentic Kernel Optimization for NPU</b></p>

华为昇腾 NPU 上的自动化 Ascend C 算子迭代优化框架。本项目是一套**纯规则文件**——不包含实际的算子代码或脚本，只定义了 agent 的工作流程、迭代规范、精度约束和停滞策略。实际的 NPU 开发知识由外部的 [CANN Skills](https://gitcode.com/cann/skills) 提供，agent 在运行时按需查阅。

提供 PyTorch 算子或已有的 Ascend C 代码，agent 会按照本项目定义的规则自动完成开发（如需要）并迭代优化至最大性能。

> **与 AKO4ALL 的关系：** AKO-NPU 将 [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) 的迭代优化方法论从 GPU 迁移到 NPU。AKO4ALL 依赖模型内化的 GPU 知识（CUDA/Triton），AKO-NPU 则利用 [CANN Skills](https://gitcode.com/cann/skills) 作为外部 NPU 知识源。

> **实践案例：** 多个算子的优化轨迹、验证结果和问题分析见 [AKO-NPU Practice](https://dev.sankuai.com/code/repo-detail/~qinjiayu02/ako-npu-practice)。

## 快速开始

### 1. 克隆本项目

```bash
git clone <AKO-NPU-repo>
cd AKO-NPU
```

### 2. 克隆 CANN Skills 仓库并安装

```bash
git clone https://gitcode.com/cann/skills.git
cd skills/teams/ops-direct-invoke
bash init.sh project claude
cd ../../..
```

### 3. 放置输入

```
input/
├── kernel.asc + CMakeLists.txt    # 已有 Ascend C 算子 → 直接优化
│   或
├── kernel.py                       # PyTorch 算子 → 先开发再优化
│   或
├── description.md                  # 文字描述 → 先开发再优化
```

### 4. 启动

```bash
claude --dangerously-skip-permissions
```

然后输入：`Follow the instructions in CLAUDE.md. Optimize for at least N iterations.`（N 替换为期望的迭代轮次，如 20、50、100）

> **多卡隔离：** 如需在指定 NPU 卡上运行（避免多任务干扰），设置 `ASCEND_RT_VISIBLE_DEVICES=<卡号> claude --dangerously-skip-permissions`

## 工作流程

```
input/ 包含 .asc 文件？
    ├── 是 → Step 3：迭代优化（AKO 循环）
    └── 否 → Step 2：开发基础版（CANNBot 工作流）→ Step 3
```

**Step 0：** Agent 检测 skills/agents 是否安装到 `.claude/` 下，未安装则自动补装。

**Step 2（如需要）：** 使用 CANNBot 的 architect → developer → reviewer 工作流，从 PyTorch 代码或文字描述生成可编译运行的 Ascend C 算子。

**Step 3：** 迭代优化算子：profiling → 修改代码 → benchmark → 记录日志 → git commit，由 NPU skills 指导优化方向。

## 项目结构

```
AKO-NPU/
├── CLAUDE.md            # 入口：路由 + 环境准备（Claude Code 启动时自动读取）
├── OPTIMIZE.md          # 迭代优化规则（AKO 循环）
├── HINTS.md             # 停滞策略 + skills 使用指引
├── ITERATIONS.md        # 迭代日志模板
├── bench-wrapper.sh     # trajectory 追踪壳模板
├── bench/               # 可选：用户自定义评测脚本
├── input/               # 用户放置算子代码
├── context/             # 可选：参考资料
├── README.md
├── .gitignore
├── .claude/
│   └── settings.local.json   # 权限配置（项目自带）
└── skills/              # 用户克隆的 CANN Skills 仓库（git-ignored）
```

## 相对于 AKO4ALL 的改动

**能力来源**：AKO4ALL 依赖模型内化的 GPU 知识直接生成优化代码。AKO-NPU 利用外部 [CANN Skills](https://gitcode.com/cann/skills) 提供的 NPU 开发能力（API 用法、Tiling 设计、profiling 方法、NPU 架构等），通过 Claude Code 的 skill 自动发现机制实时获取，规则中不硬编码具体 skill 名称。评测方法、编译流程、profiling 工具均由 agent 查阅 skills 后自行构建，而非硬编码在项目中。

**开发能力**：AKO4ALL 假设用户已有可运行的 kernel。AKO-NPU 支持两种输入：如果提供的是非 Ascend C 的代码（PyTorch 或文字描述），先利用 CANN Skills 提供的算子开发能力（architect → developer → reviewer 工作流）生成基础版 Ascend C 算子，再进入迭代优化；如果直接提供 Ascend C 代码，则跳过开发阶段直接优化。

**规则持久化**：入口从 TASK.md 改为 CLAUDE.md，Claude Code 启动时自动读取且上下文压缩后重新注入，通过 `@import` 引用 OPTIMIZE.md 和 HINTS.md，确保迭代规范在长时间运行中不丢失。

**迭代约束**：在多轮实验中人工观察到 agent 的若干不良行为，针对性地增加了约束规则：精度失败必须 revert（防止 agent 带着错误结果继续优化）、输入 shapes 不可变（防止通过改尺寸虚增加速比）、参数搜索整体算一轮迭代（防止 sweep 耗尽迭代预算）、baseline 基于纯 kernel 时间（防止计时标准不一致）、每轮必须重读 skill（防止上下文压缩后 NPU 知识丢失）。

## 致谢

- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — 迭代优化方法论
- [CANN Skills](https://gitcode.com/cann/skills) — NPU 开发知识（skills、agents、workflows）
