<h1 align="center">AKO-NPU</h1>
<p align="center"><b>Agentic Kernel Optimization for NPU</b></p>

华为昇腾 NPU 上的自动化 Ascend C 算子迭代优化框架。提供 PyTorch 算子或已有的 Ascend C 代码，agent 会自动完成开发（如需要）并迭代优化至最大性能。

> **与 AKO4ALL 的关系：** AKO-NPU 将 [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) 的迭代优化方法论从 GPU 迁移到 NPU。AKO4ALL 依赖模型内化的 GPU 知识（CUDA/Triton），AKO-NPU 则利用 [CANN Skills](https://gitcode.com/cann/skills) 作为外部 NPU 知识源。

## 快速开始

### 1. 克隆本项目

```bash
git clone <AKO-NPU-repo>
cd AKO-NPU
```

### 2. 克隆 CANN Skills 仓库

```bash
git clone https://gitcode.com/cann/skills.git
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
claude
```

然后输入：`Follow the instructions in CLAUDE.md`

Agent 会自动完成环境初始化（安装 skills/agents）、路由判断、开发（如需要）和迭代优化。

## 工作流程

```
input/ 包含 .asc 文件？
    ├── 是 → Step 3：迭代优化（AKO 循环）
    └── 否 → Step 2：开发基础版（CANNBot 工作流）→ Step 3
```

**Step 0：** Agent 自动检测并安装 skills/agents 到 `.claude/` 下。

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

## 致谢

- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — 迭代优化方法论
- [CANN Skills](https://gitcode.com/cann/skills) — NPU 开发知识（skills、agents、workflows）
