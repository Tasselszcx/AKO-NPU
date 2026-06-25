<h1 align="center">AKO-NPU</h1>
<p align="center"><b>Agentic Kernel Optimization for NPU</b></p>

Automated Ascend C kernel optimization on Huawei NPU. Provide a PyTorch operator or an existing Ascend C kernel — the agent develops (if needed) and iteratively optimizes it for maximum performance.

> **Relationship to AKO4ALL:** AKO-NPU adapts [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL)'s iteration methodology from GPU to NPU. AKO4ALL relies on model-internalized GPU knowledge; AKO-NPU externalizes NPU knowledge into [CANN Skills](https://gitcode.com/cann/skills).

## Quick Start

### 1. Clone this project

```bash
git clone <AKO-NPU-repo>
cd AKO-NPU
```

### 2. Clone CANN Skills 仓库

```bash
git clone https://gitcode.com/cann/skills.git
```

### 3. Place your input

```
input/
├── kernel.asc + CMakeLists.txt    # 已有 Ascend C 算子 → 直接优化
│   OR
├── kernel.py                       # PyTorch 算子 → 先开发再优化
│   OR
├── description.md                  # 文字描述 → 先开发再优化
```

### 4. Run

```bash
claude
```

Then say: `Follow the instructions in TASK.md`

Agent 会自动完成环境初始化（安装 skills/agents）、路由判断、开发（如需要）和迭代优化。

## What Happens

```
input/ has .asc files?
    ├── Yes → Step 3: Iterative optimization (AKO loop)
    └── No  → Step 2: Develop baseline (CANNBot workflow) → Step 3
```

**Step 0:** Agent 自动检测并安装 skills/agents 到 `.claude/` 下。

**Step 2 (if needed):** Uses CANNBot's architect → developer → reviewer workflow to create a working Ascend C kernel from PyTorch code or text description.

**Step 3:** Iteratively optimizes the kernel: profile → modify → benchmark → log → commit, guided by NPU skills.

## Project Structure

```
AKO-NPU/
├── TASK.md              # Entry point: routing + environment setup
├── OPTIMIZE.md          # Iterative optimization rules (AKO loop)
├── HINTS.md             # Stall strategies + skills usage guidance
├── ITERATIONS.md        # Iteration log template
├── bench-wrapper.sh     # Trajectory tracking shell template
├── bench/               # Optional user-provided benchmark
├── input/               # User places kernel or PyTorch code here
├── context/             # Optional reference materials
├── README.md
├── .gitignore
├── .claude/
│   └── settings.local.json   # Permissions (project-provided)
└── skills/              # User clones CANN Skills repo here (git-ignored)
```

## Acknowledgments

- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — Iteration methodology
- [CANN Skills](https://gitcode.com/cann/skills) — NPU development knowledge (skills, agents, workflows)
