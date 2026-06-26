<h1 align="center">AKO-NPU 工作区</h1>
<p align="center"><b>Agentic Kernel Optimization for NPU — 华为昇腾 NPU 算子自动迭代优化</b></p>

本仓库是 AKO-NPU 的完整工作区，包含优化框架本体、实践案例、CANN Skills 知识源三部分。AKO-NPU 把 [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) 的迭代优化方法论从 GPU 迁移到昇腾 NPU，以 [CANN Skills](https://gitcode.com/cann/skills) 作为外部 NPU 开发知识源，由 agent 在 Claude Code 中自动完成「开发 → profiling → 改代码 → benchmark → 提交」的迭代循环。

## 目录结构

```
AKO-NPU/
├── README.md                # 本文件：工作区总览
├── CHANGES.md               # 环境适配修改记录（路径差异、init.sh 适配等）
├── AKO-NPU-Overview.html    # 项目设计总览（可浏览器打开）
│
├── ako-npu/                 # ★ 优化框架本体（纯规则文件，agent 入口）
│   ├── CLAUDE.md            #   入口：路由 + 环境准备（Step 0~3）
│   ├── OPTIMIZE.md          #   迭代优化规则（AKO 循环）
│   ├── HINTS.md             #   停滞策略 + skills 使用指引
│   ├── ITERATIONS.md        #   迭代日志模板
│   ├── SETUP.md             #   ★ 环境搭建 + 跑通实验记录（见下）
│   ├── input/               #   用户放置算子代码（当前含 dsa_indexer 基线）
│   ├── bench/ context/      #   可选评测脚本 / 参考资料
│   └── .claude/             #   skills/agents 软链接（init.sh 安装，git-ignored）
│
├── ako-npu-practice/        # 实践案例库：多个算子的优化轨迹与报告
│   ├── cases/               #   每个算子一个目录（input/solution/trajectory/...）
│   ├── data/ plots/         #   加速比数据与图表
│   └── REPORT.md PLOTS.md    #   汇总报告
│
└── skills/                  # CANN Skills 仓库（git clone 进来，git-ignored）
    ├── ops/                 #   算子开发 skills（env-check、profiling、tiling…）
    ├── infra/               #   gitcode 工具链 skills
    └── plugins-official/ops-direct-invoke/   # DEV_TEAM：含 init.sh、agents、workflows
```

> `skills/`、`ako-npu/.claude/skills|agents`、`input/scripts/`、`build/`、`trajectory/` 等均被 `.gitignore` 有意忽略——它们是可复现的环境产物或用户自带文件，不进版本库。

## 快速开始

### 1. 准备环境（一次性）

```bash
# 初始化 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 安装 CANN Skills/agents 到 ako-npu/.claude/
cd skills/plugins-official/ops-direct-invoke
bash init.sh project claude /workdir/projects/AKO-NPU/ako-npu
cd -

# 注意：init.sh 会用 CANNBot 的 CLAUDE.md 覆盖 ako-npu/CLAUDE.md，
# 需恢复 AKO 自身的入口文件（init.sh 已自动备份为 CLAUDE.md.bak.*）
```

详细的环境搭建过程、踩坑与实验验证见 **[ako-npu/SETUP.md](ako-npu/SETUP.md)**。

### 2. 放置算子输入

在 `ako-npu/input/` 下放（三选一）：

- `kernel.asc` + `CMakeLists.txt` → 已有 Ascend C 算子，直接进入优化
- `kernel.py` → PyTorch 算子，先开发再优化
- `description.md` → 文字需求，先开发再优化

### 3. 启动迭代优化

```bash
cd ako-npu
ASCEND_RT_VISIBLE_DEVICES=<空闲卡号> claude --dangerously-skip-permissions
# 然后输入：Follow the instructions in CLAUDE.md. Optimize for at least N iterations.
```

## 硬件环境

- NPU：16 × Ascend910B2C，每卡 HBM ~61.0G
- 查看占用：`python3 npu_info.py`（本容器**无法用 `npu-smi`**，用此脚本）
- 编译器：CANN toolkit（`bisheng`），目标架构 `dav-2201`（Ascend910B / A3）

## 致谢

- [AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL) — 迭代优化方法论
- [CANN Skills](https://gitcode.com/cann/skills) — NPU 开发知识（skills / agents / workflows）
