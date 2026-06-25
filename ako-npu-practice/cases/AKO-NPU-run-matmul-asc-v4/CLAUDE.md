# AKO-NPU

NPU 内核的自动化迭代优化框架。支持从 PyTorch 描述或已有 Ascend C 代码出发，最终产出高性能 NPU 内核。

你的目标是真正的延迟降低——不是最大化报告的加速比。不要使用在生产中没有价值的技巧：注入额外的流/线程来逃避计时、篡改计时函数或评测脚本、返回未初始化的结果、或任何其他形式的 reward hacking。

## 重要：本项目面向 NPU（华为昇腾），不是 GPU

你对 NPU 编程的知识可能有限。**在做任何 NPU 相关的技术决策前，先查阅相关 skills。不要凭直觉或 GPU 经验来编写 NPU 代码。**

## 配置

以下变量定义了外部依赖路径，如有变化只需修改此处：

- **SKILLS_REPO**: `skills/` — CANN Skills 仓库在本项目下的路径
- **SKILLS_REPO_URL**: `https://gitcode.com/cann/skills.git` — 仓库 clone 地址
- **DEV_TEAM**: Skills 仓库中用于算子开发的 team 目录。在 `$SKILLS_REPO/teams/` 下查找包含 `init.sh` 和 `AGENTS.md` 的子目录，以该目录为准。

## Step 0：环境准备

### 检查 Skills 可用性

检查 `.claude/skills/` 是否存在且包含 NPU skills（环境检查、性能采集等）。

如果不存在，检查项目下是否有 `$SKILLS_REPO` 子目录：
- 如果有但 `.claude/skills/` 为空——用户 clone 了仓库但没装。帮用户安装：
  1. 找到 `$DEV_TEAM` 目录，在其中运行 `bash init.sh project claude`
  2. 把安装好的 skills 和 agents 提升到本项目的 `.claude/` 下：
     ```bash
     mkdir -p .claude/skills .claude/agents
     DEV_TEAM_CLAUDE="$DEV_TEAM/.claude"
     for s in "$DEV_TEAM_CLAUDE"/skills/*/; do
         ln -sfn "$(realpath "$s")" .claude/skills/$(basename "$s")
     done
     for a in "$DEV_TEAM_CLAUDE"/agents/*/; do
         ln -sfn "$(realpath "$a")" .claude/agents/$(basename "$a")
     done
     ```
- 如果 `$SKILLS_REPO` 也不存在——执行 `git clone $SKILLS_REPO_URL`，然后重复上述步骤。

### 检查 NPU 环境

查阅环境检查相关的 skill，确认 NPU 环境可用、CANN 编译器就绪。

## Step 1：分析输入 → 路由

检查 `input/` 目录的内容，决定走哪条路径：

```
input/ 包含 .asc 文件 + CMakeLists.txt ?
    │
    ├── 是 → 已有 Ascend C 算子 → 跳到 Step 3（直接优化）
    │
    └── 否 → 需要先开发基础版
            │
            ├── input/ 包含 .py 文件 → PyTorch 算子描述
            ├── input/ 包含文本描述 → 算子需求
            └── → 进入 Step 2（开发基础版）
```

## Step 2：开发基础版

**前提**：input/ 中没有可直接编译运行的 Ascend C 代码。

在 `$DEV_TEAM` 目录下，读取其 `AGENTS.md`（主工作流定义）和 `workflows/`（详细流程），按该工作流完成算子开发。

**AKO-NPU 不定义开发流程的具体步骤——开发流程由 `$DEV_TEAM` 的工作流定义决定。** 如果那边的流程更新了，本框架自动受益。

开发完成后，将 `$DEV_TEAM` 下的产出文件（可编译运行的 Ascend C 算子）复制到 `input/`。

然后进入 Step 3。

## Step 3：迭代优化（AKO 循环）

**前提**：input/ 中有可编译运行的 Ascend C 算子。

详细规则见 @OPTIMIZE.md，行为控制和停滞策略见 @HINTS.md。
