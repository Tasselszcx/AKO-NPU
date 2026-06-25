# AKO-NPU 环境适配修改记录

## 变更时间：2026-06-24

### 背景

文档（README）里的路径和实际 skills 仓库结构已不一致，导致按原步骤无法启动。

---

## 差异说明：文档 vs 实际

| 项目 | README / CLAUDE.md 描述 | 实际仓库结构 |
|------|------------------------|-------------|
| ops-direct-invoke 路径 | `skills/teams/ops-direct-invoke` | `skills/plugins-official/ops-direct-invoke` |
| skills 根目录 | 扁平结构 | 分层：`ops/`, `infra/`, `plugins-official/`, `plugins-community/`, `ops-lab/` 等 |

---

## 执行的操作

### 1. 运行 init.sh（已适配实际路径）

原文档步骤：
```bash
cd skills/teams/ops-direct-invoke   # ← 路径已失效
bash init.sh project claude
```

实际执行：
```bash
cd /home/hadoop-aipnlp/projects/AKO-NPU/skills/plugins-official/ops-direct-invoke
bash init.sh project claude /home/hadoop-aipnlp/projects/AKO-NPU/ako-npu
```

安装结果（全部成功）：
- Skills (15项)：ascendc-api-best-practices, ascendc-blaze-best-practice, ascendc-code-review, ascendc-crash-debug, ascendc-direct-invoke-template, ascendc-docs-search, ascendc-env-check, ascendc-precision-debug, ascendc-regbase-best-practice, ascendc-runtime-debug, ascendc-tiling-design, npu-arch, ops-precision-standard, ops-profiling, torch-ascendc-op-extension
- Infra Skills (4项)：gitcode-issue-gen, gitcode-issue-handler, gitcode-pr-handler, gitcode-toolkit
- Agents (4项)：ascendc-kernel-architect, ascendc-kernel-design-reviewer, ascendc-kernel-developer, ascendc-kernel-reviewer
- Workflows 软链接：`ako-npu/.claude/workflows` → `skills/plugins-official/ops-direct-invoke/workflows`

### 2. 恢复 ako-npu/CLAUDE.md

init.sh 会将 ops-direct-invoke 的 CLAUDE.md 写到目标项目目录，覆盖了 AKO-NPU 自身的 CLAUDE.md。
init.sh 已自动备份原文件为 `CLAUDE.md.bak.20260624_124102`，执行恢复：

```bash
cp ako-npu/CLAUDE.md.bak.20260624_124102 ako-npu/CLAUDE.md
```

**原因**：`ako-npu/CLAUDE.md` 是 AKO 优化框架的主工作流，不能被 ops-direct-invoke 的 CANNBot CLAUDE.md 替换。两者分工不同：
- `ako-npu/CLAUDE.md`：AKO 外层优化循环框架（Step 0~3）
- `skills/plugins-official/ops-direct-invoke/AGENTS.md`：CANNBot 算子开发内层流程（Step 1~7），由 AKO 的 Step 2 调用

---

## 已知限制

### asc-devkit 无法克隆

init.sh 的 Step 4 尝试从 `https://gitcode.com/cann/asc-devkit.git` 克隆 asc-devkit，因网络不通跳过。

**影响**：`ascendc-docs-search` skill 的 API 文档本地检索功能不可用。其他功能（环境检查、编译、测试、优化）不受影响。

**解决方法（如需）**：在有网络的环境下手动克隆，再软链到项目目录：
```bash
git clone https://gitcode.com/cann/asc-devkit.git \
    skills/plugins-official/ops-direct-invoke/asc-devkit
ln -sfn $(realpath skills/plugins-official/ops-direct-invoke/asc-devkit) \
    ako-npu/asc-devkit
```

---

## 当前状态

```
ako-npu/
├── CLAUDE.md                     ✅ AKO 主工作流（已恢复）
├── CLAUDE.md.bak.20260624_124102 （备份，可删除）
├── input/                        （空，需放入算子输入文件）
└── .claude/
    ├── settings.local.json       ✅ 原有配置
    ├── skills/ (19项)            ✅ 已安装软链接
    ├── agents/ (4项)             ✅ 已安装软链接
    └── workflows -> ...          ✅ 软链接到 ops-direct-invoke/workflows
```

---

## 正确的启动步骤

```bash
cd /home/hadoop-aipnlp/projects/AKO-NPU/ako-npu

# 在 input/ 下放置算子输入（选其一）：
# - kernel.asc + CMakeLists.txt   → 直接优化
# - kernel.py                     → 先开发再优化
# - description.md                → 先开发再优化

# 启动
claude --dangerously-skip-permissions
# 然后输入：Follow the instructions in CLAUDE.md. Optimize for at least N iterations.
```

如需指定 NPU 卡：
```bash
ASCEND_RT_VISIBLE_DEVICES=<卡号> claude --dangerously-skip-permissions
```
