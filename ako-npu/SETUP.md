# AKO-NPU 环境搭建与基线跑通记录

> 记录时间：2026-06-26 ｜ 主机：`set-hlsc-mlp-codelab-pc181` ｜ 用户：`hadoop-aipnlp`
> 目标：在已具备驱动和卡的环境上，把 AKO-NPU 框架搭起来，并跑通 `input/` 中 `dsa_indexer` 算子的 **编译 → 运行 → 精度校验** 全流程。

## 1. 结论速览

| 项目 | 结果 |
|------|------|
| 环境搭建（Step 0） | ✅ 19 skills + 4 agents + workflows 安装到 `.claude/` |
| 编译（bisheng / dav-2201） | ✅ 成功，产物 `build/dsa_indexer`（842 KB） |
| 运行（卡 15，单 AI Core） | ✅ 正常产出 topk_indices / index_score |
| 精度校验 | ✅ PASSED（index_score MERE/MARE = 0.0；topk 2048/2048 = 100%） |

## 2. 起始环境核查

| 组件 | 状态 |
|------|------|
| NPU | 16 × Ascend910B2C，每卡 HBM ~61.0G，空闲（~0.6% 占用） |
| CANN toolkit | ✅ `/usr/local/Ascend/ascend-toolkit/set_env.sh` 存在 |
| cmake / make | ✅ `/usr/bin/cmake`、`/usr/bin/make` |
| Python | `/usr/bin/python3` 3.9.9，`acl` 可用，**缺 numpy** |

> 查看 NPU 占用统一用 `python3 /workdir/projects/npu_info.py`（本容器无法用 `npu-smi`）。

## 3. 执行的操作

### 3.1 安装 CANN Skills / agents（Step 0）

```bash
# 1) 备份 AKO 自身入口（init.sh 会覆盖它）
cp ako-npu/CLAUDE.md /tmp/AKO_CLAUDE.md.safe

# 2) 在 DEV_TEAM 目录运行 init.sh
cd skills/plugins-official/ops-direct-invoke
bash init.sh project claude /workdir/projects/AKO-NPU/ako-npu

# 3) 恢复 AKO 主工作流 CLAUDE.md（被 CANNBot 的 CLAUDE.md 覆盖了）
cp /tmp/AKO_CLAUDE.md.safe ako-npu/CLAUDE.md
```

安装产物（均为软链接，git-ignored）：

- `.claude/skills/`（19 项）：ascendc-api-best-practices、ascendc-blaze-best-practice、ascendc-code-review、ascendc-crash-debug、ascendc-direct-invoke-template、ascendc-docs-search、ascendc-env-check、ascendc-precision-debug、ascendc-regbase-best-practice、ascendc-runtime-debug、ascendc-tiling-design、npu-arch、ops-precision-standard、ops-profiling、torch-ascendc-op-extension、gitcode-issue-gen、gitcode-issue-handler、gitcode-pr-handler、gitcode-toolkit
- `.claude/agents/`（4 项）：ascendc-kernel-architect、ascendc-kernel-design-reviewer、ascendc-kernel-developer、ascendc-kernel-reviewer
- `.claude/workflows` → `skills/plugins-official/ops-direct-invoke/workflows`

> **已知限制**：`asc-devkit` 因网络不通 clone 失败，仅影响 `ascendc-docs-search` 的本地 API 文档检索，编译/运行/优化不受影响。

### 3.2 补齐缺失的测试脚本

`input/run.sh` 依赖 `scripts/gen_data.py`、`scripts/verify_result.py`，但 `input/` 下无 `scripts/`（该目录被 `.gitignore` 忽略，不随仓库分发）。
当前 `input/dsa_indexer.asc` 与实践案例 `AKO-NPU-run-dsa-v5/input/dsa_indexer.asc` **完全一致（diff 无差异）**，故从该案例复制：

```bash
cp ako-npu-practice/cases/AKO-NPU-run-dsa-v5/input/scripts/{gen_data.py,verify_result.py} \
   ako-npu/input/scripts/
```

### 3.3 安装 Python 依赖

```bash
pip3 install numpy   # 脚本仅依赖 numpy，无需 torch
```

## 4. 跑通命令

```bash
cd ako-npu/input
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=15        # 隔离到空闲卡 15
bash run.sh                                 # 编译 + 生成数据 + 运行 + 校验
# 复用已编译二进制：bash run.sh --skip-build
# 自定义 shape：    bash run.sh B S_q S_kv
```

## 5. 被优化的算子：DSA Indexer

DeepSeek Sparse Attention 的 Indexer 算子。给定 query/key/weights，计算每个 query 对 key 的 index_score，并选出 top-k 个 key 下标用于稀疏注意力。

- 默认 shape（decode 场景）：`B=1, S_q=1, S_kv=4096`
- 固定参数：`N_HEADS=64, HEAD_DIM=128, INDEX_TOPK=2048`
- 数据类型：输入 bf16（uint16 存储），topk 下标 int64

## 6. 算力 / 资源用量

本次跑通是 **基线版本的一次功能验证**（非 profiling 性能采集），资源占用如下：

| 维度 | 用量 |
|------|------|
| NPU 卡 | **1 / 16** 张（卡 15，Ascend910B2C） |
| AI Core | `coreNum=1`（单核） |
| HBM workspace | `wsSysSize=16 MB` + 其它 = `wsTotalSize≈17 MB`（17,842,176 B） |
| Kernel tiling | `topkK=2048, topkInner=4096, topkTmpSize=81920` |
| 输入数据 | input_q 16 KB / input_k 1 MB / input_weights 128 B |
| 输出数据 | topk_indices 16 KB / index_score 8 KB |
| 端到端 wall-clock | ~2.7 s（3 次：2.69 / 3.70 / 2.70 s，**含进程启动 + ACL init + H2D/D2H**，非纯 kernel 时延） |
| Host 峰值 RSS | ~990 MB |
| 编译产物 | `build/` 2.5 MB，二进制 842 KB |

> **纯 kernel 时延**未在本次采集（`run.sh` 不做计时）。它属于 AKO 迭代优化阶段的工作，需借助 `ops-profiling` skill 采集，作为加速比 baseline。

## 7. 精度校验结果

```
index_score:  Max abs diff 0.000000 ｜ MERE 0.000000 (阈值 0.007812) ｜ MARE 0.000000 (阈值 0.078125)  → PASSED
topk_indices: Match rate 2048/2048 (100.00%)                                                          → PASSED
=== Verification PASSED ===
```

## 8. 文件位置索引

| 文件 / 目录 | 说明 |
|-------------|------|
| `ako-npu/input/dsa_indexer.asc` | 被优化的 Ascend C 算子（基线，30 KB） |
| `ako-npu/input/dsa_indexer.py` | 算子的 PyTorch 参考实现 |
| `ako-npu/input/CMakeLists.txt` | 编译配置（`--npu-arch=dav-2201`） |
| `ako-npu/input/data_utils.h` | 数据读写辅助头文件 |
| `ako-npu/input/run.sh` | 一键编译+运行+校验脚本 |
| `ako-npu/input/scripts/gen_data.py` | 生成测试数据 + golden（**git-ignored，从案例复制**） |
| `ako-npu/input/scripts/verify_result.py` | 精度校验（**git-ignored**） |
| `ako-npu/input/build/dsa_indexer` | 编译产物二进制（**git-ignored**） |
| `ako-npu/.claude/skills/` `agents/` | skills/agents 软链接（**git-ignored**，init.sh 安装） |
| `skills/` | CANN Skills 仓库本体（**git-ignored**，git clone 进来） |
| `skills/plugins-official/ops-direct-invoke/` | DEV_TEAM：init.sh + agents + workflows |

## 9. 复现清单（TL;DR）

```bash
cd /workdir/projects/AKO-NPU
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 装 skills（如未装）
( cd skills/plugins-official/ops-direct-invoke && \
  cp ../../../../ako-npu/CLAUDE.md /tmp/AKO_CLAUDE.md.safe && \
  bash init.sh project claude /workdir/projects/AKO-NPU/ako-npu && \
  cp /tmp/AKO_CLAUDE.md.safe ../../../../ako-npu/CLAUDE.md )
# 补脚本 + 依赖
cp ako-npu-practice/cases/AKO-NPU-run-dsa-v5/input/scripts/{gen_data.py,verify_result.py} ako-npu/input/scripts/
pip3 install numpy
# 跑通
cd ako-npu/input && ASCEND_RT_VISIBLE_DEVICES=15 bash run.sh
```
