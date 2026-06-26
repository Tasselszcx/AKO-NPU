# AKO-NPU 多 Case 验证实验记录

> 记录时间：2026-06-26 ｜ 主机：`set-hlsc-mlp-codelab-pc181` ｜ 卡：Ascend910B2C #15
> 目的：在 `ako-npu-practice/cases/` 的实践案例中挑选多个算子，验证 build→run→verify 全流程可复现，并观测单卡资源占用、理解 AKO pipeline 实际做了什么。

## 1. 结论速览

- 在 16 个案例中筛出可整套直接跑的，实测 **4 个 case 全部跑通且精度 PASS**（dsa-v5 上次已验，本次新增 matmul-py、vae-residual、attn-bwd-v2）。
- 卡占用：这些 baseline 算子是「算力密集、数据极小」型——**HBM 工作集 MB 级（占 61G 卡 < 1%）**，算力占用体现在 AI Core 核数（1~24 核）。
- 1 个 V1 老案例 `add` 因 `.asc` 文件自身缺陷（license header 用 `#`）编译失败，与环境无关，跳过。

## 2. 实验结果

| Case | 算子 / Shape | 编译 | 运行 | 精度 | AI Core | HBM workspace |
|------|-------------|------|------|------|---------|---------------|
| **dsa-v5** | DSA Indexer `[1,1,4096]` | ✅ | ✅ | ✅ PASS | 1（单核） | ~17 MB |
| **matmul-py** | MatMul+LeakyReLU `[1024,256]×[256,640]` | ✅ | ✅ | ✅ PASS | 多核(Cube+Vector) | — |
| **vae-residual** | VAE Conv+GN+SiLU 残差 `[1,256,64,64]` | ✅ | ✅ | ✅ PASS | 13 子 kernel 串行 | 数据 ~10 MB |
| **attn-bwd-v2** | Attention Backward `[4,80,256,256]` | ✅ | ✅ | ✅ PASS | 24（满核） | 13.5 MB |
| add（V1，跳过） | Add | ❌ | — | — | — | `.asc` license 用 `#` 注释，编译器报 `invalid preprocessing directive` |

### 精度细节

- **matmul-py**：`MERE=5.43e-7`（阈值 1.22e-4）/ `MARE=1.25e-4`（阈值 1.22e-3），allclose PASS，element match 655360/655360 = 100%。
- **vae-residual**：mean_abs_diff=6.9e-4，allclose(rtol=0.01,atol=0.01) PASS。`max_rel_error=5808` 是 REPORT.md 记录的**已知异常值**（个别近零点放大），mean_rel=0.018 正常。
- **attn-bwd-v2**：两个梯度输出 `grad_attn_scores` / `grad_value_states` 均 PASS，`cosine_sim≈1.0000001`、allclose 100%、rel_L2_err≈2.3e-5。（注：verify 阈值 `max_rtol=0.05`，atol 因 bf16 精度极限放宽，与 REPORT.md V2 分析一致。）

### 端到端耗时（wall-clock）

| Case | 耗时 | 说明 |
|------|------|------|
| matmul-py | ~15.7 s | 含 cmake 编译 + 数据生成 |
| vae-residual | ~16.5 s | 含 cmake 编译 + 数据生成 |
| attn-bwd-v2 | ~6.6 s | skip-build（复用二进制）|

> 绝大部分时间是 **cmake 编译 + host 端 golden 生成**，kernel 本身只占几十~几百 ms。纯 kernel 时延需 `msprof` 采集，属于 AKO 迭代优化阶段的工作。

## 3. 卡使用率观测

用 `hbm_sampler.py`（acl，150ms 间隔轮询卡 15，共 600s / 3984 采样点）：

```
HBM(G) 取值分布: {0.37: 3984}    # 全程稳定，无波动
空闲基线 = 峰值 = 0.37G (0.6%)
```

**结论**：显存维度看不出占用——算子工作集 MB 级，远低于 61G 卡容量，被基线淹没。真正反映算力占用的是 **kernel 输出里的核数 + workspace**：

- 单卡 Ascend910B2C：~61.0G HBM，约 24 个 AI Core。
- `attn-bwd-v2` 吃满 24 核但显存仅 13.5MB → **算力利用率高、显存利用率 <1%**。
- `dsa-v5` 单核 → 算力利用率约 1/24。
- 要量化 AI Core 实际利用率需 `ops-profiling` skill（msprof），baseline 验证阶段未采集。

## 4. 复现方法（关键：环境变量绕过 case 的路径假设）

practice 案例的 `run.sh` 是为特定布局写的，本容器路径不同，**不修改 case 文件**，靠环境变量绕过：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit       # run.sh 写死 $ASCEND_HOME_PATH/set_env.sh
export CMAKE_PREFIX_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/tikcpp/ascendc_kernel_cmake  # 部分 cmake .. 缺 prefix
export ASCEND_RT_VISIBLE_DEVICES=15                            # 隔离到空闲卡

cd ako-npu-practice/cases/<case>/input
# attn-bwd-v2 额外需要参考实现在 PYTHONPATH 里：
# export PYTHONPATH="$(pwd):$PYTHONPATH"
bash run.sh                # 或 bash run.sh --skip-build
```

依赖：`pip3 install numpy`（必需）；matmul-py / vae-residual / attn-bwd-v2 的 golden 生成需 `torch`（装了 CPU 版 `torch==2.8.0+cpu`，仅 host 端用）。

### 踩坑清单

| 现象 | 原因 | 解法 |
|------|------|------|
| `set_env.sh: No such file or directory` | run.sh 写死 `${ASCEND_HOME_PATH}/set_env.sh`，而顶层 set_env 把它设成了无 set_env.sh 的 `latest/` | `export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit` |
| `Could not find ASC / FindASC.cmake` | `cmake ..` 没传 prefix | `export CMAKE_PREFIX_PATH=.../ascendc_kernel_cmake` |
| `No module named 'attention_backward'` | gen_data.py 按 ako-npu 主项目层级找参考实现，practice 布局不同 | `export PYTHONPATH=<case>/input` |
| add 编译 `invalid preprocessing directive` | `.asc` 的 license header 用 `#` 行注释（V1 案例缺陷） | 跳过该 case |

## 5. AKO Pipeline 做了什么

AKO-NPU 是**纯规则驱动的 agent 自动算子优化框架**，核心是 AKO 迭代循环（`OPTIMIZE.md`）：

```
Step 0 装 skills → Step 1 路由(.asc 直接优化 / .py 先开发)
                          ↓
   Step 2(可选) CANNBot 开发: architect→design-reviewer→developer→reviewer 生成基线 .asc
                          ↓
   Step 3 AKO 循环（每轮 = 1 次代码改动 + benchmark）:
     msprof profiling 找瓶颈 → 查 skill 定优化方向 → 改 solution/ 代码
     → bench.sh 测性能+精度 → 写 ITERATIONS.md(含 References) → git commit [iter N]
```

**核心硬约束**（防 reward hacking / 跑偏，是框架精髓）：

| 约束 | 目的 |
|------|------|
| 精度是底线，失败立即 `revert` | 不许带着错误精度继续优化 |
| shapes 确定后不可变 | 防止靠改尺寸虚增加速比 |
| baseline = 纯 kernel 时间（msprof Task Duration） | 计时标准统一，排除 host I/O |
| 一次 parameter sweep = 1 轮迭代 | 防参数搜索耗尽迭代预算 |
| 入口用 `CLAUDE.md`（compact 后自动重注入） | 长时间运行规则不丢失 |

**框架演进与最佳战绩**（`REPORT.md`）：V1 子 agent（skills 发现失败）→ V2 独立进程+CANNBot → V3 规则持久化(CLAUDE.md) → V4 精度硬约束 → V5 References 字段引导查 skill。峰值加速：dsa **9.33x**、matmul-asc **6.15x**、attn-bwd **2.6-3.6x**。

## 6. 涉及文件

| 文件 | 说明 |
|------|------|
| `ako-npu-practice/cases/AKO-NPU-run-*/input/` | 各 case 的 `.asc` + `run.sh` + `scripts/`（gen_data/verify） |
| `ako-npu-practice/REPORT.md` | 框架 V1~V5 演进与历史验证结果 |
| `ako-npu/OPTIMIZE.md` | AKO 迭代循环规则与硬约束 |
| `/workdir/projects/hbm_sampler.py` | 本次新增的 HBM 占用采样器（acl，按间隔轮询指定卡） |
