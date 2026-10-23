# 腾讯混元（Tencent Hunyuan）Agent 方向开源工作调研报告

> **调研范围**：`Tencent-Hunyuan` 组织下 7 个与 Agent 直接相关的仓库 —— `UniRL`、`C3-Benchmark`、`VisualNeedle`、`RepoProbe`、`ExplorationBench`、`PlanningBench`、`Hyra-results`。
>
> **证据基线**：本报告的主体事实来自 7 份已下载到本地的仓库 README（位于 `D:\WorkSpace\workspace\Hunyuan\_research\readmes\`），并对 README 中引用的关键文件路径做了在线存在性校验。凡 README 未说明的内容，一律标注 `README 未说明`，不做推断性补全。
>
> **数据来源标注约定**：
> - 「README 内部时间线」= README 的 News / Roadmap 段落里写明的日期，属于仓库自述内容；
> - 「最近提交（外部探测）」= 本报告编写时从 GitHub 提交 atom feed（`/commits/main.atom` 的 `<updated>` 字段）读取的客观时间戳，**不属于 README 内容**；
> - 「stars（外部探测）」= 本报告编写时从 GitHub 仓库页面 HTML 读取的 `stargazerCount`，**不属于 README 内容**，会随时间变化；
> - `GitHub REST API` 全程 403（限流），因此**没有**任何数据来自该 API。

---

## 1. 概览总表

| 仓库 | 定位 | 核心能力一句话 | stars（外部探测） | 最近更新（外部探测 / README 时间线） | License | 规模（文件树特点） |
|---|---|---|---|---|---|---|
| [`UniRL`](https://github.com/Tencent-Hunyuan/UniRL) | 框架（RL 训练框架） | 用同一套 RL 后训练闭环（采样→打分→优势→更新→权重回同步）统一覆盖多模态模型族，含多轮工具调用的 agentic RL 训练路径 | 1011 | 2026-10-11（提交 feed） / README News 最新一条为 **[2026-10] `UniRL v0.2.0`** | Apache-2.0（`LICENSE` 明示 "UniRL is licensed under the Apache-2.0."） | 大型 Python 工程：分层包 `unirl/`（entrypoints / trainer / rollout / reward / algorithms / train / models / sde / types / data / distributed / utils）+ `examples/`（按 domain 分组的 Hydra recipe）+ `datasets/` + `lint/`；多个子目录自带 README（`unirl/README.md`、`unirl/models/README.md`、`unirl/rollout/env/README.md`、`unirl/types/README.md` 等） |
| [`C3-Benchmark`](https://github.com/Tencent-Hunyuan/C3-Benchmark) | 评测（Benchmark + 数据生成框架） | 用「工具关系复杂 / 关键隐藏信息 / 动态决策路径」三类挑战暴露 LLM Agent 的鲁棒性短板，并给出可复现的离线评测框架 | 38 | 2026-03-01（提交 feed：仅一条仓库迁移说明）/ README News：2026.1.26 论文被 ICLR 2026 接收 | CC BY 4.0（`LICENSE` 明示 "licensed under the Creative Commons Attribution 4.0 International Public License (CC BY 4.0)"） | 两层结构：`c3_bench/`（评测侧：`bench_test/` 推理与 `analysis_result.py`、`data/`、`tool_calls/`）+ 多智能体数据生成侧（实测存在于 `c3_bench/multi_agent/generate.py`）；附 `picture/` 文档图 |
| [`VisualNeedle`](https://github.com/Tencent-Hunyuan/VisualNeedle) | 评测（主动视觉搜索） | 把细粒度视觉感知从「一次性 VQA」改成「先裁剪定位、再看局部证据、再作答」的序列搜索过程来评测 | 0 | 2026-08-10（提交 feed） / README 未给带日期的 News；arXiv 编号 2605.26380 | Apache-2.0（`LICENSE` 明示 "licensed under the Apache-2.0 except for the third-party components listed below."） | 评测 harness 型仓库：4 个顶层入口脚本（`run_visualneedle_eval.py`、`run_crop_black_ablation.py`、`run_text_only_eval.py`、`generate_html_report.py`）+ 核心包 `visualneedle_eval/`（runner / eval_loop / judge / matching / artifacts / bbox_geometry / image_registry 等）+ `configs/models/*.yaml` + **vendored `Qwen-Agent/` 运行时** + `docs/` + `Makefile` |
| [`RepoProbe`](https://github.com/Tencent-Hunyuan/RepoProbe) | 评测（仓库级代码理解） | 以真实 GitHub Discussions 为题库、用加权 checklist 逐条验证，评测「Agent 能否解释一个代码库到底怎么运作」而非只打补丁 | 13 | 2026-08-06（提交 feed） / README 未给带日期的 News；论文 ASE 2026，arXiv:2608.04783 | 代码 Apache-2.0 + 数据 CC BY 4.0（双许可，README 与 `LICENSE` 一致） | 扁平 harness + 数据集：顶层 `evaluator.py` / `scorer.py` / `cache.py` / `dataset_loader.py` / `fetch_repos.py` / `prompt_templates_en.py`；`dataset/` 每仓库一个 CSV；`agent_configs/`（Dockerfile.base / Dockerfile / agent.json / example_agent.py / gateway.py）；`scripts_agent/` 两个 shell 入口；`docs/AGENT_INTEGRATION.md` |
| [`ExplorationBench`](https://github.com/Tencent-Hunyuan/ExplorationBench) | 评测（探索能力 / Alien Worlds） | 在规则可执行但与人熟知知识冲突的「异世界」里，只有通过四轮自主实验才能拿到解决留出任务所需的规则 | 7 | 2026-10-08（提交 feed） / README News：**2026.09** released | Apache-2.0（`LICENSE` 明示 "Code of ExplorationBench is licensed under the Apache-2.0."） | 三层结构：`explorationbench/`（`sandboxes/code/` = AlienCode、`sandboxes/logic/` = AlienLogic、`common/`、`frameworks/`、`scripts/`）+ `analysis/`（复现论文图表的脚本）+ `examples/` + `config/`；评测世界本体私有，仓库内装的是**两个 demo world** |
| [`PlanningBench`](https://github.com/Tencent-Hunyuan/PlanningBench) | 评测 + 数据构建框架 | 把真实规划场景抽象成任务/约束分类学，合成带验证 checklist 的自包含文本规划题，考「全局计划在耦合约束下是否真的可执行」 | 27 | 2026-05-29（提交 feed） / README 未给带日期的 News；arXiv:2605.20873 | **CC-BY-4.0**（`LICENSE.txt` 明示 "PlanningBench is licensed under License CC-BY-4.0"） | 极简数据仓库：`data/PlanningBench-eval.jsonl`（实测 **467 行**）+ `asset/`（logo 与两张图）+ `LICENSE.txt` + `README.md`；无评测代码（README 徽章误指向不存在的 `LICENSE-PlanningBench.txt`） |
| [`Hyra-results`](https://github.com/Tencent-Hunyuan/Hyra-results) | 研究 artifact（Agent 产出物集合） | Hyra（Hunyuan Research Agent）在数学/科学/工程/创作开放问题上的**成品解法与可复现脚本**归档 | 159 | 2026-08-19（提交 feed） / README News 最新一条亦为 **2026-08-19** | Apache-2.0（`LICENSE` 明示 "licensed under the Apache License, Version 2.0"） | 按赛道分目录的 artifact 仓库：`AI4AI/`（nanochat_autoresearch、nanogpt_speedrun、sol_execbench）、`AI4Science/`（15 个课题目录）、`AI4Fun/`（reversi、music、3d_penguin、3d_hunyuan）；每个目录放最终产物与自包含复现脚本 |

---

## 2. 逐个深挖

### 2.1 `UniRL`（框架 / RL 训练）

#### 项目定位与要解决的问题

UniRL 的自我定位是 **"A Reinforcement Learning Framework for Unified Multimodal Models"**（统一多模态模型的强化学习框架），标语为 "**U**(you)·**ni**(need)·**RL** for unified multimodal intelligence"。它要解决的问题是：把**同一个** RL 后训练闭环 —— "generate samples, score them, compute advantages, update the policy, and sync weights back to rollout workers" —— 跨越不同的多模态模型族复用，而不是每个模型族各写一套训练代码。

对 Agent 方向而言，UniRL 的价值在于它的 **`train_agentic` 训练路径**：这是把「多轮工具调用」当作 RL 环境来训练的策略优化路径，README 明确标注为 "Agentic RL | Service-scored multi-turn tool use"。

#### 核心架构 / 方法

README 描述 UniRL 是 "a layered, composable system"。关键抽象与数据流如下：

- **训练入口（entrypoint）** → 载入 **Hydra example config** → 创建对应 domain 的 **trainer**。
- RL trainer 负责协调 generation / scoring / update，跨可插拔的 **rollout engines**、**algorithms**、**model bundles**、**reward services** 与共享的 **distributed runtime**：Ray `DevicePool`、FSDP、**Transfer Queue (TQ)**、LoRA / full-weight sync。
- **SFT** 消费 supervised manifests；**async AR** 与 **diffusion** 入口把 rollout 与 training 放在**不同 GPU slab** 上重叠执行。
- 代码分层（来自 `unirl/README.md`）：**Entrypoints**（`train_diffusion.py`、`train_ar.py`、`train_pe.py`、`train_unified_model.py`、`train_agentic.py`）→ **Orchestration**（`trainer/`，含 `AgenticTrainer`）→ **Training loop**（`rollout/`、`reward/`、`algorithms/`、`train/`）→ **Foundation**（`distributed/`、`config/`、`types/`）。依赖方向被 `lint/check_core_dependencies.py` 锁定为单向向下。
- 三个部署模式：Train-side sampling / Separate rollout / Colocated rollout。

**团队自研算法（README 标为最大亮点）**：

| 算法 | 论文 | 目录 | 机制（README 原文要点） |
|---|---|---|---|
| **Flow-DPPO** | *Flow-DPPO: Divergence Proximal Policy Optimization for Flow Matching Models*（arXiv:2606.11025） | `FlowDPPO/` | Diffusion/flow RL，带 exact divergence-based trust-region mask |
| **DRPO** | *Rethinking the Divergence Regularization in LLM RL*（arXiv:2606.09821） | `DRPO/` | Token-level LLM RL，smooth advantage-weighted quadratic regularizer |
| **CPPO** | *Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning*（arXiv:2606.10968） | `CPPO/` | Token-level LLM RL，position-weighted、cumulative-prefix-budget Binary-TV mask |

同时接入标准参考算法：**(LLM's) GRPO**、**DiffusionNFT**、**DanceGRPO**、**MixGRPO**（位于 `unirl/algorithms/`，入口索引 `unirl/algorithms/README.md`）。

**Agentic 路径的具体机制（README「Agentic Workflows」+ `unirl/rollout/env/README.md`）**：

- `train_agentic` 在 AR 路径上扩展多轮工具调用：**每个 turn 是 lineage 中的一个 `Sample`**；terminal answer 由 **reward service** 打分；训练在 **colocated rollout barrier** 处等待。
- `AgenticTrainer` 在**每次 rollout 前同步当前训练权重**，并发派发 sibling trajectories，等完整 GRPO group 齐备后再打分与训练。**每条成功轨迹获得一个 group-normalized advantage，并施加到它所有生成的 assistant turn 上；失败轨迹被排除出更新**。
- 环境契约：`Environment.reset(request) -> Sample`、`Environment.step(sample) -> (observation, done, info)`、`Environment.close(sample)`（可选、幂等、不得抛异常）；`Tool` 暴露 JSON function schema 与 `execute(arguments) -> str`；`StatefulTool` 则实现 `session_start` / `execute_session` / `session_end` 生命周期。
- 轨迹结构：`input(user) -> generated(assistant) -> observation(tool) -> generated(assistant) -> ...`；`Sample.fork` 追加带采样参数的 generated `Part`，`Sample.observe` 追加 branch-one input `Part` 且**不带采样参数** —— 即 observation 是 conditioning，不是可训练的策略输出。
- 内置环境/工具：`ToolEnvironment`（解析最后一段 Hermes/Qwen 风格 `<tool_call>{...}</tool_call>`，也接受去掉闭合标签后的 balanced JSON）、`CalculatorTool`（白名单算术 AST 求值）、`SandboxTool`（**per-session 持久 Python 子进程**，带 timeout 与输出上限）、`SearchTool`（批量 Serper 或 SerpApi，需要 `SERPER_KEY_ID`）、`VisitTool`（Jina 网页阅读 + 可选 OpenAI-compatible 摘要）。
- 错误与边界语义：缺失/畸形工具调用**被当作最终答案**；未知工具、非法参数、工具异常**以错误文本返回给模型而非抛出**；`max_turns` 是引擎硬上界，环境若也暴露 `max_turns` 则生产引擎要求二者一致；token 预算耗尽记为 `overflow` outcome（**照常参与打分**），第一轮前就溢出的 prompt 记为 `failed`（无内容可打分）。

**训练模式全表（README「Training Modes」，8 条路径）**：

| Training path | Trains | Entrypoint | Built-in default recipe |
|---|---|---|---|
| Diffusion RL | Image / video diffusion models | `train_diffusion` | `examples/diffusion/sd3/sd3_trainside` |
| AR RL | VLM + text-only LLM | `train_ar` | `examples/ar/qwen_vl_grpo_geo3k_mc_4x8` |
| SFT | Supervised text / multimodal / diffusion | `train_sft` | `examples/sft/qwen3_sft` |
| Prompt enhancement | AR rewriter + diffusion reward | `train_pe` | `examples/pe/pe_trainside_pickscore` |
| Unified RL | Unified AR + diffusion | `train_unified_model` | `examples/unified_model/hi3_vllmomni` |
| **Agentic RL** | **Service-scored multi-turn tool use** | **`train_agentic`** | **`examples/deep_research/deep_research_search_judge`** |
| Async AR RL | AR with separate train / rollout workers | `train_async_ar` | `examples/ar/qwen3_grpo_4b_base_dapo_sglang_async` |
| Async diffusion RL | Diffusion with separate workers | `train_async_diffusion` | `examples/diffusion/bagel/bagel_vllmomni_async` |

#### 关键目录与文件（均为 README 真实引用，且已在线校验存在）

| 路径 | 作用 |
|---|---|
| `unirl/README.md` | 代码架构总览、module map、部署模式、runtime data flow |
| `unirl/models/README.md` | 模型维度的完整支持矩阵（package、recipe、rollout engine、限制） |
| `unirl/rollout/env/README.md` | **agentic 环境、工具与轨迹契约**（本报告 agentic 细节的主要来源） |
| `unirl/algorithms/README.md` | 各 track 的 loss 算法与 `StageAlgorithm` 契约 |
| `unirl/types/README.md` | `Sample` / `Part` / primitive / condition / segment 等共享类型契约 |
| `unirl/train_agentic.py` | agentic 训练入口（`python -m unirl.train_agentic`） |
| `unirl/rollout/harness/tool_agent.py` | `ToolAgentHarness`：worker 侧的 turn loop 控制流 |
| `unirl/rollout/engine/agentic/engine.py` | `AgenticRolloutEngine`：承载 agentic rollout 的分布式运行时 |
| `unirl/rollout/env/tool_environment.py` | `ToolEnvironment` 实现 |
| `unirl/rollout/env/tools/search.py`、`unirl/rollout/env/tools/visit.py` | `SearchTool` / `VisitTool` 实现 |
| `unirl/reward/local/llm_judge.py` | `LLMJudgeRewardScorer` / `LLMJudgeSpec` |
| `unirl/reward/local/ocr.py` | OCR reward 组件（依赖 `eval` extra 的 PaddlePaddle） |
| `lint/check_core_dependencies.py` | 锁定分层依赖方向的 lint 脚本 |
| `examples/deep_research/deep_research_search_judge.yaml` | 唯一可跑的 agentic recipe |
| `examples/run_experiment_single_node.sh` | 单机启动脚本（按 `ENTRY` 选择 Hydra 入口） |
| `INSTALL.md`、`pyproject.toml`、`AGENTS.md`、`LICENSE` | 安装、依赖与版本、仓库约定、许可证 |

#### 安装与运行方式

- **环境要求（`INSTALL.md`）**：Python **>=3.12,<3.14**、**Linux x86_64**、**NVIDIA GPUs**；官方推荐用 `uv` 安装（`pyproject.toml` 里的 CUDA index 与 override 只有 uv 生效，plain pip 会忽略 `[tool.uv]`）。
- **引擎互斥**：`vllm` 与 `sglang` 的 PyTorch 栈不兼容，**一个 virtualenv 只能装一个 engine extra，绝不要 `--all-extras`**。
- `vllm` / `sglang` 两个 extra 均为 `PyTorch 2.13.0+cu130` / `CUDA 13.0`；SGLang wheel 需要 **glibc >= 2.34**；需自行把 NVIDIA CUDA 13 forward-compat 库放到 `LD_LIBRARY_PATH`（launcher 不做这件事）。

```bash
# vllm-omni 路线
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
uv pip install -e ".[vllm,train,infer]"

# sglang 路线（先装 CUDA 13 编译器 wheel 并把 CUDA_HOME 指过去）
uv pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-runtime==13.0.*"
export CUDA_HOME="$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
uv pip install -e ".[sglang,train,infer]"
```

`INSTALL.md` 列出的 extras：`vllm`、`flash-attn`、`sglang`、`train`、`cosmos3`、`infer`、`eval`、`veomni`、`dev`、`dataset-prep`。

**运行示例（README「Getting Started」）**：

```bash
# compose-check, then launch a single-node example
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
```

**运行 agentic recipe（`examples/deep_research/deep_research_search_judge.yaml` 头部注释 + `examples/README.md`）**：

```bash
python -m unirl.train_agentic --config-name=deep_research/deep_research_search_judge num_devices=2
# 或
ENTRY=train_agentic bash examples/run_experiment_single_node.sh deep_research/deep_research_search_judge
```

该 recipe 的关键可读配置：`max_turns: 8`（rollout 与 env 两处都要求一致）、`max_new_tokens: 2048`（**per TURN**）、`stop: ["</tool_call>"]`、base model `Qwen/Qwen3-4B-Instruct`（可由 `QWEN3_INSTRUCT_PATH` 覆盖）、rollout engine `AgenticRolloutEngine` + `SGLangEngineConfig`、`env` 为 `ToolEnvironment` 且包含 `SearchTool` 与 `VisitTool`、`reward` 为 `RewardService` + `LLMJudgeRewardScorer`（`truncated_reward: keep`）、算法 `unirl.algorithms.grpo.GRPO`、weight sync 为 `TensorWeightSync`、数据源 `MultimodalRLDataSource`（data path 期望 `ASearcher jsonl` 格式 `{"prompt": ..., "metadata": {"answer": ...}}`）。所需环境变量：`SERPER_KEY_ID`、`JINA_API_KEYS`、`JUDGE_URL` / `JUDGE_MODEL`、可选 `SUMMARY_URL` / `SUMMARY_MODEL`。

**启动脚本行为（实测读取 `examples/run_experiment_single_node.sh`）**：起本机 Ray head（默认端口 6379），用 `nvidia-smi -L` 自动探测 `GPUS_PER_NODE`（探测失败回退为 8）并以其设置 `num_devices`，支持 `DRY_RUN=1` 预览命令、`INSTALL_EDITABLE=1` 时执行 `pip install --no-deps -e .`。

#### 评测 / 数据集规模

**README 未给出统一的 benchmark 分数表**。README 中出现的可核实量化信息为：

- **`UniRL v0.2.0`** 发布内容：**12 个新 model packages**，新增 **SFT / agentic / async** 三种训练模式，rollout 栈升级为 **SGLang 0.5.19 + vLLM-Omni 0.28**（来自 README News，日期 **[2026-10]**）。
- 三个团队算法各自在 **[2026-06]** 发布，对应 arXiv 编号 `2606.09821`（DRPO）、`2606.11025`（Flow-DPPO）、`2606.10968`（CPPO）。
- 模型支持矩阵规模：UniRL 根 README 的「Model Support」主表列 **24 行**模型/recipe（含 1 行 `bundle-only`、1 行 `SFT-only`、1 行 `composed`，以及 `trainside only` 的 Boogu-Image-0.1、WAN 2.2 V2V、LTX-2.3、MiniMax-H3、SenseNova-U1.5、Janus-Pro 等）。
- Trendshift 徽章标注该仓库曾获 "**#16 Python Repository of the Day**"（`trendshift.io/repositories/48953`）。
- 依赖版本下限（`INSTALL.md` 表格）：`transformers>=5.12,<5.13`、`diffusers==0.40.0`（uv override）。

#### 论文 / arXiv 链接、License

- 算法论文：arXiv [2606.09821](https://arxiv.org/abs/2606.09821)（DRPO）、[2606.11025](https://arxiv.org/abs/2606.11025)（Flow-DPPO）、[2606.10968](https://arxiv.org/abs/2606.10968)（CPPO）。
- 框架本身**没有** arXiv 编号，引用方式为 GitHub 仓库（README 给出的 BibTeX key 为 `unirl_github`，`year = {2026}`，`urldate = {2026-06-05}`）。
- 文档站：`https://unirl-project.github.io/unirl/`。
- License：**Apache-2.0**。

#### 亮点与局限

**亮点**：
1. 把 agentic RL 作为**一等训练路径**（`train_agentic`）而非外挂脚本，并提供完整的 env / tool / trajectory 契约文档。
2. 算法与模型是**两个独立正交维度**，README 明确 "any diffusion algorithm runs on a diffusion model, AR algorithms on AR models"，组合数远多于随仓库发布的 recipe 数。
3. 工程质量控制点具体：分层依赖方向由 lint 脚本锁定并**公开承认尚未满足的债务**（`models/` 的 `hunyuan_image3`、`qwen3_5`、`qwen3_moe` 反向依赖 `train/backend/veomni`，`models/cosmos3` 继承 train 侧 builder，`algorithms/` 反转 `train/lora` 适配器），明确写下 "That is migration work, not a convention to copy"。

**局限**：
1. **没有统一的评测分数**，无法从 README 判断 agentic 路径相对 baseline 的实际收益。
2. 环境门槛高：仅 Linux x86_64 + NVIDIA GPU，Python 3.12–3.13 窄区间，两个 engine extra 不可共存，SGLang 还要求 glibc ≥ 2.34 与手工 CUDA 13 环境。
3. agentic 只有**一个**可运行 recipe（deep research 搜索 + LLM judge 打分），且依赖外部付费服务（Serper / Jina / 独立 judge endpoint）。
4. README 中的 `MiniMax-H3` 备注仍标为 rollout backend 未合入（`#378 / #420`），说明部分模型支持并不完整。

#### 对我们可复用的工程实践

1. **「环境决定何时结束」的契约设计**：`Environment.step` 返回 `(observation, done, info)`，由**环境而非 turn loop** 决定 episode 是否终止；畸形工具调用一律降级为「最终答案」、工具异常降级为「错误文本回灌模型」。这套语义可以直接抄进自研 agent scaffold，避免脚手架因一次工具报错而整条轨迹崩溃。
2. **observation 不进梯度**：`Sample.observe` 追加的 tool observation **不带采样参数**，与 `Sample.fork` 产生的 generated turn 明确区分。只要做多轮 agent RL，这条区分必须一开始就设计对，否则会把工具返回文本当作策略输出训练。
3. **失败轨迹与溢出轨迹的分级处理**：`failed` 轨迹排除出 GRPO 统计、`overflow` 轨迹照常打分、slot RPC 失败则 poison 整个 manager 并 fail 掉该 step。这种「三档失败语义」比笼统的 try/except 更适合长时多轮 rollout。

---

### 2.2 `C3-Benchmark`（评测 / `C^3-Bench`）

#### 项目定位与要解决的问题

仓库标题为 **"C^3-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking"**。它要解决的问题是：现有 Agent 评测「通常通过多轮对话来评估」，但**忽略了工具间关系、环境反馈与既往决策**这些关键因素对 agent 行为的影响。为此 `C^3-Bench` "integrates attack concepts and applies univariate analysis to pinpoint key elements affecting agent robustness"，并设计三类挑战：**navigate complex tool relationships（规划复杂工具关系）**、**handle critical hidden information（处理关键隐藏信息）**、**manage dynamic decision paths（管理动态决策路径）**。

README 明确写了仓库状态：**"This repo has moved to https://github.com/yupeijei1997/WildToolBench"**（`README_ZH.md` 亦重申「这个仓库经过重构并被移动到」），并宣布论文 **"Benchmarking LLM Tool-Use in the Wild" 被 ICLR 2026 接收**（News 日期 2026.1.26）。

#### 核心架构 / 方法

**评测框架的设计原则（README「Framework」，5 条）**：

- **高可复现性**：测试数据里所有 golden answer 对应的**工具执行结果已持久化保存**，"There is no need for any website's KEY, and there are no unstable tool invocation scenarios"。
- **高评测效率**：**动态评测**。第一阶段用 `EvalByToolCallGraph` 模块，依据 action（预测工具名）是否匹配 golden answer 决定是否继续调用，过程中做**决策树剪枝**，显著减少需维护的路径数。
- **高代码复用性**：所有请求统一使用标准 **ToolCalls 协议**；并为多个原本不支持 ToolCalls 协议的开源通用/专用模型**封装了 ToolCalls 协议**，解决其他评测框架混用 Prompt 与 ToolCalls 两种调用方式造成逻辑混乱的问题。
- **多评测分析维度**：第二阶段用 `AnalysisResult` 模块做 **六个维度**的详细分析，结果落 CSV 便于 bad case 分析。
- **强扩展性**：API 模型走 `APIHandle`；新开源模型持续更新接入；自训模型参考 Handle 代码把 Prompt 调用封装成 ToolCalls 协议即可接入。

**数据构造方法（可控多智能体数据生成框架，README 列 8 项优势）**：

- **可控任务生成**：每轮可指定任务类型 —— **单工具调用 / 多工具调用 / 澄清后调用工具 / 闲聊**四类，从而遍历所有可能动作空间、构造无偏数据。
- **指定数量任务生成**：可生成指定任务数，配合前者覆盖任意任务数的所有动作空间。
- **多样化任务生成**：第一轮生成多个不同语气、长度、主题/实例、场景、角色身份的任务并随机抽取继续。
- **真多轮任务生成**：后续轮次可控生成真多轮任务，含三种核心类型 —— **implicit（指代理解）、ellipsis（省略成分）、long-term memory（长期记忆）**，并用「几十种 few-shot」示例随机引导。
- **Rich Agents**：五大类智能体 —— **User / AI / Planner / Tool / Checker**，共 **15 种二级子类型**。
- **Powerful Planner**：README 称其为「目前所有智能体框架中唯一能对复杂串行与并行多工具调用任务做决策」的 Planner，prompt 超过 **4000 字符**。
- **Reliable Checker**：README 称其为「目前唯一会检查并行调用逻辑」的 Checker，编写了**几十种规则**检查 Planner 的低级错误并给反馈让其反思；Planner + Checker 组合在**无人工干预**下决策准确率 **over 90%**。
- **Arbitrary Model Specification / Bilingual Support**：任意 LLM 可作基座；支持中英双语生成。

**数据质量流程（README 详细披露）**：初始数据由该多智能体框架生成 → 按四种动作类型划分，由 **4 名算法研究员**逐层人工查验修正（**层一层**构建，先生成一层再修正再生成下一层，避免跨层连锁返工）→ **1 名 Agent 领域高级专家**逐条点评 + 二次修正 → **交叉验证**（4 人互查互评后与专家讨论，做第三轮修正）→ 专家对全部数据做**第四轮**检查修正 → 用代码检查人工误操作导致的**参数类型错误与不合理依赖**，专家做**第五轮**修正。结论：**准确率从最初不到 60% 提升到 100% 正确性**，总计约**一个月**。

#### 关键目录与文件

README 中出现的真实路径（`✅` = 已在线校验存在；`⚠️` = README 声明但在 `main` 分支 raw 路径下 404，标注为 README 声明路径）：

| 路径 | 作用 | 校验 |
|---|---|---|
| `c3_bench/data/C3-Bench.jsonl` | 测试数据 | ⚠️ raw 404，README 声明其存在 |
| `c3_bench/requirements.txt` | 依赖清单 | ✅ |
| `c3_bench/bench_test/request_pipeline_upta.py` | **upta 格式**推理入口 | ✅ |
| `c3_bench/bench_test/request_pipeline.py` | **ua 格式**推理入口 | ✅ |
| `c3_bench/bench_test/analysis_result.py` | 评测与六维分析（输出 `data_with_details.csv`） | ✅ |
| `c3_bench/bench_test/web_server.py` | 原生 HuggingFace 部署用 web server | ✅ |
| `c3_bench/bench_test/handle/handles.py` | 模型 → handle 映射表（`tool_handle_map`） | ✅ |
| `c3_bench/multi_agent/generate.py` | 多智能体数据生成入口 | ✅（README 写的是相对路径 `cd multi_agent`） |
| `c3_bench/tool_calls/tool_model_map.py` | `tool_model_path_map` 变量所在文件 | ⚠️ raw 404，README 声明其存在 |
| `picture/first.png`、`picture/example.png`、`picture/compare.png`、`picture/framework.png`、`picture/multi_agent2.png`、`picture/agent_family.png` | 文档配图 | README 引用 |
| `README_ZH.md` | 中文版 README | ✅ |

**实测 `handles.py` 的 `tool_handle_map` 具体条目**（这是 README 提到的「view in c3_bench/bench_test/handle/handles.py」的真实内容）：
`hunyuan-turbos-latest`、`hunyuan-a13b`、`toolace`、`toolace2`、`xlam`、`xlam2-70b/32b/8b/3b/1b`、`gorilla`、`chatglm`、`fcm3.1`、`watt70b`、`watt8b`、`hammer7b/3b/1.5b/0.5b`、`llama70b/8b/3b/1b` —— 共 **24 个 key**（每个映射到 `(HandleClass, bool)`）。

#### 安装与运行方式

```bash
# 环境（README「Basic Installation」）
conda create -n C3-Bench python=3.10
conda activate C3-Bench
git clone https://github.com/Tencent-Hunyuan/C3-Benchmark.git
cd c3_bench/
pip install -r requirements.txt
```

`c3_bench/requirements.txt` 实际内容：`ipdb`、`requests`、`pandas`、`openai`、`jieba==0.42.1`、`rouge==1.0.1`、`rouge-chinese==1.0.3`、`rouge-score==0.1.2`。README **未给出 GPU 硬件要求**。

**API 模型（以 `hunyuan-turbos-latest` 为例）**：

```bash
export MODEL=hunyuan-turbos-latest
export API_KEY=xxxxxxxxx
export BASE_URL=https://api.hunyuan.cloud.tencent.com/v1

cd c3_bench/bench_test
python3 request_pipeline_upta.py \
    --model=hunyuan-turbos-latest \
    --data_path=./data/C3-Bench.jsonl \
    --output_path=./result \
    --language=en \
    --continue_file=empty.jsonl \
    --remove_role=True \
    --contain_context=True
```

（`ua format` 用 `request_pipeline.py`，其余参数相同。README 说明「若测试中途意外停止，可以修改 `continue_file` 继续测试，这将使得已预测的结果不会重复预测」。）

**开源模型（推荐 vLLM 部署以吃到 Tool Calling 与最佳 Function Call System Prompt）**：

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --host ${LOCAL_IP} \
    --port 8020 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --gpu_memory_utilization 0.92 \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --disable-log-stats \
    --enable-auto-tool-choice \
    --tool-parser-plugin /path/tool_parser/hunyuan_tool_parser.py \
    --tool-call-parser hunyuan \
    2>&1 | tee log_server.txt
```

然后用 `--model=hunyuan-a13b --model_url=MODEL_URL` 走 `request_pipeline_upta.py` / `request_pipeline.py`，并 `export MODEL=Hunyuan-A13B-Instruct`。原生 HuggingFace 路线：把模型名与地址加入 `tool_model_path_map`，再 `python3 web_server.py MODEL_NAME`。

**评测**：

```bash
cd c3_bench/bench_test
python3 analysis_result.py \
    --data_file PREDICT_DATA_FILE \
    --output_csv_flag=True \
    --output_csv_path=./data_with_details.csv
```

支持用逗号拼接多个预测文件以同时评测多模型。

**数据生成（多智能体框架）**：

```bash
export MODEL=hunyuan-turbos-latest
export API_KEY=xxxxxxxxx
export BASE_URL=https://api.hunyuan.cloud.tencent.com/v1
export LANGUAGE=zh

cd multi_agent
python3 generate.py \
    --layer_num_total 4 \
    --user_model ["hunyuan-turbos-latest"] \
    --planner_model "hunyuan-turbos-latest" \
    --tool_model "hunyuan-turbos-latest" \
    --agent_model "hunyuan-turbos-latest" \
    --checker_model "hunyuan-turbos-latest"
```

README 说明生成了三种格式的数据：raw generated data、streamlined data（去掉 checker 消息与 role-switch 消息）、每个 agent 的 captured data。

#### 评测 / 数据集规模

- **被测对象数量**：README 明确 "Extensive experiments are conducted on **49 mainstream agents**, encompassing general fast-thinking, slow-thinking and domain-specific models"（中文版：「在 **49 个主流智能体**上进行了广泛实验」）。
- **题目数量**：**README 未说明** `C3-Bench.jsonl` 的条目数。可写的事实是：任务层数可变（数据生成默认示例为 `--layer_num_total 4`；分析脚本默认 `answer_depth=4`）；README 称「第二至四轮全部是真多轮任务，覆盖率达到了 **2 个 100%**」；相对 BFCL、T-EVAL 等，其测试数据「覆盖了所有可能的动作空间」。
- **数据质量量化**：准确率从「最初不到 60%」提升到「**100% 的正确性**」。
- **生成框架的量化**：Planner prompt 超过 **4000 字符**；Planner + Checker **无人工干预决策准确率 over 90%**；5 类 agent / **15 种**子类型。
- **baseline 模型与分数**：**README 未给出任何模型分数表**（README 只给出复现命令示例中的模型名 `hunyuan-a13b`，以及一个预测文件名 `2025-06-25-15:50:37_b3b8be_hunyuan-a13b_en_remove_role_contain_context_history_with_planner_tool_.jsonl`）。
- **外部数据集事实（非 README）**：HuggingFace 数据集 `tencent/C3-BenchMark` 的 tags 含 `license:cc-by-4.0`、`language:en`、`language:zh`、`size_categories:n<1K`、`format:csv`、`arxiv:2505.18746`，`lastModified` 为 `2025-07-01T06:17:30.000Z`。

**指标名（README 声明 + 已校验的 `analysis_result.py` 实际输出）**：README 说评测结果包括「动作类型与层的矩阵准确率、动作类型与层各自的准确率、多工具调用结果分析、错误类型分析、真伪多轮准确率、真多轮子类型准确率、参数错误类型分析」。实测脚本进一步确认了下列真实指标名与分类：

- 矩阵轴：X 轴为动作类型 **`单` / `多` / `拒` / `反`** + `all` + `num`；Y 轴为层 index `0..answer_depth-1` + `all`。
- 多工具调用分析指标：`{combine, serial, cs}_tools_rate`、`serial_complete_rate{sum, serial, cs}`、`best_optical_rate{sum, combine, cs}`。
- **12 类 error type**：`reject tool calling`、`function name hallucination`、`missing information in tool calling`、`redundant tool calling`、`call function without reject`、`call the wrong function`、`early stop`、`miss required params`、`param hallucination`、`type error`、`param value hallucination`、`param value error`（前 6 类归为 action 类，后 6 类归为 param 类）。
- 其它：`turn_type_rate`、`turn_subtype_rate`、`messages_by length`、`Layer variance`、`time_out_cnt`、`error_arguments`、`case_num`。
- CSV 列（`--output_csv_flag`）：`id, idx, type, single_type, turn_type, turn_subtype, messages_length, tools, messages, ...`。

#### 论文 / arXiv 链接、License

- 论文：arXiv [2505.18746](https://arxiv.org/abs/2505.18746) — *$C^3$-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking*（作者 Peijie Yu、Yifan Yang、Jianjian Li、Zelong Zhang、Haorui Wang、Xiao Feng、Feng Zhang，2025）。
- ICLR 2026 版本：<https://openreview.net/forum?id=yz7fL5vfpn>，标题为 *Benchmarking LLM Tool-Use in the Wild*。
- 数据集：<https://huggingface.co/datasets/tencent/C3-BenchMark>。
- 后继仓库：<https://github.com/yupeijei1997/WildToolBench>。
- License：**CC BY 4.0**（`LICENSE` 第 5 行原文）。

#### 亮点与局限

**亮点**：
1. **离线可复现**是整个 benchmark 的核心设计：golden answer 的工具执行结果全部持久化，不需要任何网站 KEY —— 这是工具类 benchmark 最容易踩的坑，`C^3-Bench` 从设计上规避了。
2. **把「真多轮」拆成 implicit / ellipsis / long-term memory 三种可控子类型**，这是长上下文 Agent 评测中少见的细粒度切分。
3. **六维分析 + 12 类 error taxonomy + CSV 落盘**，直接支持 bad case 归因，而不是只给一个准确率。
4. 数据质量流程披露得非常具体（五轮修正、逐层构建的动机、交叉验证），「准确率 <60% → 100%」这一量化自述罕见地诚实。

**局限**：
1. **README 没有题目数与 baseline 分数表**，无法从中判断各模型的实际表现差异（README 只给出定性结论：agent 在工具依赖、长上下文信息依赖、频繁策略类型切换三方面存在显著缺陷）。
2. 测试数据文件在 `main` 分支 raw 路径下 404，评测需依赖 HuggingFace 数据集或完整 clone。
3. `c3_bench/tool_calls/tool_model_map.py` 同样 raw 404，原生 HF 部署路线需要从 clone 中确认实际位置。
4. **仓库已迁移**到 `yupeijei1997/WildToolBench`，本仓库实质进入维护/归档状态（提交 feed 最新一条为 2026-03-01 的迁移说明）。
5. 生成框架的「唯一」「最高」等表述均为 README 自述的比较级论断，本报告未做第三方交叉验证。

#### 对我们可复用的工程实践

1. **工具执行结果持久化 + `EvalByToolCallGraph` 动态剪枝**：用预存的 golden 工具结果替代真实 API 调用，既保证可复现又去掉外部依赖；再按「预测工具名是否匹配 golden」决定是否继续展开，配合决策树剪枝把评测路径数压下来。做内部 agent 回归评测时应直接照搬这套两段式（离线 golden trace + 动态路径裁决）。
2. **12 类 error taxonomy 与双分组（action vs param）**：任何 tool-use 评测都值得用这套分类法做失败归因，它的价值在于把「失败」拆成可分别优化的工程问题（拒答、幻觉函数名、漏参、类型错、值错……）。
3. **`--continue_file` 续跑机制**：把已完成预测写入文件并在重启时跳过，是长跑评测（尤其 49 个模型 × 多轮任务）必备的工程细节，成本极低但收益很高。

---

### 2.3 `VisualNeedle`（评测 / 主动视觉搜索）

#### 项目定位与要解决的问题

VisualNeedle 是 **"Benchmarking active visual search in information-dense scenes"**，一个 **300 题**的 benchmark，用来测试 MLLM 能否在**信息密集图像**中找到并使用**决定性的局部视觉证据**。它明确反对把细粒度感知做成一次性 VQA："Rather than treating fine-grained VQA as one-shot, it evaluates a sequential process: localize a relevant region, inspect it through a crop, and integrate that evidence into the final answer."

核心叙事是 **"Search before answering."** —— agent 从杂乱的全局视图开始，找到参考区域，然后逐步收窄裁剪范围，直到决定性细节可见。难点在于证据**既小又稀疏**：成功需要多步搜索与局部证据整合，而不是从问题措辞或全局场景里猜一个看似合理的答案。

#### 核心架构 / 方法

**Benchmark 规格（README「The Benchmark」表）**：

| Aspect | VisualNeedle |
|---|---|
| **Task** | Image-grounded, multi-step visual search rather than one-shot VQA |
| **Scenes** | Urban street views, dense documents, shelves, bookcases, and maps |
| **Evidence** | A small, decisive clue that is not reliably visible at a glance |
| **Core actions** | `Crop(bbox)` 检查局部证据，然后 `Answer` 结束 episode |

**三条反捷径要求（每条题目都要满足）**：
1. **Question-alone insufficiency**：语言先验与词汇线索不应泄露答案。
2. **Global-view insufficiency**：决定性细节不应能从单张整图观察中可靠恢复。
3. **Returned-evidence dependence**：工具使能的答案应依赖**工具返回的视觉内容**，而不只是「走了一遍工具调用轨迹」。

**四设置诊断协议（README「Diagnostic Evaluation Protocol」）** —— 这是该 benchmark 方法论上最有价值的部分：

| Setting | 模型输入与交互 | 诊断什么 |
|---|---|---|
| **Text-only** | 只有问题；无图无工具 | 措辞与语言先验是否已足够 |
| **No-tool** | 一次整图观察；直接作答 | 全局视觉语义能否解决局部线索 |
| **Tool-enabled** | 整图 + 真实局部 crop 观察 | 模型能否定位、检查并整合证据 |
| **Crop-black** | 相同的工具使用协议，但每次返回的 crop 被替换为**同尺寸黑图** | 工具带来的增益是否真的依赖返回的视觉证据 |

README 强调 **crop-black 消融是核心**：它保留交互格式与 crop 尺寸，只移除返回像素，因此它与 tool-enabled 的差距度量的是**中间视觉证据的贡献**，而不是「奖励走过工具调用轨迹」。

**官方评测 harness 的职责（README「Official Evaluation Harness」表）**：模型与工具设置（YAML 模型配置、provider 特定的坐标约定、启用的图像工具）；协议执行（标准 tool-enabled/no-tool、crop-black 消融、text-only）；可靠执行（async 或 multiprocessing、有界重试、按索引重跑、有序输出）；答案打分（先 exact/normalized 匹配，未匹配的再走可选配置的 VLM judge）；审计产物（per-sample 日志、中间工具图像、归一化 crop 坐标、token 用量、结构化 JSON、自包含 HTML 报告）。

**论文复现要求**：保持模型、工具配置、judge 与评测模式固定，并**同时报告 exact-match score、final score、judge-use count 与 tool-use statistics**。

#### 关键目录与文件（来自 README 与 `docs/evaluation.md`，已在线校验）

| 路径 | 作用 |
|---|---|
| `run_visualneedle_eval.py` | 标准评测（tool-enabled 或 no-tool） |
| `run_crop_black_ablation.py` | 消融：工具返回图替换为黑图 |
| `run_text_only_eval.py` | 基线：无图输入，text-only judge |
| `generate_html_report.py` | HTML 报告生成 |
| `visualneedle_eval/runner.py` | per-sample 评测流水线 |
| `visualneedle_eval/eval_loop.py` | async / multiprocessing 调度器 |
| `visualneedle_eval/visualneedle_agent.py` | 模型配置加载、agent 构建（默认配置路径硬编码于此） |
| `visualneedle_eval/matching.py` | 答案匹配与归一化 |
| `visualneedle_eval/judge.py` | VLM 语义 judge |
| `visualneedle_eval/artifacts.py` | 结果结构、日志、统计 |
| `visualneedle_eval/image_registry.py` | 带仿射矩阵的图像溯源追踪 |
| `visualneedle_eval/bbox_geometry.py` | 坐标变换 |
| `visualneedle_eval/report.py` | HTML 报告渲染 |
| `visualneedle_eval/dataset.py`、`cli.py`、`_worker.py`、`_retry.py`、`_types.py` | 数据加载、共享 CLI 逻辑、多进程 worker 状态、重试、`EvalMode` 类型 |
| `configs/models/think_with_images_neibu.yaml` | 起始模型配置 |
| `Qwen-Agent/` | **vendored Qwen-Agent 运行时** |
| `docs/evaluation.md` | 评测指南（模型配置、三种模式、完整 CLI、数据 schema、输出产物、Python API） |
| `requirements.txt` / `requirements-dev.txt` / `Makefile` | 依赖与快捷命令 |

`requirements.txt` 实际内容只有 6 行：`-e ./Qwen-Agent`、`datasets`、`opencc-python-reimplemented`、`pillow`、`pyyaml`、`requests`。

#### 安装与运行方式

```bash
pip install -r requirements.txt
pip install -e .

python run_visualneedle_eval.py \
  --config configs/models/think_with_images_neibu.yaml \
  --model my-model \
  --dataset-file path/to/visualneedle.jsonl \
  --concurrency 20
```

不安装包时可 `PYTHONPATH=Qwen-Agent:. python <script>.py ...`。开发模式：`pip install -r requirements-dev.txt && make test`。

**三种模式的实际命令**（`docs/evaluation.md`）：

```bash
# Tool-enabled（默认）
python run_visualneedle_eval.py --config ... --model my-model --dataset-file data/visualneedle.jsonl --concurrency 20
# No-tool（单轮 VL）
python run_visualneedle_eval.py --config ... --model my-model --dataset-file data/visualneedle.jsonl --no-tools --concurrency 20
# Crop-black 消融
python run_crop_black_ablation.py --config ... --model my-model --dataset-file data/visualneedle.jsonl --concurrency 20
# Text-only 基线
python run_text_only_eval.py --config ... --model my-model --dataset-file data/visualneedle.jsonl --concurrency 20
```

**模型配置**：全 YAML 声明，解析顺序为 `--config` CLI → `VISUALNEEDLE_MODEL_CONFIG` 环境变量 → `visualneedle_agent.py` 内编译的默认路径。支持的 `model_type` 含 `oai | claude | qwenvl_oai | ...`；工具在 `tools:` 下配置（示例：`image_zoom_in_tool_reason`，带 `bbox_order: xyxy`、`bbox_scale: 1000`）。

**并发模式（互斥，`--concurrency` 优先）**：`--concurrency N`（默认 `0` 关闭，API-bound 场景典型值 **20–50**）或 `--workers N`（默认 `1`，多进程）。

**其它关键 CLI**：`--dataset-file`、`--limit N`、`--start N`、`--indices 0,12,99`、`--log-dir`、`--auto-log-dir / --no-auto-log-dir`、`--verbose`、`--ordered-output / --no-ordered-output`、`--judge / --no-judge`、`--no-tools`、`--multiple-choice`、`--max-null-retries N`（默认 `3`）、`--data-format local|url|file`（默认 `local`）、`--retry-dir DIR`。

**图像投递模式**：`local`（默认，磁盘读入并 base64 编入请求）、`url`（`image_url` 字段透传；工具输出图经 `object_storage_upload.py` 上传，需自行实现 `upload_image_to_object_storage(local_path, object_key) -> url`）、`file`（Claude / Anthropic Files API，图片上传一次作为 `file_id` 块并跨重试复用）。

**输出产物**：

```text
<log-dir>/
├── results_summary.txt           # 人类可读 per-sample 摘要
├── structured_results.json       # 机器可读完整结果
├── sample_000000/
│   ├── sample_000000.log         # 完整推理轨迹
│   ├── original_image.*          # 输入图副本
│   └── (tool output images)
└── html/                         # generate_html_report.py 生成
    ├── index.html
    └── sample_000000.html
```

`structured_results.json` 每样本含 prediction、gold answer、correctness、**带 bbox 坐标的 tool steps**、token 用量与 judge 结果。HTML 报告可用 `--max-image-bytes`（默认 `12000000`）、`--allow-remote-images`、`--limit` 控制。

**Makefile 目标**：`install`、`install-dev`、`test`、`cli-help`、`smoke-config`（用 `--limit 0` 校验配置加载）、`clean`。

**Python API**：`from visualneedle_eval import evaluate_split`，返回 `(correct, match_correct, total, judge_used, judge_stats, tool_stats)`，`mode` 取 `"standard" | "crop_black" | "text_only"`。

#### 评测 / 数据集规模

- **题目数**：**300 questions**（README 首段与徽章区均写 "300 Questions"）。
- **能力维度**：**5 项** —— OCR、color、entity、spatial、occlusion（"OCR, color, entity, spatial, and occlusion"）。
- **诊断设置**：**4 种** —— text-only、no-tool、tool-enabled、crop-black。
- **被测模型**：**9 个**主流 MLLM（"The paper evaluates nine mainstream MLLMs on all 300 questions"，徽章区写 "**9 MLLMs** Evaluated in the paper"）。
- **结果（README「Results」表）**：

| Evaluation setting | Best accuracy | 结论 |
|---|---:|---|
| Text-only | **< 10%** | 措辞本身无法可靠揭示答案 |
| No-tool | **17.89%** | 单次整图观察对多数局部线索不足 |
| Tool-enabled | **56.01%** | 裁剪帮助显著，但证据获取仍然困难 |
| Human majority vote | **63.00%** | 即便可用 crop，模型仍存在实质性差距 |

- 徽章区另标注 "**Best Tool-Enabled 56.01%**" 与 "**63.00% Human**（Majority-vote baseline）"，与上表一致。
- **数据集发布状态**：尚未发布。README 明确写 "**Dataset release — coming soon.**" —— 将在此处发布 300 题 benchmark、源图、标注、bounding box 与评测切分。评测 harness 已公开，数据发布后把 JSONL 标注文件传给 `--dataset-file` 即可（格式见 `docs/evaluation.md#dataset-format`）。
- **数据格式（`docs/evaluation.md`）**：每行一个 JSON 对象，必填 `question`、`answer` 与一个图像字段（`image` 本地绝对路径 / `image` base64 data URI / `image_url` 远程 URL）；可选 `answer_letter`、`prompt_sn`、`question_en`、`answer_en`。
- **硬件要求**：**README 未说明**（harness 本身只是 HTTP 客户端，算力在模型 provider 侧）。

#### 论文 / arXiv 链接、License

- 论文：arXiv [2605.26380](https://arxiv.org/abs/2605.26380) — *VisualNeedle: Benchmarking Active Visual Search in Information-Dense Scenes*（Jingru Chen、Yiming Liu、Mingtao Chen、Sijie Chen、Richeng Xuan、Liang Yang、Zhichao Hu、Fanyang Lu，2026）。
- License：**Apache-2.0**（含第三方组件例外与声明）。

#### 亮点与局限

**亮点**：
1. **Crop-black 消融设计精妙**：保留交互格式与 crop 尺寸、只替换返回像素，从而把「格式合规」与「真的用上了返回证据」这两件常被混淆的事分开度量。这是本报告中最值得直接借鉴的评测设计。
2. **三条反捷径要求（question-alone / global-view / returned-evidence）是**题目准入标准**，直接在构造阶段排除掉可被语言先验或整图语义绕过的样本。
3. **审计产物完整**：每样本日志、**中间工具图像**、归一化 crop 坐标、token 用量、结构化 JSON、自包含离线 HTML 报告 —— 结果可逐题复查。
4. 复现口径明确要求四件套一起报告（exact-match score、final score、judge-use count、tool-use statistics），避免只报一个好看的数字。

**局限**：
1. **数据集尚未发布**（"coming soon"），当前只能跑 harness，**无法真正复现 300 题结果**。
2. star 数为 0、README 无带日期的 News 段落，社区采用度与迭代活跃度信息缺失。
3. `text-only` 的 "< 10%" 是一个带小于号的上界表述，README 未给精确值。
4. 图像投递的 `url` 模式需要自行实现对象存储上传回调，开箱即用只覆盖 `local` 与 Claude Files 两条路。

#### 对我们可复用的工程实践

1. **「同格式 + 空白证据」消融法**：任何带工具/检索的 agent 评测都可以加一个「工具返回同形状的空内容」对照组，用它证明收益来自**信息**而非**流程**。这条思路可以平移到文本检索（返回同长度无关文档）与代码检索（返回同大小错误文件）。
2. **审计级产物落盘**：per-sample 目录 + 中间工具图像 + 归一化坐标 + token 用量 + 结构化 JSON + 离线 HTML。这套组合让评测结果可被非作者独立复核，是内部评测平台值得直接照抄的目录结构。
3. **`--retry-dir` 断点重跑 + 有界 null 重试**：反复调用外部模型的评测必然出现 `pred=None`，`--max-null-retries`（默认 3）与按索引重跑是低成本高收益的健壮性设计。

---

### 2.4 `RepoProbe`（评测 / 仓库级代码理解）

#### 项目定位与要解决的问题

RepoProbe 的核心问题是标题所写：**"Can an LLM explain how a codebase actually works — not just patch it?"** 它是 **ASE 2026**（第 41 届 IEEE/ACM 自动化软件工程国际会议，慕尼黑，2026 年 10 月 12–16 日）论文 *RepoProbe: Benchmarking Architecture-Aware Repository Comprehension with Checklists* 的评测代码与数据。

它针对的痛点很明确：**缺陷中心型 benchmark 提供了强定位线索**（stack trace、error log、文件名），因而"let models shortcut genuine comprehension"。RepoProbe 的题目改从**真实 GitHub Discussions** 中抽取 —— 关于「代码库究竟如何工作」的开放式架构提问。

判分方法也针对另一痛点：用 **Checklist-Based Verification Protocol** 把开放式回答分解成**原子化、带权重、可验证的技术事实**，以取代不透明的标量 LLM-as-a-Judge 打分。

#### 核心架构 / 方法

**规模指标（README「At a Glance」原文）**：

| 项目 | 值 |
|---|---|
| Repositories | **50** |
| Questions | **500**（每仓库 3–30 题，中位数 **8**） |
| Programming languages | **15** |
| Question categories | Project Architecture / Business Logic / Implementation Details |
| Grading | Weighted checklist items with tiered rubrics, scored by an LLM judge |
| Agent scaffolding | **Docker-isolated, identical for every model under test** |
| Models evaluated in the paper | **20**（13 closed-weight + 7 open-weight） |

**评分方法**：每题 **10 分** —— **9 分给 knowledge 项**（事实与技术正确性）、**1 分给 clarity 项**（解释质量）。`Overall Performance` 是总分除以 10 的均值，`Perfect Solve Rate` 是拿到满分 10/10 的题目比例。README 强调所有数值均为百分比。

**两条驱动设计的关键发现**：
- **Fluency outruns correctness（流利度跑赢正确性）**：所有模型 clarity 得分远高于 knowledge —— GPT-5.2 的 clarity 达 **84.7**，而 knowledge 仅 **60.3**。答案读起来权威，但技术实质落后，**而这正是标量 LLM-as-a-Judge 打分倾向于奖励的**。
- **Full credit is rare（满分很少见）**：最高 Perfect Solve Rate 为 **27.5%**，即对前沿模型而言，大约**四分之三**的题目至少有一条 checklist 项未满足 —— 仓库级理解远未饱和。

**关键结果表（README「Key Results」）**：

| Model | Overall | Knowledge | Clarity | Perfect Solve |
|---|---|---|---|---|
| GPT-5.2 | **62.7** | **60.3** | 84.7 | 26.0 |
| Claude Opus 4.6 | 62.1 | 60.2 | 79.0 | **27.5** |
| GPT-5.4 | 60.4 | 57.6 | **85.6** | 24.2 |
| Claude Sonnet 4.6 | 60.2 | 58.3 | 78.2 | 27.2 |
| GLM-5 *(best open-weight)* | 54.8 | 52.9 | 72.2 | 21.3 |

README 说明「the full table of 20 models is in the paper」。

**Agent Scaffolding 设计（README + `docs/AGENT_INTEGRATION.md`）**：RepoProbe **评的是 agent 而不是裸模型 API**：每题由一个先探索仓库再作答的 agent 回答。harness 是 **scaffolding-agnostic** 的 —— 它只定义一个容器契约，然后启动 `agent_configs/agent.json` 声明的任何东西（自带 loop、开源框架或厂商 CLI，均无需改动 harness）。默认接好一个最小参考 agent `agent_configs/example_agent.py`，使 benchmark 开箱可跑。

**容器契约四条硬要求**：

| # | 要求 |
|---|---|
| 1 | 被测仓库位于 `/app/repo`，也是命令启动时的工作目录 |
| 2 | 问题在 `/tmp/prompt/prompt.txt`，**只读挂载** |
| 3 | 最终答案必须写到 **stdout，且 stdout 不能有别的任何东西**（其它内容会并入被打分的答案；trace 请写 stderr） |
| 4 | 成功退出码为 **0**，非零退出把该题标记为 failed |

被测模型通过环境变量 `REPOPROBE_MODEL` 提供。

**`agent.json` 字段（README 给出默认值，实测文件内容一致）**：`command`（容器内启动你的 agent 的 shell 命令，必填）、`start_model_gateway`、`gateway_startup_seconds`、`passthrough_env`（转发进容器的宿主环境变量，通常是凭证）、`env`（脚手架需要的固定变量）、`stdout_filters`（打分前从 stdout 丢弃脚手架噪声的正则，如 CLI 启动 banner）。实测 `agent_configs/agent.json` 中 `REPOPROBE_MAX_TURNS` 为 **`"30"`**，`passthrough_env` 含 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_TIMEOUT`、`OPENAI_ORG_ID`、`OPENAI_SCORING_API_KEY`、`OPENAI_SCORING_BASE_URL`。放在 `agent_configs/` 的文件会被复制进镜像 `/app`。

**内置 model gateway**：`start_model_gateway` 为 true 时，`agent_configs/gateway.py` 监听 `http://127.0.0.1:8730/api/chat`，说 OpenAI chat-completions 格式。三个好处：**统一模型访问**（gateway 从 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 解析凭证，并用 `REPOPROBE_MODEL` **覆盖 agent 自己发来的模型名**，因此硬编码模型的脚手架仍会评到目标模型）、**统一重试与超时**、**turn 计数**（gateway 每次模型调用记一行日志，结果里的 `rounds` 指标即由此计算；**绕过 gateway 则 `rounds` 为 0，但分数不受影响**）。**不支持 streaming**，须发 `"stream": false`。

#### 关键目录与文件（README 文件树，已校验存在的关键文件）

```text
.
├── README.md
├── LICENSE                              # Apache-2.0（源代码）
├── NOTICE
├── CITATION.cff
├── repos_info.json                      # 50 个仓库的元数据与快照信息
├── requirements.txt
├── dataset/                             # 每仓库一个 CSV
│   ├── LICENSE                          # CC BY 4.0（benchmark 数据）
│   ├── NOTICE
│   ├── adk-python.csv
│   ├── ...                              # 共 50 个 CSV
│   └── yasb.csv
├── docs/
│   └── AGENT_INTEGRATION.md             # 如何接入自己的 agent scaffolding
├── candidate_repos/                     # 运行时目录，git-ignored，克隆下来的仓库
├── scripts_agent/
│   ├── fetch_repos.sh                   # 克隆/更新仓库并生成 repomix 摘要
│   └── eval_models_multi_repo.sh        # 跨全部仓库跑 agent 评测
├── evaluator.py                         # 主评测编排器（Docker-based agent runner）
├── scorer.py                            # 基于 checklist 的 LLM judge 打分
├── cache.py                             # agent 答案与分数的缓存层
├── dataset_loader.py                    # CSV 数据集加载器
├── fetch_repos.py                       # 仓库克隆与 repomix 生成
├── prompt_templates_en.py               # 打分 prompt 模板
└── agent_configs/
    ├── Dockerfile.base                  # 基础镜像（Python、Node.js）
    ├── Dockerfile                       # 每仓库镜像（拷入仓库 + 配置）
    ├── build_base_image.sh
    ├── agent.json                       # 声明如何启动被测 agent
    ├── example_agent.py                 # 参考脚手架，可替换
    ├── gateway.py                       # 容器内 OpenAI-compatible 模型网关
    ├── gateway.sh
    └── model_client.py                  # OpenAI-compatible API 客户端
```

`requirements.txt` 实测仅 5 行：`fastapi`、`httpx`、`openai`、`pandas`、`uvicorn`。

**数据集格式（`dataset/` 每仓库一个 CSV，列定义）**：`repo_name`、`question_id`（格式 `<repo>-<index>`）、`discussion_id`（GitHub Discussion node ID，可回溯原始讨论串）、`taxonomy`（Project Architecture / Business Logic / Implementation Details）、`difficulty`、`question`、`answer`（讨论维护者的参考答案）、`checklist`（带权重与分级评分细则的 checklist 项）。

**checklist 条目示例（README 原文）**：

```text
(3 points) Bug Fix Confirmation and Commit Identification
- 3 points: Correctly confirms the fix is present in 8.0.2 and identifies the commit hash
- 1 point: Confirms the fix exists but does not provide the correct commit
- 0 points: Fails to confirm, or asserts the bug is unfixed
```

#### 安装与运行方式

**前置条件**：Python **3.11+**、**Docker**、**Node.js 20+**（供 `repomix`）、一个 OpenAI-compatible API endpoint 与 key。

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 构建基础 Docker 镜像
cd agent_configs && bash build_base_image.sh

# 3. 拉取评测仓库（按 repos_info.json 中记录的 snapshot commit 检出，并生成 repomix 摘要）
export DATASET_DIR=./dataset
bash scripts_agent/fetch_repos.sh

# 4. 运行评测
export OPENAI_API_KEY=<your-api-key>
export OPENAI_BASE_URL=<your-api-base-url>   # optional
export MODELS=gpt-5.2
bash scripts_agent/eval_models_multi_repo.sh
```

README 特别强调 **snapshot pinning 的重要性**："the questions and checklists were written against those snapshots, so evaluating against current upstream HEAD would score models on code the dataset never described. A repository whose snapshot commit has disappeared upstream is reported at the end of the run and should be excluded rather than evaluated at a different commit."（`GIT_DEPTH` 默认 `0` 即完整克隆；仓库**不随 benchmark 分发**，从原始来源拉取。）

**环境变量全表（README）**：`OPENAI_API_KEY`（必需）、`OPENAI_BASE_URL`（默认 `https://api.openai.com/v1`）、`OPENAI_SCORING_API_KEY` / `OPENAI_SCORING_BASE_URL`（打分 judge 独立凭证，默认回退到前者）、`MODELS`（默认 `gpt-5.2`，逗号分隔）、`SCORING_MODEL`（默认 `gpt-5.2`）、`MAX_WORKERS`（默认 `5`，并发评测容器数）、`SCORING_WORKERS`（默认 `10`）、`NUM_ITERATIONS`（默认 `1`，重复运行做方差分析）、`AGENT_CONFIG`（默认 `agent_configs/agent.json`）、`AGENT_TIMEOUT`（默认 `3600` 秒，每题容器超时）、`OPENAI_TIMEOUT`（默认 `3600` 秒）、`DATASET_DIR`（默认 `./dataset`，也可传单个 CSV 只评一个仓库）、`CANDIDATE_REPO_DIR`（默认 `./candidate_repos`）、`REPO_JSON_PATH`（默认 `./repos_info.json`）、`CACHE_DIR`（默认 `cache_experiments`）、`OUTPUT_DIR`（默认 `agent_results`）、`GIT_DEPTH`（默认 `0`）。

**容器内 harness 设置**：`REPOPROBE_MODEL`（被测模型）、`REPOPROBE_MAX_TURNS`（来自 `agent.json`）；内置 gateway 遵循 `REPOPROBE_GATEWAY_HOST`、`REPOPROBE_GATEWAY_PORT`、`MAX_ROUNDS`；参考 agent 读 `REPOPROBE_GATEWAY_URL`。

**接入自定义脚手架（`docs/AGENT_INTEGRATION.md`）**：需要额外软件时改 `agent_configs/Dockerfile.base` 并重建镜像（基础镜像已提供 **Python 3.11** 与 **Node.js 20**）。跑全量前的 4 步 checklist：① `bash agent_configs/build_base_image.sh` 成功；② 单仓库跑出非空答案（`MODELS=<model> DATASET_DIR=./dataset/Apollo.csv bash scripts_agent/eval_models_multi_repo.sh`）；③ 检查 `cache_experiments/agent_evaluation/` 下的缓存答案，若含 banner 或进度输出则往 `stdout_filters` 加正则；④ 检查结果里的 `rounds`，gateway 开启而 `rounds` 为 0 说明 agent 没走 gateway。

#### 评测 / 数据集规模

- 仓库 **50** 个、题目 **500** 道（每仓库 3–30 题，中位数 8）、覆盖 **15** 种编程语言、**3** 个 taxonomy 类目。
- 论文评测 **20** 个模型（13 闭源 + 7 开源）。
- 每题 **10 分** = 9 分 knowledge + 1 分 clarity。
- 指标名：`Overall Performance`、`Knowledge`、`Clarity`、`Perfect Solve Rate`，另有 gateway 计数的 `rounds` 指标。
- baseline 分数见上文「关键结果表」（GPT-5.2：62.7 / 60.3 / 84.7 / 26.0；Claude Opus 4.6：62.1 / 60.2 / 79.0 / 27.5；GPT-5.4：60.4 / 57.6 / 85.6 / 24.2；Claude Sonnet 4.6：60.2 / 58.3 / 78.2 / 27.2；GLM-5 开源最佳：54.8 / 52.9 / 72.2 / 21.3）。

#### 论文 / arXiv 链接、License

- 论文：arXiv [2608.04783](https://arxiv.org/abs/2608.04783)，发表于 **ASE 2026**（<https://conf.researchr.org/home/ase-2026>）。
- **双许可**：源代码 **Apache-2.0**（`LICENSE`）；benchmark 数据（`dataset/`、`repos_info.json`）**CC BY 4.0**（`dataset/LICENSE`）。
- 数据合规声明（README）：`question` 与 `answer` 字段源自公开可见的 GitHub Discussions，**版权仍属原作者**，每条记录保留 `discussion_id` 以便溯源与署名，作者可开 issue 要求移除；引用外部链接的记录由 LLM 改写以折入链接内容摘要，使任务**无需网络即可作答**；`taxonomy` 标签与 `checklist` 细则由 LLM 生成后经人类专家标注者验证；发布前已做个人数据清洗（GitHub 账号替换为 `@user`、邮箱替换为 `contact@example.com`、家目录路径中的账号名替换为 `user`，但保留代码层面的 `@` 用法与 `root` 等系统账号）。

#### 亮点与局限

**亮点**：
1. **评测对象是 agent 而非裸模型**，且 harness **完全 scaffolding-agnostic** —— 用一个极小的容器契约（仓库路径 / prompt 文件 / stdout 即答案 / 退出码）把「脚手架」与「评测逻辑」彻底解耦。这是本报告 7 个仓库中最干净的可复用接口设计。
2. **weighted checklist + tiered rubric** 把开放式答案拆成可逐项验证的原子事实，明确给出部分分（3/1/0）而非让 judge 自己揣摩，直接针对标量 LLM-as-a-Judge 的「奖励流利」偏差。
3. **9:1 的 knowledge/clarity 权重**把「说得漂亮」与「说得对」显式分离，并据此得出了 fluency-outruns-correctness 这个有说服力的结论。
4. **Docker 隔离 + snapshot commit pinning** 保证可复现性，并明确拒绝在快照丢失时改用其它 commit 评测。
5. 数据合规处理（个人数据清洗、CC BY 4.0 范围界定、NOTICE 说明）非常规范。

**局限**：
1. **论文使用的脚手架未随仓库分发**（"the experiments in the paper used the scaffolding described there, which is not redistributed here"），因此**无法直接复现论文中的 20 模型分数**。
2. README 自己承认分数**只在同一 scaffolding 下可比**："tool availability, turn budget and prompting all change how much of a repository a model actually inspects"，跨脚手架比较无意义。
3. 依赖 Docker、Node.js 20+ 与 `repomix`，且需要分别构建 base 与 per-repo 镜像，环境准备成本不低。
4. 50 个上游仓库需从原始来源克隆，快照 commit 可能已在上游消失（README 说明此类仓库会被报告并建议**排除**而非替换 commit），长期可复现性存在外部风险。
5. 评测依赖单一 LLM judge（默认 `gpt-5.2`）做 checklist 核验，README 未说明 judge 与人工标注的一致性校验数据。

#### 对我们可复用的工程实践

1. **容器契约四要素**（仓库在 `/app/repo` 且为 CWD、问题在只读 `/tmp/prompt/prompt.txt`、**stdout 只能是答案且 trace 走 stderr**、退出码 0 为成功）。这套极简契约让任何脚手架插拔式接入，是做内部 agent 评测平台时最该先定的接口。
2. **带模型名覆盖的 model gateway**：gateway 用环境变量 `REPOPROBE_MODEL` **覆盖 agent 自己发来的 model 名**，同时集中处理重试/超时，并用「每次调用记一行」顺带产出 `rounds` 指标。把「模型选择」和「调用可观测性」从脚手架里抽出来，是评测公平性的关键一招。
3. **tiered rubric 的部分分写法**：每个 checklist 项显式写出 3 分 / 1 分 / 0 分各自对应什么，而不是给一个 0–1 的模糊分。这条可以直接用于我们的技术文档/代码理解类评测集。

---

### 2.5 `ExplorationBench`（评测 / 探索能力）

#### 项目定位与要解决的问题

ExplorationBench 的定位句是 **"Measuring AI Systems' Exploration in Verifiable Alien Worlds"**。它要解决的问题是：「科学发现始于已知问题终结之处」，在那里 AI 系统必须**探索** —— 提出假设、设计实验、依据结果迭代。

关键方法是 **verifiable Alien Worlds（可验证的异世界）**：世界的规则是**可执行的**，所以每个答案被精确校验；规则又**与熟悉的知识冲突**，所以单靠 recall 解不了题。

**四轮探索协议（README「Overview」）**：系统从一个**有缺陷的手册（flawed manual）**和一组**固定 worked examples** 出发；在**四轮**中用**单一工具**探测世界，世界以**确定性反馈且不给解释**作答；每一轮之后，一份**关闭工具（tools disabled）的会话副本**报告它认为成立的规则，并对每个 held-out 任务作答**三次**；随后该副本被丢弃，**因此测试题永不泄漏进探索过程**。里程碑记为 M<sub>0</sub>–M<sub>4</sub>。

#### 核心架构 / 方法

**两个 Sandbox（README 原文要点）**：

- **AlienCode** —— 在一个小计算语言中做程序合成，语言里**看起来熟悉的算符遵循隐藏语义**：`SHATTER` 是乘法、`EMIT(100)` 打印 `127`、且**没有任何原语做加法**。**31 个 discovery targets + 70 个 held-out tasks**；每个程序由解释器在**五个私有输入**上评分。
- **AlienLogic** —— 在带 **24 条被打补丁（patched）推理规则**的一阶自然演绎系统里做形式化证明。**70 个 held-out 定理，其中 23 个不可证明**；每个证明由 proof checker 校验，**不可证明的定理只有在系统拒绝证明它时才得分**。

**Stats（README 原文）**：2 sandboxes · 55 discovery targets · 140 held-out tasks · 10 frontier systems · 每个 system 与 sandbox **3 条独立轨迹** · **every verdict from an interpreter or a proof checker, with no LLM judge**（不使用 LLM judge）。

**发布策略（README「What Is Released」）**：因为"规则公开的世界测的是 recall 而非 exploration"，**两个评测世界的规则与测试实例保持私有**。公开的内容包括：
- **按论文运行时原样发布的评测代码**：harnesses、protocol、scoring、control conditions、three answers per task、open-book answering、rescoring、audits。只有 world-specific 模块不同，`explorationbench/sandboxes/code/world_data.py` 列出了每个有差异的值。
- **两个评测引擎**：AlienCode 解释器前端与 AlienLogic proof checker，规则被抽成 **rule packs**。装入私有规则后，它们能对实验日志中**全部 7,497 个证明与 7,651 个程序**复现判定。
- **两个公开 demo world**，用同一流程构建、使用同一语言、手册、证明格式、协议与任务族，但其规则是新的、且与评测世界的每条规则都不同。
- 两个 demo world 的 **worked examples**（位于 `examples/`），以及论文图表与主表背后的 **analysis scripts**（位于 `analysis/`，它们读取私有实验日志，因此在这里只记录每个数字**如何**计算而非复现它）。

| | Demo world（已发布） | Evaluation world（私有） |
|---|---|---|
| AlienCode | 5 discovery targets · 4 worked examples · 5 held-out tasks | 31 discovery targets · 70 held-out tasks |
| AlienLogic | 3 side conditions · 5 worked examples · 5 held-out theorems（1 unprovable） | 24 patched rules · 70 held-out theorems（23 unprovable） |

**Framework track（来自 README 与实测配置/代码）**：README 提到 `explorationbench/frameworks/` 是 "framework-track interface"。实测 `config/eval.local.example.toml` 暴露了 `framework` 选项：**`baseline | reflexion | ace | evotest | gepa | agent_factory`**，`track` 选项：**`controlled | native | open`**（默认 `controlled`，注释说明 "Non-baseline frameworks are kept episode-local in the controlled track; native/open profiles are reported separately"），`budget_profile` 选项：**`c1 | 1x | 2x | 4x`**。实测 `explorationbench/frameworks/base.py` 定义了 `FrameworkSpec` dataclass（字段 `key`、`display_name`、`mechanism`、`official_url`、`default_track`、`official_compatibility`、`notes`）与 `ControlledFrameworkRuntime`（继承 `AgentClientRuntime`），后者为 **reflexion / ace / evotest / gepa / agent_factory** 各自注入一段方法性 scaffold 指令，并维护一个 **episode-local playbook**（记录 hypothesis、failed probes、reusable strategies，仅在 seen feedback 上更新，milestone 与 held-out test 期间视为冻结，**controlled track 无跨 run 记忆**）。其 docstring 明确："It deliberately does not import official framework packages by default, because controlled runs must keep the benchmark phase machine, tools, and metering intact."

#### 关键目录与文件

```text
ExplorationBench/
├── explorationbench/        evaluation code, with the public demo worlds installed
│   ├── sandboxes/code/      AlienCode: harness, protocol, interpreter front end, manual,
│   │                        demo rule pack and task set, world build
│   ├── sandboxes/logic/     AlienLogic: harness, protocol, proof checker, demo side conditions,
│   │                        demo episode, unprovability certificates
│   ├── common/              model client (routing, sessions, tool calls, closed-book copies) and runtime
│   ├── frameworks/          framework-track interface
│   └── scripts/             orchestration, control conditions, three answers per task,
│                            open-book answering, rescoring, audits, exports, HTML replays
├── analysis/                scripts behind the paper's figures and main table
├── examples/                worked examples from the demo worlds
├── config/                  template for model endpoints and credentials
└── assets/                  figures for this README
```

**README 引用的具体文件（已校验）**：`explorationbench/sandboxes/code/world_data.py`、`explorationbench/sandboxes/code/run_eval.py`、`explorationbench/sandboxes/logic/run_eval.py`、`explorationbench/scripts/answer_variance.py`、`explorationbench/scripts/logic_answer_variance.py`、`examples/render.py`、`config/eval.local.example.toml`、`LICENSE`。README 提到的模块命令还包括 `python3 -m sandboxes.code.build_world`、`python3 -m sandboxes.logic.episodes`、`python3 -m sandboxes.logic.certify`（在 `cd explorationbench` 之后执行）。

**实测 `world_data.py` 的内容**（这是公开 demo world 的真实数据）：`PUBLIC_DEMO_RULES = ("int_offset", "under_le", "strand_identity", "gauge_double")`；4 条 worked example（D01 `EMIT(32)`、D02 `EMIT(STRAND(UNDER(2, 2)))`、D03 `CRAFT count3()` + `GAUGE(xs)`、D04 `CRAFT product3(a, b, c)` 用 `PARE` 做两步乘法），每条按 `band` 分为 `apply / interact / engineer / algorithm`；另有 legacy seed 阶段 `SEED_TASKS`（S01–S04）与 `LOCKED_SEED_CODES`；以及 **5 条 ground-truth 规则槽** `R01 := (+ n 7)`、`R02 := (<= a b)`、`R03 := (get seq (+ i 1))`、`R04 := IDENTITY`、`R05 := (* (len seq) 2)`（对应 L1 字面量编码 / L3 比较运算 / L5 位置与结构 / L7 聚合）。文件还包含规则报告模板（要求严格 `RULE_ID := S-EXPRESSION` 格式、未发现写 UNKNOWN）与 `DIVERGENCE_PATTERNS`（用于识别探索笔记中「推理规则与手册背离」的中英文短语列表）。

#### 安装与运行方式

**环境要求**：Python **3.11+**。demo 检查只用标准库；调用模型需要 `httpx`；analysis 脚本需要 `numpy` 与 `matplotlib`。

**（1）检查 demo world（无需 API key）**：

```bash
cd explorationbench
python3 -m sandboxes.code.build_world     # 重建并检查 AlienCode demo task set
python3 -m sandboxes.logic.episodes       # 检查 AlienLogic demo episode
python3 -m sandboxes.logic.certify        # 认证其不可证明定理
cd ..
python3 examples/render.py                # 重新生成 examples/ 中的 worked examples
```

**（2）在 demo world 上评测模型**：harness 通过其 **OpenRouter route** 触及任何 OpenAI-compatible chat-completions endpoint。

```bash
pip install httpx
cd explorationbench
export OPENROUTER_API_KEY=<your_api_key>
# 默认走 OpenRouter；也可指向其它 OpenAI-compatible endpoint：
# export OPENROUTER_CHAT_URL=https://api.example.org/v1/chat/completions

# AlienCode：四轮探索，之后每个 milestone 对每题作答三次
EVAL_TOOL_MODE=1 ALIENCODE_PROTOCOL_V2=1 python3 -u sandboxes/code/run_eval.py \
    --model openrouter/<model> --short <name> --run-id demo_n1
python3 -u scripts/answer_variance.py --run demo_n1 --trials 3 --milestones 0,1,2,3,4

# AlienLogic
EVAL_TOOL_MODE=1 ALIENLOGIC_PROTOCOL_V2=1 python3 -u sandboxes/logic/run_eval.py \
    --model openrouter/<model> --model-short <name> --defer-tests --skip-pre --run-id demo_l1
python3 -u scripts/logic_answer_variance.py --run demo_l1 --trials 3
```

结果、transcript 与 **HTML replay** 写到 `logs/` 下。README 明确："Demo-world scores are for development only; they are not comparable to the leaderboard."

**凭证与运行设置**也可放在本地文件：复制 `config/eval.local.example.toml` 为 `explorationbench/eval.local.toml`（git 忽略），或用 `EVAL_LOCAL_CONFIG` 指向别处；**环境变量优先于文件**。实测 toml 的其他可配项：`model`（默认示例 `api_aws_third_anthropic.claude-opus-5`）、`provider`、`reasoning_effort`（`low|medium|high|xhigh|max`）、`thinking_display`（`summarized|omitted`）、`max_tokens`（示例 `4096`）、`timeout`（示例 `600` 秒）、`max_retries`（示例 `4`）、`max_empty_retries`（示例 `2`）、`cache_ttl`（`5m`/`1h`）、`[output] dir`、`[endpoints]`，以及 `[credentials]` 中的 `model_eval_api_id` / `model_eval_api_key` 等。

**（3）在私有评测世界上评测**：README 明确 **"The two evaluation worlds are not public... we do not currently provide their full contents to individual researchers."** —— 需**联系作者**，由他们按 leaderboard 同一协议运行。

#### 评测 / 数据集规模与分数

**规模**：**2** 个 sandbox、**55** 个 discovery targets、**140** 个 held-out tasks、**10** 个前沿系统、每系统每 sandbox **3** 条独立轨迹；AlienCode 31 targets / 70 held-out tasks（每题 5 个私有输入）；AlienLogic 70 held-out theorems（23 不可证明）/ 24 条 patched rules。实验日志规模：**7,497 个证明**与 **7,651 个程序**。

**指标名**：held-out accuracy（%）、**Best@3**（三条独立轨迹中最好的一条）、**Mean@3**（三条的均值）、M<sub>0</sub>（仅凭 worked examples 时 Best@3 轨迹的准确率）。"Every score is the mean of three answers per task, and every system runs at its highest reasoning setting."

**AlienCode leaderboard（M<sub>4</sub>，%）**：

| Rank | System | M<sub>0</sub> | Best@3 | Mean@3 |
|:---:|---|---:|---:|---:|
| 1 | Claude Opus 5 | 3.8 | **89.0** | 73.5 |
| 2 | GPT-5.6 Sol | 11.0 | 88.1 | 79.5 |
| 3 | Gemini 3.8 Flash | 1.4 | 79.0 | 39.7 |
| 4 | Kimi K3 | 2.4 | 79.0 | 38.7 |
| 5 | Grok 4.6 | 4.8 | 69.0 | 53.2 |
| 6 | Qwen3.8-Max | 2.4 | 65.7 | 64.0 |
| 7 | Hy4 preview | 0.5 | 63.3 | 30.2 |
| 8 | DeepSeek-V4.1-Flash | 1.4 | 59.5 | 46.3 |
| 9 | Seed2.1 Pro | 2.4 | 40.0 | 24.8 |
| 10 | DeepSeek-V4-Pro | 1.9 | 12.9 | 7.3 |

**AlienLogic leaderboard（M<sub>4</sub>，%）**：

| Rank | System | M<sub>0</sub> | Best@3 | Mean@3 |
|:---:|---|---:|---:|---:|
| 1 | Grok 4.6 | 42.9 | **81.0** | 72.1 |
| 2 | Claude Opus 5 | 43.3 | 79.5 | 75.9 |
| 3 | Qwen3.8-Max | 45.2 | 77.6 | 74.3 |
| 4 | GPT-5.6 Sol | 51.0 | 76.7 | 75.1 |
| 5 | Hy4 preview | 35.2 | 76.2 | 69.2 |
| 6 | DeepSeek-V4-Pro | 33.8 | 75.2 | 66.3 |
| 7 | Kimi K3 | 45.2 | 74.3 | 69.4 |
| 8 | DeepSeek-V4.1-Flash | 51.4 | 73.8 | 60.6 |
| 9 | Seed2.1 Pro | 37.6 | 69.0 | 59.8 |
| 10 | Gemini 3.8 Flash | 45.7 | 59.5 | 59.0 |

**Key Findings（README 原文要点，含其引用的数字）**：
1. **探索本身（而非 recall 或单纯思考）产生了任务所需的知识**：没有 AlienCode 轨迹在仅有 worked examples 后超过 **15.7%**；四轮后最好达到 **89.0%**。**同样轮数但无环境反馈**则停留在 **0.5–11.0%**。
2. **系统自己设计实验时探索效果最好**：把系统自己最好的 probes 重放给它（证据完全相同），AlienCode 中仍有 **10 个系统中的 9 个**是自主探索更好，**中位数领先 17.4 分**。
3. **两个世界卡住的位置不同**：AlienLogic 中"直接告知规则"可得 **96–100%**，高于任何系统的最好轨迹，因此瓶颈在**发现**；AlienCode 中最好轨迹对 **10 个系统中的 7 个**优于事先告知规则，因此瓶颈在**使用**。
4. **知道规则不等于会用对**：最终规则报告正确陈述了所需规则的任务，仍只有 **73.4%** 被解出。
5. **单一分数掩盖了很多东西**：同一预算下同一系统的轨迹可相差数十分（Kimi K3 在 AlienCode 为 **5.7–79.0%**）；一个世界的排名几乎无法预测另一个世界的排名（**Spearman 0.35**）；且增益会以跳跃方式到来，后续轮次还可能回退。

#### 论文 / arXiv 链接、License

- 论文：arXiv [2609.30199](https://arxiv.org/abs/2609.30199) — *ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds*（作者含 Ming Zhang、Zhenghao Xiang、Peizhong Gao、Yujiong Shen、Yuhui Wang、Zhonghan Yue、Shihan Dou、Zhangyue Yin、Junjie Ye、Shichun Liu、Weihuang Zheng、Jiahao Chen、Jiayi Chen、Hongzhang Liu、Jiaqi Shao、Tao Gui、Qi Zhang、Xuanjing Huang、Suncong Zheng、Maxm Pan，2026）。
- 官网/leaderboard/blog：<https://explorationbench.com>、<https://explorationbench.com/#leaderboard>、<https://explorationbench.com/blog/>。
- 系列相关工作：EvaLearn（NeurIPS 2025）、`Tencent-Hunyuan/CL-bench`、ExplorationBench（README 自称「本系列第三个 benchmark」）。
- License：**Apache-2.0**（`LICENSE` 限定为 "Code of ExplorationBench"）。
- 联系：Ming Zhang（mingzhang23@m.fudan.edu.cn）、Maxm Pan（maxmpan@tencent.com），署名单位为 Fudan University · Hunyuan Team, Tencent · Tsinghua University。

#### 亮点与局限

**亮点**：
1. **「可执行规则 + 与常识冲突」的世界设计**从根上排除了 recall 与语言先验，且判定完全由解释器 / proof checker 给出，**不用 LLM judge**（这是本报告 7 个仓库中唯一完全无 LLM judge 的评测）。
2. **milestone 机制（M<sub>0</sub>–M<sub>4</sub>）+ 关闭工具的副本作答**极具巧思：既能在探索过程中反复测量「当前认为的规则」和「留出任务表现」，又通过**丢弃副本**保证测试题不泄漏进探索。
3. **控制实验设计完整**：同轮数无反馈对照、把自己最好的 probes 重放给自己、open-book（事先告知规则）对照、每题三答取 Best@3/Mean@3 —— 这些控制条件使得"探索贡献了多少"成为可量化的结论而非叙事。
4. **附带真实工程质量**：解释器与 proof checker 被抽成 rule packs，装入私有规则后能复现日志中 7,497 个证明与 7,651 个程序的判定 —— 即"引擎公开、数据私有"的可信度方案。
5. **framework track** 把 reflexion / ace / evotest / gepa / agent_factory 等方法统一到 `controlled | native | open` 三档 track 与 `c1 | 1x | 2x | 4x` 预算档下比较，且明确说明 controlled track 是 episode-local、不 import 官方框架包，保证 phase machine、tools 与 metering 不被破坏。

**局限**：
1. **评测世界完全私有**（"we do not currently provide their full contents to individual researchers"），只能联系作者代跑，**无法自行复现 leaderboard**；公开的 demo world 分数明确「不可与 leaderboard 比较」。
2. 公开仓库里 `sandboxes/code/world_data.py` 的注释与部分 seed prompt 含中文，而规则报告模板要求 `RULE_ID := S-EXPRESSION` 的英文风格格式，混用两套语言约定（这是实测文件的客观情况）。
3. 部分 README 声明的脚本路径与实测不一致：README 在 `cd explorationbench` 之后写 `python3 -u scripts/answer_variance.py`，而实测该文件位于 `explorationbench/scripts/answer_variance.py`（即 `cd explorationbench` 后应为 `python3 -u explorationbench/scripts/answer_variance.py`）。
4. `common/` 目录在 README 里被描述为单一模块，实测代码显示它实际是 `common/agent_client/`（含 `types`、`JsonlTraceStore`、`ToolDefinition`）与 `common/agent_runtime/`（含 `closed_book`、`trace`、`ArtifactSnapshot`、`PhaseContext`）两个子包。
5. 主要结果依赖少量系统（10 个）与 3 条轨迹，README 自己也强调「单一分数掩盖了很多东西」。

#### 对我们可复用的工程实践

1. **里程碑式测量 + 关闭工具的副本**：在长时探索/研究型 agent 的评测中，每个阶段复制一份会话、**关掉工具**让它总结当前信念并回答留出题，然后**丢弃副本**。这套机制同时解决了「过程可观测」与「测试集不泄漏」两个矛盾需求。
2. **控制条件三件套**：①同轮数但无环境反馈；②把 agent 自己的最优 probes 重放给它；③open-book 事先告知规则。三者把「探索带来的增益」与「推理/先验带来的增益」干净地分开 —— 任何声称"我的 agent 会自己探索"的评测都该补这三组对照。
3. **「规则引擎公开、规则数据私有」的可信度方案**：把解释器/checker 抽成 rule pack 开源，把规则值与测试实例私有，并**用实验日志规模证明引擎能复现全部判定**（7,497 证明 / 7,651 程序）。这是在"必须保密防污染"与"必须可审计"之间取得平衡的实用做法，值得用于我们内部的私有评测集。

---

### 2.6 `PlanningBench`（评测 + 数据构建框架）

#### 项目定位与要解决的问题

PlanningBench 是一个**合成规划 benchmark 与数据构建框架**，用于评测与训练 LLM 完成**复杂、文本化的规划任务**。它关注的核心问题是：模型能否把**目标、约束、资源、时间窗、依赖、优先级与优化目标**协调成一个**可执行且可验证**的计划。

它与「固定的人工撰写规划样例集合」的区别在于：PlanningBench 把真实规划场景**抽象成可复用的任务分类学、约束族与难度因子**，并且被测的是**完整计划在耦合约束下的成功**，而不只是局部需求是否满足。

一个重要的定位说明：题目是 **self-contained** 的 —— "all information needed to construct and verify a plan is provided in the input"，每道实例包含一个规划问题与对应的**验证 checklist**，因此可以对模型输出做约束满足与目标质量的评测。

#### 核心架构 / 方法

**任务分类学**：整个任务空间被组织为 **6 个高层规划族、30 多个具体任务类型**，以及多个 subtask 变体。分类学**按规划结构而非表层应用域**组织，覆盖 temporal conflicts、resource assignment、coverage and fairness、route coordination、production dependencies、emergency response priorities 等规划难度来源。

**六个高层族（README 原文）**：
- **Scheduling and Timetabling**：时间冲突、时间窗、执行顺序。
- **Project and Production Operations**：里程碑、依赖、产能、连续性。
- **Routing and Travel**：路线选择、时空协调、换乘。
- **Emergency Response and Public Service**：时效性、优先级分配、重规划。
- **Allocation and Matching**：兼容性与容量限制下的资源指派。
- **Shift and Workforce Scheduling**：覆盖、轮班公平性、可用性。

**数据构建框架（constraint-driven synthesis pipeline）**：把真实规划场景抽象为任务与约束分类学 → 采样 task-constraint 配置 → 生成自包含规划问题 → 附上验证 checklist / rubric → 自动过滤 + 人工质量控制复核。pipeline 分两个主要阶段：

1. **Task and constraint taxonomy construction**：代表性规划场景被归纳为 task families、concrete tasks、subtask variants、general constraints、task-specific constraints 与 specialized **stateful** constraints。
2. **Constraint-driven synthesis and difficulty enhancement**：**Generator** 造候选规划实例，**Responder** 尝试求解，**Critic** 依据 checklist 验证输出。已解/未解实例的反馈被用于**把采样调整到合适的难度**。

README 说明该设计支持跨任务、约束与解结构的**可控多样性**；难度可通过 **constraint tightness、resource scarcity、objective conflicts、subtask dependencies、global coordination requirements** 等因子调节。

#### 关键目录与文件（README 完整给出，实测确认）

```text
PlanningBench/
├── data/
│   └── PlanningBench-eval.jsonl         # 467 synthetic evaluation instances
├── asset/
│   ├── planningbench-logo.png           # PlanningBench benchmark logo
│   ├── hunyuanlogo.png                  # Tencent Hunyuan logo
│   ├── cn_ruc_logo.png                  # Renmin University of China logo
│   ├── figure-data.png                  # Task taxonomy and data distribution figure
│   └── figure-framework.png             # Data construction framework figure
├── LICENSE-PlanningBench.txt
└── README.md
```

**实测补充事实**：`data/PlanningBench-eval.jsonl` 实际存在且包含 **467 行**（非空行亦为 467），与 README「467 synthetic planning instances」完全一致；单行 JSON 的顶层字段为 **`idx`**、**`messages`**（含 `role`/`content` 的对话数组）与 **`checklist`**（字符串形式的评分依据）；`README.md` 与 `data/PlanningBench-eval.jsonl` 均可通过 raw 路径直接获取。README 徽章链接的 `./LICENSE-PlanningBench.txt` 在 raw 路径下 **404**，而实际许可证文件为 **`LICENSE.txt`**（README 正文的「License」段落正确指向 `LICENSE.txt`）。

#### 安装与运行方式

**README 未提供任何安装与运行方式** —— 该仓库只发布数据与说明文档，**不含评测代码、跑分脚本或依赖清单**。README 未说明 Python 版本、依赖、GPU 或 Docker 要求。

#### 评测 / 数据集规模

- **实例数**：**467** 个合成规划实例（README 徽章写 "Data-467 Planning Data"，正文写 "releases **467 synthetic planning instances**"；实测 JSONL 为 467 行，双向印证）。
- **用途**：「**all 467 instances are intended for evaluation**」—— README 明确要求把它们当作 benchmark/test 数据而非训练数据。
- **任务空间**：6 个高层族、**30 多个**具体任务类型、多个 subtask 变体。
- **指标名**：**README 未说明**任何具体指标名或评分公式。README 只说明每道实例附带 verification checklists，用于「constraint satisfaction and objective quality」评测。
- **baseline 模型与分数**：**README 未给出任何 baseline 或分数**。
- **数据合规说明**：released data is written and checked manually，**不含** proprietary business data、user logs、customer data、production data 或 enterprise workflow records；human annotators 参与质量控制与复核，**但不从零创建**benchmark 实例。
- **外部事实（非 README）**：HuggingFace 数据集 `tencent/PlanningBench` 的 tags 含 `size_categories:n<1K`、`format:json`、`modality:text`、`arxiv:2605.20873`，`lastModified` 为 `2026-05-28T08:26:38.000Z`。

#### 论文 / arXiv 链接、License

- 论文：arXiv [2605.20873](https://arxiv.org/abs/2605.20873) — *PlanningBench: Generating Scalable and Verifiable Planning Data for Evaluating and Training Large Language Models*（Ziliang Zhao、Zenan Xu、Shuting Wang、Hongjin Qian、Yan Lei、Minda Hu、Zhao Wang、Shihan Dou、Zhicheng Dou、Pluto Zhou，2026）。
- 数据集：<https://huggingface.co/datasets/tencent/PlanningBench>。
- 合作方（README logo 区）：Tencent Hunyuan + Renmin University of China。
- License：**CC-BY-4.0**（`LICENSE.txt` 明示 "PlanningBench is licensed under License CC-BY-4.0. PlanningBench does not impose any additional restrictions beyond the terms of the license."）。注意：README 徽章指向的 `LICENSE-PlanningBench.txt` 路径不存在。

#### 亮点与局限

**亮点**：
1. **「按规划结构而非应用域」组织 taxonomy**，并把难度做成可调因子（约束紧度、资源稀缺、目标冲突、子任务依赖、全局协调）—— 这使数据可以**按难度梯度**生成，而不是靠人工标注难易。
2. **Generator–Responder–Critic 闭环 + 难度自适应采样**：用"已解/未解"作为反馈调节采样分布，这是一个简洁有效的自动难度校准机制。
3. **每道题自带验证 checklist**，使开放式规划任务可以离线、确定性地核验约束满足与目标质量。
4. **数据用途界定清晰**：467 条全部声明为评测集，明确不建议当训练数据使用；同时明确声明不含任何生产/用户/客户数据。
5. **规模自述与实测一致**（467 行），可信度较高。

**局限**：
1. **仓库没有代码**：无评测 harness、无依赖清单、无运行说明，指标名与评分方式 **README 未说明**，只能从 checklist 字段自行实现打分。
2. **没有 baseline 分数**，无法判断当前模型的水平区间，也无法验证 checklist 的判别力。
3. README 徽章中的 `LICENSE-PlanningBench.txt` 链接失效（实际为 `LICENSE.txt`）。
4. 数据为**中文题面为主**（实测首条实例为中文运输调度场景），README 未说明多语言支持或语言分布。
5. 提交历史很短（提交 feed 仅 2 条时间戳，最新 2026-05-29），项目处于初始发布状态。

#### 对我们可复用的工程实践

1. **「问题 + checklist」同体打包的数据格式**：把验证依据与题目放在同一条 JSONL 记录里（`messages` + `checklist`），使评测无需外部标注文件即可自包含运行。这个格式对我们自建规划/流程类评测集非常合适。
2. **按难度因子参数化生成任务**：把「约束紧度 / 资源稀缺 / 目标冲突 / 子任务依赖 / 全局协调」显式列为可调旋钮，比事后按分数分层更容易构造平滑的难度梯度。
3. **Generator–Responder–Critic 闭环做自动难度校准**：用 Critic 的通过率反馈去调采样分布，把数据生成从"一次生成"变成"闭环调节"，可用于我们内部的合成评测集扩充。

---

### 2.7 `Hyra-results`（研究 artifact）

#### 项目定位与要解决的问题

Hyra-results 是 **Hyra: Hunyuan Research Agent** 的**伴随产物仓库**。它汇总 Hyra 在**科学、数学、工程与创意设计**领域一系列开放问题上产出的**具体解法**，与 Hyra 发布贴一同释出；每个目录放**最终解法产物**，并在相关处附上**可自包含复现的脚本**。

发布贴：<https://hy.tencent.com/research/hyra>。

它的价值定位不是"框架"也不是"评测"，而是**Agent 能力的证据集**：用一批**可外部比对**的开放问题纪录改进，来证明一个研究型 agent 的实际产出能力。

#### 核心架构 / 方法

仓库本身不是可执行系统，而是**按赛道组织的 artifact 集合**：

- **AI4AI**：`nanochat_autoresearch/`、`nanogpt_speedrun/`、`sol_execbench/`。
- **AI4Science**：15 个课题目录（见下表）。
- **AI4Fun**：`reversi/`、`music/`、`3d_penguin/`、`3d_hunyuan/`。

**结果对比方式（README 原文）**：每一行把「最佳既有公开结果（**Prev best**，取自所引用的系统或 leaderboard）」与 **Hyra** 对照；箭头标出更好的方向（↓ 越低越好，↑ 越高越好）；**Hyra 胜出的值加粗**。README 明确 "Metrics are as defined by each benchmark; comparisons are against the cited published results."

**完整结果表（README 原文，共 15 行）**：

| Track | Task | Metric | Prev best | Hyra |
|---|---|---|---|---|
| **AI4AI** | [`nanochat_autoresearch`](AI4AI/nanochat_autoresearch/) | val BPB ↓ | 0.9109 <sup>e</sup> | **0.9015** |
| | [`nanogpt_speedrun`](AI4AI/nanogpt_speedrun/) | wall-clock ↓ | 77.5 s <sup>e</sup> | **76.4 s**（mean val loss: **3.280**） |
| | [`sol_execbench`](AI4AI/sol_execbench/) | score ↑ | 0.754 <sup>e</sup> | **0.771** |
| **AI4Science** | [`autocorrelation_first`](AI4Science/autocorrelation_first/) | C₁ ↓ | 1.502870 <sup>a</sup> | **1.502850** |
| | [`autocorrelation_second`](AI4Science/autocorrelation_second/) | R ↑ | 0.962694 <sup>b</sup> | **0.962901** |
| | [`erdos_min_overlap`](AI4Science/erdos_min_overlap/) | C₅ ↓ | 0.380868 <sup>b</sup> | **0.380859** |
| | [`sums_diffs`](AI4Science/sums_diffs/) | C(A) ↑ | 1.14489 <sup>b</sup> | **1.21079** |
| | [`packing_records`](AI4Science/packing_records/) | records broken | n/a <sup>f</sup> | **100** |
| | [`smallest_adder`](AI4Science/smallest_adder/) | params ↓ | 36 <sup>c</sup> | **15** |
| | [`parp1_docking`](AI4Science/parp1_docking/) | objective ↓ | −9.77 <sup>d</sup> | **−10.60** |
| | [`qubit_routing`](AI4Science/qubit_routing/) | CNOTs added ↓ | 269,037 <sup>b</sup> | **258,369** |
| | [`sunspot_symbolic`](AI4Science/sunspot_symbolic/) | forecast R² ↑ | 0.47 <sup>g</sup> | **0.78** |
| | [`3d_blaschke_lebesgue`](AI4Science/3d_blaschke_lebesgue/) | Vol/w³ bound ↑ | 0.380799 <sup>h</sup> | **0.411040** |
| | [`beurling_ahlfors_bellman`](AI4Science/beurling_ahlfors_bellman/) | C_BA ↓ | 1.575 <sup>i</sup> | **1.523958** |
| | [`partial_hadamard_counting`](AI4Science/partial_hadamard_counting/) | regime exponent ↓ | 3 <sup>j</sup> | **2** |
| | [`commutator_log_cubed`](AI4Science/commutator_log_cubed/) | log exponent ↓ | 4 <sup>k</sup> | **3** |

**Prev-best 来源（README 脚注 a–k）**：<sup>a</sup> TTT-Discover（arXiv:2601.16175）；<sup>b</sup> SimpleTES（arXiv:2604.19341）；<sup>c</sup> AdderBoard trained-weights leaderboard（github.com/anadim/AdderBoard）；<sup>d</sup> Olaparib（已获批的 PARP1 抑制剂，药物基线）；<sup>e</sup> Recursive: *First Steps Toward Automated AI Research*（github.com/recursive-org/first-steps-toward-automated-ai-research）；<sup>f</sup> Erich Friedman 的 *Packing Center*；<sup>g</sup> "copy last frame"（persistence）预测基线；<sup>h</sup> Nishioka（arXiv:2606.01754）；<sup>i</sup> Bañuelos & Janakiraman 2008；<sup>j</sup> D. Davis（arXiv:2603.30013）；<sup>k</sup> B. Bilich（github.com/bilichboris/TaoCommutators）。

**Metric notes（README 给出全部指标定义）**：**C₁** = 首个自卷积/自相关常数 `max(f∗f)/(∫f)²`（最小化）；**R** = 第二个自相关比 `‖f∗f‖₂² / (‖f∗f‖₁·‖f∗f‖∞)`（最大化）；**C₅** = Erdős 最小重叠常数（最小化）；**C(A)** = sum-vs-difference 指数 `log(|A+A|/|A|) / log(|A−A|/|A|)`（最大化）；**params** = 一个 transformer 在 ≥ 0.99 准确率下做两个 10 位数加法所需的**唯一可训练参数量**（最小化）；**objective** = PARP1 对接目标 `Vina score + 10·(1 − QED)`（最小化）；**CNOTs added** = SWAP routing 插入的额外 CNOT 数（最小化；1 SWAP = 3 CNOTs）；**forecast R²** = 月度太阳黑子数的 rolling-origin、free-running（24 个月）预测 R²（最大化）；**Vol/w³ bound** = 所有定宽 `w` 的三维凸体上 `Vol(K)/w³` 的最大已证普适下界（最大化，README 给出的猜想最优值为 `≈ 0.419860`）；**C_BA** = Beurling–Ahlfors 变换在 `‖B‖_{Lᵖ→Lᵖ} ≤ C_BA·(p*−1)` 中对所有 `1 < p < ∞` 的最小已证一致系数（最小化，猜想最优为 1）；**regime exponent** = 使 `m/nᵘ → ∞` 被证明强制 `N(n,m)` 渐近公式成立的幂律指数 `u` 的下确界（最小化，端点值本身不在主张范围内）；**log exponent** = 使交换子落在恒等元 `ε` 邻域所需 `m(ε) = O(logᵏ(1/ε))` 的最小已证 `k`（最小化，匹配下界为 `k = 1`）。

#### 关键目录与文件与时间线

**News 时间线（README 原文日期）**：
- **2026-08-19**：三项新结果，每项都带**其有限与数值核心的 Lean 4 形式化**：
  - **Beurling–Ahlfors 变换**：Iwaniec 1982 猜想预测尖锐 `Lᵖ` 算子范数为 `p*−1`；最佳已证一致系数从 **1.575**（Bañuelos & Janakiraman, 2008）降至 **1.523958**。目录 `AI4Science/beurling_ahlfors_bellman/`。
  - **Partial Hadamard matrices**：其渐近计数现已在 **`n²` 以上的每个幂律区间**成立（不再只是 `m ≫ n³` 的三次区间），临界指数从 **3**（arXiv:2603.30013）降到 **2**。目录 `AI4Science/partial_hadamard_counting/`。
  - **Commutators close to the identity**：恒等元从不是交换子，但可在 `ε` 距离内达到，代价是所需最小 `‖D‖·‖X‖` 从 Tao 的 `O(log⁵(1/ε))`（后精化为 `O(log⁴)`）降到 **O(log³(1/ε))**，对照 Popa 的 `log(1/ε)` 下界。目录 `AI4Science/commutator_log_cubed/`。
- **2026-08-17**：**三维 Blaschke–Lebesgue**。定宽 `w` 凸体的最小体积被猜想为 Meissner 的 `≈ 0.419860 w³`；最佳已证下界从 `4π/33 · w³ ≈ 0.380799 w³`（Nishioka, arXiv:2606.01754）提升到 **> 0.411040 w³**，**补齐剩余差距的 77.4%**。目录 `AI4Science/3d_blaschke_lebesgue/`。README 另给出 Hyra 的已证界精确表达式 `(130838246407123/10¹⁵)·π > 0.411040473721188`。
- **2026-07-29**：**sum-vs-difference 问题被解决**。对有限 `A ⊆ ℤ`，联系 `|A+A|` 与 `|A−A|` 的最优指数的上确界**恰为 2**，可任意逼近但永不达到；这是一份完整的、机器校验的解决，超越了上表中的 `sums_diffs` 记录。证明仓库：<https://github.com/linhaowei1/sum-diff-proof>。

**AI4Fun 部分（README 说明「创意与博弈 demo，无 leaderboard 对比」）**：
- `AI4Fun/reversi/`：面向 [Botzone](https://www.botzone.org.cn/) 8×8 竞技场的 **AlphaZero 风格黑白棋（Reversi/Othello）bot**（C++ pattern/n-tuple net + PUCT-MCTS + 精确终局求解器）。
- `AI4Fun/music/`：**望春風**（*Bāng-chhun-hong*, 1933, 鄧雨賢）的五声部编曲。
- `AI4Fun/3d_penguin/`：由**单个** [Blender](https://docs.blender.org/api/current/) `bpy` 脚本构建的程序化 **3D QQ 企鹅**。
- `AI4Fun/3d_hunyuan/`：程序化的 **3D 腾讯混元 logo 球体**。

**汇总说明（README 的 Note）**：表内结果以 **2026-07-10** 为基准时点，另加 `3d_blaschke_lebesgue`（2026-08-17）与三条 **2026-08-19** 条目。README 还主动提示：「Several of these problems live on public, continuously-updated leaderboards; later entries there may **warm-start from Hyra's published solutions** to reach still-better numbers.」—— 即 Hyra 的纪录可能被后来者以它为起点超越。`packing_records` 的具体口径为：Hyra 的破纪录 packing 在 Erich Friedman 的 Packing Center 上以 **"Found by Haowei Lin"** 署名，**100** 项纪录分布于 **28 个 shape-in-shape 家族**，每一项都超过此前列出的最优值。

#### 安装与运行方式

README 说明每个目录持有最终解法产物，并在相关处提供**自包含的复现脚本**，但**未给出统一的环境要求、依赖安装步骤或运行命令**。具体复现方式需进入各课题目录自行查看（README 未说明）。

#### 评测 / 数据集规模

- 本仓库**不是 benchmark**，没有题目数、任务数或统一指标；其"规模"体现为 **15 条有外部对照的纪录改进**（AI4AI 3 条 + AI4Science 12 条）+ **4 个 AI4Fun demo**。
- 其中若干条目带**形式化验证**：2026-08-19 的三项结果「each with a Lean 4 formalization of its finite and numerical core」；2026-07-29 的 sum-vs-difference 结果为 "complete, machine-checked resolution"。
- 量化幅度举例（均为 README 原文数字）：`packing_records` 破 **100** 项纪录；`smallest_adder` 参数量 36 → **15**；`3d_blaschke_lebesgue` 补齐 **77.4%** 的剩余差距；`partial_hadamard_counting` 临界指数 3 → **2**；`commutator_log_cubed` 指数 4 → **3**；`sunspot_symbolic` forecast R² 0.47 → **0.78**。

#### 论文 / arXiv 链接、License

- 发布贴：<https://hy.tencent.com/research/hyra>。
- 引用的外部论文/仓库：arXiv [2601.16175](https://arxiv.org/abs/2601.16175)、[2604.19341](https://arxiv.org/abs/2604.19341)、[2606.01754](https://arxiv.org/abs/2606.01754)、[2603.30013](https://arxiv.org/abs/2603.30013)；<https://github.com/anadim/AdderBoard>、<https://github.com/recursive-org/first-steps-toward-automated-ai-research>、<https://github.com/bilichboris/TaoCommutators>、<https://github.com/linhaowei1/sum-diff-proof>、<https://erich-friedman.github.io/packing>。
- 本仓库**没有自己的 arXiv 编号**；引用方式为 BibTeX key `hyra2026`（`author = {{Hyra Team}}`，`year = {2026}`）。
- License：**Apache-2.0**，但 README 与 `LICENSE` 都附带限定："Unless otherwise specified, all Tencent-authored materials in this repository are licensed under the Apache License, Version 2.0. Third-party materials listed below remain subject to their respective original license terms."

#### 亮点与局限

**亮点**：
1. **每个结论都给出可外部比对的 Prev best 与出处脚注（a–k）**，并明确说明「比较对象是所引用的已发表结果」「指标按各 benchmark 自身定义」—— 方法论透明度在同类"agent 成果展示"里相当高。
2. **指标定义完整且给出精确表达式**（如 C₁、R、C(A)、objective、Vol/w³ 界），避免只报一个孤立的数字。
3. **部分结果带 Lean 4 形式化 / 机器校验**，把"agent 声称证明了一个定理"的可信度问题往前推了一大步。
4. **主动披露不利信息**：提醒公开 leaderboard 上的后来者可能**以 Hyra 的解法为起点**做得更好，这种自我限制在宣传性仓库里少见。
5. 覆盖面广（模型训练/推理加速、组合数学、算子理论、蛋白对接、量子线路路由、太阳黑子预测、以及创意设计），展示了通用研究型 agent 的宽度。

**局限**：
1. **仓库不含 agent 本身**：没有 Hyra 的架构、脚手架、提示策略、搜索算法或评测协议的任何说明 —— 对外部复现者而言，这些是**结果而非方法**。
2. **无法自行复现**：README 未给统一环境要求与运行命令，各课题的复现脚本细节 `README 未说明`；且部分结果依赖仍在更新的外部 leaderboard（如 Packing Center、AdderBoard），存在**记录被刷新**的时效性。
3. **未说明运行成本、调用量、成功/失败率或尝试次数**，无法评估这些结果背后需要多少算力与试错。
4. `AI4Fun` 部分明确无 leaderboard 对比，属于展示性质。
5. 结果基准时点不统一（以 2026-07-10 为基准 + 后续追加），README 自己用 Note 说明，但跨条目比较时需要注意时点差异。

#### 对我们可复用的工程实践

1. **「Prev best + 出处脚注 + 指标定义」三件套的结果登记表**：任何 agent 产出型项目（自动科研、自动调优、自动写代码）都应该用这种表来记录结果 —— 明确对照对象、明确指标数学定义、明确方向（↓/↑），比只报"我们的 agent 达到了 X"可信得多。
2. **关键结论附机器可检查的形式化**：对数学/算法类结论，附 Lean 4 形式化或机器校验的证明仓库链接，把 agent 输出的可信度从"叙述"提升到"可验证"。这条对任何做 AI4Science 类 agent 的团队都是高价值的验收标准。
3. **主动声明结论的时效性与可被超越的路径**（后来者可从你的公开解法 warm-start），既降低误用风险，也避免"纪录"被反复刷榜时产生误导。

---

## 3. 横向对比：Agent 评测类仓库的多维对比

本节只对比 **5 个 agent 评测类仓库**：`C3-Benchmark`、`VisualNeedle`、`RepoProbe`、`ExplorationBench`、`PlanningBench`。（`UniRL` 是训练框架、`Hyra-results` 是研究 artifact，不属评测类，但 `UniRL` 的 agentic 环境契约在对比表中作为参照列出。）

| 维度 | `C^3-Bench` | `VisualNeedle` | `RepoProbe` | `ExplorationBench` | `PlanningBench` | （参照）`UniRL` agentic |
|---|---|---|---|---|---|---|
| **评测对象** | LLM Agent 的工具使用能力（49 个主流 agent） | MLLM 的主动视觉搜索能力（9 个 MLLM） | **agent**（先探索仓库再作答），20 个模型（13 闭源 + 7 开源） | **AI System** 的探索能力（10 个前沿系统） | LLM 的复杂文本规划能力 | 被训练的 policy 模型（Qwen3-4B-Instruct） |
| **任务类型** | 多任务多轮工具调用（串行/并行/澄清后调用/闲聊四类动作；三类挑战：工具关系、隐藏信息、动态决策路径） | 图像密集场景中的局部证据定位与整合（OCR / color / entity / spatial / occlusion 五能力） | 仓库级架构理解问答（Project Architecture / Business Logic / Implementation Details 三类） | 在规则隐藏的 Alien World 中通过实验发现规则，再做程序合成（AlienCode）或形式化证明（AlienLogic） | 文本化规划（6 大族 / 30+ 任务类型：排程、生产运营、路由、应急、分配匹配、排班） | 多轮工具调用（deep research：search + visit） |
| **交互轮数** | 多轮，层数可变（数据生成默认 4 层；分析默认 `answer_depth=4`）；第二至四轮为真多轮 | 多步：`Crop(bbox)` 反复收窄 → `Answer` 结束（具体轮数上限 **README 未说明**） | 多轮 agent 探索；默认 turn 预算 **30**（`REPOPROBE_MAX_TURNS="30"`） | **4 轮**探索（M<sub>0</sub>–M<sub>4</sub>），每轮后关闭工具作答；每 task **3 次**作答 | **单轮**（self-contained 输入 → 输出完整计划） | `max_turns: 8`（rollout 与 env 两处必须一致） |
| **是否工具调用** | **是**，核心即工具调用；统一 ToolCalls 协议；golden 工具结果持久化，离线无 KEY | **是**，核心动作 `Crop(bbox)`；配置可启用 zoom / crop / flip / sharpen 等图像工具 | **是**（agent 可探索仓库）；工具集由被测脚手架决定，harness 本身不限定 | **是**，探索期通过**单一工具**探测世界；milestone 期**关闭工具** | **否**（纯文本推理，无工具调用） | **是**，`ToolEnvironment` + `CalculatorTool` / `SandboxTool` / `SearchTool` / `VisitTool` |
| **评分方式** | `EvalByToolCallGraph` 动态裁决 + 决策树剪枝；`AnalysisResult` 六维分析；指标含矩阵准确率、`*_tools_rate`、`serial_complete_rate`、`best_optical_rate`、12 类 error type、`turn_type_rate`、`turn_subtype_rate` | exact / normalized 匹配优先，未匹配再走**可选 VLM judge**；四设置对比（text-only / no-tool / tool-enabled / **crop-black**）；报告 exact-match score + final score + judge-use count + tool-use statistics | **加权 checklist 逐项核验 + LLM judge**；每题 10 分（9 knowledge + 1 clarity）；指标 `Overall` / `Knowledge` / `Clarity` / `Perfect Solve Rate` | **interpreter 或 proof checker 给出判定，无 LLM judge**；指标 held-out accuracy、**Best@3**、**Mean@3**、M<sub>0</sub> | 每实例附带 verification checklist，用于 constraint satisfaction 与 objective quality（**具体指标名 README 未说明**） | `RewardService` + `LLMJudgeRewardScorer`（LLM-as-judge 打分 terminal answer），GRPO group-normalized advantage |
| **是否含 agent scaffold** | **含**（评测框架自带推理 pipeline 与 handle 层；不是可插拔脚手架） | **含**（vendored `Qwen-Agent` 运行时 + `visualneedle_eval` agent 构建） | **含，且显式解耦**：Docker 容器契约 + `agent.json` 声明，**任何脚手架可插拔**，默认接 `example_agent.py`；**Docker 隔离，所有被测模型用同一 scaffolding** | **含 framework track**：`baseline` + reflexion / ace / evotest / gepa / agent_factory，分 `controlled` / `native` / `open` 三档 track 与 `c1` / `1x` / `2x` / `4x` 预算档 | **不含**（无评测代码） | **含**（`ToolAgentHarness` + `AgenticRolloutEngine`，env / tool / trajectory 三层契约） |
| **规模** | 49 个被测 agent；题目数 README 未说明；数据准确率 <60% → 100% | **300** 题 / 5 能力 / 4 设置 / 9 MLLM | **50** 仓库 / **500** 题 / 15 语言 / 20 模型 | **2** world / **55** targets / **140** held-out tasks / 10 系统 / 3 轨迹 | **467** 实例（实测 467 行）/ 6 族 / 30+ 任务类型 | 单 recipe（deep research），`max_turns=8` |
| **最佳成绩（README 数字）** | README **未给分数表** | Tool-enabled **56.01%** vs Human **63.00%**；No-tool **17.89%**；Text-only **< 10%** | GPT-5.2 Overall **62.7**，最高 Perfect Solve **27.5%** | AlienCode Best@3 **89.0%**；AlienLogic Best@3 **81.0%** | README **未给分数** | README **未给分数** |
| **可复现性关键设计** | golden 工具结果持久化 → 无需任何网站 KEY；`--continue_file` 续跑 | harness 已公开但**数据集 coming soon** | snapshot commit pinning；快照丢失的仓库建议**排除而非换 commit**；论文脚手架未分发 | 评测世界私有（需联系作者代跑）；demo world 分数不可与 leaderboard 比较；引擎可复现全部 7,497 证明 / 7,651 程序 | 数据即全部（无代码可复现） | extra 互斥、版本钉死；仅 1 个可运行 agentic recipe |
| **License** | CC BY 4.0 | Apache-2.0 | 代码 Apache-2.0 + 数据 CC BY 4.0 | Apache-2.0（code） | CC-BY-4.0 | Apache-2.0 |

**几点值得注意的对比结论**：

1. **「是否用 LLM judge」几乎与仓库的年代/成熟度同步演进**：`C3-Benchmark` 走确定性 golden 结果比对（2025），`RepoProbe` 明确批判标量 LLM-as-a-Judge 并改用 weighted checklist + LLM judge（2026），`ExplorationBench` 则完全弃用 LLM judge，改由 interpreter / proof checker 给出判定（2026.09）。
2. **「防止评测集被污染」的手段各有不同**：`C3-Benchmark` 靠持久化 golden trace 去掉外部依赖；`RepoProbe` 靠 Docker + snapshot commit 钉死代码版本；`ExplorationBench` 靠"世界与规则私有 + 副本丢弃"从机制上防止测试题泄漏进探索。
3. **只有 `RepoProbe` 把 agent scaffold 抽象成了可插拔契约**（容器四要素 + `agent.json` + model gateway），其余评测仓库都自带或隐含特定脚手架；`ExplorationBench` 的 framework track 是另一种思路（在同一 harness 内做方法性对比，且 controlled track 不 import 官方框架）。
4. **只有 `C^3-Bench` 与 `RepoProbe` 给出完整的失败归因体系**（12 类 error type / knowledge-clarity 分离），`VisualNeedle` 与 `ExplorationBench` 更偏向"用对照实验解释分数"。
5. **`PlanningBench` 是 5 个评测仓库中唯一完全无代码、无 baseline 的**，其贡献集中在数据与分类学；与之相对，`VisualNeedle` 是唯一"代码全公开、数据未公开"的。

---

## 4. 对我们可复用的点（汇总索引）

按主题汇总各节末尾给出的工程实践，便于按需检索：

| 主题 | 出处仓库 | 可复用做法 |
|---|---|---|
| **Agent scaffold 契约** | `UniRL`、`RepoProbe` | `Environment.reset/step/close` 三段式（**环境决定 episode 何时结束**）；`Tool.execute` / `StatefulTool` 会话生命周期；畸形工具调用降级为最终答案、工具异常降级为错误文本回灌；容器契约四要素（仓库路径 / 只读 prompt 文件 / stdout 仅答案且 trace 走 stderr / 退出码 0） |
| **模型调用与公平性** | `RepoProbe` | model gateway 用环境变量**覆盖脚手架自报的模型名**，集中处理重试与超时，并把「每次调用记一行」顺带产出 `rounds` 指标 |
| **可插拔脚手架** | `RepoProbe` | 只定义最小契约、不发布特定 agent；`agent.json` 声明 command / gateway / 环境变量 / stdout 过滤正则 |
| **Checklist 打分** | `RepoProbe`、`PlanningBench` | 加权 checklist + **tiered rubric**（显式写出 3/1/0 分各对应什么）；knowledge 与 clarity 分离（9:1）；把验证依据与题目打包在同一条记录里 |
| **对照实验设计** | `VisualNeedle`、`ExplorationBench` | **crop-black 式"同格式 + 空白证据"消融**；控制条件三件套（同轮数无反馈 / 重放自己的最优 probes / open-book 告知规则）；每题多答取 Best@3 与 Mean@3 |
| **防污染与防泄漏** | `ExplorationBench`、`RepoProbe`、`C3-Benchmark` | milestone 时复制会话、**关闭工具**、总结信念并答留出题、**然后丢弃副本**；Docker 隔离 + snapshot commit pinning；golden trace 持久化以去掉外部 API 依赖 |
| **Tool trace 与审计产物** | `VisualNeedle`、`UniRL`、`ExplorationBench` | per-sample 目录（日志 + 中间工具图像 + 归一化坐标 + token 用量 + 结构化 JSON + 离线 HTML 报告）；`Sample.parts` 线性轨迹链；JSONL trace store + artifact snapshot digest |
| **失败归因体系** | `C3-Benchmark` | 12 类 error taxonomy 并分为 action / param 两组；六维分析结果落 CSV 便于 bad case 分析 |
| **长跑评测健壮性** | `C3-Benchmark`、`VisualNeedle`、`UniRL` | `--continue_file` 跳过已预测样本；`--retry-dir` 断点重跑 + `--max-null-retries`；`failed` / `overflow` / 引擎级失败三档语义分开处理 |
| **难度可控的数据生成** | `PlanningBench`、`C3-Benchmark` | 按难度因子（约束紧度 / 资源稀缺 / 目标冲突 / 子任务依赖 / 全局协调）参数化生成；Generator–Responder–Critic 闭环自动校准难度；可控 + 指定数量 + 多样化的任务生成以覆盖全部动作空间 |
| **规模与可信度表达** | `Hyra-results`、`PlanningBench` | 结果登记表「Prev best + 出处脚注 + 指标数学定义 + 方向箭头」三件套；关键结论附 Lean 4 / 机器校验；**主动声明结论时效性与可被超越的路径**；规模自述与实测数字互相印证 |
| **长时 agent 状态管理** | `ExplorationBench` | framework track 的 **episode-local playbook**（记录 hypothesis / failed probes / 可复用策略，仅在可见反馈上更新，milestone 与 test 期间视为冻结，无跨 run 记忆） |

---

## 5. 参考链接

### 仓库主页（7 个）

- <https://github.com/Tencent-Hunyuan/UniRL>
- <https://github.com/Tencent-Hunyuan/C3-Benchmark>
- <https://github.com/Tencent-Hunyuan/VisualNeedle>
- <https://github.com/Tencent-Hunyuan/RepoProbe>
- <https://github.com/Tencent-Hunyuan/ExplorationBench>
- <https://github.com/Tencent-Hunyuan/PlanningBench>
- <https://github.com/Tencent-Hunyuan/Hyra-results>

### 仓库内直接引用的文档（已校验可访问）

- UniRL：<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/INSTALL.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/unirl/README.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/unirl/models/README.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/unirl/rollout/env/README.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/unirl/algorithms/README.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/unirl/types/README.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/examples/README.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/examples/deep_research/deep_research_search_judge.yaml>、<https://raw.githubusercontent.com/Tencent-Hunyuan/UniRL/main/examples/run_experiment_single_node.sh>
- UniRL release：<https://github.com/Tencent-Hunyuan/UniRL/releases/tag/v0.2.0>；文档站：<https://unirl-project.github.io/unirl/>
- C3-Benchmark：<https://raw.githubusercontent.com/Tencent-Hunyuan/C3-Benchmark/main/README_ZH.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/C3-Benchmark/main/c3_bench/bench_test/handle/handles.py>、<https://raw.githubusercontent.com/Tencent-Hunyuan/C3-Benchmark/main/c3_bench/bench_test/analysis_result.py>、<https://raw.githubusercontent.com/Tencent-Hunyuan/C3-Benchmark/main/c3_bench/requirements.txt>
- VisualNeedle：<https://raw.githubusercontent.com/Tencent-Hunyuan/VisualNeedle/main/docs/evaluation.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/VisualNeedle/main/requirements.txt>、<https://raw.githubusercontent.com/Tencent-Hunyuan/VisualNeedle/main/Makefile>
- RepoProbe：<https://raw.githubusercontent.com/Tencent-Hunyuan/RepoProbe/main/docs/AGENT_INTEGRATION.md>、<https://raw.githubusercontent.com/Tencent-Hunyuan/RepoProbe/main/requirements.txt>、<https://raw.githubusercontent.com/Tencent-Hunyuan/RepoProbe/main/agent_configs/agent.json>
- ExplorationBench：<https://raw.githubusercontent.com/Tencent-Hunyuan/ExplorationBench/main/config/eval.local.example.toml>、<https://raw.githubusercontent.com/Tencent-Hunyuan/ExplorationBench/main/explorationbench/frameworks/base.py>、<https://raw.githubusercontent.com/Tencent-Hunyuan/ExplorationBench/main/explorationbench/sandboxes/code/world_data.py>
- PlanningBench：<https://raw.githubusercontent.com/Tencent-Hunyuan/PlanningBench/main/LICENSE.txt>、<https://raw.githubusercontent.com/Tencent-Hunyuan/PlanningBench/main/data/PlanningBench-eval.jsonl>

### 论文 / arXiv

- `C^3-Bench`：<https://arxiv.org/abs/2505.18746>；ICLR 2026 版本 <https://openreview.net/forum?id=yz7fL5vfpn>
- `VisualNeedle`：<https://arxiv.org/abs/2605.26380>
- `RepoProbe`：<https://arxiv.org/abs/2608.04783>；会议主页 <https://conf.researchr.org/home/ase-2026>
- `ExplorationBench`：<https://arxiv.org/abs/2609.30199>
- `PlanningBench`：<https://arxiv.org/abs/2605.20873>
- `UniRL` 团队算法：DRPO <https://arxiv.org/abs/2606.09821>；Flow-DPPO <https://arxiv.org/abs/2606.11025>；CPPO <https://arxiv.org/abs/2606.10968>
- `Hyra-results` 引用：<https://arxiv.org/abs/2601.16175>、<https://arxiv.org/abs/2604.19341>、<https://arxiv.org/abs/2606.01754>、<https://arxiv.org/abs/2603.30013>

### 数据集 / Leaderboard / 官网

- C3-Bench 数据集：<https://huggingface.co/datasets/tencent/C3-BenchMark>
- PlanningBench 数据集：<https://huggingface.co/datasets/tencent/PlanningBench>
- ExplorationBench 官网 / leaderboard / blog：<https://explorationbench.com>、<https://explorationbench.com/#leaderboard>、<https://explorationbench.com/blog/>
- Hyra 发布贴：<https://hy.tencent.com/research/hyra>

### 相关 / 后继项目

- C3-Bench 后继仓库：<https://github.com/yupeijei1997/WildToolBench>
- ExplorationBench 系列：EvaLearn <https://github.com/ByteDance-Seed/EvaLearn>、CL-bench <https://github.com/Tencent-Hunyuan/CL-bench>
- Hyra 结果的外部对照来源：<https://github.com/anadim/AdderBoard>、<https://github.com/recursive-org/first-steps-toward-automated-ai-research>、<https://github.com/bilichboris/TaoCommutators>、<https://github.com/linhaowei1/sum-diff-proof>、<https://erich-friedman.github.io/packing>
- UniRL 致谢的基础设施：<https://github.com/vllm-project/vllm>、<https://github.com/sgl-project/sglang>、<https://github.com/THUDM/slime>、<https://github.com/volcengine/verl>
- VisualNeedle 相关对外服务/工具文档：<https://docs.vllm.ai/en/stable/features/tool_calling.html>

---

## 附录：本报告的证据边界与已知不一致

1. **stars 与「最近更新」不是 README 内容**。概览总表中的 stars 取自 GitHub 仓库页面 HTML 的 `stargazerCount`，「最近更新」取自 `<repo>/commits/main.atom` 的 `<updated>` 字段，均为本报告编写时的一次性快照。GitHub REST API 全程返回 403（限流），本报告**没有**使用该 API 的任何数据。
2. **已在线校验为 404 的 README 声明路径**（README 声称存在，但 `main` 分支 raw 路径下取不到）：
   - `C3-Benchmark`：`c3_bench/data/C3-Bench.jsonl`、`c3_bench/tool_calls/tool_model_map.py`
   - `PlanningBench`：`LICENSE-PlanningBench.txt`（实际许可证文件为 `LICENSE.txt`）
3. **README 相对路径与实测目录不一致**：
   - `C3-Benchmark` README 写 `cd multi_agent`，实测入口位于 `c3_bench/multi_agent/generate.py`。
   - `ExplorationBench` README 写 `cd explorationbench` 后运行 `python3 -u scripts/answer_variance.py`，实测该文件位于 `explorationbench/scripts/answer_variance.py`。
4. **README 模块描述与实测代码结构存在差异**：`ExplorationBench` 的 `common/` 在代码中实际为 `common/agent_client/` 与 `common/agent_runtime/` 两个子包（由 `explorationbench/frameworks/base.py` 的 import 语句可见）。
5. **未做第三方交叉验证的 README 自述**（属仓库自身论断，本报告如实转述并标注来源）：`C3-Benchmark` 的「唯一」「最高」「over 90% 决策准确率」「准确率 <60% → 100%」；`ExplorationBench` 的「no LLM judge」与引擎可复现「7,497 proofs / 7,651 programs」；`Hyra-results` 的全部 Prev best 对比（其对照对象本身来自第三方 leaderboard 与论文）。
6. **本报告未获取到的信息**（均为 `README 未说明`）：`C^3-Bench` 的题目数量与 baseline 分数；`RepoProbe` 论文所用脚手架的实际实现；`PlanningBench` 的评测指标名与 baseline；`VisualNeedle` 的硬件要求与 text-only 精确值；`UniRL` agentic 路径的评测分数；`Hyra` 自身的架构与运行成本。
