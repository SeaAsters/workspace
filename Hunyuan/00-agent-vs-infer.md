# 混元 Agent 线 vs 推理线：对比分析

> **文档定位**：本文件是对 [`agent/`](agent/) 与 [`infer/`](infer/) 四份报告的**横向拉通解读**，回答两个问题：Agent 方向主要在做什么？推理方向主要在做什么？两者的性格差异与耦合关系是什么？
>
> **证据来源**：所有论断均可回溯到 `_research/` 下的原始素材（101 份 README / LICENSE / HF 模型卡 / 子文档快照）与四份明细报告。未在原文出现的数字一律不写。
>
> **配套文档**：[`INDEX.md`](INDEX.md) 总索引 · [`agent/01`](agent/01-agent-frameworks-and-benchmarks.md) · [`agent/04`](agent/04-embodied-and-agentic-3d.md) · [`infer/02`](infer/02-inference-optimization-libs.md) · [`infer/03`](infer/03-model-release-deployment-matrix.md)

---

## 0. 一句话结论

**两条线性格截然不同：Agent 线在定义「怎么衡量能力」，推理线在解决「怎么把能力跑起来」。**

而且这两条线在混元内部**几乎是割裂的** —— 各自建团队、各自出成果，暂时没有把它们缝成一个端到端 agent 产品的意图。唯一的交汇点是 `Hy-Embodied-VLM-1.0` 复用 `Hy3-A3B` 骨干。

| | Agent 线 | 推理线 |
|---|---|---|
| 核心目的 | 定义**怎么衡量能力** | 解决**怎么把能力跑起来** |
| 主要产出形态 | benchmark + harness + rubric | 镜像 + 启动命令 + 量化权重 |
| 仓库数 | 14 | 56 |
| Stars 区间 | 0 – 1377（多为论文配套） | 最高 2036（`HunyuanOCR`）、1234（`Hy-MT2`），有生产用户 |
| 成熟度信号 | 论文驱动，脚手架常不完整 | 镜像化交付，可直接上线 |
| 与另一条线的耦合 | 几乎没有 | 几乎没有 |

---

# 第一部分：Agent 方向

## 1. 投入分布本身就是结论

14 个仓库里，**6 个是评测基准，只有 1 个是通用框架**（`UniRL`），剩下是 5 个具身智能 + 1 个 Agentic 3D。

这个比例不是偶然 —— 混元在 Agent 上的战略重心是**测量失败模式**，而不是提供脚手架。

| 仓库 | 子类 | Stars | 攻的失效点 |
|---|---|---:|---|
| [`UniRL`](https://github.com/Tencent-Hunyuan/UniRL) | Agent 训练框架 | 1011 | 唯一的通用框架 |
| [`C3-Benchmark`](https://github.com/Tencent-Hunyuan/C3-Benchmark) | 能力评测 | 38 | 工具依赖关系、隐藏信息、频繁策略切换下的鲁棒性 |
| [`VisualNeedle`](https://github.com/Tencent-Hunyuan/VisualNeedle) | 能力评测 | 0 | 高信息密度场景能不能**主动找**证据（而非一次性 VQA） |
| [`RepoProbe`](https://github.com/Tencent-Hunyuan/RepoProbe) | 能力评测 | 13 | 仓库级**架构理解**（而非打补丁） |
| [`ExplorationBench`](https://github.com/Tencent-Hunyuan/ExplorationBench) | 能力评测 | 7 | 探索增益 vs 推理增益的分离 |
| [`PlanningBench`](https://github.com/Tencent-Hunyuan/PlanningBench) | 能力评测 | 27 | 规划（⚠️ 仓库**无任何代码**） |
| [`ArtifactsBenchmark`](https://github.com/Tencent-Hunyuan/ArtifactsBenchmark) | 能力评测 | 279 | 视觉交互式代码产物的保真度与可交互性 |
| [`Hyra-results`](https://github.com/Tencent-Hunyuan/Hyra-results) | 研究产物 | 159 | 递归自我改进 / AI4Science |
| `HY-Embodied` 系列 ×5 | 具身智能 | 67–876 | 感知-动作闭环（独立技术栈） |
| [`Hunyuan3D-WorldClaw`](https://github.com/Tencent-Hunyuan/Hunyuan3D-WorldClaw) | Agentic 3D | 1377 | ⚠️ 仅 README + 论文，无可运行内容 |

## 2. `UniRL`：唯一真正的框架，它回答了 agentic RL 最难的题

`UniRL` 把 agentic 做成了**一等训练路径**（`train_agentic` 是 8 条训练入口之一），而不是「顺手支持工具调用」。三个设计点真正解决了问题：

### 2.1 `Environment.reset/step/close` 契约 —— 由环境而非 turn loop 决定 episode 何时结束

这直接对应**稀疏奖励下的信用分配**问题。多轮工具调用里，如果轮数由训练循环写死，reward 就无法归因到「哪一步决策错了」。让环境掌握终止权，才能把最终奖励正确地分摊回每一步。

### 2.2 `Sample.fork` vs `Sample.observe` —— 在类型层面分开「模型生成的」与「环境返回的」

| API | 带采样参数 | 是否进梯度 |
|---|---|---|
| `Sample.fork` | ✅ | ✅ 可训练 |
| `Sample.observe` | ❌ | ❌ observation 不进梯度 |

这是多轮 agent RL **最容易搞脏梯度的地方**：工具返回的文本如果混进 policy 的 loss，训练就是错的。用类型系统强制区分，比靠约定可靠得多。

### 2.3 `StatefulTool` 会话生命周期

工具可以有状态（浏览器、数据库连接），而不只是无状态的纯函数。

### 2.4 工程现实

- 唯一可跑的 recipe：`examples/deep_research/deep_research_search_judge.yaml`
  - `max_turns: 8`、每轮 `max_new_tokens: 2048`
  - 奖励：`LLMJudgeRewardScorer`；算法：GRPO
  - 依赖：Serper / Jina / 独立 judge endpoint（**三个外部服务**）
- 环境约束：**仅 Linux x86_64 + NVIDIA GPU**，Python 3.12–3.13
- ⚠️ **`vllm` 与 `sglang` 两个 extra 不可共存**
- `UniRL v0.2.0`（README 标注 [2026-10]）：12 个新 model packages + SFT/agentic/async 三种训练模式 + SGLang 0.5.19 / vLLM-Omni 0.28

## 3. 评测方法论走完了一条清晰的演进路

这是整批 Agent 工作里**最有借鉴价值**的部分：

| 代次 | 代表 | 判定方式 |
|---|---|---|
| 2025 | `C3-Benchmark` | 持久化 golden 工具结果做**确定性比对**（用 `EvalByToolCallGraph` + 决策树剪枝；**无需任何网站 KEY**） |
| 2026 | `RepoProbe` | **加权 checklist + 分级 rubric** 取代 LLM-as-a-Judge 标量打分 |
| 2026.09 | `ExplorationBench` | **完全弃用 LLM judge**，判定全部来自 interpreter / proof checker |

### 3.1 checklist 换掉标量分，立刻暴露被掩盖的现象

`RepoProbe` 每題 10 分 = 9 knowledge + 1 clarity，并显式写出 3/1/0 分各对应什么。结果：

> **clarity 84.7 vs knowledge 60.3** —— 模型回答得很流畅，但事实正确率差 **24 个点**（"流畅度跑赢正确性"）。最高 Perfect Solve 仅 27.5%。

标量 LLM-as-a-Judge 会把这两个维度糊成一个数，把这个现象完全掩盖。

### 3.2 两个可以直接搬走的设计

**(a) `VisualNeedle` 的 crop-black 消融** —— 保留交互格式和 crop 尺寸，只把返回图替换为**同尺寸黑图**，从而把「格式合规」与「真的用上了返回的证据」分开度量：

| 条件 | 成绩 |
|---|---|
| Tool-enabled | **56.01%** |
| No-tool（单轮 VL） | 17.89% |
| Text-only | 更低 |
| 人类 | 63.00% |

> **平移到文本检索场景**：返回**同长度但无关的文档**，即可复刻这个消融。
> ⚠️ 注意：数据集目前仍 `coming soon`，只有 harness 开源。

**(b) `ExplorationBench` 的 milestone 机制** —— 每轮后复制会话、**关闭工具**让它总结当前信念并回答留出题、**然后丢弃副本**。一个动作同时满足「过程可观测」与「测试集不泄漏」。

其控制条件三件套把探索增益与推理增益干净分离：

| 条件 | Best@3 |
|---|---|
| 四轮探索 | **89.0%**（AlienCode） |
| 同轮数但无反馈 | 仅 0.5–11.0% |

## 4. 具身智能：一条完全独立的线

5 个仓库由 **Tencent Robotics X × HY Vision Team** 主导，技术上与「LLM 工具调用型 agent」**不共享任何东西**：

- 那边是**感知-动作闭环**，这边是**文本工具调用**
- 且已下沉到 [`Hy-Embodied-RoboFusion`](https://github.com/Tencent-Hunyuan/Hy-Embodied-RoboFusion) 这种**电机/传感器 UDP 通信协议规范**（PDO/SDO，C 实现）—— 这是机器人控制层，不是 AI 层

### 最强的技术证据：跨本体迁移消融

`Hy-Embodied-0.5-VLA` 在 RoboTwin 2.0（50 任务 × 100 rollouts）：

| 方法 | Clean | Randomized |
|---|---|---|
| **Hy-VLA** | **90.9%** | **90.1%** |
| π₀.₅ | 82.7% | 76.8% |
| Qwen-VLA | 86.1% | 87.2% |
| starVLA | 88.2% | 88.3% |

**Track B 跨本体**（仅用 UMI 示范微调、无目标机遥操作）：

| 条件 | Clean | Randomized |
|---|---|---|
| Hy-VLA | 90% | 89% |
| **去掉 UMI 预训练** | **38%** | **44%** |

→ 这是「**大规模 UMI 语料 = 本体无关的动作先验**」的硬证据（UMI 数据规模：**10,000+ 小时**，70+ 任务）。

### ⚠️ 架构坑

**0.5 代是 MoT，1.0 代 `Hy-Embodied-VLM-1.0` 是 MoE（8 of 128 experts）** —— 两者**不是同一条升级线**。`Hy-VLA` 基于 0.5 代 MoT，**不能假定 VLM-1.0 可直接替换其 backbone**。

### ⚠️ 两个数字缺口

- `HY-Embodied-0.5-X` 宣称 10 个 benchmark / 7 个第一 + 内部 AI2Thor 1,011 任务，但**所有具体分数只存在于 `assets/*.png` 图里**，正文无数字
- `Hy-Embodied-RxBrain-1.0`（~6.2B）评测章节只写「详见技术报告」，`RxBrain-Bench` **尚未开源**，全仓库**无任何精度指标**

---

# 第二部分：推理方向

## 1. 稀疏注意力是投入最重的方向，且分成两条互不干扰的路线

压力源不同，所以解法不同：

- **LLM 侧**：上下文变长（Hy3 支持 256K）
- **生成侧**：分辨率 / 时长变高（Prism 做到 2K 联合视频-音频）

两者都是 O(n²)，但稀疏化的**决策依据**完全不同：

| | 通用 LLM 长上下文 | 视频 / 多模态生成 |
|---|---|---|
| 代表 | [`HiLS-Attention`](https://github.com/Tencent-Hunyuan/HiLS-Attention) | [`Prism`](https://github.com/Tencent-Hunyuan/Prism) + [`flex-block-attn`](https://github.com/Tencent-Hunyuan/flex-block-attn) |
| 决策依据 | 压缩 chunk key 估计的 chunk-mass surrogate | 视频通道方差 × audio→video 交叉注意力范数 |
| 决策性质 | **可端到端学习**（在 LM loss 下） | 内容驱动的动态规则，**为每个 macro-zone 分配不同 block 形状** |
| 稀疏度 | 固定 top-k（page = 64） | **自适应**：top-k + top-p CDF 混合（`bsa_cdf_threshold=0.20`），per-query 不同 |
| 落地形态 | SGLang HSA 后端（`--attention-backend hsa`） | CUDA 算子库（STA/MoBA/SSTA），**Hopper SM90 + CUDA 12.8** |
| 硬件门槛 | CUDA 12.8（torch 2.8.0+cu128） | flex-block-attn: 仅 Hopper；Prism: 推理 1–4×80GB，训练 32–64×80GB |

**两条线共享同一批核心作者**（`Jiangfeng Xiong`、`Zhao Zhong` 同时出现在 `flex-block-attn` 与 `Prism`），说明这是**有组织的投入**而非零散论文。

### 1.1 最值得学的一条共同原则：训推一致

- `HiLS-Attention`：让 top-k 选择在 LM loss 下**可学**，而非训练完再套 kernel
- `Prism`：`enable_bsa` 与 `enable_ivpq_dynamic_block` 在**训练和推理配置里同时存在**

> **不做「训练完再套一个稀疏 kernel」的两段式** —— 那是稀疏注意力落地最容易踩的坑，会引入训推不一致。

### 1.2 另一个可复用的 trick：混合稀疏度

`Prism` 用 `bsa_sparsity`（top-k）+ `bsa_cdf_threshold`（top-p）组合，使**每个 query 有不同稀疏度**。固定稀疏率对信息分布极不均匀的输入（如 2K 视频）是纯浪费。

## 2. 真正的统一发生在工程侧：量化 / 投机解码 / 部署形态

模型发布库那 48 个仓库**技术含量参差**（有的纯 README），但**工程收敛度很高**。三个信号：

### 2.1 量化工具链统一收敛到自研 `AngelSlim`

`Hy3`、`Hy4-preview`、`Hunyuan-A13B`、`Hy-MT2` 全部指向 [AngelSlim](https://github.com/Tencent/AngelSlim)，覆盖：

| 方案 | 说明 |
|---|---|
| FP8-static | 8-bit 浮点，少量校准数据（无需训练）预定量化 scale |
| GPTQ / AWQ W4A16 | 逐层量化，逼近 Hessian 逆矩阵 |
| **1.25-bit 极低比特** | 端侧极限压缩 |

**最佳样本 —— `Hy-MT2-1.8B`**：1.25-bit 量化后**存储仅 440MB、推理提速 1.5×**。这是全批**唯一带数字的端侧量化方案**。

> ⚠️ 坑：依赖**尚未合并**的 llama.cpp STQ kernel [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836)。

### 2.2 投机解码从「可选」变「默认」

| 模型 | 机制 | 启动参数 |
|---|---|---|
| `Hy4-preview` | native MTP（10B 总参 / 0.7B 激活） | `--speculative-config '{"num_speculative_tokens":3,"method":"mtp"}'` + `--attention-backend FLASHMLA_SPARSE` |
| `Hy3` | 1 层 3.8B MTP | `--speculative-config.method mtp --speculative-config.num_speculative_tokens 2` |
| `HunyuanOCR` | **DFlash** | `--spec-type draft-dflash --spec-draft-n-max 15` |

**两个坑**：

1. **SGLang 侧算法名没统一** —— `Hy3` 系写 `EAGLE`，`Hy4-preview` 写 `NEXTN`
2. **三个旗舰 README 都没给加速倍数** —— 唯一有公开数字的是 `HunyuanOCR` 的 DFlash：

| 平台 | 结果 |
|---|---|
| H20 80GB | **2.14×** 端到端（latency 3.032s → 1.408s，0.330 → 0.706 page/s），token 差异 < 0.15% |
| Apple M5 Pro | decode **84.6 → 167.1 t/s（~2×）**，prefill 232 → 192 t/s（慢 ~17%），**输出 byte-identical** |

### 2.3 部署形态覆盖「云 → 端」全谱

这是混元做得比多数开源项目好的地方 —— 不只给 server 方案：

```
H100 服务器 (vLLM/SGLang)  →  消费级 GPU (CPU offload)  →  Apple Silicon (MLX)  →  纯 CPU (llama.cpp/GGUF)
```

**最完整的样本 —— `AuK`**（MIT）：开源**一周内**拿到 SGLang-Omni Day-0 + vLLM-Omni 支持，加上 GGUF(audio.cpp)、MLX、ComfyUI、Xinference。实测显存（A800-SXM4-80GB，bf16）：

| 模型 | 输入 | CPU offload 关 | 开 | 节省 |
|---|---|---:|---:|---:|
| AuK | 纯文本 1.5s | 24.78 GiB | 16.75 GiB | **8.03 GiB (32.4%)** |
| AuK | 5s 参考音频 | 25.00 GiB | 16.98 GiB | **8.02 GiB (32.1%)** |

→ 开启后峰值约 **17 GiB**，可落到 24GB 消费级卡区间。

**反面教材 —— `HiLS-Attention`**：SGLang 后端放在**个人 fork**（`alexzms/SGLang-HiLS`），用户要装 fork + 额外装 `tilelang`，还受三重约束：

> `--page-size` **必须**等于模型 `chunk_size`（64）；prefill **仅支持单序列**；overlap scheduler **必须关闭**。

→ 对高并发在线服务是**吞吐约束**。**落地摩擦比 `AuK` 差了一个量级。**

## 3. 关键策略取舍：不自研引擎

混元**没有做自己的推理引擎**，而是深度适配 vLLM / SGLang 主流框架。

| | 收益 | 代价 |
|---|---|---|
| 选择 | 避免自研引擎的长期维护成本；随上游自动获得新特性（如 MTP、FP8 KV cache） | **主动权在别人手里** |

**代价的具体体现**：

- `Hy3` / `Hy3-preview` / `Hy-MT2` **必须源码编译**（无官方镜像）
- **TensorRT-LLM 是明显短板** —— 只剩 `Hunyuan-A13B`、`Hunyuan-7B`、`HY-MT1.5` 等**老产品线**；四代旗舰（`Hy3`、`Hy3-preview`、`Hy4-preview`）与 `Hy-MT2`、`HunyuanOCR` **均无 TRT 路径**
- 官方一等公民已收敛为 **vLLM + SGLang**

**唯一的镜像例外**：`Hy4-preview` 独享双官方镜像（`vllm/vllm-openai:hy4-preview` + `lmsysorg/sglang:hy4-preview`，SGLang 版还是 **multi-arch，含 Arm**），且是唯一需要专用 attention backend 的模型（`--attention-backend FLASHMLA_SPARSE`，对应 Gated DSA）。

---

# 第三部分：两条线的关系

## 1. 唯一的交汇点

`Hy-Embodied-VLM-1.0` 用 **`Hy3-A3B` 语言骨干 + `Hy-ViT2` 视觉编码器**，只激活 ~3B 参数换取延迟敏感场景的推理效率。

→ **推理侧的效率追求，服务的是 agent 侧的部署可行性。**

除此之外，两条线各自独立。换句话说：混元把「能力」和「效率」当成**两个独立问题在解**，各自建团队、各自出成果，暂无把它们缝成端到端 agent 产品的迹象。

## 2. 不对称性：Agent 线明显更薄

| 观察 | 证据 |
|---|---|
| 框架稀缺 | 6 个评测 vs 1 个框架 |
| 能力覆盖有缺口 | **没有**多轮对话记忆、RAG、computer-use / GUI agent、通用 agent runtime |
| 甚至有工作迁出 | `C3-Benchmark` 已迁至外部仓库 `WildToolBench` |
| 论文配套多 | `PlanningBench`、`Hunyuan3D-WorldClaw`、`HunyuanVision`、`DRIVE-RLVR`、`evolve-scaler` **无任何代码** |

推理线则相反：镜像化交付、量化权重、生产用户齐全。

---

# 第四部分：如果要继续深挖

按价值排序，我会看这三个方向：

## ① `UniRL` 能否跑通（**最高价值**）

它是唯一可运行的 agent 训练框架，但只给了 1 个 recipe，且依赖 Serper / Jina / judge endpoint **三个外部服务**。实际跑一遍能验证它宣称的 agentic 训练契约（`Environment` / `Sample.fork` vs `observe` / `StatefulTool`）是否**真的可复用**，还是只对 deep-research 这一种形态有效。

## ② 两条稀疏注意力路线能否合流

`HiLS-Attention`（可学习 chunk 选择，面向 LLM 服务）与 `Prism`（内容驱动 block 形状，面向生成）**共享核心作者但代码库完全独立**。技术上存在合流空间：前者的「可学习选择」+ 后者的「按内容分配粒度」如果结合，可能同时适用于长上下文与高分辨率生成。

## ③ Agent 侧的「缺失」是「不做」还是「没开源」

`C3-Benchmark` 迁出到 `WildToolBench` 是个信号。需要确认：混元是否有**未开源**的通用 agent runtime / memory / RAG 工作，还是战略上就不做这一层。

---

## 附录：本文件引用的一手数据来源

| 论断 | 来源 |
|---|---|
| RoboTwin 90.9% / 90.1%、消融 38% / 44%、UMI 10,000+ 小时 | `_research/readmes/Hy-Embodied-0.5-VLA.md` L28, L358 |
| Hy-Embodied ~30B / ~3B、38 benchmarks / 19 个第一 | `_research/readmes/HY-Embodied.md` L33, L36, L178–179 |
| DFlash 2.14×、3.032s→1.408s | `_research/extra/hyocr_benchmark.md` L18–23 |
| Hy-MT2 1.25-bit 440MB / 1.5× | `_research/readmes/Hy-MT2.md` L29 |
| Hy4-preview 双官方镜像与启动参数 | `_research/readmes/Hy4-preview.md` L158–198 |
| Hy3 部署与 MTP 参数 | `_research/readmes/Hy3.md` L141–203 |
| HiLS-Attention SGLang 三条约束 | `_research/readmes/HiLS-Attention.md` L225–264 |
| AuK 显存实测表 | `_research/readmes/AuK.md` L392–402 |
| Prism 显存门槛与稀疏参数 | `_research/readmes/Prism.md` L316–342 |
| 85 仓库分类与许可证 | `_research/repos-classification.csv`、`_research/licenses.csv` |

> 完整清单见 [`INDEX.md`](INDEX.md) 第七节「报告索引」。
