# 混元推理侧增量扫描 · 2026-10-26 → 2026-11-09

> **本次任务**：在既有基线（[`INDEX.md`](../../INDEX.md) + [`infer/2026-10-26/06-infer-world-model-and-serving.md`](../2026-10-26/06-infer-world-model-and-serving.md)）之上，扫描 **2026-10-26 之后**混元在推理方向的新增仓库、代码变更、公开信息，并按需求方要求纳入**本地商发分支 `D:\WorkSpace\vllm-ascend-hy4`** 的实测与 **Hy4 正式版**专项排查。
>
> **窗口**：`2026-10-26T10:30+08:00` → `2026-11-09T11:20+08:00`。
> **基线**：`_research/2026-10-26/org-repos-2026-10-26.csv`（90 仓库，Infer 59）。
> **本次实测**：组织现有 **95 个公开仓库**（+5，Infer +2）；HF `tencent` 窗口内新模型 **1 个**（非混元团队的 DSpine 27B）；arXiv 核验 **4 篇**；本地商发分支 **127 commits 中 11 个 HY4 相关提交**逐一取证。

**数据来源标注约定**（沿用前两轮，新增"本地代码实测"一类）

| 标记 | 含义 |
|---|---|
| **仓库实测** | raw.githubusercontent.com / `gh api`（本轮起认证）读原文 |
| **GitHub API** | commits / releases（`published_at` 为权威发布时间） |
| **HF API / 模型·数据集卡** | huggingface.co/api 一手元数据 |
| **arXiv API** | 标题/日期核验（本轮 4 篇全部通过） |
| **本地代码实测** | `D:\WorkSpace\vllm-ascend-hy4\repo` git 历史 + 源码（内部制品，一手） |
| **官方公告（媒体逐字引用）** | 人民网/央广网/财联社转述 Hy4 发布口径 |
| **媒体转述/自媒体** | 单独标注，内容农场不采信 |
| `README 未说明` | 不做推断补全 |

---

## 0. 一句话结论

窗口内推理侧的主线**不是新模型权重**（截至 11-09 HF 无新混元权重），而是四件事：

1. **Hy4 正式版（11-04）以"API/产品先行、权重后置"的节奏发布**——首个引入视觉理解与 Computer Use 的混元旗舰，幻觉率 12.95%→7.2%，定价不变；**权重承诺"近期开源"但截至 11-09 未上 HF**。GitHub 侧 `Hy4-preview` 仓库自 8-28 后零推送。
2. **稀疏注意力完成"架构→训练系统→内核"的全栈开源**：新仓库 `FlashTopK`（11-05）是 HyScale 长上下文训练系统的核心内核（1.7-3.3× over dense scoring + FlashInfer top-k），**支持矩阵明确点名 Hy4-preview 的 32×128 indexer**——与 Hy4 README 的 Gated DSA + IndexCache + top-k 2048 完全互证。
3. **`Hy-SimulMT`（10-26 发布，恰在上轮扫描后 2 小时）补齐语音推理线**：30B-A3B 同传模型 6 个权重（BF16/FP8/GGUF × 两种输出模式），StreamLAAL 约 3 秒，三条部署路径（vLLM / llama.cpp PR / SGLang-Omni）。
4. **本地商发分支证实：昇腾侧对 Hy4（含视觉）的 day0 适配早于正式版两周**——`vllm-ascend-hy4`（hy4-v0.30.0 分支）在 **10-19/10-20** 已完成 HY4 MXFP8 文本与图像/视频链路验证（8×Ascend950DT），视觉塔规格可从代码读出；官方 NPU 镜像 `quay.io/ascend/vllm-ascend:hy4-a3` 同期存在。

---

## 1. 增量总览

### 1.1 组织级增量（推理侧相关）

| 仓库 | 类型 | created / 公开 | stars（现） | License | 与推理主线的关系 |
|---|---|---|---:|---|---|
| **Hy-SimulMT** | **补录** | 10-22 建 / **10-26 12:53 导入**（上轮快照后 2 小时） | 263 | API 报 NOASSERTION（README 未标注，待核 LICENSE 文件） | ⭐ 实时同传：30B-A3B + SimulInfer/SimulTrain |
| **FlashTopK** | **新** | 11-05 | 10 | Apache-2.0 | ⭐ 稀疏注意力 fused top-k 内核（HyScale） |
| Hy-Nomas | 触及 | 10-21 | 540（+380） | Apache-2.0 | ⭐ 技术报告上 arXiv + 社区集成爆发 |
| Hy-Memory | 触及 | 10-15 | 2,012（+730） | Apache-2.0 | SDK 2.0.6→2.0.14（见 agent 报告 §3） |
| AuK | 触及 | 08-19 | 1,589（+43） | MIT | docs：社区微调 model cards |
| Prism | 触及 | 09-21 | 1,513（+368） | MIT | 仅 README 署名调整（热度继续与代码脱钩） |
| 其余（UniRL/UniCom/DiagSafe/RoboFusion/Hunyuan3D-1/OmniWeaving/ExplorationBench/Hyra-results） | 触及 | — | — | — | 见 agent 报告 §1.2 / §5 |

> **补录说明**：`Hy-SimulMT` 的 `created_at=2026-10-22` 早于上轮快照时刻，但上轮组织 API 未返回它（当时不可见），且首条内容导入 commit 为 **2026-10-26T04:53:43Z**（= 北京时间 12:53，上轮扫描 10:30 之后）。**实质是"上轮扫描后 2 小时公开"**，性质同上轮 HYDRA（补录），不是上轮漏采。

### 1.2 关注度增量

| 仓库 | 10-26 | 11-09 | Δ | 同期代码变更 |
|---|---:|---:|---:|---|
| Hy-Memory | 1,282 | 2,012 | **+730** | ✅ 295 commits（真实迭代支撑） |
| Hy-Nomas | 160 | 540 | **+380** | ✅ TR + 社区集成 + demo 图像参考 |
| Prism | 1,145 | 1,513 | +368 | ❌ 仅 3 条 README 署名提交 |
| Hy-SimulMT | — | 263 | （4 天） | ✅ 发布 + llama.cpp 文档 |
| AuK | 1,546 | 1,589 | +43 | docs |
| UniRL | 1,025 | 1,045 | +20 | fix 提交 |

---

## 2. 平台级事件：Hy4 正式版（11-04）—— 推理与部署视角

> Agent 能力细节见 [`agent/2026-11-09/07-...`](../../agent/2026-11-09/07-agent-benchmarks-and-hy4-agentic.md) §2。本节只写**推理侧事实**。

| 项 | 事实 | 来源级别 |
|---|---|---|
| 发布日期 | **2026-11-04** | 官方公告（人民网/央广网/财联社逐字一致） |
| 形态 | **API/产品侧发布**：腾讯云 TokenHub、WorkBuddy/CodeBuddy、元宝、企微"大圆"、ima；API 接入 **OpenRouter、Hermes、OpenCode** | 官方公告 |
| 参数 | "770B"（系列口径）；"770B 总参/49B 激活 MoE"仅 AI产品库转述（B 级），与 preview 相同——**正式版规格无一手源** | 中 |
| 新能力 | **视觉理解**（首次）+ **Computer Use**（首次）+ 长程智能体/指令遵循提升 | 官方公告 |
| 质量口径 | 幻觉率 12.95%→7.2%、猜测错误率 23.1%→5.6%（自建测试）；GameDev +19.1% / GameLogic +9.8%（vs preview） | 官方自评 |
| 定价 | 与 preview 持平：6 / 18 / 0.3 元每百万 tokens；TokenHub 5 折至 12-31；WorkBuddy/CodeBuddy 11-04~15 免费 | 官方公告 |
| **权重** | **"近期开源模型权重 + 模型系统卡片"——截至 2026-11-09 未开源**（HF `tencent` 组织最新混元权重仍是 10-24 的 `Hy-Nomas-Diffusers`；`Hy4-preview` / `-FP8` 停在 8-27） | HF API 实测 |
| 后续 | **Hy4 flash** 近期发布（低成本版）；**Hy5** 规模继续扩大；"平均每两个月一个大版本"节奏 | 官方公告 |
| 开源安全口径 | 主动剔除攻击性网络安全训练数据；生物安全持续评估；"更负责任的开源" | 官方公告 |

**推理侧含义（三条）**：

1. **发布节奏改变**：Hy3/Hy4-preview 是"权重当天开源"；Hy4 正式版是**"产品/API 先行，权重承诺后置"**。对下游推理栈（vLLM/SGLang recipe、NPU 适配）意味着：**正式版的部署验证全部悬空**——现有 vLLM recipe（`recipes.vllm.ai/tencent/Hy4-preview`）、SGLang cookbook、NPU 镜像全部只对 preview 有效。正式版是否带视觉塔、MTP 层是否变化，将直接决定这些 recipe 是否需要重写。
2. **视觉能力 = 新推理负载**：Hy4 正式版首次引入视觉理解，意味着推理栈要支持图文混合 prefill + 视觉编码器。**本地商发分支已提前就绪**（见 §6）。
3. **"每两个月一个大版本"**：按此节奏 Hy4 preview（8-28）→ Hy4 正式版（11-04）间隔 10 周；Hy5 预计年底/明年初。**下一次扫描（12 月上旬）重点观察 Hy4 权重开源与 Hy4 flash。**

---

## 3. 核心增量 A：`FlashTopK` —— 稀疏注意力内核，从"模型架构选择"到"全栈工程线"

- 仓库：<https://github.com/Tencent-Hunyuan/FlashTopK>（created 11-05，v1 11-06，Apache-2.0）
- 论文：**HyScale: Scaling Long-Context Training for Learned Sparse Attention**，arXiv **2610.32057**（v1 **10-23**，v3 11-04——本轮 API 核验）
- 团队署名：**"from HyScale (Tencent Hy)"**；论文作者 11 人（Yangshen Deng 等，bibtex `deng2026hyscale`）

### 3.1 内核事实（README 照抄）

| 项 | 值 |
|---|---|
| 定位 | 快速、**精确**、**确定性**的 fused top-k 选择内核，用于 learned sparse attention 的**训练与推理** |
| 性能 | full prefill 上比 dense scoring + FlashInfer top-k（DeepGEMM/cuDNN scoring）快 **1.7-3.3×**（trained indexer）、**1.2-2.0×**（untrained）；H800 实测 |
| 确定性 | 同输入同 bit；k-th 分数平局取最小 key index |
| 精确性 | 采样阈值漏掉的行在 GPU 上重打分（"the true top-k"） |
| CP 兼容 | 每行阈值来自自身窗口的 strided sample，**context parallelism 无需跨 rank 选择** |
| 支持模型 | **GLM-5.1 / 5.2 / 5.3 / 5.3-Flash、Hy4-preview**（indexer 32 heads × 128），token-level selection 同 DeepSeek Sparse Attention（DSA） |
| 硬件 | **NVIDIA Hopper（SM90）**：H800 / H20 / H100 / H200 |
| 操作数 | FP8（带 per-key scale）或 BF16；`topk` ≤ 4096 |
| 依赖 | Python 3.10+ / torch 2.6+ / **Triton 3.3+ / CuTe DSL（nvidia-cutlass-dsl）4.5+** |
| 上游 | 基于 LiteTopK（gate 校准、候选选择）与 DoSA（剪枝分数界），NOTICE 列明派生文件 |
| Roadmap | block-level LSA（DeepSeek-V4 的 CSA、Qwen3.8-Flash-Next 的 QSA）；**paged KV cache**（当前要求连续 KV）；64-head indexer（如 DeepSeek-V3.2） |

### 3.2 HyScale 论文（arXiv 摘要要点）

- 核心洞察："**selection 的输出不是 activation，而是一组 indices**" → ① **FlashTopK**：无 score table 的 on-the-fly 精确 top-k；② **selection replay**：recomputation 时复用前向选择结果。
- 端到端：**GLM-5.1 在 1M-token 序列上训练加速 2.07×（vs Megatron-LM，512×H20）**；**DeepSeek-V4-Flash 的 CSA 层 2.01×、Qwen3.8-Flash-Next 的 QSA 层 1.57×**（1M tokens，32×H800）。
- 摘要称 learned sparse attention "自 DeepSeek-V3.2 起在生产中广泛采用"。

### 3.3 为什么这条线重要（本轮新增的结构性事实）

1. **Hy4 用稀疏注意力拿到内核级实证**：Hy4-preview README 写明 attention 为 **Gated DSA + IndexCache（Indexer 32 heads / 128 dim，top-k 2048）**——与 FlashTopK 支持矩阵逐字吻合。上轮我们只知道"Hy4 的 vLLM 启动参数用 `FLASHMLA_SPARSE`"；本轮起，**其 indexer 形状、top-k、内核实现、训练系统全部开源可复算**。
2. **跨厂商模型统一支持**：FlashTopK 同时支持 GLM-5.x 与 Hy4（同为 32×128 indexer），roadmap 指向 DeepSeek CSA 与 Qwen QSA——混元把"DSA 生态的公共基础设施"当作开源标的，**标准输出策略**：让自家实现成为多家模型共用的内核，反哺自身架构路线的行业地位。
3. **稀疏注意力仓库线的第三代**：`HiLS-Attention`（层间层次稀疏 + SGLang 后端）→ `flex-block-attn`（块稀疏 CUDA）→ **`FlashTopK`（fuse + 精确 + 确定性 + CP 兼容）**。三代一条主线：**给"训练 + 推理"两端的 selection 瓶颈做系统**。

---

## 4. 核心增量 B：`Hy-SimulMT` —— 同传作为独立推理负载（补录）

- 仓库：<https://github.com/Tencent-Hunyuan/Hy-SimulMT>（created 10-22，内容 10-26 12:53 导入）
- 权重：HF collection `tencent/hy-simulmt`（6 个，**createdAt 2026-10-22**——上轮扫描时 HF API 未返回，应为当时私有）· ModelScope 同步
- 发布口径：README News "2026.10.26 发布"；百度百科/媒体报道同日确认
- 架构（第三方读 config.json，转引自机器学习AI算法工程公众号转载文，**与 README 部署段互证**）：`HYV3SpeechForCausalLM`（model_type `hy_v3`）、48 层、hidden 2048、32 头/4 KV 头、**max_position 262144（256K）**、128 路由专家 + 1 共享、top-8 激活、首层 dense；音频编码器 24 层、128 维 mel、16 kHz、单请求上限 **30 秒**（`max_samples=480000`）

### 4.1 资产矩阵（6 权重，README 照抄）

| 输出模式 | BF16 | FP8 | GGUF |
|---|---|---|---|
| 仅译文（target-only） | `Hy-SimulMT-30B-A3B` | `-FP8` | `-GGUF` |
| 原文+译文（Interleave） | `-Interleave` | `-Interleave-FP8` | `-Interleave-GGUF` |

下载量（11-09 实测）：GGUF 两版合计 **8,438**（4434 + 4004），远超 BF16/FP8——端侧/本地需求为主。

### 4.2 评测口径（README 数字照抄）

- **RealSI 长音频**：StreamLAAL **约 3 秒**（参评系统最低）；GEMBA 高于对比开源模型、超过 5 个商用 API 中的 4 个（打分模型 GPT6.1-Sol）。
- **短音频 4 集**：FLEURS **90.3** / CoVoST2 75.5 / ACL6060 **81.2** / RealSI-Sentence **84.7**（全表最高）；短板是 CoVoST2（75.5 vs Qwen3.8-LiveTranslate 82.1、Gemini3.5 74.4——**开源对比模型里 Qwen3.8 更高**）。
- **人类对照**：18 段 2026-09 后中英音频（每段约 6 分钟）× 6 位持证译员（3×CATTI 一级、3×二级）：**Hy 3.3 秒 vs 一级 4.5 秒 vs 二级 4.7 秒**；质量"达到人类同传专家水平"。
  > **口径提醒**：3 秒（RealSI 集）与 3.3 秒（18 段自制集）是**两个不同集合**的数字；StreamLAAL 衡量流式滞后，**不等于首次开口延迟**——README 明确标注。
- 背书：广外高翻副院长王巍巍（"先开口、后修补"）、仲伟合教授。范围限定声明：仅适用本次样本/译员/设置。

### 4.3 部署三路（README 照抄）

| 路线 | 覆盖 | 关键约束 |
|---|---|---|
| **vLLM + SimulInfer**（主推） | 全部 6 权重 | Linux、Python 3.12+、vLLM **≥0.31** 且支持 `HYV3ForCausalLM` + `simulst` 包；`deploy.sh --install-server` 一键；网页 8080 / 模型 8800；支持 WebSocket 实时与 TTS 音色克隆（TTS 另配） |
| **llama.cpp** | GGUF | 架构支持以 **PR #31198** 提交官方 llama.cpp，**尚未合并**；当前 `llama-mtmd-cli` 可推理、`llama-server` 待适配；桥接端口 8094 |
| **SGLang-Omni** | 仅 BF16 target-only | BF16 权重约 **57 GiB，单张 H200** 可部署；`sgl-omni serve --enable-realtime`；实测比 Transformers 最高 **6.5×** 加速（H200、bs1，cookbook 口径）；单请求 ≤30 秒、每 1.6 秒刷新假设、30 秒切分段；**FP8/GGUF/Interleave 不支持** |

配套 **SimulTrain**（从 checkpoint 继续训练，`.env` 配置）与 **Hy 翻译 App**（首发渠道）。

> **与语音线的关系**：上轮的 `AuK` 是语音**生成/编辑**基座；`Hy-SimulMT` 是**同传**负载。两者共用 **hy_v3 架构栈 + 30B-A3B MoE 模板**（见 §7 判断⑧）——混元语音方向已经形成"基座 + 应用负载"的双仓结构。

---

## 5. 其他推理相关增量

### 5.1 `Hy-Nomas`：技术报告落地 + 社区集成爆发（stars +380）

- **技术报告**：PDF 10-26 进仓库（`Hy_Nomas_Tech_Report_v1.pdf`），**10-27 上 arXiv 2610.36729**《Hy-Nomas: Towards Free-form Interactive World Generation》（本轮 API 核验，v1）。**上轮风险表 #1"TR Coming Soon"解除**——但**质量/速度指标仍以仓库 README 与模型卡为准**，TR 的第三方复测仍未出现。
- **风险表 #2 更新**：`Hy-Nomas-Bench` 仍无官方模型分数。
- **社区集成**（README News）：SGLang **PR #46124**、ComfyUI（社区）、**WanGP v18.81**（10-25，键盘探索 + 文本事件）、Reactor Runtime、**Google TPU demos**（10-27，社区贡献）、LTX-2.5 给视频配音（11-03，社区）、demo 支持**图像参考初始化**（10-31 官方）。
- **读法**：单卡 4090 路线（上轮 §2.4）+ Diffusers 打包 + 多运行时社区移植，使 Hy-Nomas 成为混元生成模型里**社区生态最宽**的一个；TPU 移植说明对硬件的绑定是松的。

### 5.2 `Qwen3.8-27B-DSpine-b16`（11-07，HF，非混元团队）

- 窗口内 HF `tencent` 组织唯一新模型。DSpine 投机解码 drafter 从上轮的 8B（Qwen3-8B）扩展到 **27B**（Qwen3.8-27B 基座）。
- 归属沿用上轮口径：**作者与仓库均未自述混元，按"腾讯其他团队"处理**。写入本报告仅因其是 SGLang 生态的投机解码工作，与混元 LLM 的部署栈相邻。

### 5.3 一句话增量（MAINTENANCE/TRIVIAL 级，附证据）

- **AuK**（10-27）：docs——共享社区微调结果的 model cards（PR #30）。蒸馏管线无变化。
- **Prism**（10-31）：3 条 README 提交，作者署名加星标/顺序调整；**+368★ 无代码支撑**（传播事件，见 agent 报告 §5）。
- **UniCom**（11-05）：camera-ready 作者列表/致谢（论文定稿级）。
- **DiagSafe**（10-26~11-02）：docs badge、恢复被覆盖的 89% 一致率数字、judge 客户端参数校验、页面文案。
- **Hunyuan3D-1**（11-02）：社区 PR #67 加固本地 demo 默认值与权重加载。
- **OmniWeaving**（10-30）：README 两连更。
- **RoboFusion**（11-03）：**新增 `LICENSE.txt`（MIT）**——上轮"无 LICENSE 12 个"高风险清单修订为 11 个（⚠️ 又是 `License.txt` 大小写变体，GitHub API 仍报 NOASSERTION）。
- **EmoReward**（11-02/11-05）：情感视频生成 RL 的自进化奖励校准（arXiv + HF badges；README 明确写了"给 agent 的使用说明"章节，但定位是视频生成训练组件）——归"其余"，不进推理主线。

---

## 6. 本地商发分支实测：`D:\WorkSpace\vllm-ascend-hy4`（需求方指定）

> 该目录是 2026-10-26 从 `srv177-8-21` 搬回本机的 **vLLM-Ascend 商发分支**阅读环境：`repo\`（git 克隆，分支 `hy4-v0.30.0`，基线上游 tag **v0.30.0rc1**，`v0.30.0rc1..HEAD` 共 **127 commits**，其中 **11 个 HY4 直接相关**）。本轮对其做了完整取证。

### 6.1 HY4 适配提交链（git log 实测，时间倒序）

| commit | 内容 |
|---|---|
| `a74b08e3f` / `7b7f7e3f2` | attention sink scale 的 FP32 处理修复与回退（迭代对） |
| `a35d92121` | **perf(hy4)**：小 batch iHC 在 Ascend 上融合 |
| `251d23f17` | **fix(hy4)**：MXFP8 activation 量化对齐 GPU 语义 |
| `9acae310d` | 从主 checkpoint 加载 HY4 ModelSlim MTP |
| `7b3a8b927` | 多模态位置下保留 HY4 MTP embeddings |
| `aa431fece` | **HY4 day0 loading** 与 ModelSlim 视觉初始化对齐 |
| `4eafbf16b` | 遵循 HY4 checkpoint 预处理与视频像素上限 |
| `274ccff73` | **feat：HY4 图像/视频推理上 Ascend** |
| `187a35a0d` / `394b07cd0` | 记录 HY4 多模态/GPQA 验证结果 |
| `c4714a5b1` | **feat：HY4 MXFP8 文本推理适配 Ascend** |

### 6.2 两个验证报告的关键事实（`tools/hy4/REPORT.md` / `MULTIMODAL_REPORT.md`，一手）

| 项 | 文本报告（10-19） | 多模态报告（10-20） |
|---|---|---|
| 权重 | `/mnt/weight/Hy4-preview-FP8` 的 **ModelOpt MXFP8** 版，约 **757.88 GiB（含 MTP）** | 同 MXFP8 文本+MTP 权重；**缺失的 337 个视觉张量用固定种子 20261020 随机初始化**（仅验链路，非能力评测） |
| 硬件/栈 | **8×Ascend950DT（96 GiB/卡）**、CANN 9.1.0、torch 2.10 / torch-npu 2.10.0.post4、transformers 5.14.1、vLLM 0.30.0 | 同左 |
| 接入点 | 共享 TopK、interleaved RoPE、attention sink、gate、上游 **iHC**；复用 Ascend **SFA** 执行路径；`kv_b_proj` 恢复浮点以被 MLA 吸收 | 组合 HY4 文本 + 视觉编码器 + 投影 + 图像/视频处理器 + **三维 MRoPE**（sections `[12,10,10]` interleaved）；SFA/indexer/MTP 独立旋转位置 |
| **视觉塔规格（代码可读）** | — | **27 层、hidden 1152、16 heads × head dim 72、patch 16、spatial merge 2、newline、BF16 视觉计算** |
| 分阶段验证 | eager → 异步 → FULL_DECODE_ONLY ACLGraph → 1-token MTP → **PD 分离（MooncakeConnectorV1，TP4+TP4）** → 快速单实例；全部 5/5 通过 | eager → async → graph → MTP → PD → 快速单实例；文本 5/5、多模态 6/6 全通过 |
| 精度 | GPQA Diamond 198 题（no_think、约束选项、T=0；**不与公开 CoT 分数可比**） | —（明确声明随机视觉权重不评语义） |

### 6.3 与 Hy4 正式版的关系（本轮最重要的交叉推断）

1. **时间线**：多模态链路验证完成于 **10-20 01:14**（北京时间），比 Hy4 正式版发布（11-04）**早 15 天**，比上轮扫描早 6 天。即：**当混元对外还在预热 Hy4 正式版时，昇腾生态内部已在为"带视觉的 Hy4"做 day0 适配**。
2. **"day0" 是仓库自述用词**（`aa431fece fix: align HY4 day0 loading...`）——商发分支以"发布日当天可用"为目标工程化。
3. **`HYV4VLForConditionalGeneration` / `hy_v4_vl` config**（`vllm_ascend/patch/hy4_vl/`，1832 行）：Hy4 的视觉-语言变体在 transformers config 层已有注册名。**公开 HF 上不存在该架构的权重**——它是为正式版/内部版本预留的接入面。**推断（标注：据本地代码推断）**：Hy4 正式版"首次视觉理解"的载体很可能就是这个 Hy4-VL 形态（文本塔 + 视觉塔 27 层/1152/16×72/patch16 + MRoPE [12,10,10]），**以官方权重开源为准**。
4. **官方 NPU 路径（`docs/source/tutorials/models/Hy4-preview.md`）**：镜像 `quay.io/ascend/vllm-ascend:hy4-a3`，**仅 Atlas A3**（单机 16 NPU TP16；双机 DP2 至 96k 上下文；**单机仅约 1k 上下文**——762GB W8A8 权重 `Eco-Tech/Hy4-preview-w8a8`@ModelScope 挤占显存）；支持 MTP / ACLGraph / NPUGraph EX / 前缀缓存；代码未合并进 vLLM-Ascend 上游。**注意**：该教程页是 preview 口径，正式版权重开源后预计同步更新。
5. **对 vllm-ascend 主线的影响**：商发分支的 iHC 融合、MXFP8 语义对齐、SFA 复用等 127 个提交，是 **Hy4 系模型在 Atlas A3 上 day0 可用的工程底座**；其向上游的合并进度是下次扫描的观察点（上轮 `Hy3-preview` 页已有 "discontinued since v0.30.0" 记录——混元模型在 vllm-ascend 上的支持是**按版本窗口滚动**的）。

---

## 7. 对照 10-26 报告的推理侧判断

| 10-26 判断 | 本轮（11-09）证据 | 结论 |
|---|---|---|
| **④ 世界模型复制 LLM 推理栈，但"交互延迟"缺口留白** | TR 上线（2610.36729）但仍无吞吐/延迟数字；社区移植爆发（TPU/SGLang/ComfyUI/WanGP） | **成立**——TR 补的是方法叙事，延迟缺口依旧 |
| **"LLM 侧投机解码默认开"** | DSpine drafter 扩展至 27B（非混元）；本地分支继续调 MTP（对齐/嵌入修复）；Hy4 教程维持 MTP 推荐位 | **成立并扩展** |
| **"生成模型默认加速 = 少步蒸馏 + 量化"** | Hy-SimulMT 6 权重含 FP8/GGUF 四档；Hy-Nomas 无新量化 | **成立** |
| **"量化三轨"（AngelSlim / 仓库自研 W8-FP8 / llama.cpp imatrix）** | 本轮无新量化配方；Hy-SimulMT 的 FP8/GGUF 归入既有分轨 | **成立** |
| **"官方一等公民 = vLLM + SGLang"** | Hy-SimulMT 主推 vLLM（≥0.31）+ SGLang-Omni 二路；Hy-Nomas 社区 SGLang PR；FlashTopK 是内核层（超越框架叙事） | **成立并下沉**——竞争焦点从"框架支持"下移到"内核/算子层" |
| **"没有任何官方吞吐/延迟数字"** | Hy-SimulMT 给了 SGLang-Omni 6.5×（vs Transformers，H200 bs1）与 StreamLAAL 3 秒——**同传负载首次有官方延迟口径**；Hy4/Nomas 仍无 | **部分修订**——分负载看：语音负载已给延迟口径，旗舰 LLM 与世界模型仍无 |

**本轮新增判断**：

> **⑬ Hy4 正式版确立"API/产品先行、权重后置"发布模式。**
> 与 preview 的"当天开源"不同，正式版先在 TokenHub/WorkBuddy/元宝/ima/OpenRouter 全量上线，权重与系统卡片"近期开源"。**含义**：① 下游推理栈对正式版的适配将集中在一个滞后的窗口完成（等权重）；② 混元开始用"产品可用性"而非"开源热度"作为发布首日的叙事载体；③ 按两个月节奏，**下一次扫描（12 月上旬）大概率覆盖 Hy4 权重开源 + Hy4 flash，并可能触到 Hy5 预热**。
>
> **⑭ 稀疏注意力成为"架构→系统→内核→NPU"四层全栈线。**
> 模型层：Hy4 Gated DSA + IndexCache（32×128 indexer / top-k 2048）；训练系统层：HyScale（2.07× @1M tokens）；内核层：FlashTopK（精确/确定性/CP 兼容，Apache-2.0）；NPU 层：本地商发分支的 SFA/indexer 接入。**且内核同时支持 GLM-5.x/DeepSeek CSA/Qwen QSA 的 roadmap**——混元在把 DSA 做成跨厂商公共设施，这是比"又一个加速库"更进取的生态卡位。
>
> **⑮ "30B-A3B 工作负载专用小模型"成为可复制模板。**
> Hy-Memory-30B-A3B（记忆写入，~3B 激活）→ Hy-SimulMT-30B-A3B（同传，128+1 专家 top-8）——同一套"hy_v3 栈 + 30B 总参/约 3B 激活 + 专用任务"模板连续复用，都带 FP8/GGUF 双量化线与 vLLM 优先部署。**含义**：混元的开源权重策略是"旗舰开源树标杆 + 专用小模型铺负载"，后者才是产业落地的主力形态（GGUF 下载量 8.4k vs BF16 千级即为佐证）。
>
> **⑯ 旗舰模型的昇腾 day0 是制度化动作。**
> 本地商发分支在正式版发布前两周完成 Hy4（含视觉）全链路验证 + 官方 NPU 镜像同期存在 + "day0 loading" 作为 commit 主题——**Hy 系旗舰的"开源日/发布日"同时是"Atlas A3 day0 日"**。对国产算力选型的含义：Hy 系模型的 NPU 可用性不再是"社区后补"，而是发布工程的一部分。

---

## 8. 风险与待核实

| # | 事项 | 现状 | 建议动作 |
|---|---|---|---|
| 1 | **Hy4 正式版权重未开源** | 官方承诺"近期"；截至 11-09 HF 无 | 12 月上旬复扫第一优先级；核对正式版架构（视觉塔/MTP/参数）与 preview recipe 的适用性 |
| 2 | **正式版参数口径** | "770B/49B"仅 AI产品库转述且与 preview 相同 | 一手源出现前不写死正式版参数 |
| 3 | **Hy-SimulMT 的 LICENSE** | GitHub API 报 NOASSERTION，本轮未拉取其 LICENSE 文件原文 | 引用许可前先读仓库 LICENSE 文件（注意大小写变体） |
| 4 | **Hy-SimulMT 架构细节** | 48 层/256K/128+1 专家等来自第三方读 config（转载文） | 用 HF 权重 config.json 亲自复核后再引用 |
| 5 | **SimulMT 与 Qwen3.8-LiveTranslate 的 CoVoST2 差距** | 75.5 vs 82.1，官方表格如实列出 | 官方未解释；评测口径（打分模型）未披露细节 |
| 6 | **FlashTopK 性能数字** | 1.7-3.3× 为官方报告（含 `reproduce/` 复现脚本） | 复现脚本齐全，可信度较高；引用注明 H800/训练后 indexer 前提 |
| 7 | **Hy-Nomas 仍无分数与吞吐** | TR 已上 arXiv 但交互延迟缺口未填 | 自建"指令→下一段可见"时延基线（上轮建议继续有效） |
| 8 | **DSpine 27B 归属** | 沿用"腾讯其他团队"口径 | 需归属口径时先向官方确认 |
| 9 | **TailorRig 论文** | arXiv 2611.02902 未核验（API 503） | 下次扫描补核；归 3D 线 |
| 10 | **paooo.com 内容农场** | "GEMBA 85.2 分"等 README 没有的精确数字 | 不采信 |
| 11 | **本地商发分支的时效** | `repo\` 是 10-26 快照，127 commits 停在该日 | 若需跟踪 day0 后的适配进展，需重新拉取 bundle（`fetch-bundle.ps1` 可复用） |

---

## 9. 证据来源与采集脚本

| 产物 | 位置 | 说明 |
|---|---|---|
| 组织全量快照（95 仓库） | [`../../_research/2026-11-09/org-repos-2026-11-09.csv`](../../_research/2026-11-09/org-repos-2026-11-09.csv) | created/pushed/stars/license/default_branch |
| 增量表（17 行） | [`../../_research/2026-11-09/delta-2026-11-09.csv`](../../_research/2026-11-09/delta-2026-11-09.csv) | NEW 4 / TOUCHED 12 / 补录 1 |
| 提交/Release（认证 API） | [`../../_research/2026-11-09/commits-summary.json`](../../_research/2026-11-09/commits-summary.json) · `releases-summary.json` | 17 仓库窗口内提交 |
| HF 模型/数据集 | [`../../_research/2026-11-09/hf-tencent-models.json`](../../_research/2026-11-09/hf-tencent-models.json) · `hf-tencent-datasets.json` | models 100 条 / datasets 26 条 |
| README 快照（16 份） | [`../../_research/2026-11-09/readmes/`](../../_research/2026-11-09/readmes/) | FlashTopK / Hy-SimulMT×2 / HyperLogic / AppCIF / Hy-Nomas / Hy-Memory×2 / Hy4-preview / HunyuanOCR 等 |
| arXiv 核验 | 本报告 §3/§5/§6（2610.32057、2605.19597、2610.36729 逐一经 API 核验；2611.02902 因 503 未核验） | — |
| 本地商发分支 | `D:\WorkSpace\vllm-ascend-hy4\repo`（git log 实测 + `tools/hy4/REPORT.md`、`MULTIMODAL_REPORT.md`、`vllm_ascend/patch/hy4_vl/`、`docs/source/tutorials/models/Hy4-preview.md`） | 一手内部制品 |
| 采集脚本 | `scan_org.py` / `scan_detail.py` / `fetch_readmes.py` | 同目录可重跑（gh 已认证） |

**关键链接**：<https://github.com/Tencent-Hunyuan/FlashTopK> · <https://github.com/Tencent-Hunyuan/Hy-SimulMT> · <https://github.com/Tencent-Hunyuan/Hy-Nomas> · <https://arxiv.org/abs/2610.32057>（HyScale） · <https://arxiv.org/abs/2610.36729>（Hy-Nomas TR） · <https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16> · <https://www.cls.cn/detail/2525324>（Hy4 正式版）
