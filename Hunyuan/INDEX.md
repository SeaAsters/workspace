# 腾讯混元（Tencent-Hunyuan）开源仓库调研 · Agent 与推理方向

本目录是对 GitHub 组织 [Tencent-Hunyuan](https://github.com/Tencent-Hunyuan) **全部 85 个公开仓库**的系统扫描，聚焦其中 **Agent** 与 **推理（infer）** 两条主线。

> **说明**：本次交付为**调研报告**（不包含仓库克隆）。所有结论均基于各仓库 README / LICENSE 原文，关键数字与命令照抄原文并注明来源；原文未说明的一律标注 `README 未说明`。
> 仓库在持续更新，命令与版本请以仓库当前 README 为准。

---

## 目录结构

```
D:\WorkSpace\workspace\Hunyuan\
├── INDEX.md                              ← 本文件（总索引 + 范围界定 + 许可证风险）
├── 00-agent-vs-infer.md                  ← ⭐ 先读这个：两条线的对比分析（结论性总览）
├── clone-repos.ps1                       ← 按分类批量浅克隆脚本（需要时再跑）
│
├── agent\                                ← Agent 方向交付物
│   ├── 01-agent-frameworks-and-benchmarks.md   （框架 + 评测，7 个核心仓库）
│   └── 04-embodied-and-agentic-3d.md           （具身智能 + Agentic 3D，7 个仓库）
│
├── infer\                                ← 推理方向交付物
│   ├── 02-inference-optimization-libs.md       （稀疏注意力 / 高效推理库，8 个）
│   └── 03-model-release-deployment-matrix.md   （模型发布库部署矩阵，10 个重点 + 全量索引）
│
└── _research\                            ← 原始素材（可溯源，约 5.1 MB）
    ├── repos-classification.csv          ← 85 个仓库完整分类表（机器可读）
    ├── licenses.csv                      ← 85 个仓库许可证实读结果（含 default branch）
    ├── readmes\                          ← 33 份仓库 README 原文快照
    └── extra\                            ← 68 份补充证据（LICENSE 原文 / HF 模型卡 / 子文档 / 脚本）
```

> **阅读顺序建议**：先看 [`00-agent-vs-infer.md`](00-agent-vs-infer.md) 拿到全局结论与两条线的性格对比，再按需深入 `agent/` 与 `infer/` 的明细报告。
>
> 所有报告结论均可回溯到 `_research\` 下的原始素材：README 快照、LICENSE 全文、HuggingFace 模型卡、仓库内子文档（如 `hy3_rl.md`、`hyocr_llama_cpp.md`）与实际脚本（如 `UniRL__train_agentic.py`、`hyocr_serve_dflash.sh`）。

---

## 一、范围界定

经与需求方确认，本次调研的口径为：

| 方向 | 收录范围 | 仓库数 |
|---|---|---|
| **Agent** | Agent 框架 + Agent 能力评测 + 具身智能（Embodied，服务于物理世界 agent）+ 机器人通信协议 | **14** |
| **Infer** | ① 推理优化/高效注意力库 ② **全部模型发布库**（含 vLLM / SGLang / TensorRT-LLM 部署路径与量化方案） | **56** |
| 其余 | RL / RLVR 后训练、扩散模型训练算法、代码/上下文评测基准、奖励模型、论文项目页等 | 15 |
| **合计** | | **85** |

> 分类明细见 [`_research/repos-classification.csv`](_research/repos-classification.csv)。

---

## 二、Agent 方向仓库（14 个）

| 仓库                                                                                    | 子类         | Stars | 定位一句话                                                                 |
| ------------------------------------------------------------------------------------- | ---------- | ----- | --------------------------------------------------------------------- |
| [UniRL](https://github.com/Tencent-Hunyuan/UniRL)                                     | Agent 训练框架 | 1011  | 统一多模态模型的 RL 后训练框架，含 SFT / **agentic** / async 训练模式                    |
| [C3-Benchmark](https://github.com/Tencent-Hunyuan/C3-Benchmark)                       | Agent 能力评测 | 38    | `C^3-Bench`：LLM Agent 工具使用鲁棒性评测（已迁至 WildToolBench）                    |
| [VisualNeedle](https://github.com/Tencent-Hunyuan/VisualNeedle)                       | Agent 能力评测 | 0     | 高信息密度场景的**主动视觉搜索** agent 评测（300 题）                                    |
| [RepoProbe](https://github.com/Tencent-Hunyuan/RepoProbe)                             | Agent 能力评测 | 13    | 仓库级代码理解 agent 评测，checklist 打分（ASE 2026）                               |
| [ExplorationBench](https://github.com/Tencent-Hunyuan/ExplorationBench)               | Agent 能力评测 | 7     | 探索能力评测                                                                |
| [PlanningBench](https://github.com/Tencent-Hunyuan/PlanningBench)                     | Agent 能力评测 | 27    | 规划能力评测                                                                |
| [ArtifactsBenchmark](https://github.com/Tencent-Hunyuan/ArtifactsBenchmark)           | Agent 能力评测 | 279   | `ArtifactsBench`：视觉交互式代码生成评测，1,825 任务 + MLLM-as-Judge + 细粒度 checklist |
| [Hyra-results](https://github.com/Tencent-Hunyuan/Hyra-results)                       | Agent 研究产物 | 159   | 递归自我改进 / AI4Science / AI4Fun 的研究产物                                    |
| [HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied)                         | 具身智能       | 876   | `Hy-Embodied-VLM-1.0`：面向物理世界 agent 的高效 MoE 视觉-语言基座（~3B 激活 / ~30B 总）   |
| [HY-Embodied-0.5-X](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X)             | 具身智能       | 67    | 增强版具身基座，聚焦「理解-推理-行动」闭环                                                |
| [Hy-Embodied-0.5-VLA](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA)         | 具身智能       | 307   | 从 VLA 模型到真实机器人学习栈（HY-VLA-0.5，2000+ 小时 UMI 数据）                         |
| [Hy-Embodied-RxBrain-1.0](https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0) | 具身智能       | 131   | 联合语言-视觉推理与想象的具身认知基座                                                   |
| [Hy-Embodied-RoboFusion](https://github.com/Tencent-Hunyuan/Hy-Embodied-RoboFusion)   | 具身智能       | 35    | RoboFusion 机器人控制系统 UDP 通信协议规范（PDO/SDO，C 实现）                           |
| [Hunyuan3D-WorldClaw](https://github.com/Tencent-Hunyuan/Hunyuan3D-WorldClaw)         | Agentic 生成 | 1377  | `WorldClaw`：Agentic 3D 开放世界生成（⚠️ 仓库极简，仅 README + 论文）                  |

**观察**：混元的 Agent 工作呈现**「框架少、评测多、具身强」**的结构 —— 唯一可运行的通用 Agent 训练框架是 `UniRL`，但评测类占了 6 个；而**具身智能**是一条独立且投入很重的线（Tencent Robotics X × HY Vision Team，5 个仓库，且已下沉到机器人控制系统通信协议层）。

---

## 三、推理方向仓库（56 个）

### 3.1 推理优化 / 高效注意力库（8 个）— 详见 [`infer/02-inference-optimization-libs.md`](infer/02-inference-optimization-libs.md)

| 仓库                                                                                                    | Stars | 定位一句话                                                   |
| ----------------------------------------------------------------------------------------------------- | ----- | ------------------------------------------------------- |
| [HiLS-Attention](https://github.com/Tencent-Hunyuan/HiLS-Attention)                                   | 150   | 层次化稀疏注意力 + **SGLang HSA 后端**（唯一直接面向 LLM 在线服务的）          |
| [flex-block-attn](https://github.com/Tencent-Hunyuan/flex-block-attn)                                 | 132   | 块稀疏注意力 CUDA 库（STA / MoBA / SSTA），Hopper SM90            |
| [Prism](https://github.com/Tencent-Hunyuan/Prism)                                                     | 174   | 2K 联合视频-音频生成的**动态稀疏注意力**（2.5× 训练加速）                     |
| [Rosetta-inference](https://github.com/Tencent-Hunyuan/Rosetta-inference)                             | 9     | 可组合原生多模态预训练（15 个 checkpoint + 全套评测）                     |
| [AuK](https://github.com/Tencent-Hunyuan/AuK)                                                         | 1487  | 语音生成与编辑基座（vLLM-Omni / SGLang-Omni / GGUF / MLX 全覆盖，MIT） |
| [Simple-Attention-Sparsification](https://github.com/Tencent-Hunyuan/Simple-Attention-Sparsification) | 66    | SAS：端到端上下文排序优化的注意力稀疏化（NeurIPS 2026）                     |
| [FlashVDM](https://github.com/Tencent-Hunyuan/FlashVDM)                                               | 335   | Vecset 扩散模型快速形状生成（ICCV'25 Highlight）                    |
| [DisCa](https://github.com/Tencent-Hunyuan/DisCa)                                                     | 25    | 蒸馏兼容的可学习特征缓存，加速视频 DiT                                   |

### 3.2 模型发布库（48 个）— 详见 [`infer/03-model-release-deployment-matrix.md`](infer/03-model-release-deployment-matrix.md)

按代次与模态分组：

| 分组 | 代表仓库 | 推理框架支持要点 |
|---|---|---|
| **旗舰 LLM** | `Hy4-preview`(770B-A49B)、`Hy3`(295B-A21B)、`Hy3-preview` | vLLM + SGLang，均开 **MTP/投机解码**；Hy4 有**官方预构建镜像** |
| **翻译模型** | `Hy-MT2`、`Hy-MT`、`Hunyuan-MT` | vLLM / SGLang + **AngelSlim 1.25-bit → 440MB** + GGUF |
| **中小 LLM** | `Hunyuan-A13B`、`Hunyuan-7B`、`Hunyuan-4B/1.8B/0.5B`、`Hunyuan-TurboS(-3)`、`Tencent-Hunyuan-Large`、`Tencent-Hunyuan-7B-0124` | **部署文档最全**：TensorRT-LLM + vLLM + SGLang + FP8/Int4 |
| **多模态理解** | `HunyuanOCR`、`HunyuanVision` | OCR：**DFlash 投机解码 + llama.cpp PC 端**；Vision：仅 README |
| **图像生成 / 编辑** | `HunyuanImage-3.0/2.1`、`HunyuanDiT`、`iFSQ`、`HY-WU` | 扩散推理；`HY-WU` 含推理代码与权重 |
| **视频生成** | `HunyuanVideo(-1.5/I2V/Avatar/Foley)`、`HunyuanCustom`、`Hunyuan-GameCraft-1.0`、`HunyuanPortrait`、`InstantCharacter`、`OmniWeaving` | 扩散推理 |
| **3D 生成** | `Hunyuan3D-2.1/2/1/Omni/Part/Buffalo1.0`、`bpt` | 扩散推理 |
| **世界模型** | `HY-World-2.0`、`HunyuanWorld-1.0(-Voyager/-Mirror)`、`HY-WorldPlay` | 扩散推理 |
| **其他生成** | `HY-Motion-1.0`、`R-DMesh`、`UniCom`、`HY-SOAR` | 扩散推理 |

---

## 四、未收录（15 个，out-of-scope）

| 分组 | 仓库 |
|---|---|
| RL / RLVR 后训练（8） | `Precise`、`MixGRPO`、`SRPO`、`SAGE-GRPO`、`HY-Video-PRFL`、`GradLoc`、`DRIVE-RLVR`、`Thinking-Free_Policy_Initialization` |
| 生成模型训练算法（2） | `GEAR`、`Twins` |
| 评测基准 / 数据集（4） | `AutoCodeBenchmark`、`CL-bench`、`HY3D-Bench`、`evolve-scaler` |
| 奖励模型（1） | `UnifiedReward` |

> 这些仓库与 Agent 框架/推理部署无直接关系（多为训练算法、数据集或论文 artifact），如后续需要可单独成篇。

---

## 附：无代码仓库清单

以下仓库**只有 README / 论文 / 项目页，没有可运行实现**，扫描时不要期待有代码：

`HunyuanVision`、`Hunyuan-TurboS`、`Hunyuan-TurboS3`、`PlanningBench`、`DRIVE-RLVR`、`Hunyuan3D-WorldClaw`、`evolve-scaler`、`OmniWeaving`（无 README）

> ⚠️ 若目标是「拿到推理代码」，请优先跳过以上仓库。

---

## 五、跨方向关键发现

### 5.1 推理侧的三个明确信号

1. **官方镜像化交付成为新标准，但覆盖不均是明确分水岭**
   `Hy4-preview` 直接给出 `vllm/vllm-openai:hy4-preview` 与 `lmsysorg/sglang:hy4-preview` 官方预构建镜像（后者还是 multi-arch，含 Arm）；`Hunyuan-7B`/`Hunyuan-A13B` 提供 TensorRT-LLM / vLLM / SGLang 的预构建 docker。但 **`Hy3` / `Hy3-preview` / `Hy-MT2` 仍必须源码编译**。
   **TensorRT-LLM 是明显短板**：README 中提供 TRT 路径的只有 `Hunyuan-A13B`、`Hunyuan-7B`、`HY-MT1.5` 等**老产品线**；四代旗舰（`Hy3`、`Hy3-preview`、`Hy4-preview`）与 `Hy-MT2`、`HunyuanOCR` **均无 TRT 路径**。官方一等公民已收敛为 **vLLM + SGLang**。

2. **投机解码（MTP）默认开启**
   `Hy4-preview`（`--speculative-config.method mtp` + `--attention-backend FLASHMLA_SPARSE`）、`Hy3`（`--speculative-config.method mtp`）、`HunyuanOCR`（**DFlash** 投机解码）都把投机解码写进推荐启动命令，而非可选项。

3. **量化工具链统一收敛到 AngelSlim**
   `Hy3`、`Hy4-preview`、`Hunyuan-A13B`、`Hy-MT2` 都指向腾讯自研压缩工具 [AngelSlim](https://github.com/Tencent/AngelSlim)，覆盖 FP8-static / GPTQ-AWQ W4A16 / **1.25-bit 极低比特**；`Hy-MT2-1.8B` 经 1.25-bit 量化后**存储仅 440MB**、推理速度 **1.5×**。

### 5.2 Agent 侧的三个观察

1. **评测先行，框架稀缺** —— **6 个** Agent 能力评测基准（`C3-Benchmark`、`VisualNeedle`、`RepoProbe`、`ExplorationBench`、`PlanningBench`、`ArtifactsBenchmark`）vs **仅 1 个**通用 Agent 训练框架（`UniRL`）。混元在 Agent 上的投入明显偏向「测量能力」而非「提供框架」。
2. **具身智能是独立重线** —— **5 个**具身仓库由 Tencent Robotics X 主导，已下沉到真实机器人学习栈与**通信协议层**（`Hy-Embodied-RoboFusion` 是电机/传感器 UDP 协议规范）。它与「LLM 工具调用型 Agent」是**两套技术栈**：前者是感知-动作闭环，后者是文本工具调用。
   > 最强证据：`Hy-Embodied-0.5-VLA` 在 RoboTwin 2.0 取得 **90.9% / 90.1%**（Clean/Randomized），而**去掉 UMI 预训练的消融塌陷到 38% / 44%** —— 大规模 UMI 语料构成「本体无关的动作先验」。
3. **评测方法论在向「可验证」演进** —— `RepoProbe` 用**加权 checklist + 分级 rubric** 替代 LLM-as-a-Judge 的标量打分（并据此发现 clarity 84.7 vs knowledge 60.3 的「流畅度跑赢正确性」现象）；`ExplorationBench` **完全弃用 LLM judge**，判定全部来自 interpreter / proof checker；`VisualNeedle` 用 **crop-black 消融**（同格式、把返回图换成同尺寸黑图）把「格式合规」与「真用上证据」分开度量（Tool-enabled 56.01% vs No-tool 17.89% vs 人类 63.00%）。这类设计比单一分数更能定位模型缺陷。

### 5.3 ⚠️ 许可证风险提示

对 **全部 85 个仓库**逐一读取了 LICENSE 文件原文（结果存于 [`_research/licenses.csv`](_research/licenses.csv)）。**分布如下**：

| License 类型 | 数量 | 商用判断 |
|---|---:|---|
| **Tencent 社区许可** —— `TENCENT (HUNYUAN/HY/HY-WORLD...) COMMUNITY LICENSE AGREEMENT` | **38** | ⚠️ **中 — 原文含明文地域排除**：`THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA`。在 EU/UK/韩国不可用，商用前必读全文（还含 Acceptable Use Policy 与附加商业条款） |
| **Apache-2.0** | **18** | ✅ 低 |
| **MIT** | **13** | ✅ 低 |
| **CC-BY-4.0**（数据集/benchmark 类） | **3** | ✅ 低（记得署名） |
| **Custom（需人工审阅）** | **1** | ⚠️ 见下 |
| **❌ 无 LICENSE 文件** | **12** | 🔴 **高 — 默认保留全部权利，法律上不可商用** |

**明细**：

- **MIT（13）**：`AuK`、`Prism`、`UniCom`、`UnifiedReward`、`MixGRPO`、`SRPO`、`Precise`、`HY-SOAR`、`HunyuanPortrait`、`InstantCharacter`、`Twins`、`Hyra-results`、`Thinking-Free_Policy_Initialization`
- **Apache-2.0（18）**：`Hy3`、`Hy4-preview`、`Hy-MT2`、`flex-block-attn`、`UniRL`、`RepoProbe`、`VisualNeedle`、`ExplorationBench`、`Rosetta-inference`、`HY-Embodied`、`Hy-Embodied-0.5-VLA`、`Hy-Embodied-RxBrain-1.0`、`AutoCodeBenchmark`、`CL-bench`、`GEAR`、`GradLoc`、`HY-Video-PRFL`、`iFSQ`
  > `flex-block-attn` 的 `LICENSE.txt` 原文：「flex block attn is licensed under **Apache-2.0**, except for the third-party components」——宽松许可只覆盖本项目代码，**第三方组件沿用各自原许可**。
- **CC-BY-4.0（3）**：`ArtifactsBenchmark`、`C3-Benchmark`、`PlanningBench` —— 注意这 3 个**不是** Apache-2.0
- **Custom（1）**：`Hunyuan3D-1`（自定义条款，需人工审阅全文）
- 🔴 **无 LICENSE（12）**：`HiLS-Attention`、`DisCa`、`Simple-Attention-Sparsification`、`R-DMesh`、`Hunyuan3D-Buffalo1.0`、`Hunyuan3D-WorldClaw`、`Hy-Embodied-RoboFusion`、`Hunyuan-TurboS`、`Hunyuan-TurboS3`、`HunyuanVision`、`DRIVE-RLVR`、`evolve-scaler`
  → 含**本次重点关注的 `HiLS-Attention`**。这 12 个仓库若要用在商业产品中，必须先联系作者取得授权。

**两个易踩的坑**（本次实测踩到）：

1. **文件大小写不统一** —— `Hy-MT`、`Hunyuan-MT`、`HY-Motion-1.0`、`Hunyuan3D-Omni`、`HunyuanWorld-Mirror` 等用的是 `License.txt`（而非 `LICENSE`）。只查 `LICENSE` 会**误判为无许可证**。
2. **默认分支不统一** —— ⚠️ **5 个仓库的默认分支是 `master` 而不是 `main`**：`HY-Embodied`、`HY-Embodied-0.5-X`、`HY-Motion-1.0`、`SAGE-GRPO`、`Twins`。任何按 `main` 抓 raw 文件的自动化都会在这 5 个仓库上**静默漏掉所有文件**。
   > 这正是 `HY-Embodied`（实为 **Apache-2.0**）与 `HY-Embodied-0.5-X`（实为 **Tencent 社区许可**）初次扫描被判为「无许可证」的原因，已核实修正。

**关于 README 自述的免责说明**：README 徽章/文字经常与仓库实际 LICENSE 不一致（例如 `HY-Embodied-0.5-X` 的仓库结构树把 `LICENSE` 注释成 `# Apache-2.0`，实际文件却是 Tencent 社区许可）。**一律以 LICENSE 文件与 HuggingFace 模型卡为准。**

> 另注：`LICENSE` 里的宽松许可通常只覆盖**代码**，并常附加「except for the third-party components」；**权重**可能适用单独协议。请以模型卡为准。

---

## 六、如何使用克隆脚本

报告只是扫描结果。若后续需要把代码拉到本地：

```powershell
cd D:\WorkSpace\workspace\Hunyuan

# 只拉 Agent 方向（12 个仓库）
pwsh -File clone-repos.ps1 -Category agent

# 只拉推理方向（51 个仓库）
pwsh -File clone-repos.ps1 -Category infer

# 两个方向都拉
pwsh -File clone-repos.ps1 -Category agent,infer

# 需要完整 Git 历史（占用大、耗时长）
pwsh -File clone-repos.ps1 -Category infer -FullHistory

# 更新已有仓库
pwsh -File clone-repos.ps1 -Category agent -Pull
```

脚本特性：

- **幂等** —— 已存在的目录自动跳过，可反复执行；
- **默认浅克隆** `--depth 1`，且设 `GIT_LFS_SKIP_SMUDGE=1` **跳过 LFS 大文件**（模型权重、演示视频）。⚠️ 注意 `HunyuanWorld-Voyager`（~850MB）、`HY-World-2.0`（~340MB）、`Hunyuan3D-Buffalo1.0`（~294MB）等仓库体积较大，浅克隆 + 跳过 LFS 后仍可能较重；
- 克隆目标路径为 `<分类>\<仓库名>`，即 `agent\UniRL`、`infer\Hy3` 等；
- 失败会在结尾汇总并以退出码 1 结束。

---

## 七、报告索引

| 文件 | 内容 | 规模 |
|---|---|---|
| [`00-agent-vs-infer.md`](00-agent-vs-infer.md) | ⭐ **对比分析（建议先读）**：Agent 线在「定义怎么衡量能力」vs 推理线在「解决怎么把能力跑起来」；`UniRL` 的三个 agentic RL 设计、评测方法论三阶段演进、稀疏注意力双路线、量化/投机解码/部署形态收敛、两条线的唯一交汇点与后续深挖方向 | 全篇 |
| [`INDEX.md`](INDEX.md) | 本文件：范围界定、85 仓库分类、许可证风险、跨方向发现、克隆指南 | — |
| [`agent/01-agent-frameworks-and-benchmarks.md`](agent/01-agent-frameworks-and-benchmarks.md) | Agent 框架与能力评测深度分析：`UniRL`（agentic RL 一等路径）、`C3-Benchmark`、`VisualNeedle`、`RepoProbe`、`ExplorationBench`、`PlanningBench`、`Hyra-results` | 780 行 |
| [`agent/04-embodied-and-agentic-3d.md`](agent/04-embodied-and-agentic-3d.md) | 具身智能与 Agentic 3D：`HY-Embodied` 系列 5 个 + `Hunyuan3D-WorldClaw` + `Hunyuan3D-Buffalo1.0`，含具身栈全景图与「与 LLM 工具调用型 Agent 的差异」 | 1,108 行 |
| [`infer/02-inference-optimization-libs.md`](infer/02-inference-optimization-libs.md) | 推理优化与高效注意力库深度分析：`HiLS-Attention`、`flex-block-attn`、`Prism`、`Rosetta-inference`、`AuK`，含环境/命令/显存/性能原文摘录 | 747 行 |
| [`infer/03-model-release-deployment-matrix.md`](infer/03-model-release-deployment-matrix.md) | 模型发布库推理框架支持矩阵（vLLM / SGLang / TensorRT-LLM / llama.cpp）、部署命令摘录、量化与投机解码专题、选型建议 | 1,691 行 |
| [`_research/repos-classification.csv`](_research/repos-classification.csv) | 85 仓库完整分类表（repo / category / subcategory / stars / license / 备注） | 85 行 |
| [`_research/licenses.csv`](_research/licenses.csv) | 85 仓库许可证实读结果（license / license_file / **default branch**） | 85 行 |
| `_research/readmes/` | 33 份仓库 README 原文快照 | — |
| `_research/extra/` | 68 份补充证据（LICENSE 原文、HF 模型卡、子文档、评测脚本） | — |
