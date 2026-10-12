# 腾讯混元（Tencent Hunyuan）具身智能与 Agentic 生成方向开源工作调研报告

> **调研范围**：`Tencent-Hunyuan` 组织下 7 个「具身智能（Embodied）+ Agentic 3D 生成」相关仓库 —— `HY-Embodied`、`HY-Embodied-0.5-X`、`Hy-Embodied-0.5-VLA`、`Hy-Embodied-RxBrain-1.0`、`Hy-Embodied-RoboFusion`、`Hunyuan3D-WorldClaw`、`Hunyuan3D-Buffalo1.0`。
>
> **与系列其他报告的关系**：本报告是 Agent 方向系列的第 4 篇。第 1 篇《[01-agent-frameworks-and-benchmarks.md](./01-agent-frameworks-and-benchmarks.md)》已覆盖 `UniRL`、`C3-Benchmark`、`VisualNeedle`、`RepoProbe`、`ExplorationBench`、`PlanningBench`、`Hyra-results` 七个「LLM 工具调用型 Agent」仓库，本报告**不重复**这部分内容，两者互补。
>
> **证据基线**：主体事实来自 7 份仓库 README。其中 `HY-Embodied.md`、`HY-Embodied-0.5-X.md`、`Hunyuan3D-WorldClaw.md` 来自本地既有目录 `D:\WorkSpace\workspace\Hunyuan\_research\readmes\`，其余 4 份（`Hy-Embodied-0.5-VLA.md`、`Hy-Embodied-RxBrain-1.0.md`、`Hy-Embodied-RoboFusion.md`、`Hunyuan3D-Buffalo1.0.md`）由本报告从 `raw.githubusercontent.com` 现取并落盘到同一目录。凡 README 未说明的内容，一律标注 `README 未说明`，不做推断性补全。
>
> **数据来源标注约定**：
> - 「README 自述」= README 正文/表/README 内嵌图片之外可读的文字内容，属于仓库自述；
> - 「README 时间线」= README 的 News / Updates / 版本历史段落里写明的日期，同样属于仓库自述；
> - `[model-card]` = 数据来自 Hugging Face 模型卡（`huggingface.co/tencent/<MODEL>/raw/main/README.md`），**不是** GitHub README 内容，两者出现分歧时本报告会同时列出；
> - 「stars（外部探测）」/「最近推送（外部探测）」= 本报告编写时从工作区既有仓库元数据快照 `_research/repos-classification.csv` 读取的数值，**不属于 README 内容**，会随时间变化；
> - `GitHub REST API` 全程 403（限流），因此**没有**任何数据来自该 API；star 数只使用上述本地快照，不做二次抓取。
>
> **本轮现场校验（原文件实读，非 README 转述）**：
> - `HY-Embodied`、`HY-Embodied-0.5-X`、`Hy-Embodied-RoboFusion`、`Hunyuan3D-WorldClaw`、`Hunyuan3D-Buffalo1.0` 五个仓库的默认分支是 **`master`** 而不是 `main`（`raw.githubusercontent.com/.../main/...` 对这些仓库全部 404，这正是本报告需要现场核实的原因）。
> - `HY-Embodied/LICENSE`、`HY-Embodied/Hy-Embodied-VLM-1.0/LICENSE`（均为 11,813 字节，内容一致）、`HY-Embodied-0.5-X/LICENSE`（15,979 字节）、`Hy-Embodied-0.5-VLA/LICENSE`（11,624 字节）、`Hy-Embodied-RxBrain-1.0/LICENSE`（11,700 字节）已逐一下载实读；`Hy-Embodied-RoboFusion`、`Hunyuan3D-WorldClaw`、`Hunyuan3D-Buffalo1.0` 在 `main` 与 `master` 两个分支上均**不存在** `LICENSE` / `LICENSE.txt` / `LICENSE.md`。
> - **一处工作区缓存纠错**：既有快照 `_research/licenses.csv` 把 `HY-Embodied-0.5-X` 与 `HY-Embodied` 标为「❌ 无 LICENSE 文件」。实读证明两者**都**带 `LICENSE`，只是位于 `master` 分支；`HY-Embodied` 确为 Apache-2.0，而 `HY-Embodied-0.5-X` 是**带地域限制的 Tencent HY Community License**（详见 2.2 与 5.3）。

---

## 1. 概览总表

| 仓库 | 定位 | 模型规模（总/激活） | stars（外部探测） | 最近更新（外部探测 / README 时间线） | License | 备注 |
|---|---|---|---|---|---|---|
| [`HY-Embodied`](https://github.com/Tencent-Hunyuan/HY-Embodied) | 具身基座模型（`Hy-Embodied-VLM-1.0`）+ 0.5 代归档目录 | `Hy-Embodied-VLM-1.0`：**~30B 总 / ~3B 激活**（8 of 128 experts + 1 shared，MoE）；`Hy-Embodied-0.5 MoT-2B`：4B 总 / 2B 激活 | 876 | 2026-07-15 / README Updates 最新一条为 **[2026-07-15] 发布 Hy-Embodied-VLM-1.0** | Apache-2.0（实读 `master/LICENSE`，11,813 字节，含 Apache-2.0 全文 + Tencent 声明头） | 默认分支 `master`（`main` 404）；仓库为**多版本布局**，0.5 代文档已迁入 `Hy-Embodied-0.5/`；不提供微调代码（Plannings 中 `Fine-tuning Code` 未勾选） |
| [`HY-Embodied-0.5-X`](https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X) | 具身基座模型（增强版，面向「理解-推理-行动」闭环） | 4B 总 / 2B 激活（基于 `HY-Embodied-0.5 MoT-2B` 架构） | 67 | 2026-05-14 / README Updates：**[2026-04-24] 发布 HY-Embodied-0.5-X** | **Tencent HY Community License**（实读 `master/LICENSE`，15,979 字节，标题为 "TENCENT HY COMMUNITY LICENSE AGREEMENT"，Release Date: April 22, 2026，含地域与 Acceptable Use Policy 限制）。**README 未说明 License**；`[model-card]` 元数据标 `license: other` | 默认分支 `master`；**7 个仓库中唯一带完整 SFT 微调代码 + 文档体系（`docs/`）的具身基座**；评测分数字在图片里，README 正文无数字 |
| [`Hy-Embodied-0.5-VLA`](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA) | VLA / 机器人学习栈（`Hy-VLA`，`HY-VLA-0.5`） | 未给整体参数量；action expert：**370M 参数**（`[model-card]`：dual-tower，hidden=1024，intermediate=2048） | 307 | 2026-08-04 / README Updates 最新一条为 **[2026-07-09] 新增 RoboDojo benchmark 支持** | Apache-2.0（README 明示 "Released under the **Apache-2.0 License**"；实读 `main/LICENSE` 为 Apache-2.0 全文 + Tencent 声明头） | 默认分支 `main`；**唯一带真实机器人实机评测数字的仓库**；含 2K+ 小时 UMI 数据；FlowPRO 训练算法标注「under review, code coming soon」 |
| [`Hy-Embodied-RxBrain-1.0`](https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0) | 具身认知基座（联合语言-视觉推理与想象） | **~6.2B**（`[model-card]` 补充：MoT 三流 text/vision/generation）；外挂 FLUX VAE `ae.safetensors` 83.8M | 131 | 2026-07-28 / README News 最新一条为 **[2026-07] 发布微调代码**（无具体日） | Apache-2.0（README 明示 "Released under **Apache License 2.0**"；实读 `main/LICENSE` 为 Apache-2.0 全文 + Tencent 声明头，`[model-card]` 元数据亦为 `apache-2.0`） | 默认分支 `main`；**评测章节只写「详见技术报告」，无任何 benchmark 数字**；`RxBrain-Bench` 尚未开源 |
| [`Hy-Embodied-RoboFusion`](https://github.com/Tencent-Hunyuan/Hy-Embodied-RoboFusion) | 通信协议层（RoboFusion 机器人控制系统 UDP 协议规范，C 实现） | **不适用（无神经网络模型）** | 35 | 2026-05-15 / README 文档信息：创建 2026-04-13、最后更新 2026-04-14、版本 1.0 | `README 未说明`（README 全文无 License 章节；实读 `main`/`master` 均无 `LICENSE` 等文件） | 默认分支 `main`；语言为 C；README 即协议规范正文（PDO/SDO、设备类型、端口、C 结构体、控制模式表） |
| [`Hunyuan3D-WorldClaw`](https://github.com/Tencent-Hunyuan/Hunyuan3D-WorldClaw) | Agentic 3D 开放世界生成（研究发布） | `README 未说明`（README 未给任何模型名/参数量） | 1377 | 2026-08-13 / README News：**2026.08.07 Paper 与 project page 发布** | `README 未说明`（实读 `main`/`master` 均无 `LICENSE`） | 默认分支 `master`；**README-only 仓库**：现场读取文件树仅 `README.md`、`.gitignore`、`assets/`，**无任何代码**；README 仅 40 行，Method 章节只有一张 `assets/pipeline.jpg` |
| [`Hunyuan3D-Buffalo1.0`](https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0) | 统一多模态 3D 生成 / 理解 / 编辑（研究发布） | `README 未说明` | 256 | 2026-08-10 / README News：**2026.08.05 Paper 与 project page 发布** | `README 未说明`（实读 `main`/`master` 均无 `LICENSE`） | 默认分支 `main`；**README-only 仓库**：现场读取文件树仅 `README.md`、`assets/`，**无任何代码**；README 仅 49 行 |

> **规模口径提示**：`Hy-Embodied-VLM-1.0` 的 README 在 Abstract 里提到上一代基线为「the previous-generation model with **32B activated parameters**」，同时 Model Card 表明确写 `Total parameters ~30B` / `Activated parameters per token ~3B`。**30B 是总参数、3B 是每 token 激活**，二者不可混用；「上一代 32B 激活」指的是上一代系统的激活量口径，不要误读成 VLM-1.0 的总参数。

---

## 2. 逐个深挖

### 2.1 `HY-Embodied` — Hy-Embodied-VLM-1.0

#### 项目定位与要解决的问题

项目自我定位为 **"Efficient Physical-World Agents"**，出品方为 *Tencent Robotics X × Hy Vision Team × Futian Laboratory*。要解决的问题是：具身 agent 不只需要多模态感知理解，还需要「对行动进行推理、适应变化中的情境、与物理世界交互」的 agentic 能力 —— 而这类能力**必须从预训练阶段就开始培养**，而非只在后训练阶段外挂。

为此作者定义了 **action-centric capability taxonomy**（以行动为中心的能力分类法），并围绕它系统性地设计数据管线与预训练/后训练数据混合，同时用 MoE 把推理成本压到延迟敏感场景可接受的水平。

#### 核心架构 / 方法

- **骨干**：`Hy3-A3B` 语言骨干 + `Hy-ViT2` 视觉编码器，组合为 MoE 架构。
- **模型类名**（README Model Card 表）：`HYV3VLForConditionalGeneration`（VL wrapper，内部为 `HYV3ForCausalLM` MoE LLM + `Hy-ViT2`）；`Model type` 为 `hy_v3_vl`。vLLM 插件 README 进一步说明插件向 vLLM 注册的正是 `HYV3VLForConditionalGeneration` 与 `HYV3ForCausalLM` 两个模型类。
- **效率口径**：每 token 仅激活 ~3B 参数（8 of 128 experts + 1 shared），约为上一代 A32B 系统激活参数的十分之一，而整体性能接近。
- **能力分类法（三层递进，全报告的骨架）**：
  1. **Action-Relevant State Understanding** —— 准确理解 agent 自身与所处环境的状态；
  2. **Action–Transition Reasoning** —— 理解动作、规划动作、推理动作后果；
  3. **Sequential and Adaptive Reasoning** —— 长程规划、反思、修复与恢复。
- **自演化后训练（Self-Evolving Post-Training）**：用一个「强化学习 ↔ 拒绝采样微调」耦合的自演化循环来培养具身 agentic 推理，种子是一小批精选的高质量 thinking traces。最后有一个 **reward-specialized 阶段**：分别训练 continuous-reward 与 discrete-reward 两种 RL 策略再融合，以同时获得「锐利的几何精度」与「稳健的决策/规划/反思质量」。
- **混合推理模式**：同一份权重内同时训入 thinking / non-thinking 两种模式，按请求用 chat-template kwarg 切换（详见下文命令）。

#### 关键目录与文件（仅列 README/子文档真实出现的路径）

| 路径 | 说明 |
|---|---|
| `Hy-Embodied-VLM-1.0/inference/transformers/infer_hf.py` | HuggingFace `transformers` 单实例推理 demo（单图 + 小 batch，thinking / 非 thinking 两种模式） |
| `Hy-Embodied-VLM-1.0/inference/vllm/` | vLLM 插件（`uv pip install -e` 安装，注册 `HYV3VL` 模型与解析器） |
| `Hy-Embodied-VLM-1.0/inference/vllm/serve.sh` | 启动 OpenAI 兼容服务（默认 `TP=4`） |
| `Hy-Embodied-VLM-1.0/inference/vllm/README.md` | 插件与服务完整选项 |
| `Hy-Embodied-VLM-1.0/inference/vllm/example_client.py` | OpenAI-SDK Python 客户端示例（含图片输入与流式） |
| `Hy-Embodied-VLM-1.0/figures/1-teaser.png`、`2-taxonomy.png`、`3-navigation-agent.png` | README 引用图 |
| `Hy-Embodied-VLM-1.0/figures/r2r-ce-instruction-following-navigation.webp`、`matterport3d-object-goal-navigation.webp` | 导航 demo 视频 |
| `hy_embodied_vlm_1_0_tech_report.pdf` | 技术报告 PDF（仓库根） |
| `Hy-Embodied-0.5/` | 0.5 代（`HY-Embodied-0.5 MoT-2B`）文档与代码归档目录 |
| `LICENSE`、`Hy-Embodied-VLM-1.0/LICENSE` | 两份内容一致（均 11,813 字节，实读验证） |

> ⚠️ README 的安装命令写的是 `git clone .../HY-Embodied` 后 `cd HY-Embodied`，但仓库**实际默认分支是 `master`**；README 里的相对链接（如 `./Hy-Embodied-VLM-1.0/inference/vllm/README.md`）在 GitHub 页面上被解析为 `blob/master/...`。用脚本按 `main` 抓取会 404。

#### 安装 / 推理命令（照抄 README）

依赖口径被固定为 `vllm==0.14.1` + `transformers==4.57.6` + `torch==2.9.1`。

```bash
# Install uv once (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a fresh venv (Python 3.10+)
uv venv --python 3.12
source .venv/bin/activate
```

**Option A — vLLM path（推荐用于服务化）**：

```bash
uv pip install vllm==0.14.1 --torch-backend auto
uv pip install -e Hy-Embodied-VLM-1.0/inference/vllm/    # our plugin
```

**Option B — HuggingFace `transformers` path（单实例推理）**：

```bash
uv pip install torch==2.9.1 torchvision==0.24.1 --torch-backend auto
uv pip install transformers==4.57.6 accelerate pillow
```

**Quick Start**：

```bash
git clone https://github.com/Tencent-Hunyuan/HY-Embodied
cd HY-Embodied

# transformers 单实例
cd Hy-Embodied-VLM-1.0/inference/transformers
python infer_hf.py

# vLLM 服务
MODEL_PATH=./Hy-Embodied-VLM-1.0-weights bash Hy-Embodied-VLM-1.0/inference/vllm/serve.sh
```

**调用 OpenAI 兼容端点**：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hy_a3b",
    "messages": [{"role": "user", "content": "Describe how to grasp a cup."}],
    "max_tokens": 512,
    "chat_template_kwargs": {"enable_thinking": true}
  }'
```

**验证插件已注册**（来自 vLLM 插件 README）：

```bash
python -c "
from vllm import ModelRegistry
assert 'HYV3VLForConditionalGeneration' in ModelRegistry.get_supported_archs()
print('OK: HYV3VLForConditionalGeneration registered')
"
```

**推理模式切换**（同一权重，按请求切换）：

| `enable_thinking` | Prompt suffix | 适用场景 |
|---|---|---|
| `True`（默认） | `<think>` | 复杂空间推理、规划、多步任务 |
| `False` | `<think></think>` | 直接作答、低延迟单轮问答 |

> README 专门解释了为什么用 `enable_thinking`（Qwen3 惯例）而不是 `reasoning_effort`：vLLM v0.22 之前存在顶层 `request.reasoning_effort` 字段**静默覆盖** `chat_template_kwargs["reasoning_effort"]` 的问题（由 [vllm-project/vllm#43401](https://github.com/vllm-project/vllm/pull/43401) 修复），`enable_thinking` 可规避该覆盖并在所有 vLLM 版本上一致工作。这是一个很典型的「上游字段冲突导致配置静默失效」的踩坑记录。

**vLLM 服务环境变量**（来自 `inference/vllm/README.md`）：

| Var | Default | Notes |
|---|---|---|
| `MODEL_PATH` | `tencent/Hy-Embodied-VLM-1.0` | HuggingFace id 或本地目录 |
| `SERVED_NAME` | `hy_a3b` | `/v1/models` 暴露的名字 |
| `PORT` | `8080` | HTTP 端口 |
| `TP` | `4` | 张量并行度（每副本 GPU 数） |
| `GPU_MEM_UTIL` | `0.85` | 预留显存比例 |
| `MAX_MODEL_LEN` | `32768` | 最大上下文长度 |
| `CHAT_TEMPLATE` | `$MODEL_PATH/chat_template.jinja` | 随权重一同发布 |
| `EXTRA_ARGS` | （空） | 额外 `vllm serve` 参数 |

就绪检查（首次加载约需 1–5 分钟）：`curl -sf http://127.0.0.1:8080/v1/models`；停止服务：`pkill -9 -f "vllm serve"`。

#### 模型权重地址与所需硬件

- **权重**：HuggingFace [`tencent/Hy-Embodied-VLM-1.0`](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0)；代码首次运行会自动从 Hub 下载。
- **Model Card（README 表）**：

| Field | Value |
|---|---|
| Architecture | `HYV3VLForConditionalGeneration`（VL wrapper over `HYV3ForCausalLM` MoE LLM + `Hy-ViT2` vision encoder） |
| Model type | `hy_v3_vl` |
| Total parameters | ~30B |
| Activated parameters per token | ~3B（8 of 128 experts + 1 shared） |
| Context length | 32,768 tokens |
| Precision | BF16 |
| Vision inputs | Image（单 prompt 最多 128 张）；native aspect ratios |
| Chat template | 随权重发布的统一 `chat_template.jinja`（支持 `enable_thinking` kwarg） |

- **硬件（README）**：训练/全精度推理需单节点 8×80 GB（H100 / H20 / A100 80G），权重 BF16 约 **~86 GB**，建议 tensor-parallel size 4–8；服务化推荐每副本 4×80 GB（tp=4）；开发调试任意 CUDA GPU。磁盘约 **~120 GB**（含 cache）。
- 前置条件：Linux（推荐）、Python 3.10+、CUDA 12.x（H100 / H20 / A100 实测）、PyTorch 2.4+。
- **训练代码**：README Plannings 显示 `Fine-tuning Code` **未勾选**，即本仓库不提供微调代码。

#### 评测（只写 README 真实出现的数字）

README 声明：在 **38 个**具身相关 benchmark 上评测，其中 **19 个排名第一**、**另外 11 个排名第二**；相对 `Qwen3.6-A3B` 平均领先 **+4.4%**（Abstract 中表述为 "substantially outperforms"）；相对上一代 `Hy-Embodied-0.5 MoT-2B` 平均性能提升 **8.4%**；在 R2R-CE（RGB-only 设定）上达到 SOTA，在 Matterport3D Object Goal Navigation 上有较强 zero-shot 表现。评测对象列包括 `Hy-Embodied 0.5 MoT-2B`、`Qwen3.6-A3B`、`Embodied-R1.5 8B`、`Cosmos3-Nano 8B`、`Hy-Embodied VLM-1.0 A3B` 五列。评测说明：Hy-Embodied 系列与 Qwen3.6-A3B 以 thinking 模式评测；`Embodied-R1.5-8B` 仅有 Instruct 配置；`Cosmos3-Nano-8B` 以 non-thinking 模式报告（开启 thinking 会显著降低其表现）。

**① Action-Relevant State Understanding（23 项）**

| Benchmark | Hy-Embodied 0.5 MoT-2B | Qwen3.6-A3B | Embodied-R1.5 8B | Cosmos3-Nano 8B | Hy-Embodied VLM-1.0 A3B |
|---|---|---|---|---|---|
| BLINK | 82.7 | **87.9** | 77.8 | 82.4 | 87.3 |
| CV-Bench | 89.2 | 88.6 | 86.8 | 88.0 | **89.7** |
| PixMo-Points | 51.4 | 57.5 | 57.1 | 59.8 | **64.6** |
| PointBench | 69.0 | 35.1 | 59.1 | 39.2 | **71.7** |
| Depth-InHouse | 45.7 | 63.0 | 52.0 | 47.0 | **67.6** |
| 3DSRBench | **57.0** | 49.9 | 42.6 | 31.9 | 52.6 |
| All-Angles-Bench | 55.1 | **64.0** | 48.4 | 51.9 | 63.4 |
| DA-2K | **92.3** | 81.4 | 80.5 | 82.8 | 83.2 |
| ERQA | 54.5 | 57.5 | 37.3 | 45.0 | **60.8** |
| EmbSpatial-Bench | 82.8 | **83.2** | 76.0 | 80.0 | 82.7 |
| MMSI-Bench | 33.2 | **41.9** | 29.8 | 34.0 | 41.8 |
| MindCube | 66.3 | 55.0 | 27.9 | 32.8 | **70.0** |
| SAT | 76.7 | **80.7** | 60.7 | 54.0 | 78.0 |
| SIBench-mini | 58.2 | 60.9 | 51.9 | 52.5 | **64.5** |
| SITE-Bench-Image | 62.7 | 71.7 | 60.3 | 59.6 | **72.3** |
| ViewSpatial-Bench | 53.1 | 49.0 | 43.7 | 52.0 | **53.3** |
| OpenEQA | 54.4 | **73.2** | 53.9 | 53.8 | 63.1 |
| PartAfford | 30.1 | 25.5 | **82.6** | 32.2 | 63.7 |
| RoboAfford | 73.5 | 66.7 | 60.6 | **76.2** | 71.5 |
| RoboRefIt | 82.8 | 78.5 | 77.2 | 55.4 | **88.2** |
| RefSpatial-Bench | 45.8 | 53.1 | 52.4 | 44.4 | **53.4** |
| RoboSpatial-Home | 55.7 | **70.9** | 69.1 | 58.3 | 69.4 |
| Where2Place | 68.0 | 70.0 | **73.0** | 71.0 | 65.0 |

**② Action–Transition Reasoning（8 项）**

| Benchmark | Hy-Embodied 0.5 MoT-2B | Qwen3.6-A3B | Embodied-R1.5 8B | Cosmos3-Nano 8B | Hy-Embodied VLM-1.0 A3B |
|---|---|---|---|---|---|
| FineBench | 56.9 | 76.9 | 67.1 | 63.5 | **80.3** |
| CrossHOI-Bench | 40.7 | 58.0 | 55.1 | 51.0 | **63.2** |
| PIO | 54.6 | 47.9 | 61.6 | 54.4 | **65.3** |
| VABench-Point | 26.0 | 50.5 | **61.4** | 45.2 | 59.7 |
| VABench-Visual-Trace | 75.0 | 80.3 | **89.8** | 81.6 | 79.7 |
| ShareRobot-Bench-Affordance | 26.8 | **28.2** | 25.2 | 23.0 | 26.7 |
| ShareRobot-Bench-Trajectory | 73.3 | 68.9 | 69.2 | 65.5 | **76.7** |
| RoboBench-MCQ | 49.2 | 59.1 | 41.1 | 43.5 | **61.2** |

**③ Sequential and Adaptive Reasoning（7 项）**

| Benchmark | Hy-Embodied 0.5 MoT-2B | Qwen3.6-A3B | Embodied-R1.5 8B | Cosmos3-Nano 8B | Hy-Embodied VLM-1.0 A3B |
|---|---|---|---|---|---|
| SITE-Bench-Video | 63.5 | **71.1** | 59.1 | 57.6 | 69.2 |
| VSIBench | **60.5** | 57.5 | 59.2 | 50.4 | 58.9 |
| EgoPlan2 | 45.5 | 49.9 | **61.0** | 42.6 | 49.6 |
| Cosmos | 54.3 | 67.8 | **68.6** | 67.1 | 66.9 |
| VLABench | 16.2 | 49.9 | 39.4 | 48.9 | **51.1** |
| RoboBench-Planning | 54.2 | 53.9 | 39.4 | 41.5 | **54.9** |
| RoboFAC | 35.6 | 41.4 | 43.9 | 34.4 | **51.0** |

> 以上三表共 38 项，与 README 声明的「38 benchmarks」一致；三张表的表头列数（5 数据列 + 1 benchmark 列）与分隔行列数逐一对应。

**Navigation Agent**：README 声明 Hy-Embodied-VLM-1.0 用**单一模型**支撑闭环导航（语言 grounding、空间记忆、规划、目标验证），给出两个场景 demo：Instruction-Following Navigation（R2R-CE, val-unseen）与 Zero-Shot Object-Goal Navigation（Matterport3D）——**demo 以 webp 视频给出，README 正文未给这两项的分数数字**（除 Abstract 的定性表述 "State-of-the-art on R2R-CE vision-and-language navigation (RGB-only setting)"）。

#### 论文 / arXiv、License

- 论文：*Hy-Embodied-VLM-1.0: Efficient Physical-World Agents*，**arXiv:2607.12894**，2026；技术报告 PDF 随仓库发布（`hy_embodied_vlm_1_0_tech_report.pdf`）。
- 上一代论文（同仓库 Citation 内）：*HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents*，**arXiv:2604.07430**，2026。
- License：**Apache License 2.0**。README 明示 "Released under **Apache License 2.0**. See `LICENSE` and `Hy-Embodied-VLM-1.0/LICENSE`."；两份 LICENSE 实读一致且含 Apache-2.0 全文。
- Acknowledgements：基于 [Hy3](https://huggingface.co/tencent/Hy3) MoE LLM 骨干与 [Hy-ViT2](https://huggingface.co/tencent/HunyuanOCR) 视觉编码器构建。

#### 亮点与局限

**亮点**
- **能力分类法先行**：三层 taxonomy（状态理解 → 动作-转移推理 → 序列/自适应推理）不是事后评测分组，而是明确用于「指导数据管线与数据混合的构建」——把评测维度和数据构建维度统一，这是具身基座里比较少见的方法论自觉。
- **激活参数效率**：~3B 激活 / ~30B 总，宣称达到上一代 32B 激活系统的接近性能，对机器人端侧/边缘部署意义直接。
- **推理/训练分离的插件化部署**：vLLM 插件通过 `vllm.general_plugins` entry-point 注册模型类与解析器，**不需要 fork vLLM**；且 README 明确承诺「上游 vLLM 原生合并后可直接卸载插件、用户侧代码零改动」。
- **混合推理模式做在同一份权重里**，且把 vLLM 字段冲突的坑写进文档。
- **明确提供 tool-call parser**：插件注册 `ToolParserManager.register_module("hy_v3", HYV3ToolParser)` 解析模型的 **XML 风格 tool-call 输出**，说明该具身基座在服务层同时具备工具调用通道。

**局限**
- **不提供微调代码**（Plannings 中 `Fine-tuning Code` 未勾选），也**没有 Online Gradio Demo**（同样未勾选）；自演化后训练循环在 README 中只有文字描述，无法复现。
- **硬件门槛高**：全精度需 8×80 GB 单节点，磁盘 ~120 GB，不适合个人开发者直接上手。
- 评测分数字**全部由团队自测并直接列在 README**，无第三方复现路径；`Where2Place`（65.0 vs 上一代 68.0）与 `PartAfford`（63.7 vs Embodied-R1.5 的 82.6）等项明显低于对照，README 未对这些落后项做分析。
- 3D/导航 demo 只有媒体文件，没有可运行的复现脚本，也没有导航项的具体分数。

#### 可复用要点

1. **「能力分类法」先于数据工程**：把 taxonomy 明确写成 README 的一等公民（并配 `figures/2-taxonomy.png`），再让预训练/后训练的数据混合围绕它组织。这套做法可以直接套用到任何垂直领域基座的数据配比设计上。
2. **插件优先、fork 兜底的上游集成策略**：用 `vllm.general_plugins` entry-point 注册 `ModelRegistry` + `ToolParserManager` + `ReasoningParserManager`，并预留「上游合并即卸载插件」的退出路径。这比维护一个 vLLM fork 的长期成本低得多。
3. **部署参数的默认值要写进文档表格**：把 `TP` / `GPU_MEM_UTIL` / `MAX_MODEL_LEN` / `CHAT_TEMPLATE` 的默认值与含义列成表，并给出就绪检查命令（`curl -sf .../v1/models`）与首次加载耗时预期（1–5 分钟）。运维侧直接受益。

---

### 2.2 `HY-Embodied-0.5-X`

#### 项目定位与要解决的问题

自我定位为 **"An Enhanced Embodied Foundation Model for Real-World Agents"**，出品方 *Tencent Robotics X × HY Vision Team*。它构建在 `HY-Embodied-0.5 MoT-2B` 架构之上（**4B 总参数 / 仅 2B 激活**），专门针对真实机器人的核心闭环 —— **"understand, reason, and act"** —— 做优化，目标是把模型从 *"seeing"* 推向 *"doing"*。

README 明确了六个专项改进方向：**细粒度操作理解（fine-grained manipulation understanding）、空间推理、动作预测、风险评估、多模态指代 grounding、长程规划**。

#### 核心架构 / 方法

- **骨干**：`HY-Embodied-0.5 MoT-2B`（**MoT = Mixture-of-Transformers**，注意与 2.1 的 MoE 不是同一概念）。`[model-card]` 给出底层实现细节：`transformers` 的指定 commit 原生注册 `HunYuanVLMoT{Config,Model,Processor}` 与 `HunYuanDenseV1Config`，因此**仓库不含本地建模代码**（来自 `docs/architecture.md`）。
- **数据构成**（三源合一）：自采第一人称机器人操作数据 + 机械臂操作数据 + 开源具身数据。具体覆盖：
  - **机械臂 / 人手轨迹** —— 用于状态理解、下一动作预测、操作风险评估、故障诊断、成对候选动作比较；
  - **第一人称具身任务** —— 细粒度动作识别、子任务进度估计、手部空间定位、深度估计、相对空间推理、相机位姿推断等；
  - **多模态交互指代 grounding** —— 围绕 *"put this over there"* 这类含歧义的真实指令，结合语音与手势线索。
  - 所有核心样本都配 **chain-of-thought (CoT) 标注**，并走完整的 **"generate → verify → correct → eval-regression"** 数据质量闭环；具身/互联网/3D 三类数据经标准化重建管线统一为一致的具身推理数据。
- **训练策略**：**"Validate → Scale → Full-Run"** 三段式 —— (1) 先在小规模高质量子集上快速验证训练配置与数据清洗；(2) 逐步放大数据与算力；(3) 只有在最优数据混合与训练策略确认后才启动全量训练。
- **能力闭环验证**：与 **PlaygroundX** 仿真框架（构建于 Tairos 之上）集成，对 *"throw the potato into the trash"*、*"close the fridge door"*、*"put the tomato in the fridge"* 等家务指令产出完整计划，并能根据环境反馈调整执行、**在初始计划失败时即时重规划**，形成完整的 **ReAct loop**：*reason → execute → detect failure → replan*。
- **坐标与输出格式契约**：Point 为 `(x, y)` 或 `[(x1, y1), (x2, y2)]`；Box 为 `[xmin, ymin, xmax, ymax]`；坐标归一化到整数区间 **(0, 1000)**。thinking 模式响应结构为 `<think>[reasoning]</think><answer>[answer]</answer>`；direct 模式（默认）只含答案。

#### 关键目录与文件（README 与 `docs/architecture.md` 真实出现的路径）

```
HY-Embodied-0.5-X/
├── README.md                 # this file
├── LICENSE                   # Apache-2.0   ← 注意：README 未说明 License，此处为 README 的目录树注释原文
├── pyproject.toml            # packaging + console scripts
├── requirements.txt          # full pinned dependency list
├── setup_env.sh              # one-click env setup
├── src/hy_embodied/          # Python package
│   ├── cli/                  # `python -m hy_embodied.cli.train / .infer / .server`
│   ├── training/             # SFT trainer, data pipeline, chat template
│   └── inference/            # HyEmbodiedPipeline
├── configs/
│   ├── sft/                  # training config (example_small.yaml)
│   ├── accelerate/           # accelerate launcher configs
│   ├── deepspeed/            # ZeRO configs
│   └── fsdp/                 # FSDP configs
├── scripts/                  # shell launchers (1-node / 4-node)
├── data_examples/            # per-capability sample JSONLs (+ README)
├── docs/                     # data_format / training / inference / architecture
├── assets/                   # images used in docs / README
├── ckpts/                    # (gitignored) `hf download` target
├── outputs/                  # (gitignored) training run outputs
└── inference.py              # backward-compat shim for the legacy CLI
```

`docs/architecture.md` 进一步给出包内文件级清单：`src/hy_embodied/cli/train.py`、`cli/infer.py`、`training/config.py`（dataclasses：`ModelConfig`/`DataConfig`/`TrainConfig`）、`training/chat_template.py`、`training/data.py`（`LazySupervisedDataset` + `SFTDataCollator`）、`training/callbacks.py`（`SaveInferenceArtifactsCallback`、`ProgressLoggingCallback`）、`training/trainer.py`（`run_sft` 入口）、`inference/pipeline.py`（`HyEmbodiedPipeline` + `GenerationConfig`）；配置目录含 `configs/sft/example_small.yaml`、`configs/accelerate/`、`configs/deepspeed/`、`configs/fsdp/`；脚本含 `scripts/run_sft_1node_8gpu.sh`、`scripts/run_sft_4node_8gpu.sh`。

> ⚠️ **一处 README 内部不一致（原样记录，不做修正）**：README 的仓库结构树把 `LICENSE` 注释为 `# Apache-2.0`，但 README 本身**没有 License 章节**；而仓库 `master/LICENSE` 实读内容为 **TENCENT HY COMMUNITY LICENSE AGREEMENT**（带地域限制），`[model-card]` 元数据也标 `license: other`。使用前请以 `LICENSE` 文件原文为准。

**`docs/architecture.md` 的导入规则**（对二次开发很有价值）：
- 面向用户的代码只放 `src/hy_embodied/`；**永不**从 `configs/`、`scripts/`、`data_examples/`、`docs/` 导入。
- `training/` 与 `inference/` **互不导入**，两个子系统仅共享 chat template。
- `cli/*.py` **只做入口**：解析参数、配置日志、委托给 `training/` 或 `inference/` 的库代码。
- 依赖方向：`cli/train.py → training/trainer.py → {config, chat_template, data, callbacks}`；`cli/infer.py → inference/pipeline.py`。
- **模型加载哲学**：因为上游 `transformers` 已原生注册 `HunYuanVLMoT*`，所以 (1) 在 `AutoProcessor`/`AutoModelForImageTextToText` 调用处**始终禁用 `trust_remote_code`**；(2) 训练用的 chat template 只在训练期使用；(3) 保存 checkpoint 时把原始 `chat_template.jinja` 复制回去并恢复 `model_type` / `auto_map`，使产物与上游发布**可互换**。结论：升级 transformers 即透明升级模型实现，不 fork、不 vendor 任何建模代码。
- **新增一种能力的四步流程**：按 `docs/data_format.md` 写 JSONL → 往 `data_examples/` 放 ≤5 条代表样本并在 `data_examples/README.md` 记录 → 在训练配置的 `data.train_data_paths` 与 `data.train_data_sampling_ratios` 加一节 → 若引入新 token 或结构化输出，在 `docs/data_format.md` 记录。

#### 安装 / 推理 / 训练命令（照抄 README）

```bash
# 一键环境（flash_attn 从源码编译，约 10–20 分钟）
bash setup_env.sh
conda activate hy_embodied_x

# （可选）把包装成 console script
pip install -e .
```

前置条件：OS Linux；Python 3.12；CUDA 12.6；PyTorch 2.10.0；GPU 为 **NVIDIA GPU，≥ 16 GB VRAM**。关键依赖：`transformers`（[specific commit](https://github.com/huggingface/transformers/commit/9293856c419762ebf98fbe2bd9440f9ce7069f1a)，提供原生 HY-Embodied 支持）、`flash_attn==2.8.3`、`accelerate`、`deepspeed`、`timm`、`liger-kernel`（详见 `setup_env.sh` 与 `requirements.txt`）。

```bash
# 下载权重
hf download tencent/HY-Embodied-0.5-X \
    --local-dir ckpts/HY-Embodied-0.5-X
```

```bash
# 单图推理（默认关闭 thinking）
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image ./assets/demo.jpg \
    --prompt "Describe this image"

# 开启 thinking 模式
python -m hy_embodied.cli.infer \
    --model ckpts/HY-Embodied-0.5-X \
    --image ./assets/demo.jpg \
    --prompt "Describe this image" \
    --enable-thinking
```

```bash
# OpenAI 兼容服务
bash scripts/run_server.sh

python -m hy_embodied.cli.server \
    --model ckpts/HY-Embodied-0.5-X \
    --host 0.0.0.0 --port 8080

# pip install -e ".[serve]" 之后
hy-embodied-server --model ckpts/HY-Embodied-0.5-X --port 8080
```

```bash
# SFT 微调
# 单卡冒烟测试 —— 不需要 torchrun / DeepSpeed（≥ 16 GB VRAM）
CUDA_VISIBLE_DEVICES=0 python -m hy_embodied.cli.train \
    --config configs/sft/example_small_single_gpu.yaml
# 或直接：
bash scripts/run_sft_single_gpu.sh

# 1 node × 8 GPUs + DeepSpeed ZeRO-2
bash scripts/run_sft_1node_8gpu.sh

# 4 nodes × 8 GPUs
bash scripts/run_sft_4node_8gpu.sh
```

提供的两份参考配置：`configs/sft/example_small_single_gpu.yaml`（**单卡配置**，DeepSpeed 关闭，可用纯 `python -m` 启动，适合快速验证与调试）与 `configs/sft/example_small.yaml`（**多卡配置**，启用 DeepSpeed ZeRO-2，必须经 `torchrun` 或 `accelerate` 启动，其训练/优化器默认值即发布配方）。两份配置都随仓库附带 `data_examples/data_demo.jsonl`（**14 条样本，覆盖 6 种能力**，图片内置），因此默认命令可**无外部数据端到端跑通**。

`docs/inference.md` 另给出：`--no-thinking` 关闭思考、`--batch` 批量演示、`hy-embodied-infer` console script，以及 `python inference.py` 作为向后兼容 shim。

#### 模型权重地址与所需硬件

- 权重：HuggingFace [`tencent/HY-Embodied-0.5-X`](https://huggingface.co/tencent/HY-Embodied-0.5-X)；`*.safetensors` 被 git-ignore，预期放在 `ckpts/HY-Embodied-0.5-X/`；推理与训练代码也接受直接传 Hub repo id（触发 `transformers` 按需下载）。
- 硬件门槛是这套仓库里最低的：**≥ 16 GB VRAM 的 NVIDIA GPU** 即可做单图推理与单卡 SFT 冒烟测试。
- 项目页：<https://tairos.tencent.com/openSourceModels/hy-embodied>；平台：TAIROS。

#### 评测（只写 README 真实出现的数字）

- README 声明：在 **10 个主流具身任务规划 benchmark** 上达到 SOTA，其中 **7 个在边缘侧领域模型中排名第一**。覆盖 planning、spatial reasoning、embodied QA、visual reference、trajectory understanding。
- **AI2Thor 具身规划 benchmark（内部构建）**：**1,011 个任务**，覆盖四类家居场景（厨房、卧室、客厅、浴室），评测导航、抓取、放置、家电操作、食物切割。README 只给出定性结论「在长程操作、自我认知、空间理解上有明显增益」；**具体分数只存在于 `assets/Results-Planning-benchmark.png` 图中，README 正文无数字**。
- 另外两组结果同样只有图片、无正文数字：`assets/Results-All-benchmarks.png`（10 个开源 benchmark 总览）、`assets/Results-Comparison_with_Open-Source_Models.png`（同尺寸开源模型对比）。
- 运行说明：`python -m hy_embodied.cli.infer ...` 默认 thinking 关闭；Python API 示例用了 `GenerationConfig(max_new_tokens=32768, temperature=0.05)`。

#### 论文 / arXiv、License

- **arXiv：`README 未说明`**。README 的 Citation 为 `@article{tencent2026hyembodied05x, title={HY-Embodied-0.5-X: An Enhanced Embodied Foundation Model for Real-World Agents}, author={Tencent Robotics X and HY Vision Team}, year={2026}}` —— **无 journal / eprint / arXiv 编号字段**。项目页为 <https://tairos.tencent.com/openSourceModels/hy-embodied>。
- License：**README 未说明**（无 License 章节）。仓库 `master/LICENSE` 实读为 **TENCENT HY COMMUNITY LICENSE AGREEMENT**（Tencent HY Embodied 0.5-X，Release Date: April 22, 2026），关键限制包括：**不适用于欧盟、英国、韩国**（Territory = 全球排除 EU/UK/韩国），**Section 5(c) 明确禁止在 Territory 之外使用、复制、修改、分发或展示**该作品及其输出，另附 20 条 Acceptable Use Policy（含禁止军事用途、禁止高风险自动化决策等）；Section 4 为 "ADDITIONAL COMMERCIAL TERMS"。`[model-card]` 元数据为 `license: other`，与此一致。

> 这一条与 README 目录树里 `LICENSE # Apache-2.0` 的注释**直接冲突**，是本报告发现的最需要注意的合规风险点。

#### 亮点与局限

**亮点**
- **7 个仓库中工程完备度最高的具身基座**：唯一同时提供 CLI 推理、Python API、OpenAI 兼容 server、单卡/多卡 SFT、accelerate/DeepSpeed/FSDP 四套启动配置、以及 `docs/` 四份文档（`architecture.md` / `data_format.md` / `training.md` / `inference.md`）。
- **零外部依赖即可跑通**：附 `data_examples/data_demo.jsonl`（14 样本 / 6 能力 / 图片内置），默认命令开箱端到端，这对「先验证环境再谈数据」的实践非常友好。
- **不 fork 上游建模代码**：靠 `transformers` 的原生 `HunYuanVLMoT*` 支持 + 保存时恢复 `model_type`/`auto_map` 保证产物可互换，架构文档把导入规则写死（training 与 inference 互不导入、cli 只做入口）。
- **数据质量闭环显式化**："generate → verify → correct → eval-regression" 四步 + CoT 标注；训练策略 "Validate → Scale → Full-Run" 明确反对一上来就全量训练。
- **ReAct 闭环已有集成验证**：与 PlaygroundX 集成，具备 *reason → execute → detect failure → replan* 的完整失败恢复链路。
- **极低部署门槛**：4B 总 / 2B 激活，≥16 GB VRAM 可跑。

**局限**
- **License 风险最高**：实际授权为带地域限制的社区许可，与 README 目录树的 "Apache-2.0" 注释不一致，商用/跨境部署前必须自行核对 LICENSE 原文。
- **评测章节几乎不可比对**：README 只给「10 个 benchmark、7 个第一」的结论性数字与内部 AI2Thor 的 1,011 任务规模，**全部具体分数都在 PNG 图片里**，无法文本检索、无法复核，也未列出对照模型清单。
- **无 arXiv 论文**可追溯方法细节（Citation 无编号）。
- 数据来源描述为「自采 + 开源」，但**数据量、数据配比、清洗后规模均未披露**。
- README 未给出 vLLM/SGLang 等高性能服务端路径（只有自带 server）。

#### 可复用要点

1. **「开箱可跑」的最小数据切片**：随仓库附带一份 ≤ 几百 KB 的 `data_examples/data_demo.jsonl`（覆盖每种能力的少量样本 + 图片内置），让默认训练命令无需外部数据即可端到端跑通，再用它承接「新增能力」的文档流程。这直接消除了新人上手最大的摩擦点。
2. **把架构约束写成可执行的导入规则**：`training/` 与 `inference/` 互不导入、`cli/*` 只做入口、永不从非包目录导入，并画出依赖方向图。对一个既要训练又要推理的仓库，这是防止腐化最便宜的手段。
3. **不 fork 上游、靠 checkpoint 元数据保证互操作**：禁用 `trust_remote_code`、训练用模板只在训练期存在、保存时恢复 `model_type`/`auto_map` —— 升级上游即透明升级自身实现。

---

### 2.3 `Hy-Embodied-0.5-VLA`

#### 项目定位与要解决的问题

标题即定位：**"From Vision-Language-Action Models to a Real-World Robot Learning Stack"**，出品方 *Tencent Robotics X × Tencent Hy Team*。要解决的不是「再训一个 VLA」，而是**打通整条真实机器人学习栈**：数据采集 → 模型设计 → 预训练 → 监督微调 → RL 后训练 → 实机部署。

技术上要解决三个具体痛点：
1. **动作表示与本体耦合** —— 换一台机器人就要重训策略；
2. **多帧历史带来 token 预算膨胀** —— 时序上下文与推理成本难以两全；
3. **仿真到实机的落差与实机失败数据的利用** —— 实机干预数据稀疏，传统 RL 需要奖励模型。

**模型代号 `Hy-VLA`**（README 亦称 `Hy-Embodied-0.5-VLA`）；`HY-VLA-0.5` 是 `HY-Embodied` 仓库 Updates 里对 2026-06-15 发布版本的称呼。

#### 核心架构 / 方法

三个组件（README "Model" 章节）：

1. **Backbone — `Hy-Embodied-0.5 MoT`**：Mixture-of-Transformers 架构，**modality-adaptive computation** —— 视觉 token 走专用视觉参数，文本 token 走原语言参数，跨模态交互**仅限共享 self-attention 层**。骨干通过 `Hy-ViT 2.0` 以原生分辨率编码图像。
2. **Action Expert — Dual-Tower Flow Matching**：**不把动作离散成语言 token**，而是用**独立的 370M 参数 action expert** 通过 conditional flow matching 建模连续动作分布。VLM tower 与 action expert tower **共享 attention**，使有 grounding 的视觉-语言上下文能引导连续动作生成。推理时对学到的速度场积分 **10 步 Euler**，并使用 **KV-cached 观测前缀**。
3. **Compact Memory Encoder**：在 ViT 编码器内**交错插入 temporal-spatial attention** 来压缩 K 帧多视角历史。temporal attention（跨帧因果）与 spatial attention（帧内双向）**复用同一套 QKV 投影 —— 相对单图编码器零新增参数**。时序混合后**丢弃过去帧 token**，因此无论历史多长，VLM 的 token 数恒定。

其他关键设计：
- **Delta-Chunk Action Representation**：动作预测为「相对当前帧的末端执行器 delta chunk」，**将策略与本体特定运动学解耦**，从而实现跨本体迁移。
- **异步部署栈**：producer-consumer 推理 + **三次贝塞尔（cubic Bézier）chunk 拼接**，在异构机器人平台上实现高频闭环控制。
- **FlowPRO 后训练**（*under review*）：**critic-free 偏好优化算法**，把真实机器人的失败干预转化为快速策略改进，**不需要奖励模型**。RPRO loss 直接对每个状态的 preferred / dispreferred action chunks 做对比，并用**对称 proximal 正则项**防止 reward hacking。迭代三步：(1) 通过遥操作 intervention-and-rollback **采集**偏好对；(2) 用 **Smooth Interpolation** 把稀疏纠正转成密集 per-state 元组；(3) 在混合 batch 上用 **RPRO loss** 优化。

**训练阶段**（README "Training & Evaluation"）：

| 阶段 | 关键设置 |
|---|---|
| 预训练 | VLM tower 从 `tencent/HY-Embodied-0.5` 初始化，action expert 随机初始化；用完整 10K 小时 UMI 语料，flow-matching 目标；**history length K=1**，**action chunk horizon H=50**，**10 Hz**；**200K steps**，**global batch size 1,024**；`[model-card]` 补充：learning rate **5e-5**（linear warmup 1K → 160K 内衰减到 5e-6 → 其后 constant 40K）、AdamW + bfloat16 混合精度、**硬件 64 GPUs（8 nodes × 8）** |
| SFT | 从预训练 checkpoint 出发，**激活 compact memory encoder（K=6 帧）**，在任务示范上微调；实机部署训练 **60K steps / batch size 32**；RoboTwin 2.0 用 **batch size 128 + action downsampling（stride 3）** |

`[model-card]` 额外给出：action expert hidden=1024、intermediate=2048；动作表示为 **relative-to-first-frame delta EEF chunk（每臂 10 维：xyz + rot6d + gripper）**；预训练期 video encoder 为 **single-frame 模式（K=1）**，memory encoder 在 SFT 阶段才激活。

#### 关键目录与文件（README 真实出现的路径）

```
Hy-Embodied-0.5-VLA/
├── hy_vla/                      # 核心模型定义、训练与推理
│   ├── modeling_hy_vla.py       # HyVLA 模型类
│   ├── modeling_dual_tower.py   # 双塔 transformer（VLM + action expert）
│   ├── configuration_hy_vla.py  # 模型配置
│   ├── space_time_attention.py  # memory encoder 的时序-空间注意力
│   ├── data/                    # dataloader 与 dataset 工具
│   ├── config/                  # YAML 配置文件
│   └── hunyuan_vl_mot/          # vendored Hy-Embodied VLM backbone（fallback）
├── scripts/                     # 训练、评测、预处理脚本
│   ├── quick_start.py           # 已发布 checkpoint 的快速冒烟测试
│   ├── train_umi_vlm.sh         # Stage-1 预训练启动脚本
│   ├── train_robotwin_vlm.sh    # Stage-2 SFT（从 VLM backbone 起）
│   ├── train_robotwin_umi.sh    # Stage-2 SFT（从 UMI 预训练起）
│   ├── train_robodojo_umi.sh    # Stage-2 SFT（从 UMI 预训练起，RoboDojo HDF5）
│   ├── train_table_vlm.sh       # 单表快速迭代训练
│   ├── eval_robotwin_test.sh    # RoboTwin 快速回归（6 个任务）
│   ├── eval_robotwin_full.sh    # RoboTwin 全量扫描（50 任务 × 100 rollouts）
│   ├── compute_norm_umi.py      # 从 UMI 数据预计算 norm stats
│   ├── compute_norm_robotwin.py # 从 RoboTwin 数据预计算 norm stats
│   ├── compute_norm_robodojo.py # 从 RoboDojo 数据预计算 norm stats
│   ├── vis_umi_episode.py       # 把 UMI episode 渲染成 MP4
│   ├── vis_robotwin_episode.py  # 把 RoboTwin episode 渲染成 MP4
│   └── vis_robodojo_episode.py  # 把 RoboDojo episode 渲染成 MP4
├── robotwin_eval/               # RoboTwin 评测适配器
├── robodojo_eval/               # RoboDojo 评测适配器
├── assets/                      # 示例数据与索引文件
└── pyproject.toml               # Python 项目配置（uv/pip）
```

README 另提及的具体模块/配置：`hy_vla.data.umi_dataset.LanceTableReader`、`hy_vla.data.robodojo_dataset.RoboDojoVLADataset`、`hy_vla/config/dataset/robodojo_hdf5.yaml`、`robodojo_eval/deploy_policy.yml`、`robodojo_eval.deploy_policy`。

#### 安装 / 推理 / 训练命令（照抄 README）

**安装**：

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA
cd Hy-Embodied-0.5-VLA

# One-off: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Materialize the virtual environment
uv sync
```

或 `pip install -r requirements.txt`。

**Quick Start（冒烟测试）**：

```bash
python scripts/quick_start.py
```

等价的 Python 代码（README 原文）：

```python
import torch
from huggingface_hub import snapshot_download
from hy_vla import HyVLA, HyVLAConfig

ckpt = snapshot_download("tencent/Hy-Embodied-0.5-VLA-RoboTwin")

config = HyVLAConfig.from_pretrained(ckpt)
policy = HyVLA.from_pretrained(ckpt, config=config)
policy.enable_video_encoder_if_needed()
policy = policy.to(device="cuda", dtype=torch.bfloat16).eval()

# (B, K, C, H, W); K=6 history slots
img = torch.zeros(1, 6, 3, 224, 224, device="cuda", dtype=torch.bfloat16)
# Normalized dual-arm EEF: [xyz(3) + rot6d(6) + gripper(1)] * 2
state = torch.zeros((1, config.max_state_dim), device="cuda", dtype=torch.bfloat16)
batch = {
    "observation.images.top_head":   img,
    "observation.images.hand_left":  img,
    "observation.images.hand_right": img,
    "observation.state": state,
    "task": ["pick up the bottle"],
}

with torch.no_grad():
    actions = policy.forward_evaluate(batch)["pred"]
    actions = actions[..., : config.action_feature.shape[0]]
print(actions.shape)
```

**预训练**：

```bash
# Single-table fast iteration
export TABLE_NAME=table_001
bash scripts/train_table_vlm.sh

# Full corpus (64 GPUs)
export CHIEF_IP=<chief-ip> INDEX=0
bash scripts/train_umi_vlm.sh
```

**SFT 与评测**：

```bash
# --- Training ---
export CHIEF_IP=<chief-ip> INDEX=0
bash scripts/train_robotwin_umi.sh   # Fine-tune from Hy-VLA-UMI on RoboTwin
bash scripts/train_robodojo_umi.sh   # Fine-tune from Hy-VLA-UMI on RoboDojo HDF5

# --- Evaluation ---
export ROBOTWIN_DIR=/path/to/RobotTwin
export CKPT_PATH=tencent/Hy-Embodied-0.5-VLA-RoboTwin

# Quick regression (6 tasks × 10 rollouts)
bash scripts/eval_robotwin_test.sh

# Full sweep (50 tasks × 100 rollouts, 8 GPUs)
bash scripts/eval_robotwin_full.sh
```

> README 提示：评测脚本会自动把 `Hy-VLA/robotwin_eval/` 软链到 `RoboTwin/policy/hy_vla`，使 RoboTwin 的 `eval_policy.py` 无需手工配置即可发现 Hy-VLA policy adapter。

**RoboDojo 微调**：

```bash
python scripts/compute_norm_robodojo.py \
    --hdf5-dir /path/to/robodojo/hdf5 \
    --output /path/to/experiments/hy_vla_robodojo_umi/norm_stats.pkl \
    --downsample-rate 1 \
    --chunk-size 25 \
    --umi-coord-frame

CHIEF_IP=127.0.0.1 INDEX=0 NUM_MACHINES=1 NPROC_PER_NODE=8 \
HDF5_DIR=/path/to/robodojo/hdf5 \
EXP_ROOT=/path/to/experiments \
NORM_PATH=/path/to/experiments/hy_vla_robodojo_umi/norm_stats.pkl \
bash scripts/train_robodojo_umi.sh
```

**预计算归一化统计量**（首次训练前必需）：

```bash
python scripts/compute_norm_umi.py \
    --lance-source tencent/Hy-Embodied-0.5-VLA-Data \
    --output norm_stats.pkl
```

> 已发布 checkpoint 自带预计算的 norm stats，仅在自定义数据上训练时才需要跑这个脚本。（`[model-card]` 中该脚本名为 `scripts/compute_norm_lance.py`，与 GitHub README 的 `compute_norm_umi.py` 不一致，以 GitHub README 为准。）

**数据加载**：

```python
from hy_vla.data.umi_dataset import LanceTableReader

# Local directory
reader = LanceTableReader(root="./table_000")

# HF Hub
reader = LanceTableReader(
    repo_id="tencent/Hy-Embodied-0.5-VLA-Data",
    table_name="table_000",
)

# Access
frame = reader[42]                        # single frame dict
episode = reader.get_episode(3)           # all frames of episode 3
```

**Episode 可视化**：

```bash
python scripts/vis_umi_episode.py -t table_000 -e 666
python scripts/vis_umi_episode.py /path/to/Hy-Embodied-0.5-Data -e 0 --no-3d
```

#### 模型权重地址与所需硬件

**已发布 checkpoint**（两个，均为 self-contained，自带 `tokenizer.json`、`vlm_config_dict`、`chat_template.jinja`、`preprocessor_config.json` 与预计算 `norm_stats.pkl`）：

| Model | HuggingFace | ModelScope | 说明 |
|---|---|---|---|
| **Hy-VLA-UMI** | [`tencent/Hy-Embodied-0.5-VLA-UMI`](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI) | [`Tencent-Hunyuan/Hy-Embodied-0.5-VLA-UMI`](https://modelscope.cn/models/Tencent-Hunyuan/Hy-Embodied-0.5-VLA-UMI) | 在 Hy-UMI-10K 语料上预训练；作为微调的通用起点 |
| **Hy-VLA-RoboTwin** | [`tencent/Hy-Embodied-0.5-VLA-RoboTwin`](https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-RoboTwin) | [`Tencent-Hunyuan/Hy-Embodied-0.5-VLA-RoboTwin`](https://modelscope.cn/models/Tencent-Hunyuan/Hy-Embodied-0.5-VLA-RoboTwin) | 在 RoboTwin 2.0 的 50 任务 benchmark 上后训练 |

**硬件**：前置条件为 Linux（推荐）、Python 3.12（推荐且测试过）、CUDA 12.x、PyTorch ≥ 2.4、**NVIDIA GPU（建议 ≥ 16 GB VRAM）**。但预训练规模实际为 **64 GPUs**（`[model-card]`：8 nodes × 8）。

**数据（Hy-Embodied-0.5-VLA-Data）**：

| Attribute | Value |
|---|---|
| Format | Lance（LeRobot v3.0 schema） |
| Resolution | 240 × 424 px |
| Cameras | `cam_high`（头部）、`cam_left_wrist`、`cam_right_wrist` |
| State | 16 维双机械臂 EEF（每臂 pos + quat + gripper） |
| FPS | 30 |
| Total Episodes | 250K+ |
| Total Duration | 2000+ hours |

README 正文另述：数据集共 **250K+ episodes / 10K+ 小时**双机械臂遥操作轨迹，**开源约五分之一（2K+ 小时）**，切分为 **22 个 Lance 格式表**，兼容 LeRobot v3.0；采集方式为**定制指尖 UMI 装置 + 光学动作捕捉系统**，覆盖 **70+ 任务**。`[model-card]` 的口径为「完整 10K 小时 UMI 语料，约 **~1M episodes**，70+ 任务」——两个来源的 episode 数口径不同（250K+ vs ~1M），本报告同时列出。

**坐标系统**（README 专门章节，提供 UMI ↔ RoboTwin 的轴映射与变换矩阵）：

| Dataset | World Forward | World Left | World Up | Local Forward | Local Left | Local Up | Gripper Open | Gripper Close |
|---|---|---|---|---|---|---|---|---|
| **UMI** | +X | +Y | +Z | +Z | +X | +Y | 0 | 90 |
| **RoboTwin** | +Y | -X | +Z | +X | +Y | +Z | 1 | 0 |

> 夹爪值表示平行夹爪的行程距离（mm）：`0` 完全张开，`90` 完全闭合。

变换方式：设 RoboTwin 的位置向量与旋转矩阵为 $p_{\text{native}}$、$R_{\text{native}}$，四元数为 $q_{\text{native}}$，则到 UMI 的映射为 $p_{\text{umi}} = W p_{\text{native}}$、$R_{\text{umi}} = W R_{\text{native}} P$、$q_{\text{umi}} = q_W \otimes q_{\text{native}} \otimes q_P$；反向（UMI → RoboTwin）因 $W$、$P$ 为正交矩阵，逆即转置：$p_{\text{native}} = W^T p_{\text{umi}}$、$R_{\text{native}} = W^T R_{\text{umi}} P^T$、$q_{\text{native}} = q_W^{-1} \otimes q_{\text{umi}} \otimes q_P^{-1}$。其中

$$W = \begin{bmatrix} 0 & 1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & 1 \end{bmatrix}, \quad P = \begin{bmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$

> README 特别说明：**已发布 checkpoint 不做任何特殊预处理** —— 「the network naturally adapts to different coordinate conventions after sufficient training iterations」，上述变换仅作为对齐 UMI 坐标系的参考。

#### 评测（只写 README 真实出现的数字）

**RoboTwin 2.0（仿真，成功率 %，每任务 100 rollouts × 50 任务取平均）**

| Method | Clean | Randomized |
|---|---|---|
| π₀ | 65.9 | 58.4 |
| ABot-M0 | 81.2 | 80.4 |
| π₀.₅ | 82.7 | 76.8 |
| Qwen-VLA | 86.1 | 87.2 |
| LingBot-VLA | 86.5 | 85.3 |
| starVLA | 88.2 | 88.3 |
| Motus | 88.7 | 87.0 |
| JoyAI-RA | 90.5 | 89.3 |
| **Hy-VLA** | **90.9** | **90.1** |

**Track A — Intra-Embodiment（Dobot X-Trainer，同机器人微调+评测，4 个任务）**

| Method | Set the Table | Fold & Store Glasses | Zip Up the Pen Case | Insert Bottles |
|---|---|---|---|---|
| π₀ | 79% | 67% | 48% | 80% |
| π₀.₅ | 88% | 75% | 57% | 84% |
| Hy-Embodied（w/o UMI pretrain） | 80% | 65% | 43% | 70% |
| **Hy-VLA (Ours)** | **83%** | **94%** | **73%** | **94%** |

**Track B — Cross-Embodiment（仅用 UMI 示范微调，无目标机器人遥操作数据）**

| Method | JAKA K1 · Organize Accessory | Astribot S1 · Clean Up Table |
|---|---|---|
| π₀ | 88% | 87% |
| π₀.₅ | 81% | 89% |
| Hy-Embodied（w/o UMI pretrain） | 38% | 44% |
| **Hy-VLA (Ours)** | **90%** | **89%** |

> README 结论：无 UMI 预训练的消融在 Track B 上「塌陷」（38% / 44%），确认大规模 UMI 语料对「本体无关的动作先验」是必需的。

**FlowPRO / RPRO 后训练结果（X-Trainer 实机，成功率均值 ± 标准差 / 平均完成时间，K=3 轮后训练，3 seeds × 100 rollouts）**

| Method | Bottle | Cap | USB | Zip |
|---|---|---|---|---|
| DAgger | 93 ± 2.1% / 27s | 88 ± 1.8% / 29s | 86 ± 2.4% / 25s | 83 ± 2.0% / 55s |
| π₀.₆* | 95 ± 1.5% / 24s | 95 ± 1.2% / 27s | 95 ± 1.4% / 23s | 89 ± 1.6% / 45s |
| **RPRO (Ours)** | **99 ± 0.6% / 16s** | **99 ± 0.7% / 21s** | **98 ± 0.9% / 22s** | **94 ± 1.1% / 37s** |

**实机平台**：README 声明跨 **4 个真实机器人平台**验证（Track A：Dobot X-Trainer；Track B：JAKA K1、Astribot S1）。

#### 论文 / arXiv、License

- 论文：*Hy-Embodied-0.5-VLA: From Vision-Language-Action Models to a Real-World Robot Learning Stack*，**arXiv:2606.14409**，2026（README 徽章与 Citation 一致）。
- 项目页：<https://tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla>。
- License：**Apache-2.0**。README 明示 "Released under the **Apache-2.0 License**"；实读 `main/LICENSE`（11,624 字节）为标准 Apache-2.0 全文加 Tencent 声明头（"Hy-Embodied-0.5-VLA is licensed under the Apache-2.0."）。`[model-card]` 元数据亦为 `apache-2.0`。
- Acknowledgements：构建于 [Hy-Embodied-0.5](https://github.com/Tencent-Hunyuan/HY-Embodied)、[FlowPRO](https://wuyeyexvnainai.github.io/flowpro/)、[OpenPi](https://github.com/Physical-Intelligence/openpi) 与 [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin) 之上；RoboDojo 支持由 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) 贡献。

#### 亮点与局限

**亮点**
- **唯一提供真实机器人实机评测数字的仓库**，且同时给仿真（RoboTwin 2.0 / RoboDojo）与实机（Track A/B）两层结果，Track B 的跨本体消融（38%/44% → 90%/89%）说服力很强。
- **动作表示的解耦设计**：delta-chunk（相对当前帧的 EEF delta）+ 与运动学无关，是跨本体迁移能成立的关键前提，README 把它作为独立 Key Feature 列出并给出消融证据。
- **零新增参数的 memory encoder**：temporal/spatial attention 复用同一套 QKV 投影，且时序混合后丢弃过去帧 token，使 VLM token 数**与历史长度无关** —— 这是很干净的成本控制设计。
- **数据工程完整闭环**：`compute_norm_*.py`（三种数据源各自的归一化统计量脚本）+ `vis_*_episode.py`（三种数据源的 episode 渲染）+ `train_table_vlm.sh`（单表快速迭代）—— 从「单表跑通」到「全量 64 卡」的路径是通的。
- **评测规模写得非常具体**：50 任务 × 100 rollouts；快速回归 6 任务 × 10 rollouts；FlowPRO 结果标注 3 seeds × 100 rollouts。
- **FlowPRO 直面实机失败数据**：critic-free、无需奖励模型、把遥操作 intervention-and-rollback 转成偏好对，工程上比通用 RLHF 更贴近机器人场景。

**局限**
- **FlowPRO 代码未开源**：README 标注 "under review, code coming soon"，RPRO loss、Smooth Interpolation 均无实现可读。
- **数据开源比例为五分之一**（2K+ / 10K+ 小时），复现论文级预训练不现实。
- **预训练成本高**：64 GPUs、200K steps、global batch 1,024。
- **本报告发布时无独立 License 冲突，但存在文档不一致**：`[model-card]` 的 `compute_norm_lance.py` 与 GitHub README 的 `compute_norm_umi.py` 脚本名不同；episode 数口径 250K+ 与 ~1M 不同。
- 未给推理延迟/吞吐数字；「10 步 Euler + 异步 producer-consumer」的具体频率指标 `README 未说明`。

#### 可复用要点

1. **「单表快速迭代」训练入口**：除了全量 64 卡脚本，额外提供 `train_table_vlm.sh`（`export TABLE_NAME=table_001`）作为快速迭代通道。任何大规模数据训练项目都应提供这样一条「分钟级」回路，否则调参只能靠排队。
2. **归一化统计量与可视化脚本按数据源一一对称**：三套 `compute_norm_*.py` + 三套 `vis_*_episode.py`，结构完全平行。新接一个数据集时照抄一个文件即可，这是把「接入新数据集」从设计问题降级为填空题。
3. **把坐标变换矩阵与轴映射表写进 README**：跨本体/跨仿真器项目里，坐标系约定不清是最贵的沟通成本。这里直接给出 W、P 矩阵、正向/反向公式、夹爪开合数值定义，并明确「checkpoint 不做特殊预处理」——避免使用者自行发明转换。

---

### 2.4 `Hy-Embodied-RxBrain-1.0`

#### 项目定位与要解决的问题

**RxBrain**（仓库名 `Hy-Embodied-RxBrain-1.0`）自我定位为 **"a unified multimodal foundation model for embodied cognition"**（具身认知统一多模态基座），出品方 *Tencent Robotics X × Futian Laboratory × Tencent Hy Team*。

它要解决的是「理解」与「想象」被拆成两套模型的问题 —— 用一个模型同时提供三种能力：

1. 🤖 **Embodied Understanding & Reasoning** —— 对图像与多帧视频做问答与 chain-of-thought；
2. 🔮 **World State Prediction** —— 想象某个动作在物理世界中产生的近未来帧；
3. 🧩 **Joint Subgoal Planning** —— 把任务分解成步骤，**每一步同时输出「下一动作（语言）」与「应达到的目标图像（视觉）」**。

核心技术主张是 **interleaved generation（交错生成）**：在**单条自回归序列**内交替产出推理文本与 flow-matched 想象帧，由一个**学习得到的 `<Image>` token** 决定何时进行想象 —— 于是具身计划把 *what to do* 与 *what the world should look like* 逐步绑定在一起。

#### 核心架构 / 方法

- **统一 Mixture-of-Transformers (MoT)**：**~6.2B 参数**骨干，含**模态专属通路（text / vision / generation）**，让理解与图像合成分享同一个自回归模型，而不是各自独立成塔（`[model-card]` Key Features；README 的仓库结构树也表述为 "3 streams: text / vision / generation"）。
- **Flow-Matching Image Head**：想象帧由 flow-matching head 生成，解码进**冻结的 FLUX VAE latent 空间**，从而支持文生图、多帧世界模型 rollout、目标图像规划（`[model-card]`）。
- **交错推理 + 想象**：文本推理与生成帧在同一条序列中产出（`[model-card]`）。
- **自定义解码循环**：RxBrain **不使用标准 `model.generate` API**，而是使用自定义的交错文本/图像解码循环；`vqa_inference.py` 中的 `answer(...)` helper 是对理解场景的封装（`[model-card]`）。
- **微调：理解与生成在一次 packed sequence 内联合训练**（来自 `train/README.md`）：understanding 用 cross-entropy，图像生成在 VAE latent 上做 flow matching，二者可在同一 run 内按整数比例混合，也可分开训。四类任务与数据格式：

| 任务 | 数据 | 说明 |
|---|---|---|
| ① VQA / understanding | jsonl（或 parquet） | 仅 cross-entropy |
| ② Text-to-image | parquet | 仅 flow matching |
| ③ Multi-frame generation | jsonl | 4 帧，一次 joint forward |
| ④ Interleaved planning | jsonl **+ VAE mirrors** | 每步文本 + 帧；需要 Stage 2 |

- **任务由样本形状自动推断（无需声明）**，这一点设计得相当巧：

| shape | task | 追加指令 |
|---|---|---|
| 无 `generated_image` | understanding / VQA | （无） |
| 1 个 `generated_image` | single-frame generation | `Generate future image of the goal` |
| ≥2 个 `generated_image` | multi-frame generation | `Generate future video of the goal` |
| 1 个 `generated_image` + `metadata.step_index` | interleaved planning | `Generate interleave goal planning` |

- **多帧生成**：目标帧放在**同一个** assistant turn 内、在**一次 forward 中联合去噪**（"autoregression along time, diffusion across space"），自动包在一个 `<Video> … </Video>` 块里。发布模型用 **4 帧**；代码本身接受任意 N ≥ 2。
- **交错规划的数据构造**：**Y 步轨迹展开为 Y 个独立样本**（不是一条）。第 k 个样本的 user turn 放观测、任务、以及已经产出的步骤（文本, 帧）交替对 —— **prior frames 通过 vision tower 进入，与自回归推理时完全一致**；assistant turn **只放第 k 步**。`metadata` 是必需字段：`step_index`（选择交错指令并标记 obs/prior 边界）、`num_context_frames`（user 侧总帧数 = 观测 + 前序步骤帧）、`num_steps`（记账/过滤）。切分依据 `num_obs = num_context_frames - (step_index - 1)`，**依赖这个算术而不是文本位置**，因此文本为空或被 CFG 丢弃时仍然正确。**缺任一字段，样本会静默退化为单帧生成**。
- **Stage 2 VAE mirror（README 称之为 essential）**：推理时上下文里的每个前序帧都是模型自己**有损**的 `vae.decode` 输出，而原始训练数据里是**清晰的 GT JPEG** —— 模型学着读清晰帧却被要求读自己的重建，从第 2 步起条件分布就是错的，且每步输出成为下一步输入，误差会**累积**。这个图像侧 **exposure bias** 是长交错 rollout 的主要失效模式。修法：对每个可能作为前序帧出现的 GT 帧预计算其 VAE `encode → decode` 镜像，再以概率 `prior_frame_vae_roundtrip_prob` 把前序帧替换为镜像。构建器要求：在**转换后的** jsonl 上扫描（顺序不能颠倒）、每帧一张无损 PNG、可续跑幂等、生成分辨率 resize + `[-1,1]` 归一化 + **posterior mean**（不是采样，因为推理是确定性 ODE 输出）+ bf16；**只替换前序步骤帧**，观测保持真实，FM 目标永远是真 GT 帧；缺镜像会**静默 no-op**。

#### 关键目录与文件（README 与 `train/README.md` 真实出现的路径）

```
model/        UnifiedMoT model definition (3 streams: text / vision / generation)
vae_model/    FLUX VAE autoencoder loader
inference/    inference entry points  — T2I, multi-frame, VQA, interleaved planning
train/        fine-tuning pipeline    — see train/README.md
demo_cases/   bundled scenes for the interleaved-planning demo
```

`train/README.md` 另出现：`train/data_prep/convert_interleave.py`、`train/data_prep/convert_multiframe.py`、`train/data_prep/build_vae_roundtrip.py`、`train/configs/example.yaml`、`train/configs/deepspeed_zero2.yaml`、`train/finetune.py`、`train/launch.sh`、`inference/utils.py`（指令字符串与训练侧保持一致的地方）、`vqa_inference.py`（`[model-card]`）。

README 强调：`model/` 与 `vae_model/` 被 `inference/` 与 `train/` **共享**，因此微调出的 checkpoint 可以**不改动推理脚本**直接运行。

#### 安装 / 推理 / 训练命令（照抄 README）

**环境**：

```bash
git clone https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0.git
cd Hy-Embodied-RxBrain-1.0
pip install -r requirements.txt
```

要求：**Python 3.10+**、**CUDA GPU**（`flash-attn` 需要）。

**下载权重**：

```bash
pip install -U "huggingface_hub[cli]"
hf download tencent/Hy-Embodied-RxBrain-1.0 --local-dir ./Hy-Embodied-RxBrain-1.0
```

> `[model-card]` 提示：**`--ckpt` 必须是本地路径，不能是 Hub repo id**（loader 直接读 checkpoint 文件）；VQA 路径**只需**主权重，图像生成（T2I / world-model rollout / 交错规划）**额外需要**外部 FLUX VAE `ae.safetensors`。

**① Text-to-Image (T2I)**：

```bash
python inference/text2image.py \
    --ckpt ./ckpts/rxbrain --vae /path/to/ae.safetensors \
    --prompt "a watercolor painting of a cat" \
    --height 256 --width 256 --num_steps 25 --out out.png

# with classifier-free guidance
python inference/text2image.py \
    --ckpt ./ckpts/rxbrain --vae /path/to/ae.safetensors \
    --prompt "a watercolor painting of a cat" \
    --cfg_scale 5.0 --num_steps 50 --out out.png
```

**② Multi-Frame World-Model Rollout**（从一次观测想象 4 帧未来）：

```bash
python inference/multiframe.py \
    --ckpt ./ckpts/rxbrain --vae /path/to/ae.safetensors \
    --frames /path/to/obs.jpg --task "imagine the next frames" \
    --num_frames 4 --num_steps 50 --out_dir multiframe_out
```

**③ Visual Question Answering (VQA)**：

```bash
python inference/vqa.py \
    --ckpt ./ckpts/rxbrain \
    --images demo_cases/bridgev2_move_toy/input/obs_1.jpg \
    --question "What objects are on the stovetop, and where is the green toy?" \
    --max_new_tokens 256
```

**④ Interleaved Embodied Planning**：

```bash
CASE=umi_fold_sock
python inference/interleave.py \
    --ckpt ./ckpts/rxbrain --vae /path/to/ae.safetensors \
    --frames  demo_cases/$CASE/input/*.jpg \
    --task    "$(cat demo_cases/$CASE/prompt.txt)" \
    --max_frames 5 --num_steps 50 --out_dir out_$CASE
```

**微调（`train/README.md`）**：

```bash
pip install -r requirements.txt          # from the repository root

# interleave task only: trajectories -> training jsonl -> VAE mirror frames
python train/data_prep/convert_interleave.py --in <traj>.jsonl --out interleave.jsonl
python train/data_prep/build_vae_roundtrip.py --data_glob 'interleave.jsonl' \
    --src_root <frames_root> --dst_root <mirror_root> --vae ae.safetensors --gpus 0,1,2,3

cp train/configs/example.yaml train/configs/my_run.yaml   # edit paths + ratios
bash train/launch.sh                                       # single or multi node

# 等价显式写法
accelerate launch --config_file train/configs/deepspeed_zero2.yaml \
    train/finetune.py --config train/configs/example.yaml
```

多节点：由调度器导出 `WORLD_SIZE`、`RANK`、`GPU_NUM`、`MASTER_ADDR`、`MASTER_PORT`，`launch.sh` 会转发它们并**覆盖** accelerate yaml 中的节点数，因此同一脚本可无改动扩展；每个节点会做 GPU 预检，若无可见 GPU 则**快速中止并报出主机名**。`configs/deepspeed_zero2.yaml` 默认 1 node × 8 GPUs；ZeRO-2 分片优化器状态与梯度但复制参数，每 rank 约需 `(bf16 模型) + (优化器状态 / ranks)`；**完整 checkpoint 约 12 GB**，`output_dir` 需容纳 `live_keep_count` 份加最终保存。

#### 模型权重地址与所需硬件

- 权重：HuggingFace [`tencent/Hy-Embodied-RxBrain-1.0`](https://huggingface.co/tencent/Hy-Embodied-RxBrain-1.0)。
- Demo：HuggingFace Space [`hugging-apps/rxbrain-embodied-cognition`](https://huggingface.co/spaces/hugging-apps/rxbrain-embodied-cognition)。
- 模型规模（README 表）：

| Model | Params | Download |
|---|---|---|
| **Hy-Embodied-RxBrain-1.0** | ~6.2 B | [`tencent/Hy-Embodied-RxBrain-1.0`](https://huggingface.co/tencent/Hy-Embodied-RxBrain-1.0) |
| FLUX VAE (`ae.safetensors`) | 83.8 M | 从 [FLUX](https://github.com/black-forest-labs/flux) 发行版获取 |

- 硬件与依赖：`[model-card]` 给出 Linux（推荐）、Python 3.10+、**CUDA 12.x**（`flash-attn` 需要 NVIDIA GPU）、**PyTorch 2.10**；并且**必须安装指定 commit 的 transformers**：

```bash
pip install git+https://github.com/huggingface/transformers@9293856c419762ebf98fbe2bd9440f9ce7069f1a
```

> `[model-card]` 说明：**正式发布的 `transformers` 尚不包含 `hunyuan_vl_mot`**，必须用这个 pinned commit，后续会合入 Transformers main。这与 `HY-Embodied-0.5-X` 用的是**同一个 commit hash**（`9293856`），说明 Hy-Embodied 系列的 MoT 支持都挂在这一个上游 PR/commit 上。
- **README 未给出所需显存/GPU 数量**（只说明「CUDA GPU，`flash-attn` 需要」）；微调侧可参照的间接数字是「完整 checkpoint 约 12 GB」。

#### 评测（只写 README 真实出现的数字）

- README 的评测章节只有一句：*"RxBrain is evaluated on embodied understanding, spatial reasoning, and imagination/generation benchmarks. For detailed metrics and methodology, please refer to our [technical report](https://arxiv.org/abs/2607.14187)."* —— **无任何 benchmark 名称或分数数字**。
- TODO 列表显示 **`RxBrain-Bench`（a comprehensive benchmark for embodied cognition）尚未开源**（未勾选）。
- 因此本仓库的**全部可引用数字只有：模型 ~6.2B 参数、FLUX VAE 83.8M 参数、发布模型用 4 帧、checkpoint 约 12 GB、`--num_steps` 默认 25/50、`--cfg_scale 5.0`、Stage-2 替换概率示例 0.5、`--height 144 --width 256`（示例）** —— 无精度类指标。

#### 论文 / arXiv、License

- 论文：*RxBrain: Embodied Cognition Foundation Model with Joint Language-Visual Reasoning and Imagination*，**arXiv:2607.14187**，2026。
- 项目页：<https://tairos.tencent.com/openSourceModels/hy-embodied-rxbrain-1.0>。
- License：**Apache License 2.0**。README 明示 "Released under **Apache License 2.0**. See `LICENSE` for details."；标题徽章亦为 Apache-2.0；实读 `main/LICENSE`（11,700 字节）为标准 Apache-2.0 全文 + Tencent 声明头（额外一句 "For the avoidance of doubt, Hy-Embodied-RxBrain-1.0 refers to the code and model parameter weights made publicly available by Tencent."）。`[model-card]` 元数据为 `apache-2.0`。
- GitHub 与 HF 模型卡在 TODO 状态上**不一致**：GitHub README 把「开源 VQA / Multi Frames Generation / Interleave Generation 微调代码」标为**已完成**（`[x]`，指向 `train/`），而 `[model-card]` 仍将其标为未完成 `[ ]`。以 GitHub 仓库为准（`train/` 与 `train/README.md` 实际存在，本报告已实读）。

#### 亮点与局限

**亮点**
- **「动作 + 目标图像」成对输出**：联合子目标规划同时给出语言动作与视觉目标图，这是与纯文本规划最本质的区别，也让「计划对不对」有了视觉可验证性。
- **学习得到的 `<Image>` token 控制何时想象**：把「何时该画图」变成模型自己学的决策，而不是外部规则调度，是交错多模态生成里比较优雅的做法。
- **一次 forward 内联合去噪多帧**："autoregression along time, diffusion across space"。
- **训练配方文档质量很高**：`train/README.md` 明确写出任务由样本形状自动推断的映射表、`metadata` 每个字段「为什么必需」、以及 **Stage 2 VAE mirror 的完整推理（exposure bias 如何从第 2 步起污染条件分布并逐步累积）**，还给出「顺序不能颠倒」「缺镜像是静默 no-op」「只看 done/skip/fail 计数」等操作性警告。这是本批次 7 个仓库里最有信息量的一份训练文档。
- **训练/推理共享模型代码**：`model/` 与 `vae_model/` 被两侧共享，微调产物无需改推理脚本即可运行。
- **多节点启动脚本可无改动扩展**：调度器环境变量覆盖 accelerate yaml 的节点数，且每节点做 GPU 预检并快速失败报主机名。

**局限**
- **没有任何评测数字**：README 与 `[model-card]` 都只说「详见技术报告」，`RxBrain-Bench` 未开源，因此无法在本仓库内验证任何能力主张。
- **依赖非发布版 `transformers`**：必须从 GitHub 特定 commit 安装，版本漂移风险高（正式发布尚不含 `hunyuan_vl_mot`）。
- **推理路径繁琐**：`--ckpt` 必须本地路径；生成类任务另需从 FLUX 发行版自行获取 `ae.safetensors`（83.8M）。
- **`metadata` 字段缺失是静默失败**：缺 `step_index` 或 `num_context_frames` 时样本会悄悄退化为单帧生成、Stage 2 修正变成 no-op，只有靠人工核对才能发现。
- **未给显存/GPU 需求**，微调成本只能从「checkpoint ~12 GB + ZeRO-2」间接推断。
- Stage 2 需要为每个 GT 帧存一张无损 PNG，**磁盘预算需自行估算**（README 也提醒 "budget disk accordingly"）。
- GitHub README 的推理脚本名（`inference/vqa.py`、`inference/text2image.py`、`inference/multiframe.py`、`inference/interleave.py`）与 `[model-card]`（`vqa_inference.py`、`text2image_inference.py`、`multiframe_inference.py`、`interleave_inference.py`，仓库根）不一致，两个来源并列记录。

#### 可复用要点

1. **「任务由样本形状推断」的数据契约**：不引入 task 字段，而是用「有没有 `generated_image`、有几个、有没有 `metadata.step_index`」来确定任务与追加指令，并在 `inference/utils.py` 里保持指令字符串与训练侧完全一致。这大幅减少数据标注里的任务标错问题；代价是要像 README 那样把「训练与推理指令字符串必须同步」写成显式约束。
2. **把「训练/推理分布不一致」当成一等工程问题处理**：自回归式图像 rollout 里，前序帧在推理时是自身有损重建、在训练时是清晰 GT —— 这个 exposure bias 有通用的修法（预计算 VAE encode→decode 镜像，按概率替换前序帧，观测与目标保持真实）。任何「生成结果会回流为下一步输入」的系统都该照此设计。
3. **长轨迹展开为多样本时的切分依据要用算术而非位置**：`num_obs = num_context_frames - (step_index - 1)` —— 这样即使文本为空或被 CFG 丢弃，切分依然正确。比「按文本位置找边界」稳健得多。

---

### 2.5 `Hy-Embodied-RoboFusion`

#### 项目定位与要解决的问题

**这是一个通信协议规范文档，不是模型仓库。** README 标题即《Robo Fusion 通信协议规范》，文档信息为：协议版本 **1.0**、创建日期 **2026-04-13**、最后更新 **2026-04-14**。仓库语言为 C（外部元数据），联系方式 `RoboticsX@tencent.com`。

它要解决的是**机器人控制系统里「上位机 / 电机驱动 / 传感器」之间的实时通信标准化**问题：Robo Fusion 协议是**基于 UDP 的应用层通信协议**，专为机器人控制系统设计，支持电机控制、传感器数据采集与设备状态监控，采用**结构化二进制格式**，支持多种设备类型与数据交换模式。

两种通信模式：

- **PDO 通信**：周期性传输实时数据（电机控制指令与状态、传感器数据等），采用**事件驱动机制**；
- **SDO 通信**：客户端/服务器模式，用于传输关键配置参数与设备状态信息，主要用于初始化配置，采用**请求-响应机制**；
- **端口分配**：不同服务使用独立端口号，便于上层应用开发与管理。

协议特性：支持多种电机控制模式（MIT、PP、PV、PT、CSP、CSV、CST）；支持多类型传感器数据采集（IMU、力传感器、编码器）；紧凑二进制格式；支持实时状态监控与故障诊断。

**在具身技术栈中的位置**：它是整条栈里唯一触达「硬件时序层」的组件 —— 上面 4 个仓库决定「做什么动作」，RoboFusion 决定「动作指令如何以 1 kHz 频率可靠地送达电机」。

#### 核心架构 / 方法（协议规范要点）

**设备类型（类型值 → 设备类型）**

| 类型值 | 设备类型 | 描述 |
|---|---|---|
| 0x00 | 未使用 | 保留类型 |
| 0x80 | 同步帧 | 用于时序协调的同步帧 |
| 0x10 | 电机SDO | 电机服务数据对象 |
| 0x20 | 电机RXPDO | 电机接收过程数据对象 |
| 0x30 | 电机TXPDO | 电机发送过程数据对象 |
| 0x60 | 力传感器TXPDO | 力传感器发送数据 |
| 0x61 | IMU TXPDO | 惯性测量单元发送数据 |
| 0x62 | 编码器TXPDO | 编码器发送数据 |

**PDO 数据包结构**（多个 PDO 类型可组合在单个 UDP 包中）：

```
+--------------+---------------+---------------+---------------+---------------+---------------+
|  类型 (1字节) |  长度 (2字节) |   数据 (N字节) |  类型 (1字节) |  长度 (2字节) |   数据 (N字节) | ...
+--------------+---------------+---------------+---------------+---------------+---------------+
```

| 字段 | 长度 | 描述 |
|---|---|---|
| 类型 | 1字节 | PDO 类型标识符（0x20: 电机RXPDO，0x30: 电机TXPDO，0x60-0x62: 传感器TXPDO） |
| 长度 | 2字节 | 后续数据字段的长度（字节数） |
| 数据 | N字节 | 具体 PDO 数据内容，格式取决于 PDO 类型 |

**SDO 数据包结构**：

```
+--------------+---------------+------------+-----------------+----------------+-------------------+---------------+
|  类型 (1字节) |  长度 (2字节) |  ID (1字节) |  操作码 (1字节) | 操作长度 (1字节) | 寄存器地址 (2字节) |   数据 (N字节) | ...
+--------------+---------------+------------+-----------------+----------------+-------------------+---------------+
```

| 字段 | 长度 | 描述 |
|---|---|---|
| 类型 | 1字节 | SDO 类型标识符 (0x10) |
| 长度 | 2字节 | 后续数据字段的长度（字节数） |
| ID | 1字节 | 设备标识符 |
| 操作码 | 1字节 | 操作类型（0x01: 读, 0x02: 写, 0x03: 响应, 0x04: 错误） |
| 操作长度 | 1字节 | 后续数据字段的长度（字节数） |
| 寄存器地址 | 2字节 | 目标寄存器地址 |
| 数据 | N字节 | 读写的数据内容 |

**通信端口**

| 端口号 | 端口类型 | 描述 |
|---|---|---|
| 9110 | 电机PDO | 电机过程数据对象通信端口 |
| 8880 | 电机SDO | 电机服务数据对象通信端口 |
| 8887 | CM7调试 | Cortex-M7 处理器调试端口 |
| 8884 | CM4调试 | Cortex-M4 处理器调试端口 |

**通信示例**（README 原文）：

```
PDO: UDP数据包: [0x30][0x0018][电机TXPDO数据][0x61][0x0020][IMU数据]
SDO 写操作: [0x10][0x000C][0x01][0x01][0x04][0x1000][0x3F800000] // 写入电机参数
SDO 读操作: [0x10][0x0008][0x01][0x00][0x04][0x1000][00000000]   // 读取电机参数
```

**C 数据结构定义**（README 给出完整 typedef，主要枚举/结构体名称）：

`ROFU_TYPE_T`（设备类型枚举）、`ROFU_PORT_T`（端口枚举）、`ROFU_PDO_T`（通用 PDO：`type`/`len`/`pdata`）、`ROFU_SDO_T`（通用 SDO：`type`/`len`/`id`/`op`/`op_len`/`addr`/`pdata`）、`ROFU_SDO_OP_T`（读 0x01 / 写 0x02 / 响应 0x03 / 错误 0x04）、`ROFU_SDO_STATE_T`（`FRAME_OK` 0x00 / `NO_DEFINED` 0x01 / `LEN_ERROR` 0x02 / `OUTTIME` 0x03）、`ROFU_SDO_ADDR_T`（寄存器地址表，见下）。

**SDO 寄存器地址分组**（`ROFU_SDO_ADDR_T`）：

| 分组 | 地址范围 | 内容 |
|---|---|---|
| 电机 SDO | 0x1000–0x1009 | `MOTOR_ALL`、`POLE`（极对数）、`PHASE_SEQ`（相序）、`REDU_RATIO`（减速比）、`RS`、`LQ`、`LD`、`KT`、`INERTIA`、`KE` |
| 三环 PID | 0x2000–0x2005 | `PID_ALL`、`CUR_KP`、`CUR_KI`、`VEL_KP`、`VEL_KI`、`POS_KP` |
| 保护阈值 | 0x2010–0x2016 | `THR_ALL`、`VBUS_HIGH`、`VBUS_LOW`、`IPHASE_HIGH`、`MOS_TEMP_MAX`、`MOTOR_TEMP_MAX`、`MOTOR_SPEED_MAX` |

**电机控制字 `ROFU_MOTOR_CTRL_T`**（位域联合体）：`enable : 1`（0 disable / 1 enable）+ `mode : 7` + 8 位保留，共 16 位。

**电机控制模式（README 3.2 节）**

| 模式值 | 模式名称 | 描述 |
|---|---|---|
| 0 | MIT | 力矩控制模式 |
| 1 | PP | 位置轮廓控制模式 |
| 2 | PV | 速度轮廓控制模式 |
| 3 | PT | 力矩轮廓控制模式 |
| 4 | CSP | 循环位置控制模式 |
| 5 | CSV | 循环速度控制模式 |
| 6 | CST | 循环力矩控制模式 |

> ⚠️ README 内部**两处数字不一致，原样记录**：1.1 节写「支持多种电机控制模式（MIT、PP、PV、PT、CSP、CSV、CST）」共 **7 种**；6.2 节技术说明写「电机控制支持 **8 种**不同的控制模式」。3.2 节的模式表实际列出 **0–6 共 7 种**。使用时请以 3.2 节模式表为准。

**电机头与 PDO 结构**：`ROFU_MOTOR_RXHEAD_T`（`id` / `seq` / `ctrl`）、`ROFU_MOTOR_TXHEAD_T`（`id` / `seq` / `state`）、`ROFU_MOTOR_RXPDO_T`（header + union 七种模式载荷）、`ROFU_MOTOR_TXPDO_T`（header + `ROFU_MOTOR_TXPDO0_T`）。

**各模式 RXPDO 载荷**：`ROFU_MOTOR_RXPDO_MIT_T`（`kp`/`pos_target`/`kd`/`vel_target`/`taff`）、`_PT_T`（`tor_target`/`tor_profile`）、`_PV_T`（`vel_target`/`vel_max`/`acc_profile`/`dec_profile`）、`_PP_T`（`pos_target`/`vel_target`/`acc_profile`/`dec_profile`）、`_CST_T`（`tor_target`）、`_CSV_T`（`vel_target`）、`_CSP_T`（`pos_target`）。

**TXPDO 电机状态** `ROFU_MOTOR_TXPDO0_T`：`pos_real`（实际位置）、`vel_real`（实际速度）、`tor_real`（实际扭矩）、`mos_temper`（MOS 温度）、`motor_temper`（电机温度）、`error_code`（故障码）。

**SDO 配置参数结构**：`ROFU_MOTOR_SDO_PID_T`（`cur_kp`/`cur_ki`/`vel_kp`/`vel_ki`/`pos_kp`）、`ROFU_MOTOR_SDO_CONFIG_T`（`pole`/`phase_seq`/`redu_ratio`/`Rs`/`Lq`/`Ld`/`Kt`/`inertia`/`Ke`）、`ROFU_MOTOR_SDO_THR_T`（`vbus_high`/`vbus_low`/`iphase_high`/`mos_temper`/`motor_temper`/`speed_max`）。

**传感器 TXPDO 结构**：`ROFU_IMU_TXPDO_T`（`id`/`seq`/`sample_time_fine`/`quaternion[4]`/`acceleration[3]`/`rate_of_turn[3]`/`state`/`err_code`）、`ROFU_FT_TXPDO_T`（`id`/`seq`/`fx`/`fy`/`fz`/`mx`/`my`/`mz`/`err_code`）、`ROFU_ENCODE_TXPDO_T`（`id`/`seq`/`pos`/`state`）。

**字节序与时序**：所有数据包采用**小端字节序**，结构体打包对齐为 **1 字节**（`#pragma pack(1)`）；错误码为 **32 位无符号整数**且**厂家自定义**；建议通信频率 **1 kHz**、超时时间 **100 ms**。

**编译环境**：CubeIDE 版本 V1.18.0、CubeMX 版本 V6.14.0、芯片型号 **STM32H755**。

#### 关键目录与文件

`README 未说明` —— README 是协议规范正文，**没有列出任何仓库文件路径**，也没有安装/构建/运行章节。（本报告未对该仓库做目录树抓取，故不对文件构成下结论。）

#### 安装 / 推理 / 训练命令

`README 未说明` —— 该 README 不包含任何 shell 命令、构建指令或使用示例代码；所有 C 代码段均为**数据结构定义**（typedef / enum / union），不是可运行程序。唯一可执行层面的线索是 6.3 节的编译环境（CubeIDE V1.18.0 / CubeMX V6.14.0 / STM32H755）。

#### 模型权重地址与所需硬件

- 模型权重：**不适用**（本仓库无神经网络模型、无权重发布）。
- 硬件：README 6.3 节指明目标芯片为 **STM32H755**（含 Cortex-M7 与 Cortex-M4 双核，与 8887/8884 两个调试端口对应）；建议通信频率 1 kHz、超时 100 ms。
- 设备侧硬件类型：支持 IMU、力传感器（6 轴：fx/fy/fz/mx/my/mz）、编码器三类传感器。

#### 评测

`README 未说明` —— 无 benchmark、无任务数、无任何性能数字（唯一的量化参数是协议侧的建议频率 1 kHz 与超时 100 ms，以及端口号）。

#### 论文 / arXiv、License

- 论文 / arXiv：`README 未说明`（README 无 arXiv 链接、无 Citation 段落）。
- License：`README 未说明`（README 无 License 章节）。实读校验：`main` 与 `master` 分支上均**不存在** `LICENSE`、`LICENSE.txt`、`LICENSE.md`。
- 版本历史（README 6.1 节）：**v1.0 / 2026-04-13 / 初始版本**。

#### 亮点与局限

**亮点**
- **协议规范即 README**：把设备类型表、PDO/SDO 线格式（含 ASCII 报文示意）、端口分配、C 结构体定义、寄存器地址表、控制模式表、字节序与打包对齐、编译工具链版本全部写在一页里，是「文档即交付物」的范例 —— 硬件/嵌入式联调时，对方只需要一个 URL。
- **给出了可直接照抄的报文示例**（PDO 组合包、SDO 读写），比只描述字段表实用得多。
- **寄存器地址分组有章法**：0x1000 段电机参数、0x2000 段三环 PID、0x2010 段保护阈值，预留了 `*_ALL` 批量读写地址。
- **工程参数明确**：小端 + `#pragma pack(1)` + 1 kHz + 100 ms 超时 —— 这四项是联调期最容易扯皮的点，全部前置写死。

**局限**
- **无任何可执行代码或示例程序**：全部 C 片段都是类型定义，没有 client/server 实现、没有 Makefile / CMake、没有测试工具，因此**无法据此仓库独立跑通通信**。
- **无 License**，商用与再分发存在法律不确定性。
- **控制模式数量自相矛盾**（1.1/3.2 节 7 种 vs 6.2 节 8 种）。
- **错误码未定义**：只说「32 位无符号整数、厂家自定义」，没有错误码表，故障诊断能力实际上取决于设备厂商。
- **未说明 CRC/校验、重传、丢包处理**等 UDP 可靠性机制；仅给超时时间。对 1 kHz 周期控制而言这是关键缺口。
- 无版本协商 / 兼容性策略说明（只有一条 v1.0 记录）。
- 文档语言为中文，且无英文版；`README 未说明` 是否有 `README_zn.md` 之外的本地化版本。

#### 可复用要点

1. **「文档即交付」的协议规范写法**：字段表 + ASCII 报文示意 + 语言级结构体定义 + 端口/地址规划 + 工具链版本，五件套放同一页。做任何跨团队/跨厂商接口时都可以照这个模板写。
2. **把最容易扯皮的物理层约定前置写死**：字节序（小端）、打包对齐（`#pragma pack(1)`）、建议频率（1 kHz）、超时（100 ms）—— 这四项在协议文档第一屏就写明，能省掉大量联调返工。
3. **寄存器地址按功能分段并预留 `*_ALL` 批量地址**（0x1000 电机 / 0x2000 PID / 0x2010 阈值），便于后续扩展与批量配置。

---

### 2.6 `Hunyuan3D-WorldClaw`

#### 项目定位与要解决的问题

标题：**"WorldClaw: Agentic 3D Open-world Generation at Scale"**（规模化 agentic 3D 开放世界生成）。README 徽章指向三个外部资源：[arXiv:2608.05248](https://arxiv.org/abs/2608.05248)、[项目页](https://tencent-hunyuan.github.io/Hunyuan3D-WorldClaw/)、[HuggingFace papers](https://huggingface.co/papers/2608.05248)。

**仓库性质：README-only 研究发布，无代码。** 现场读取仓库文件树仅包含 `README.md`、`.gitignore` 与 `assets/` 目录（`assets/teaser.jpg`、`assets/pipeline.jpg`）。README 全文 40 行。

#### 核心架构 / 方法

**`README 未说明`。**

README 的 "📖 Method" 章节**只有一张图**（`assets/pipeline.jpg`），既无文字描述、无公式、无模块清单，也无模型名/参数量。可确证的只有：News 段落写明 **2026.08.07 Paper 与 project page 发布**；标题中的定位词是 "Agentic 3D Open-world Generation at Scale"。

> 按本报告「只写 README 真实出现的内容」的硬性约束，本节**无法给出 backbone、能力分类法、训练阶段或 RL 策略的任何描述** —— 这些信息只存在于论文与项目页，不在仓库内。任何关于方法的叙述都需要另行阅读 arXiv:2608.05248，本报告不做推测。

#### 关键目录与文件（README 真实出现的路径）

| 路径 | 说明 |
|---|---|
| `assets/teaser.jpg` | README 顶部 teaser 图（宽 95%） |
| `assets/pipeline.jpg` | Method 章节流水线图（宽 95%） |
| `README.md` | README 全文（40 行） |
| `.gitignore` | 现场文件树读取到的另一顶层文件 |

#### 安装 / 推理 / 训练命令

`README 未说明` —— README 不含任何安装、推理、训练、评测命令，也不含环境依赖说明。**因为没有代码，所以不存在可照抄的命令。**

#### 模型权重地址与所需硬件

- 权重：`README 未说明`（未给出任何 HuggingFace 模型仓库地址；三个徽章中只有 HuggingFace **papers** 链接，不是模型链接）。
- 所需硬件：`README 未说明`。
- 模型规模：`README 未说明`。

#### 评测

`README 未说明` —— README 无 Evaluation / Result 章节，**无任何 benchmark 名称、任务数或分数**。

#### 论文 / arXiv、License

- 论文：**arXiv:2608.05248** —— *WorldClaw: Agentic 3D Open-World Generation at Scale*，作者 `Guo, Chunchao and Li, Jinpeng and Li, Yang and Huang, Zilong`，2026（Citation 为 `@article{guo2026worldclaw, journal={arXiv preprint arXiv:2608.05248}}`）。
- 项目页：<https://tencent-hunyuan.github.io/Hunyuan3D-WorldClaw/>。
- License：`README 未说明`（无 License 章节）；实读校验 `main` 与 `master` 分支均无 `LICENSE` / `LICENSE.txt` / `LICENSE.md`。

#### 亮点与局限

**亮点**
- **外部资源三件套齐全且都在首屏**：arXiv、项目页、HF papers 链接并列，配合 teaser 图，作为「论文发布引流仓库」完成度是够的。
- star 数在本报告 7 个仓库中最高（**1377**，外部探测），说明 "Agentic 3D 开放世界生成" 这个题目本身关注度很高。
- **News 有明确日期**（2026.08.07），时间线可核对。

**局限**
- **完全无代码、无权重、无文档**：README 40 行，Method 只有一张图。除了「论文已发布」之外，仓库本身**不提供任何可复现内容**。
- 无 License、无评测数字、无硬件说明、无 Citation 之外的任何技术细节。
- 仓库体量（外部元数据约 180 MB）主要来自 `assets/` 图片，而非代码。
- 标题含 "Agentic" 但 README 完全没有说明 agent 体现在哪里（规划器？工具调用？迭代生成？）—— 需要读论文才能判断这是不是「LLM 工具调用意义上的 agent」。

#### 可复用要点

1. **论文引流型仓库的最小可用结构**：首屏三链接（arXiv / 项目页 / HF papers）+ teaser 图 + News 带日期 + Method 流水线图 + Citation。若目标只是「让论文可被检索和引用」而非「让结果可复现」，这个 40 行模板已经够用。
2. **但对工程团队要有预期管理**：这类仓库**不能作为技术选型依据** —— 本报告将其明确标注为 README-only，避免下游误以为有可跑的 pipeline。

---

### 2.7 `Hunyuan3D-Buffalo1.0`

#### 项目定位与要解决的问题

标题：**"Hunyuan3D-Buffalo 1.0 — A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing"**（面向可扩展 3D 生成、理解与编辑的统一多模态模型）。出品方标注为 *Tencent Hunyuan3D*。

要解决的问题从标题即可读出：把 3D 的**生成、理解、编辑**三类任务统一到一个多模态模型里，并强调 **scalable**（可扩展）。

**仓库性质：README-only 研究发布，无代码。** 现场读取仓库文件树仅包含 `README.md` 与 `assets/` 目录（`assets/pipeline.png`、`assets/result.jpg`）。README 全文 49 行。

#### 核心架构 / 方法

**`README 未说明`。**

README 的 "📖 Method" 章节**只有一张图**（`assets/pipeline.png`），无任何文字描述；"🏆 Result" 章节同样**只有一张图**（`assets/result.jpg`）。README 未给出模型规模、backbone、视觉编码器、训练阶段或数据构成。

> 同样按硬性约束，本节不做任何推测性补全。方法细节需阅读 arXiv:2608.02711。

#### 关键目录与文件（README 真实出现的路径）

| 路径 | 说明 |
|---|---|
| `assets/pipeline.png` | Method 章节流水线图（宽 95%） |
| `assets/result.jpg` | Result 章节结果图（宽 95%） |
| `README.md` | README 全文（49 行） |

> README 首行有一段被 HTML 注释包裹的标题（`<!-- ... -->`），因此页面上不显示，但源码中可见标题原文为 "Hunyuan3D-Buffalo 1.0：A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing"。

#### 安装 / 推理 / 训练命令

`README 未说明` —— README 不含任何命令、代码块或依赖说明（全文唯一的代码块是 BibTeX Citation）。**因为没有代码，不存在可照抄的命令。**

#### 模型权重地址与所需硬件

- 权重：`README 未说明`（无 HuggingFace 模型链接；仅有 HF **papers** 徽章指向 `huggingface.co/papers/2608.02711`）。
- 模型规模（总/激活）：`README 未说明`。
- 所需硬件：`README 未说明`。

#### 评测

`README 未说明` —— "🏆 Result" 章节**只有 `assets/result.jpg` 一张图**，README 正文**无任何 benchmark 名称、任务数或分数**。（分数若存在，只存在于图片像素中，本报告不做 OCR 猜测。）

#### 论文 / arXiv、License

- 论文：**arXiv:2608.02711** —— *Hunyuan3D-Buffalo 1.0: A Unified Multimodal Model for Scalable 3D Generation, Understanding, and Editing*，第一作者 `Ye, Junliang`，2026。
- 项目页：<https://tencent-hunyuan.github.io/Hunyuan3D-Buffalo1.0/>。
- News：**2026.08.05 Paper 与 project page 发布**。
- License：`README 未说明`（无 License 章节）；实读校验 `main` 与 `master` 分支均无 `LICENSE` / `LICENSE.txt` / `LICENSE.md`。

#### 亮点与局限

**亮点**
- **统一「生成 / 理解 / 编辑」的题目定位**清晰，且标题直接点明 "Scalable"，与同族的 `Hunyuan3D-2`、`Hunyuan3D-Omni`、`Hunyuan3D-Part`、`HY3D-Bench` 等形成 3D 方向的完整矩阵。
- 首屏三链接（arXiv / 项目页 / HF papers）+ Method 图 + Result 图，作为论文配套仓库结构规整。
- News 日期明确（2026.08.05）。

**局限**
- **无代码、无权重、无 License、无任何量化信息** —— 从工程角度看，仓库当前价值等同于一篇论文的落地页。
- Method 与 Result 都只有图片，**README 文本零技术信息**，无法据此判断与 `Hunyuan3D-Omni`（可控 3D 生成）、`Hunyuan3D-Part`（部件分割与生成）的能力边界。
- 无评测数字（`assets/result.jpg` 内容不可作为可引用数据）。
- 无 Citation 之外的作者/机构信息。

#### 可复用要点

1. **与 2.6 同构**：论文发布期先用极简 README 占位（三链接 + 两张关键图 + Citation），后续再补代码与权重。对使用者而言，遇到这类仓库应第一时间去看论文而不是找代码。
2. **「Result 只有图」的仓库不要纳入可复现清单**：本报告统一将其标为 README-only，避免在技术选型或对比表中被误当作可用方案。

---

## 3. 具身智能技术栈全景

把 5 个具身仓库（`HY-Embodied`、`HY-Embodied-0.5-X`、`Hy-Embodied-0.5-VLA`、`Hy-Embodied-RxBrain-1.0`、`Hy-Embodied-RoboFusion`）串起来看，它们恰好构成一条**自下而上的完整栈**：从「看懂世界」到「说出计划」到「产生连续动作」到「用真实机器人数据训练」到「把指令以 1kHz 送到电机」。

### 3.1 栈层定位与依赖关系表

| 栈层 | 仓库 / 模型 | 角色（做什么） | 输入 → 输出 | 依赖的下层 | 被谁依赖 / 谁在其上 |
|---|---|---|---|---|---|
| ① 基座模型（感知-语言） | `HY-Embodied` → `Hy-Embodied-VLM-1.0`（~30B 总 / ~3B 激活 MoE，`Hy3-A3B` + `Hy-ViT2`） | 物理世界的高效视觉-语言基座；提供具身状态理解、动作-转移推理、序列自适应推理三层能力 | 图像（单 prompt 最多 128 张）/ 视频 + 文本 → 文本（可含 CoT） | 无（栈底，仅依赖 `Hy3` LLM 骨干与 `Hy-ViT2` 视觉编码器） | 概念上是全栈的能力上限来源；`HY-Embodied` 仓库内同时归档 `Hy-Embodied-0.5/`，是 0.5 代的发布容器 |
| ② 基座模型（轻量边缘） | `HY-Embodied-0.5-X`（4B 总 / 2B 激活，基于 `Hy-Embodied-0.5 MoT-2B`） | 「理解-推理-行动」闭环的增强版边缘基座；提供单卡可训的 SFT 全流程 | 图像 + 文本 → 文本（`<think>…</think><answer>…</answer>`）/ 归一化坐标（0–1000） | `HY-Embodied` 仓库发布的 `HY-Embodied-0.5 MoT-2B` 架构与权重（`tencent/HY-Embodied-0.5`） | 与 ① 同层但面向 ≥16 GB VRAM 的端侧；是与 PlaygroundX 仿真集成的规划层 |
| ③ 认知与想象层 | `Hy-Embodied-RxBrain-1.0`（~6.2B MoT 三流 + FLUX VAE 83.8M） | 联合语言-视觉推理与**想象**：不仅输出文本计划，还输出**目标图像**，并用学习到的 `<Image>` token 决定何时想象 | 图像 / 多帧视频 + 任务文本 → 交错序列（推理文本 + flow-matched 想象帧） | 冻结的 FLUX VAE latent 空间（外部 `ae.safetensors`）；`transformers` 指定 commit 的 `hunyuan_vl_mot` 骨干 | 向下可为 ④ 提供「目标图像」形式的中层表征；与 ① 共用 MoT/`hy_vl_mot` 血统 |
| ④ 动作模型（VLA） | `Hy-Embodied-0.5-VLA` → `Hy-VLA`（`HY-VLA-0.5`；action expert 370M） | 端到端 VLA 与真实机器人学习栈：数据采集 → 预训练 → SFT → RL 后训练（FlowPRO）→ 实机部署 | `observation.images.{top_head,hand_left,hand_right}` + `observation.state`（16 维双机械臂 EEF）+ `task` → 连续 action chunk（delta EEF，H=50 @ 10 Hz） | **`Hy-Embodied-0.5 MoT` backbone**（VLM tower 从 `tencent/HY-Embodied-0.5` 初始化）；UMI 数据（`Hy-Embodied-0.5-VLA-Data`）；LeRobot v3.0 / Lance 格式 | 由 RoboTwin 2.0 / RoboDojo 提供仿真评测环境；实机侧对接 Dobot X-Trainer、JAKA K1、Astribot S1 |
| ⑤ 通信协议层 | `Hy-Embodied-RoboFusion`（协议 v1.0，C 实现，STM32H755） | 机器人控制系统 UDP 应用层协议：PDO 周期实时数据 + SDO 请求-响应配置；设备/传感器/电机状态统一编解码 | UDP 二进制包（PDO/SDO）→ 电机控制指令 / 传感器读数（IMU、力传感器、编码器） | 无（硬件层，依赖 STM32H755 + CubeIDE/CubeMX 工具链） | 承载 ④ 输出的连续动作指令；1 kHz 建议频率、100 ms 超时 |

### 3.2 依赖关系链（文字版）

```
                 ┌──────────────────────────────────────────────┐
                 │  HY-Embodied 仓库（多版本容器，默认分支 master）│
                 │   ├─ Hy-Embodied-VLM-1.0  (~30B/3B MoE)      │  ← ① 能力上限
                 │   └─ Hy-Embodied-0.5/  (MoT-2B 归档)          │
                 └───────────────┬──────────────────────────────┘
                                 │ 提供 HY-Embodied-0.5 MoT-2B 架构与权重
              ┌──────────────────┴───────────────────┐
              ▼                                      ▼
 ┌────────────────────────┐            ┌──────────────────────────────┐
 │ HY-Embodied-0.5-X      │            │ Hy-Embodied-0.5-VLA          │
 │ (4B/2B, 规划+理解+行动) │            │ VLM tower 从 HY-Embodied-0.5 │
 │ ② 边缘基座 / 规划层      │            │ 初始化                        │
 └────────────────────────┘            │ ④ VLA / 动作模型（370M expert）│
              │                        └───────────┬──────────────────┘
              │ PlaygroundX 仿真集成                │ 连续 action chunk
              │ (reason→execute→detect→replan)      │
              │                                     ▼
              │                        ┌──────────────────────────────┐
              │                        │ Hy-Embodied-RoboFusion       │
              │                        │ ⑤ UDP PDO/SDO, 1 kHz         │
              │                        │ → STM32H755 → 电机/传感器     │
              │                        └──────────────────────────────┘
              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ Hy-Embodied-RxBrain-1.0  (③ 认知与想象层, ~6.2B MoT 三流)          │
 │ 文本计划 + 目标图像成对输出；与 ①② 同属 MoT / hunyuan_vl_mot 血统   │
 └──────────────────────────────────────────────────────────────────┘
```

**几点需要说清楚的观察**：

1. **真正被代码级复用的依赖只有一条**：`Hy-Embodied-0.5-VLA` 明确写明 VLM tower 从 `tencent/HY-Embodied-0.5` 初始化（README Abstract 与 `[model-card]` 的 `base_model: tencent/HY-Embodied-0.5` 一致）。其余仓库之间的「栈关系」是**功能互补**，README 中**并未声明**存在代码依赖或权重继承。
2. **0.5 代与 1.0 代在架构上是两套东西**：0.5 代是 **MoT（Mixture-of-Transformers）**，1.0 代 `Hy-Embodied-VLM-1.0` 是 **MoE（Mixture-of-Experts，8 of 128 experts）**。因此 `Hy-VLA`（基于 0.5 代 MoT）与 `Hy-Embodied-VLM-1.0`（1.0 代 MoE）**不是同一条升级线上的上下游**，不能假定「VLM-1.0 能直接替换 Hy-VLA 的 backbone」。
3. **RxBrain 是横向能力扩展而非纵向层**：它引入「想象未来帧」与「目标图像」这两种 ① ② 都不产出的输出模态，因此更适合看作与基座并列的**认知/想象能力分支**。它与 ① 的联接点是共同的 MoT / `hunyuan_vl_mot` 血统与同一个 pinned transformers commit（`9293856`）。
4. **两个 3D 仓库不属于这条具身栈**：`Hunyuan3D-WorldClaw` 与 `Hunyuan3D-Buffalo1.0` 面向 3D 资产/开放世界的生成，且都是 README-only。它们与具身栈的潜在接口是「为具身 agent 提供 3D 资产或仿真环境」，但**README 中没有任何一方声明这种集成**。
5. **通信协议层是整条栈的地基，但没有任何仓库声明与它对接**：`Hy-Embodied-0.5-VLA` 的文档只到「部署到 Dobot X-Trainer / JAKA K1 / Astribot S1」这一层，未提及 RoboFusion 协议。把 ⑤ 放进栈里是基于它的协议定位与目标芯片（STM32H755）做出的**结构性判断**，而非 README 声明 —— 请按此口径引用。

### 3.3 栈的横切面：评测与数据

| 栈层 | 评测环境 / benchmark | 数据 |
|---|---|---|
| ① `Hy-Embodied-VLM-1.0` | 38 个具身 benchmark（23 状态理解 + 8 动作-转移推理 + 7 序列/自适应推理）；R2R-CE；Matterport3D Object Goal Navigation | README 未披露数据量与配比 |
| ② `HY-Embodied-0.5-X` | 10 个开源具身规划 benchmark；内部 AI2Thor 规划 benchmark（**1,011 任务**，厨房/卧室/客厅/浴室四场景）；PlaygroundX 仿真 | 自采第一人称 + 机械臂 + 开源具身数据，CoT 标注；数据量未披露 |
| ③ `Hy-Embodied-RxBrain-1.0` | `README 未说明`（只指向技术报告）；`RxBrain-Bench` **未开源** | 四类任务数据（VQA jsonl/parquet、T2I parquet、多帧 jsonl、交错规划 jsonl+VAE 镜像）；原始机器人轨迹→中间记录的转换**未随仓库发布** |
| ④ `Hy-VLA` | RoboTwin 2.0（50 任务 × 100 rollouts）、RoboDojo（HDF5）、实机 Track A/B | Hy-UMI-10K：250K+ episodes / 10K+ 小时，开源 2K+ 小时、22 个 Lance 表、240×424、30 FPS、16 维 state |
| ⑤ `Hy-Embodied-RoboFusion` | `README 未说明`（无评测） | 不适用（协议规范） |
| — `Hunyuan3D-WorldClaw` | `README 未说明` | `README 未说明` |
| — `Hunyuan3D-Buffalo1.0` | `README 未说明`（Result 只有图） | `README 未说明` |

#### 可复用要点

1. **分层设计要允许「同层不同架构」并存**：0.5 代 MoT 与 1.0 代 MoE 同时在线，各自服务不同的部署预算（2B 激活端侧 vs 3B 激活高性能）。把「架构代际」和「产品分层」解耦，避免一刀切升级。
2. **跨仓库依赖只保留一条明线**：整个具身栈里只有 `Hy-VLA ← HY-Embodied-0.5` 一条真实的权重继承关系，且写进了 README Abstract 与 `[model-card]` 的 `base_model` 字段。依赖越少越容易维护 —— 其余能力靠「功能互补 + 统一评测口径」而非代码耦合成栈。
3. **每一层都要有独立的可验证输出**：① 有 38 benchmark，② 有 AI2Thor 1,011 任务，④ 有 RoboTwin 50 任务 × 100 rollouts 与实机 Track A/B。每一层都能单独被验证，是这条栈能被工程采纳的关键 —— 反面案例就是 ③（无数字）与 ⑤（无评测）。

---

## 4. 与「LLM 工具调用型 Agent」的差异

本系列第 1 篇报告覆盖的 7 个仓库（`UniRL`、`C3-Benchmark`、`VisualNeedle`、`RepoProbe`、`ExplorationBench`、`PlanningBench`、`Hyra-results`）代表的是**文本工具调用型 Agent** 技术栈；本报告的 5 个具身仓库代表**感知-动作闭环**技术栈。两者在 6 个维度上分野明显：

| 维度 | LLM 工具调用型 Agent（报告 01 覆盖） | 具身智能栈（本报告覆盖） | 证据来源 |
|---|---|---|---|
| **行动空间** | 离散符号调用：工具名 + JSON 参数（如 `CalculatorTool` 白名单 AST 求值、`SandboxTool` 持久 Python 子进程、`SearchTool`/`VisitTool`），或输出文本计划 | 连续动作向量：delta EEF chunk（`[xyz(3)+rot6d(6)+gripper(1)]*2`），经 flow matching 积分 10 步 Euler 生成 | 报告 01 的 `UniRL` 环境契约；本报告 2.3 的 action expert 与 `observation.state` 定义 |
| **感知输入** | 以文本为主；视觉作为「工具返回的观测」或一次性 VQA 输入（`VisualNeedle` 是主动裁剪视觉搜索，但仍以文本为最终答案） | 原生多模态连续流：多视角相机（`cam_high` / `cam_left_wrist` / `cam_right_wrist`）、K 帧历史（K=6）、16 维本体状态、30 FPS | 本报告 2.3 数据表与 `compact memory encoder` |
| **闭环形态** | 多轮：`input(user) → generated(assistant) → observation(tool) → generated → …`，由 reward service 对最终答案打分 | 物理闭环：reason → execute → detect failure → replan（ReAct）；异步 producer-consumer + 三次贝塞尔 chunk 拼接做高频控制；VLA 在 10 Hz 上闭环 | 报告 01 的 `AgenticTrainer` 轨迹结构；本报告 2.2 的 PlaygroundX ReAct loop、2.3 的异步部署栈 |
| **输出模态** | 文本 / 代码 / HTML 等 artifact（`Hyra-results` 归档成品解法；`ArtifactsBenchmark-related` 评测视觉交互式代码生成） | 除文本外还有**连续动作**与**想象帧 / 目标图像**（RxBrain 的 flow-matching head 输出未来帧与 goal image） | 本报告 2.4 的三项能力 |
| **训练信号** | token 级 RL：GRPO / DPPO / DRPO / CPPO，group-normalized advantage 施加到所有 assistant turn；失败轨迹排除出更新 | flow matching 监督 + 偏好优化：预训练用 conditional flow matching；后训练用 **FlowPRO / RPRO**（critic-free、无需奖励模型、对称 proximal 正则防 reward hacking），把实机 intervention-and-rollback 转成偏好对 | 报告 01 的 `UniRL` 算法表；本报告 2.3 的 FlowPRO 章节 |
| **时间尺度与失败代价** | 秒级到分钟级；失败 = 答案错、可重试、无物理后果 | 毫秒级到秒级控制（协议层建议 1 kHz、超时 100 ms；策略 10 Hz）；失败 = 物体掉落/碰撞/任务中断，**不可无损重试**；因此需要「风险评估、故障诊断、失败反思、重规划」成为模型的一等能力 | 本报告 2.5 协议时序、2.2 的六项专项改进、2.4 的图像侧 exposure bias |
| **评测口径** | 任务成功率 / checklist / judge（`C3-Benchmark` 三类挑战、`RepoProbe` 加权 checklist、`PlanningBench` 467 行 eval 集） | 成功率 + **物理指标**：RoboTwin Clean/Randomized 双设定（90.9% / 90.1%）、每任务 100 rollouts × 50 任务、实机 mean ± std / **完成时间**（RPRO 99±0.6% / 16s） | 本报告 2.3 全部评测表 |
| **「Agentic」一词的含义** | 指多轮工具调用与自主决策路径（`UniRL` 的 `train_agentic` = "Service-scored multi-turn tool use"） | 指**长程规划 + 反思/修复/恢复 + 动作后果推理**（`Hy-Embodied` 的 "Sequential and Adaptive Reasoning"） | 报告 01 的训练模式表；本报告 2.1 taxonomy |

**关键分野的一句话总结**：LLM 工具调用型 Agent 把「世界」抽象成一组可调用的 API 与返回的文本观测，错误可以回滚；具身智能栈面对的是一个**连续的、不可回滚的、毫秒级时钟驱动的物理世界**，因此它必须把「感知→状态估计→动作生成→失败检测→重规划」压进同一个模型或同一条紧耦合流水线，并且动作输出必须是连续的（flow matching / delta chunk）而非离散 token。

**两者的交汇点**（值得注意）：`Hy-Embodied-VLM-1.0` 的 vLLM 插件同时注册了 **tool-call parser**（`ToolParserManager.register_module("hy_v3", HYV3ToolParser)`，解析模型 XML 风格的 tool-call 输出）与 **reasoning parser**（`hunyuan_v3`，切分 `<think>…</think>`）。也就是说，一个具身基座在服务层同时暴露了「工具调用」与「思维链」两条通道 —— 具身模型并不排斥工具调用范式，而是把它作为附加能力；真正不可替代的是连续动作输出那一层。

#### 可复用要点

1. **同一套服务接口承载两种 Agent 范式**：一个 OpenAI 兼容端点内，用 chat-template kwarg 切 thinking / non-thinking，用独立 parser 分别处理 tool-call（XML 风格）与 reasoning（`<think>` 块）。工程上不必为两种范式部署两套服务。
2. **评测口径必须匹配行动空间的代价结构**：文本 Agent 可以用 judge/checklist 打分，具身栈必须报**每任务多次 rollout 的成功率**（100 rollouts × 50 任务）以及**完成时间**（16s vs 27s）。跨范式比较时不要拿「任务成功率」直接对比 —— 分母的物理含义完全不同。
3. **失败恢复能力要进能力分类法，不要只当兜底逻辑**：`Hy-Embodied` 把「反思、修复、恢复（reflection, repair, and recovery）」写进第三层 taxonomy，`HY-Embodied-0.5-X` 把「failure reflection / risk judgement」列为专项改进 —— 在不可回滚的场景里，恢复能力就是核心能力本身。

---

## 5. 可复用要点汇总

前面每节末尾已给出 1–3 条针对性实践；本节按「跨仓库通用」再收敛为 6 条，并附上来源仓库，便于直接检索原文。

### 5.1 模型与架构

1. **能力分类法先行，并让数据管线围绕它组织**（来源：`HY-Embodied`）。三层 taxonomy（状态理解 → 动作-转移推理 → 序列/自适应推理）不只是评测分组，而是明确用于指导预训练与后训练的数据混合设计。垂直领域基座都值得先写这一层。
2. **动作表示与本体解耦 = 跨本体迁移的前提**（来源：`Hy-Embodied-0.5-VLA`）。delta-chunk（相对当前帧的 EEF delta，每臂 10 维 xyz+rot6d+gripper）+ 与运动学无关，配合 UMI 大规模预训练，使 Track B 跨本体从 38%/44% 提升到 90%/89%。反之，去掉 UMI 预训练的消融会「塌陷」。
3. **零新增参数的历史压缩设计**（来源：`Hy-Embodied-0.5-VLA`）。memory encoder 的 temporal（跨帧因果）与 spatial（帧内双向）注意力复用同一套 QKV 投影，时序混合后丢弃过去帧 token，使 VLM token 数与历史长度无关。任何「多帧历史 → 单帧表示」的模块都可以照此设计。

### 5.2 训练与数据

4. **把「训练/推理分布不一致」当一等工程问题**（来源：`Hy-Embodied-RxBrain-1.0`）。自回归 rollout 中前序帧在推理时是自身有损 VAE 重建、训练时是清晰 GT，这个 exposure bias 从第 2 步起污染条件分布并累积。通用修法：预计算 VAE encode→decode 镜像、按概率替换前序帧，观测与 FM 目标保持真实。凡「生成结果回流为下一步输入」的系统都适用。
5. **「Validate → Scale → Full-Run」+ 单表快速迭代回路**（来源：`HY-Embodied-0.5-X`、`Hy-Embodied-0.5-VLA`）。先用小规模高质量子集验证配置与数据清洗，再逐步放大，确认最优混合后才全量训练；同时保留一条「单表 / 单卡冒烟」的分钟级回路（`train_table_vlm.sh`、`configs/sft/example_small_single_gpu.yaml`），避免调参只能靠排队。

### 5.3 工程与合规

6. **服务化靠插件、不 fork 上游，并把默认参数与就绪检查写进文档**（来源：`HY-Embodied`）。通过 `vllm.general_plugins` entry-point 注册 `ModelRegistry` + `ToolParserManager` + `ReasoningParserManager`，并预留「上游合并后零改动卸载插件」的退出路径；同时把 `TP`/`GPU_MEM_UTIL`/`MAX_MODEL_LEN`/`CHAT_TEMPLATE` 的默认值列成表，给出 `curl -sf .../v1/models` 就绪检查与首次加载耗时预期（1–5 分钟）。
7. **License 必须读文件原文，不能读 README 徽章或目录树注释**（来源：`HY-Embodied-0.5-X` vs 其余）。本报告的实证：`HY-Embodied-0.5-X` 的 README 无 License 章节、其目录树却把 `LICENSE` 注释为 `# Apache-2.0`，而 `master/LICENSE` 实读是**带地域限制的 TENCENT HY COMMUNITY LICENSE AGREEMENT**（排除 EU/UK/韩国，Section 5(c) 禁止 Territory 外使用）。同时，既有工作区快照 `licenses.csv` 把 `HY-Embodied` 与 `HY-Embodied-0.5-X` 都误标为「无 LICENSE 文件」，实际原因是**默认分支是 `master` 而非 `main`**，按 `main` 抓取全部 404。
8. **跨仓库/跨厂商接口用「文档即交付」五件套**（来源：`Hy-Embodied-RoboFusion`）。字段表 + ASCII 报文示意 + 语言级结构体定义 + 端口/寄存器地址规划 + 工具链版本写在同一页；并把最易扯皮的物理层约定（小端、`#pragma pack(1)`、1 kHz、100 ms 超时）放在第一屏。同时应补上该 README 缺失的部分：错误码表、CRC/重传/丢包策略、版本协商。

---

## 6. 参考链接

### 6.1 仓库与代码

| 仓库 | 链接 | 默认分支（实读校验） |
|---|---|---|
| `HY-Embodied` | <https://github.com/Tencent-Hunyuan/HY-Embodied> | `master`（`main` 404） |
| `HY-Embodied-0.5-X` | <https://github.com/Tencent-Hunyuan/HY-Embodied-0.5-X> | `master`（`main` 404） |
| `Hy-Embodied-0.5-VLA` | <https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA> | `main` |
| `Hy-Embodied-RxBrain-1.0` | <https://github.com/Tencent-Hunyuan/Hy-Embodied-RxBrain-1.0> | `main` |
| `Hy-Embodied-RoboFusion` | <https://github.com/Tencent-Hunyuan/Hy-Embodied-RoboFusion> | `main` |
| `Hunyuan3D-WorldClaw` | <https://github.com/Tencent-Hunyuan/Hunyuan3D-WorldClaw> | `master`（`main` 404） |
| `Hunyuan3D-Buffalo1.0` | <https://github.com/Tencent-Hunyuan/Hunyuan3D-Buffalo1.0> | `main` |

### 6.2 论文 / 技术报告 / 项目页

| 工作 | arXiv | 项目页 / 其他 |
|---|---|---|
| `Hy-Embodied-VLM-1.0`（Efficient Physical-World Agents） | [arXiv:2607.12894](https://arxiv.org/abs/2607.12894) | 技术报告 PDF 随仓库发布：`hy_embodied_vlm_1_0_tech_report.pdf` |
| `HY-Embodied-0.5`（Embodied Foundation Models for Real-World Agents） | [arXiv:2604.07430](https://arxiv.org/abs/2604.07430) | — |
| `HY-Embodied-0.5-X` | `README 未说明`（Citation 无 arXiv 字段） | <https://tairos.tencent.com/openSourceModels/hy-embodied> |
| `Hy-Embodied-0.5-VLA` / `Hy-VLA` | [arXiv:2606.14409](https://arxiv.org/abs/2606.14409) | <https://tairos.tencent.com/openSourceModels/hy-embodied-0.5-vla> |
| `Hy-Embodied-RxBrain-1.0`（RxBrain） | [arXiv:2607.14187](https://arxiv.org/abs/2607.14187) | <https://tairos.tencent.com/openSourceModels/hy-embodied-rxbrain-1.0>；Demo <https://huggingface.co/spaces/hugging-apps/rxbrain-embodied-cognition> |
| `Hunyuan3D-WorldClaw`（WorldClaw） | [arXiv:2608.05248](https://arxiv.org/abs/2608.05248) | <https://tencent-hunyuan.github.io/Hunyuan3D-WorldClaw/>；HF papers <https://huggingface.co/papers/2608.05248> |
| `Hunyuan3D-Buffalo1.0` | [arXiv:2608.02711](https://arxiv.org/abs/2608.02711) | <https://tencent-hunyuan.github.io/Hunyuan3D-Buffalo1.0/>；HF papers <https://huggingface.co/papers/2608.02711> |
| `FlowPRO`（Hy-VLA 后训练算法） | `README 未说明` | <https://wuyeyexvnainai.github.io/flowpro/> |

### 6.3 模型权重与数据集

| 资产 | 链接 |
|---|---|
| `tencent/Hy-Embodied-VLM-1.0`（权重） | <https://huggingface.co/tencent/Hy-Embodied-VLM-1.0> |
| `tencent/HY-Embodied-0.5`（0.5 代 MoT-2B 权重） | <https://huggingface.co/tencent/HY-Embodied-0.5> |
| `tencent/HY-Embodied-0.5-X`（权重） | <https://huggingface.co/tencent/HY-Embodied-0.5-X> |
| `tencent/Hy-Embodied-0.5-VLA-UMI`（预训练 checkpoint） | <https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-UMI> · [ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy-Embodied-0.5-VLA-UMI) |
| `tencent/Hy-Embodied-0.5-VLA-RoboTwin`（后训练 checkpoint） | <https://huggingface.co/tencent/Hy-Embodied-0.5-VLA-RoboTwin> · [ModelScope](https://modelscope.cn/models/Tencent-Hunyuan/Hy-Embodied-0.5-VLA-RoboTwin) |
| `tencent/Hy-Embodied-0.5-VLA-Data`（UMI 数据，2K+ 小时） | <https://huggingface.co/datasets/tencent/Hy-Embodied-0.5-VLA-Data> · [ModelScope](https://modelscope.cn/datasets/Tencent-HunYuan/Hy-Embodied-0.5-VLA-Data) |
| `tencent/Hy-Embodied-RxBrain-1.0`（权重，~6.2B） | <https://huggingface.co/tencent/Hy-Embodied-RxBrain-1.0> |
| FLUX VAE `ae.safetensors`（RxBrain 生成路径必需，83.8M） | <https://github.com/black-forest-labs/flux> |

### 6.4 依赖 / 相关开源项目

| 项目 | 用途 | 链接 |
|---|---|---|
| `Hy3` | `Hy-Embodied-VLM-1.0` 的 MoE LLM 骨干 | <https://huggingface.co/tencent/Hy3> |
| `Hy-ViT2` | `Hy-Embodied-VLM-1.0` 的视觉编码器（指向 `tencent/HunyuanOCR`） | <https://huggingface.co/tencent/HunyuanOCR> |
| `transformers` 指定 commit `9293856` | 提供 `HunYuanVLMoT{Config,Model,Processor}` / `hunyuan_vl_mot` 原生支持（`HY-Embodied-0.5-X` 与 `RxBrain` 共用） | <https://github.com/huggingface/transformers/commit/9293856c419762ebf98fbe2bd9440f9ce7069f1a> |
| `vllm-project/vllm#43401` | 修复顶层 `reasoning_effort` 覆盖 `chat_template_kwargs["reasoning_effort"]` 的问题 | <https://github.com/vllm-project/vllm/pull/43401> |
| `LeRobot` v3.0 | UMI 数据的 schema 与工具链 | <https://github.com/huggingface/lerobot> |
| `lerobot-lancedb` | Lance 格式的 LeRobot 数据读取 | <https://github.com/lancedb/lerobot-lancedb> |
| `RoboTwin 2.0` | Hy-VLA 的主要仿真评测基准 | <https://github.com/robotwin-Platform/RoboTwin> |
| `RoboDojo` | 双臂操作仿真评测（由 XPolicyLab 贡献支持） | <https://github.com/robodojo-benchmark/RoboDojo> |
| `XPolicyLab` | RoboDojo 支持的贡献方 | <https://github.com/XPolicyLab/XPolicyLab> |
| `OpenPi` | Hy-VLA 致谢中提及的构建基础 | <https://github.com/Physical-Intelligence/openpi> |

### 6.5 本报告的本地证据文件

| 文件 | 内容 |
|---|---|
| `_research\readmes\HY-Embodied.md` | `HY-Embodied` README 全文（本地既有） |
| `_research\readmes\HY-Embodied-0.5-X.md` | `HY-Embodied-0.5-X` README 全文（本地既有） |
| `_research\readmes\Hunyuan3D-WorldClaw.md` | `Hunyuan3D-WorldClaw` README 全文（本地既有，40 行） |
| `_research\readmes\Hy-Embodied-0.5-VLA.md` | `Hy-Embodied-0.5-VLA` README 全文（本报告现取） |
| `_research\readmes\Hy-Embodied-RxBrain-1.0.md` | `Hy-Embodied-RxBrain-1.0` README 全文（本报告现取） |
| `_research\readmes\Hy-Embodied-RoboFusion.md` | `Hy-Embodied-RoboFusion` README 全文（本报告现取） |
| `_research\readmes\Hunyuan3D-Buffalo1.0.md` | `Hunyuan3D-Buffalo1.0` README 全文（本报告现取，49 行） |
| `_research\extra\HYEmbodied__LICENSE.txt` | `HY-Embodied` `master/LICENSE`（11,813 字节，Apache-2.0 全文 + Tencent 头） |
| `_research\extra\HY05X__LICENSE.txt` | `HY-Embodied-0.5-X` `master/LICENSE`（15,979 字节，Tencent HY Community License） |
| `_research\extra\HyVLA__LICENSE.txt` | `Hy-Embodied-0.5-VLA` `main/LICENSE`（11,624 字节） |
| `_research\extra\RxBrain__LICENSE.txt` | `Hy-Embodied-RxBrain-1.0` `main/LICENSE`（11,700 字节） |
| `_research\extra\HYEmbodied__vllm_README.md` | `Hy-Embodied-VLM-1.0/inference/vllm/README.md` 全文 |
| `_research\extra\05X__docs_architecture.md` 等 | `HY-Embodied-0.5-X` 的 `docs/architecture.md`、`docs/inference.md`、`docs/training.md` |
| `_research\extra\RxBrain__train_README.md` | `Hy-Embodied-RxBrain-1.0` 的 `train/README.md` 全文 |
| `_research\extra\hf_Hy-Embodied-VLM-1.0.md`、`hf_HY-Embodied-0.5-X.md`、`hf_Hy-Embodied-0.5-VLA-UMI.md`、`hf_Hy-Embodied-RxBrain-1.0.md` | 四份 Hugging Face 模型卡全文（报告中 `[model-card]` 标注的来源） |

---

## 附录 A：本报告的已知不确定性清单

按「引用时必须带上下文」的原则，集中列出全部存疑点，供后续核实：

| # | 事项 | 冲突内容 | 本报告处理 |
|---|---|---|---|
| A1 | `HY-Embodied-0.5-X` 的 License | README 结构树注释 `LICENSE # Apache-2.0` vs `master/LICENSE` 实读为 Tencent HY Community License（含地域限制）vs `[model-card]` 标 `license: other` | 三源并列，以 LICENSE 文件原文为准；标注为最高合规风险点 |
| A2 | `HY-Embodied-0.5-X` 的评测分数字 | README 正文无数字，全部在 `assets/*.png` 图中 | 只写 README 明示的「10 benchmark、7 个第一、AI2Thor 1,011 任务」，不猜图中数字 |
| A3 | `Hunyuan3D-WorldClaw` 与 `Hunyuan3D-Buffalo1.0` 的技术细节 | README 的 Method / Result 章节只有图片 | 全部标 `README 未说明`；未做图片 OCR 或论文代读 |
| A4 | `Hy-Embodied-0.5-VLA` 的数据规模 | README「250K+ episodes / 10K+ 小时」vs `[model-card]`「~1M episodes」 | 两个口径同时列出并标明来源 |
| A5 | `Hy-Embodied-0.5-VLA` 的 norm 脚本名 | GitHub README `scripts/compute_norm_umi.py` vs `[model-card]` `scripts/compute_norm_lance.py` | 以 GitHub README 为准，两者并列记录 |
| A6 | `Hy-Embodied-RxBrain-1.0` 的推理脚本名与路径 | GitHub README `inference/vqa.py` 等 vs `[model-card]` 仓库根 `vqa_inference.py` 等 | 两者并列记录；本地实测 GitHub README 版本 |
| A7 | `Hy-Embodied-RxBrain-1.0` 微调代码状态 | GitHub README 标 `[x]` 已完成（指向 `train/`）vs `[model-card]` 标 `[ ]` 未完成 | 以 GitHub 为准（`train/` 与 `train/README.md` 已实读存在） |
| A8 | `Hy-Embodied-RoboFusion` 的电机控制模式数 | 1.1 节与 3.2 节为 7 种 vs 6.2 节为 8 种 | 以 3.2 节模式表（0–6，共 7 种）为准，冲突原样记录 |
| A9 | `Hy-Embodied` 仓库的默认分支 | README 与徽章链接在 GitHub 上解析为 `master`，但按 `main` 抓取 raw 文件全部 404 | 全文统一标注默认分支为 `master` |
| A10 | 工具能力与 `ArtifactsBenchmark` 的归属 | `ArtifactsBenchmark` 与本次 7 仓库无关，但报告 01 的 Context 提及 "ArtifactsBenchmark-related" | 本报告不涉及，避免与报告 01 重复 |
