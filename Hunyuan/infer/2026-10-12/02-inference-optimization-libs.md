# 混元推理优化 & 高效注意力库 深度分析

> 调研范围：Tencent-Hunyuan 组织下**直接做推理加速 / 高效注意力 / 高效服务**的代码库。
> 数据来源：各仓库 README（已下载快照存于 `_research/readmes/`）。文中所有数字、命令、链接均来自 README 原文；README 未说明的一律标注 `README 未说明`。
> 生成时间：本次调研会话。仓库在持续更新，命令与版本请以仓库当前 README 为准。

## 目录

- [0. 一句话结论](#0-一句话结论)
- [1. 总览对比表](#1-总览对比表)
- [2. HiLS-Attention — 层次化稀疏注意力 + SGLang 长上下文服务](#2-hils-attention--层次化稀疏注意力--sglang-长上下文服务)
- [3. flex-block-attn — 面向 HunyuanVideo 的块稀疏注意力 CUDA 库](#3-flex-block-attn--面向-hunyuanvideo-的块稀疏注意力-cuda-库)
- [4. Prism — 2K 联合视频-音频生成的动态稀疏注意力](#4-prism--2k-联合视频-音频生成的动态稀疏注意力)
- [5. Rosetta-inference — 可组合原生多模态（推理/评测发布）](#5-rosetta-inference--可组合原生多模态推理评测发布)
- [6. AuK — 语音生成与编辑基座模型的推理部署](#6-auk--语音生成与编辑基座模型的推理部署)
- [7. 横向对比与技术脉络](#7-横向对比与技术脉络)
- [8. 可复用要点](#8-可复用要点)
- [9. 参考链接](#9-参考链接)

---

## 0. 一句话结论

| 库 | 一句话定位 |
|---|---|
| **HiLS-Attention** | 用「压缩 chunk key 估计 chunk 质量」把块稀疏注意力的 top-k 选择变成**可端到端学习**的模块，并配了 **SGLang HSA 后端**，是这批仓库里**唯一直接面向 LLM 在线服务**的推理优化工作。 |
| **flex-block-attn** | 基于 **ThunderKittens** 写的**块稀疏注意力 CUDA 算子库**，支持 STA / MoBA / SSTA（STA+MoBA 混合），已用于 HunyuanVideo 的**训练和推理**，要求 Hopper (SM90) + CUDA 12.8，head dim 固定 128。 |
| **Prism** | 面向 **2K 分辨率联合视频-音频生成**的**动态稀疏注意力训练/推理框架**，把 token 序列划成时空 macro-zone，按「视频通道方差 + 音视频交叉注意力范数」为每个 zone **自适应分配 block 形状**，README 报告 **2.5× 训练加速**且生成质量反超全注意力。 |
| **Rosetta-inference** | 不是推理加速库，而是**多模态预训练成果的推理/评测发布包**：15 个 checkpoint（3 架构 × 5 训练阶段）+ 全套 benchmark 脚本，核心贡献是 `Composable FFN` + `MAOP` 解决「模态扩展导致灾难性遗忘」。 |
| **AuK** | 1.5B **语音生成 / 编辑**基座模型的推理工程：统一的自然语言指令接口覆盖 13 类任务，且**推理后端生态最全** —— vLLM-Omni、SGLang-Omni、audio.cpp(GGUF)、MLX(Apple Silicon)、CUDA CPU-offload、ComfyUI。 |

---

## 1. 总览对比表

| 仓库 | 类别 | 核心机制 | 是否含推理算子 | 是否含服务后端 | 硬件要求 | 最近更新 | License |
|---|---|---|---|---|---|---|---|
| HiLS-Attention | 注意力算法 + 服务 | 压缩 chunk key 估计 chunk-mass surrogate；inter/intra-chunk softmax 分解 | ✅ `ops/` | ✅ SGLang-HiLS (`--attention-backend hsa`) | CUDA 12.8 (torch 2.8.0+cu128) | 2026-08-13 | ⚠️ **仓库无 LICENSE 文件** |
| flex-block-attn | CUDA 算子库 | STA / MoBA / SSTA 块稀疏；PyTorch-like mask 表达式 | ✅ 核心即算子 | ❌ | **Hopper SM90** + CUDA 12.8 | 2025-12-26 | **Apache-2.0**（`LICENSE.txt`，声明「licensed under Apache-2.0, except for third-party components」） |
| Prism | 生成模型训练/推理框架 | 时空 macro-zone + 各向异性动态 block 形状 + top-k/top-p 混合块选择 | ✅ Block Sparse Attention (BSA) | ❌（脚本式推理） | CUDA ≥ 12.4；推理 720p 需 1×80GB，1080p/2K 需 ≥4×80GB | 2026-10-06 | **MIT**（`LICENSE`，声明「Prism is licensed under MIT, except for the third-party components」） |
| Rosetta-inference | 模型发布 + 评测 | Unified Attention + Composable FFN + MAOP | ❌ | ❌ | Python 3.12+ / CUDA 12.8+；评测需 8 GPU | 2026-07-02 | Apache 2.0 |
| AuK | 语音模型推理 | 统一指令接口 + 扩散 Transformer + 4-step 蒸馏 (Flash) | ❌（用现成后端） | ✅ vLLM-Omni / SGLang-Omni / audio.cpp / MLX | NVIDIA A800-SXM4-80GB 实测；bf16 | 2026-10-10 | **MIT** |

---

## 2. HiLS-Attention — 层次化稀疏注意力 + SGLang 长上下文服务

- 仓库：<https://github.com/Tencent-Hunyuan/HiLS-Attention>
- 论文：*Hierarchical Sparse Attention Done Right: Toward Infinite Context Modeling*，[arXiv:2607.02980](https://arxiv.org/abs/2607.02980)
- 模型：<https://huggingface.co/tencent/HiLS-Attention-7B>
- 作者（BibTeX）：Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui Han, Yushi Bai, Kewei Tu, Haitao Mi, Leo Liang

### 2.1 要解决的问题

朴素块稀疏注意力要选 top-k chunk，必须**先算出所有 chunk 的精确质量（chunk mass）**，而这需要完整的 QK 计算 —— 稀疏化省下的算力又被选择过程吃掉了。HiLS-Attention 的做法是：

1. 用**压缩后的 chunk key** 估计一个 **chunk-mass surrogate**（代理指标），避免全量 QK；
2. 把注意力**因式分解为 inter-chunk 与 intra-chunk 两级 softmax**；
3. 于是 chunk 选择这一步可以**在 next-token prediction loss 下端到端学习**（native sparse training），而不是训练完再套一个稀疏 kernel。

### 2.2 效果（README 原文）

仅 **50B continued-training tokens** 后：

- 继承全注意力的能力；
- **超长上下文外推能力强**：超过 YaRN 扩展 4× 的长度仍有效；
- **推理更快**（README 的 Figure (b)）；
- 短/中上下文任务性能可比，覆盖原始训练长度与 YaRN 外推范围。

> README 给出的是性能曲线图（`assets/performances.png`），**未给出具体 tokens/s 或倍数数字**，建议直接看论文。

### 2.3 工程结构（README 提到的真实路径）

```
HiLS-Attention/
├── assets/            # 架构图与性能图
├── configs/           # hils_attention/ 与 olmo3_7B/ 等配置
│   ├── hils_attention/config_hils_attn_8KA2K_HoPE_345M_prop3p1_qcal_r64.json
│   └── olmo3_vocab/   # Olmo3 tokenizer
├── data/              # tokenized 数据（含 mbpp_plus/）
├── eval/              # eval_ppl.py, eval_ruler.py, configs/
├── models/
├── ops/               # 稀疏注意力算子
├── preprocess/
├── scripts/
│   ├── pretrain/345M_exp_dist/pretrain_hils_attn_8KA2K_HoPE_345M_prop3p1_qcal_r64.sh
│   ├── cpt/cpt_olmo3_8KA2K_HoPE_qcal.sh
│   ├── ckpt_transfer/dcp_hf_transfer.sh
│   ├── convert_hils_checkpoint.py
│   ├── run_hsa_infer.py
│   └── eval/          # LongBench / OpenCompass / RULER 批量脚本
├── tasks/
├── unittests/
├── utils/
├── train.sh / train_dist.sh
├── pyproject.toml / requirements.txt / uv.lock
```

### 2.4 环境安装

两种方式，**uv 为推荐**：

```bash
# uv（推荐）
git clone https://github.com/Tencent-Hunyuan/HiLS-Attention.git
cd HiLS-Attention
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# 或 pip
conda create -n hils python=3.11 -y
conda activate hils
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 2.5 训练

```bash
# 从零训练
export CORPUS_PATH=/path/to/tokenized/data
export OUTPUT_DIR=outputs/checkpoints/hils_attn_8KA2K_HoPE_345M_prop3p1_qcal_r64
bash scripts/pretrain/345M_exp_dist/pretrain_hils_attn_8KA2K_HoPE_345M_prop3p1_qcal_r64.sh

# 继续预训练
export MODEL_PATH=/path/to/base/hf_ckpt
export CORPUS_PATH=/path/to/tokenized/data
export OUTPUT_DIR=outputs/checkpoints/olmo3_8KA2K_HoPE_qcal
bash scripts/cpt/cpt_olmo3_8KA2K_HoPE_qcal.sh

# landmark-token tuning（MODEL_PATH 指向 base checkpoint 目录）
bash scripts/cpt/cpt_olmo3_8KA2K_lmk_token_tuning.sh
```

### 2.6 评测

训练产物是 **DCP 分布式 checkpoint**，PPL/RULER 可直接读 DCP；只有走 HuggingFace 生成/评测时才需要转换：

```bash
DCP_PATH=/path/to/checkpoints/global_step_xxx bash scripts/ckpt_transfer/dcp_hf_transfer.sh
# 转换结果：/path/to/checkpoints/global_step_xxx/hf_ckpt
```

| 评测 | 命令 / 关键参数 |
|---|---|
| PPL | `python eval/eval_ppl.py --config_path ... --checkpoint_path ... --use_dcp_checkpoint --data_path ... --max_seq_len 8192 --last_k_tokens 512` |
| RULER | `python eval/eval_ruler.py --config_path ... --corpus_path ... --max_seq_len 8192 --task_id 0`（`0: S-N, 1: MK-MQ, 2: VT`） |
| LongBench v1 | `bash scripts/eval/eval_olmo3_longbench_v1.sh`，默认 `MAX_LENGTH=65536`，全部 21 个任务，`GPU_IDS="0 1 2 3 4 5 6 7"` |
| OpenCompass 11 项 | `bash scripts/eval/eval_olmo3_opencompass.sh`（MMLU/GPQA/HellaSwag/ARC-c/BoolQ/RACE 用 few-shot PPL；GSM8K/CMath 用 CoT 生成；HumanEval+/MBPP+/CRUXEval-O 用代码生成，HumanEval+/MBPP+ 事后用 `evalplus` 重新打分） |
| PPL + RULER 批量 | `bash scripts/eval/eval_olmo3_ruler_ppl.sh`，可用 `EVAL_MODE=ppl` / `EVAL_MODE=ruler` 拆开 |

PPL 长度覆盖 `64 … 256K`（HiLS 模型会跳过 ≤ `chunk_size` 的长度），RULER 长度 `8K, 16K, 32K, 128K`，`PPL_MAX_SAMPLES=100` / `RULER_MAX_SAMPLES=50`。

### 2.7 ⭐ 推理服务（本仓库对工程最直接的价值）

HiLS-Attention 有**原生 SGLang 后端**，把层次化稀疏注意力实现为**一等公民 attention backend**，因此可以直接用标准 SGLang server + OpenAI 兼容 API 提供服务，并随序列长度增长获得相对 dense attention 的长上下文加速。

后端代码**不在本仓库**，而在 fork：<https://github.com/alexzms/SGLang-HiLS>（HSA README 在 `python/sglang/srt/layers/attention/hsa/README.md`）。

```bash
# 1. 安装带 HSA 后端的 SGLang
git clone https://github.com/alexzms/SGLang-HiLS.git
cd SGLang-HiLS && pip install -e "python[all]"
# HSA 选择 kernel 还需要 tilelang（见该 fork 的 HSA README）

# 2. 下载 checkpoint
huggingface-cli download tencent/HiLS-Attention-7B --local-dir ./HiLS-Attention-7B

# 3. 转换 checkpoint（只翻译 config.json；权重用软链接）
python scripts/convert_hils_checkpoint.py --src ./HiLS-Attention-7B --dst ./HiLS-Attention-7B-sglang

# 4a. 起服务（OpenAI 兼容 API，http://localhost:30000）
python -m sglang.launch_server --model-path ./HiLS-Attention-7B-sglang \
    --attention-backend hsa --page-size 64 --trust-remote-code

# 4b. 或离线批量跑
python scripts/run_hsa_infer.py --model-path ./HiLS-Attention-7B-sglang \
    "The capital of France is" "Water is made of hydrogen and"
```

**关键约束（务必注意）**：

> `--page-size` **必须**等于模型的 `chunk_size`（即 64）。后端会**自动配置**所需的 HSA 服务约束（单序列 prefill；关闭 overlap scheduler），无需额外 flag。

### 2.8 亮点 / 局限

**亮点**
- 把「稀疏选择」变成可学习模块，训练即稀疏，避免训推不一致；
- 提供**开箱即用的 SGLang 服务路径**（`--attention-backend hsa`），不是只发论文代码；
- 评测体系完整（PPL / RULER / LongBench / OpenCompass 11 项），可直接复现长上下文结论。

**局限**
- 服务后端在**个人 fork**（`alexzms/SGLang-HiLS`）而非官方 SGLang 主线，跟进上游版本有维护风险；
- `--page-size` 必须等于 `chunk_size`，且 prefill 只支持单序列、overlap scheduler 必须关闭 —— 对高并发在线服务是**吞吐约束**；
- README 未标注 License，**且仓库根目录确实不存在 LICENSE 文件**（已核实），商用前必须联系作者确认；
- 性能收益只给了曲线图，没有绝对数字。

---

## 3. flex-block-attn — 面向 HunyuanVideo 的块稀疏注意力 CUDA 库

- 仓库：<https://github.com/Tencent-Hunyuan/flex-block-attn>
- 论文：*flex-block-attn: an efficient block sparse attention computation library*
- 作者：Yuanbo Peng\*, Penghao Zhao\*, Jiangfeng Xiong, Fang Yang, Songtao Liu, Jianbing Wu, Zhao Zhong, Key, Linus, Peng Chen, Jie Jiang

### 3.1 定位

专为 **Hunyuan Video** 设计的高效块稀疏注意力**计算库**，支持多种稀疏策略：

| 策略 | 说明 |
|---|---|
| **STA** | Sliding Tile Attention（滑动 tile 注意力） |
| **MOBA** | MoBA（MoonshotAI 提出） |
| **SSTA** | Selective and Sliding Tile Attention —— **STA 与 MOBA 的混合**，README 称其为 novel 机制 |

**训练和推理都支持**。底层基于 **ThunderKittens** 的 attention demo 实现，为 **Hopper 架构**优化，提供 **PyTorch-like mask 表达式**，兼顾易用性与稀疏 mask 生成效率。

### 3.2 硬性约束（README「Notes」原文要点）

- **head dim 必须是 128**；
- `q` tile_size 可为 **16 的任意倍数**；`k/v` tile_size 可为 **64 的任意倍数**，**推荐 384**（该尺寸做了额外优化）；
- q 与 kv 的序列长度必须能被各自的 tile size 整除；
- `attention_mask` **只支持 block 级 mask**；`block_mask` 支持两种形状：`[seq_len, seq_len]` 或 `[batch, head_num, seq_len, seq_len]`；
- 选中的 block 内部执行**完整注意力**。

### 3.3 环境与安装

- Hopper (SM90) GPU，或其他支持 SM90 PTX ISA 的架构；
- **Python ≥ 3.8**；
- **CUDA 12.8**；
- 安装方法（注意需要拉 submodule）：

```bash
git submodule update --init --recursive
python setup.py install
```

### 3.4 用法

```python
from flex_block_attn import flex_block_attn_func
from benchmark.utils.utils import create_sparse_mask

# 以 (block_size*2) x (block_size*2) 为例，block size 可取 64/128/192...
selected_blocks = [[0, 1], [1, 0]]
sparse_mask = create_sparse_mask(q, block_size, selected_blocks)

# block_size=64 时，torch mask 形状为 [128, 128]
output = flex_block_attn_func(query, key, value, q_block_size, k_block_size, block_mask)
```

README 特别对比了本库的**块级 mask 表示**与**原始 torch 逐元素 mask** 的差异：本库用 `[[0,1],[1,0]]` 这种紧凑块表示，而原始 torch mask 需要展开成 `[128,128]` 的重复行列块，显存和生成开销大得多。

### 3.5 性能与 benchmark

benchmark 代码在 `benchmark/`，覆盖三类：**full attention / sparse static attention / sparse dynamic attention**，测量维度为 **mask 创建时间、forward/backward 执行时间、GPU 显存占用**。结果按硬件分目录存放：

- 全注意力：`benchmark/full/results/`
- 静态稀疏：`benchmark/static/swa/results/`
- 动态稀疏：`benchmark/dynamic/random/results/`

测试硬件：**H800** 与 **H20**。

动态稀疏测试参数（README 原文）：

- 序列长度：`11520, 19200, 30720, 38400, 46080, 53760, 61440, 69120`
- **Block size 384**
- **Sparse rate 0.6**

结论：**mask 创建 + forward + backward 合计耗时优于主流稀疏注意力库**（README 未给出具体倍数，需看 `benchmark/` 结果与图 `assets/h800_dynamic_time_merge.png`、`assets/h20_dynamic_time_merge.png`）。

### 3.6 致谢中透露的技术脉络

- **ThunderKittens** —— 提供核心计算引擎（本项目在其之上加逻辑层）；
- **STA (Sliding Tile Attention)** —— 来自 `hao-ai-lab/FastVideo`；
- **MoBA** —— 来自 `MoonshotAI/MoBA`；两者是 HunyuanVideo 训练中稀疏注意力的灵感来源；
- 对比/借鉴对象：`flex attention`(attention-gym)、`flash-attention`、`MagiAttention`(SandAI)、`SpargeAttn`(thu-ml)、`Triton`。

### 3.7 亮点 / 局限

**亮点**
- 稀疏策略**可插拔**（STA / MoBA / SSTA），不是单一 kernel；
- **块级 mask + PyTorch-like 表达式**，把稀疏 mask 的表达成本降到接近 PyTorch 语义；
- 训推同源（训练和推理用同一套稀疏注意力），benchmark 横跨 H800/H20 两种卡，参考价值高；
- **Apache-2.0**（`LICENSE.txt` 明确「flex block attn is licensed under Apache-2.0, except for the third-party components」，第三方组件沿用原许可）；
- 有 submodule（`tk` = ThunderKittens），依赖关系清晰。

**局限**
- **硬件门槛高**：仅 Hopper SM90，其他架构只能靠 PTX ISA 兼容性「碰运气」；
- **CUDA 12.8 + 编译安装**（`python setup.py install`），部署成本不低；
- **head dim 锁死 128**，非常规 head dim 的模型无法直接用；
- SSTA 的完整代码 README 说「near future 开源」，属**部分开源**状态（截至 2025-11-19 更新）。

---

## 4. Prism — 2K 联合视频-音频生成的动态稀疏注意力

- 仓库：<https://github.com/Tencent-Hunyuan/Prism>
- 项目页：<https://francis-rings.github.io/Prism>
- 论文：[arXiv:2610.05416](https://arxiv.org/abs/2610.05416)
- 权重：<https://huggingface.co/FrancisRing/Prism>
- 单位：复旦大学 / 腾讯混元 / 浙江大学

### 4.1 动机与核心思路

高分辨率原生训练联合视频-音频模型能学到更丰富的视觉细节与更锐利的运动，但：

- **全注意力是二次成本**；
- 分辨率升高后，注意力被**越来越冗余的 token 稀释**，信息量大的内容学习信号被削弱，还会破坏预训练先验；
- 已有稀疏注意力方法要么只做 **training-free 加速**，要么**忽略了联合视频-音频数据的独特结构** —— 跨模态交互天然集中在**发声区域**附近。

**Prism 的做法**（动态稀疏注意力框架）：

1. 把 token 序列组织成**时空 macro-zone**，让注意力结构适配局部内容；
2. 对每个 zone，用两个信号估计局部信息结构：
   - **视频特征沿通道的方差**（视觉内容变化的方向性）；
   - **audio-to-video 交叉注意力的特征范数**（音频对每个视觉区域的影响强度）；
3. 据此为每个 zone **动态分配定制的 block 形状** —— 在视觉内容变化剧烈的轴、以及音视频耦合强的轴上做**更细的分割**（anisotropic dynamic block assignment，即 `enable_ivpq_dynamic_block`）；这样每个 block 内的 token 保持语义一致，block 级特征能同时表达视觉内容与联合音视频交互模式；
4. 再采用**混合块选择策略**（top-k 结合 top-p CDF 阈值）动态决定**每个 query 的稀疏度**。

**结果**：README 称相比全注意力实现 **2.5× 训练加速**，且**生成质量反超**全注意力。

### 4.2 资源开放状态

- [x] Prism-preview-alpha (stable)
- [x] Prism-preview-beta (motion)
- [x] 数据预处理代码（latent 提取与解码）
- [x] 训练代码
- [x] 全量微调代码
- [x] 推理代码
- [ ] Prism-pro（未发布）

预览模型支持 **720p / 1080p / 2K** 的原生联合视频-音频生成**与训练**。

### 4.3 环境安装

前置：**Python ≥ 3.10、CUDA ≥ 12.4、系统级 `ffmpeg`**。

```bash
# 1 — PyTorch (CUDA 12.4)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu124
# 2 — 依赖
pip install -r requirements.txt
# 3 — FlashAttention（需 CUDA）
pip install flash-attn --no-build-isolation
# 4 — ffmpeg（系统级）
sudo apt-get install ffmpeg
# 5 — 校验
python -c "import torch, diffusers, transformers, flash_attn, triton; \
from audiotools import AudioSignal; from decord import VideoReader; print('All imports OK')"
```

多节点集群可用 `pdsh` 一次性并行装包（README 给出 `pdsh -f 256 -w $node_ip "..."` 的写法，涉及 `diffusers==0.33.0`、`descript-audiotools`、`decord`、`flash-attn` 等）。国内网络可 `export HF_ENDPOINT=https://hf-mirror.com`。

### 4.4 权重下载与目录结构

```bash
pip install "huggingface_hub[cli]"
cd Prism && mkdir checkpoints
huggingface-cli download FrancisRing/Prism --local-dir ./checkpoints
```

```
Prism/
├── configs/  ├── hymm/  ├── scripts/  ├── tools/
├── checkpoints/
│   ├── pretrained_models/MOVA-360p/        # 基座 MOVA 预训练模型
│   ├── preview_alpha/diffusion_pytorch_model.safetensors
│   └── preview_beta/diffusion_pytorch_model.safetensors
├── mova_infer.sh  ├── mova_infer_fsdp.sh  ├── requirements.txt
```

### 4.5 数据准备（CSV 单文件 manifest）

训练只吃一个 CSV（每行一个 clip）。**必需列**：`latent_path`（缓存的 WAN VAE latent `.npy`，形状 `[C,T,H,W]`）、`video_path`、`audio_path`、`caption`、`num_frames`†、`height`†、`width`†、`fps`†。**可选列**：`video_id`、`audio_caption`、`latent_frames/latent_height/latent_width`、`best_frame_index`、`ref_frame_index`。

> † 当 CSV 里存在 `latent_frames/latent_height/latent_width` 时，分桶优先用它们，`num_frames/height/width` 变为可选；`fps` 缺省取 `dataloader_config.video_fps`（24）。

`caption` 建议用结构化 JSON（`style_features` / `content_summary` / `background_audio` / `shots[]`），由 `video_caption_sample_ratio` 控制哪些字段拼进 prompt。

**Step 1 — 提取 WAN VAE latent**（离线，每视频一次）：

```bash
python hymm/wan_vae_latent_extraction.py \
    --input_csv /path/i2va_raw.csv --output_dir /path/latents \
    --model_path /path/checkpoints/pretrained_models/MOVA-360p/video_vae \
    --sample_n_frames 289 --fps 24
```

**Step 2 — 多分辨率/多时长分桶**（时间桶范围**必须与训练配置一致**）：

| config | `latent-resolution` | `temporal-max-length` |
|---|---|---|
| 720p | `720p` | `289` |
| 1080p | `1080p` | `289` |
| 2k | `2k` | **`265`** |

```bash
python tools/build_dataset_csv.py --input /path/i2va_raw.csv --output /path/i2va_720p.csv \
    --latent-resolution 720p --temporal-min-length 49 --temporal-max-length 289 \
    --temporal-interval 12 --scan-latents --workers 32
# 1080p 同上，改 --latent-resolution 1080p；2k 改 --temporal-max-length 265
```

`--scan-latents` 从 `.npy` header 填 `latent_frames/height/width`，并打印每个桶的直方图，便于启动前预判 batch 填充情况。

### 4.6 推理

两个脚本：`prism_infer.sh`（单卡或 DeepSpeed 多卡，**推荐 720p**）、`prism_infer_fsdp.sh`（**FSDP 参数分片多卡，推荐 1080p / 2K**）。

```bash
bash prism_infer.sh          # 单卡 / DeepSpeed 多卡（720p）
bash prism_infer_fsdp.sh     # FSDP 多卡（1080p / 2K）
```

核心参数（README 表格摘录）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt` | — | MOVA 基座模型路径（如 `checkpoints/pretrained_models/MOVA-360p`） |
| `--resume_ckpt` | — | Prism checkpoint 路径 |
| `--config` | `configs/train/t2va_config/mova_infer.yaml` | 推理配置（扩散步数、默认分辨率、seed） |
| `--height` / `--width` | `480` / `848` | 输出分辨率。推荐 720p `720×1280`、1080p `1072×1920`、2K `1440×2560` |
| `--num_frames` | `205` | 帧数，必须满足 `(n − 1) % 4 == 0`，否则自动对齐 |
| `--prompt` / `--audio_prompt` | — | 文本提示；支持 `<music>`、`<sfx>`、`<speech>` 结构化标签 |
| `--ref_path` | — | i2v 参考图 |
| `--sp_size` | `4` 或 `8` | **序列并行度**（多少张 GPU 共享一个样本的 token 序列） |
| `--offload` | `cpu` | 把冻结的 encoder 卸载到 CPU 省显存 |
| `--seed` | `42` | 随机种子 |
| `VISUAL_SHIFT` / `AUDIO_SHIFT` | `9.0` / `7.0` | 噪声调度 shift。推荐 720p `7.0`、1080p `9.0–13.0`、2K `13.0–17.0` |
| `CFG_SCALE` | `5.0` | 分类器无关引导强度（`1.0` = 不引导） |
| `ENABLE_TILING` | `false` | VAE 解码空间分块，避免高分辨率 OOM（配合 `TILE_SAMPLE_MIN_SIZE` / `TILE_SAMPLE_STRIDE`） |
| `ENABLE_BSA` / `BSA_SPARSITY` | `true` / `0.93`\|`0.85`\|`0.75` | 启用 Prism Block Sparse Attention 及 top-k 稀疏率 |
| `ENABLE_IVPQ_DYNAMIC_BLOCK` | `true` | 启用**推理阶段的动态 block 形状分配** |

> README 提示：`mova_infer.sh` 用 `--deepspeed` 启动，适合**单张 80GB 卡跑 720p**（配 `--offload cpu`）；`mova_infer_fsdp.sh` 用 `torchrun` + FSDP `FULL_SHARD`，适合 **4 卡以上跑 1080p / 2K**，因为参数分片而非每卡复制全模型。仓库还提供了 `assets/ti2va_cases` 现成用例。

### 4.7 训练（高分辨率原生联合训练）

两级配置体系：

- **顶层 YAML** `configs/train/wan_ti2va.yaml` —— 优化器、并行、checkpoint、**Prism 稀疏注意力行为**；
- **每分辨率 inject-config YAML** `configs/train/t2va_config/wan_15B_ti2va_*_init_top_k_p.yaml` —— 预训练模型路径、loss 权重、噪声调度、**全部 dataloader 设置**（`dataloader_config.video_csv_file` 放 Step 2 生成的 CSV 路径，支持列表合并多数据集）。

```bash
# 720p
bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_720p_init_top_k_p"
# 1080p
bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_1080p_init_top_k_p"
# 2K
bash scripts/pretrain/pdsh_train.sh "scripts/pretrain/train_wan_ti2va.sh configs/train/wan_ti2va.yaml wan_15B_ti2va_2k_init_top_k_p"
```

**顶层配置关键参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `sp_size` | 8（1080p/2K）/ 4（720p） | 序列并行度 |
| `lr` / `lr_scheduler` / `lr_warmup_steps` | `1e-5` / `constant_with_warmup` / `200` | 学习率策略 |
| `optimizer` / `weight_decay` / `max_grad_norm` | `adamw` / `0.01` / `1.0` | 优化器与梯度裁剪 |
| `gradient_accumulation_steps` | `1` | 梯度累积 |
| `fsdp_strategy` | `full` | `full`（参数+梯度+优化器状态全分片）/ `hybrid`(HSDP) / `none` |
| `checkpointing_steps` | `50` | 存 checkpoint 间隔 |
| `train_full_model` | `true` | `false` = 只训注意力 |
| `offload_frozen_encoders` | `true` | 每步 forward 后把冻结的 text encoder / audio VAE / video VAE 挪回 CPU |
| `empty_cache_interval` | `1` | 每 N 步 `torch.cuda.empty_cache()` |
| `loss_spike_threshold` / `grad_norm_spike_threshold` | `1.5` / `1e6` | 超阈值**跳过该步优化器更新**，防脏 batch 打崩权重 |
| `enable_bsa` | `true` | 启用 Prism Block Sparse Attention（视频自注意力） |
| `bsa_sparsity` | `0.75/0.80/0.85/0.93` | top-k 稀疏率（`0.75` 表示每 query block 只 attend 25% key block） |
| `bsa_chunk_3d_shape_q` / `_k` | `4 4 4` | 时空 macro-zone 的 `(T,H,W)` 形状 |
| `bsa_cdf_threshold` | `0.20` | 混合块选择的 top-p CDF 阈值，与 top-k 结合实现**自适应 per-query 稀疏度** |
| `enable_ivpq_dynamic_block` | `true` | **Prism 核心**：各向异性动态 block 形状分配 |

**inject config 关键参数**：`video_loss_weight`/`audio_loss_weight` = `0.75`/`0.25`；`visual_shift`/`audio_shift` = `9.0`/`5.0`；`video_size` = `[960,960]`/`[1440,1440]`/`[1920,1920]`；`sample_n_frames` = `289`（2K 为 `265`）；`video_micro_batch_size` = `1`；`video_uncond_p`/`audio_uncond_p` = `0.1`/`0.2`；时间桶 `49 → 289`（2K `→265`）步长 `12`。

**分桶机制**（README 说明）：数据集按 bucket key 分组，`DistributedBucketBatchSampler` 让每个 mini-batch **只从一个桶取样本**，保证同 batch 内张量形状完全一致，避免 DP rank 间形状不匹配。空间桶 = 枚举所有 patch 数等于 `(base_size/patch_size)²` 的 `(H,W)` 裁剪，样本匹配**宽高比最接近**的桶；时间桶 = 从 `temporal_min_length` 到 `temporal_max_length`、步长 `temporal_interval` 的 `4k+1` 帧数序列，样本分到**能容纳它的最大**时间桶。

### 4.8 显存需求（README「VRAM Requirement」原文）

| 场景 | 最低要求 |
|---|---|
| 720p 原生高分辨率**训练** | **≥ 32 × NVIDIA 80GB**（A100 或 H100） |
| 1080p / 2K **训练** | 各需 **≥ 64 × NVIDIA 80GB** |
| 720p **推理** | **单张 80GB** |
| 1080p / 2K **推理** | **≥ 4 × NVIDIA 80GB** |

### 4.9 亮点 / 局限

**亮点**
- 稀疏模式**由内容驱动**（方差 + 跨模态注意力范数），而非固定 sliding window，这是它与 STA/MoBA 类方法的本质差异；
- 稀疏度是**自适应的**（top-k + top-p CDF 混合），不同 query 可以有不同稀疏度；
- **训推一致**：`enable_bsa` 与 `enable_ivpq_dynamic_block` 在训练和推理配置里同时存在；
- 工程细节扎实：桶采样保证同 batch 同形状、loss/grad spike 跳过、encoder CPU offload、VAE 分块解码防 OOM、`pdsh` 多机部署；
- 给出了**明确的显存门槛**，选型时不用猜；
- **MIT License**（`LICENSE` 声明「Prism is licensed under MIT, except for the third-party components listed below」，权重亦在 MIT 范围内），商用友好。

**局限**
- **资源门槛极高**：2K 训练要 64×80GB，个人/小团队基本无法复现训练，只能做推理；
- 推理脚本是**改 shell 脚本 + 环境变量**的形态，没有 server/API 封装，不适合直接做在线服务；
- 依赖较旧（`torch==2.7.1`、`diffusers==0.33.0`），且 `diffusers` 版本需降级安装；
- README 的性能结论（2.5× 训练加速、质量反超）**只有文字，无表格数字**；
- 不是通用 LLM 推理加速方案，仅限视频-音频联合生成这个模态。

---

## 5. Rosetta-inference — 可组合原生多模态（推理/评测发布）

- 仓库：<https://github.com/Tencent-Hunyuan/Rosetta-inference>
- 项目页：<https://rosetta-lmm.github.io/>
- 论文：*Rosetta: Composable Native Multimodal Pretraining*，[arXiv:2607.00293](https://arxiv.org/abs/2607.00293)
- 权重：<https://huggingface.co/tencent/Rosetta-inference>
- 单位：HKUST / 腾讯混元 | **License: Apache 2.0**

### 5.1 定位说明

**这不是推理加速库**，而是多模态预训练成果的**推理与评测发布包**（仓库名带 `-inference` 是因为只放推理/评测，训练代码在 README 中被注释掉、指向 `TRAIN.md`）。它的价值在于：**15 个 checkpoint 的完整对照实验 + 全套复现脚本**。

### 5.2 要解决的问题：遗忘-协同困境

论文标题是 *Escaping the Forgetting-Synergy Dilemma in Native Multimodal Pretraining*（逃离原生多模态预训练中的遗忘-协同困境）。

README 的 Figure 1 说明：在可组合预训练各阶段（**LM → +MMU → +T2I**）上，标准 MoE 和结构隔离的 MoT 在**引入连续生成目标时会出现灾难性的路由坍塌（catastrophic routing collapse）**；而 Rosetta 在**所有阶段都保持稳定的语义锚点（semantic anchor）**。

### 5.3 三大机制（Rosetta FFN）

| 机制 | 作用 |
|---|---|
| **Unified Attention** | 全局共享 QKV 投影，保留密集的跨模态交互 |
| **Composable FFN** | 模态专属的**即插即用专家**（Text / ViT / VAE），由一个 **Global Shared Expert** 桥接，锚定基础知识 |
| **Conflict-Free Optimization (MAOP)** | **零显存开销**地「外科手术式」中和破坏性梯度 |

所有模型在 **Transfusion 框架**上、以 **Qwen3-0.6B-Base** 为语言骨干训练，数据与超参完全一致以保证公平对比。

### 5.4 Checkpoint 矩阵（3 架构 × 5 阶段 = 15 + 1）

**最终 3 个 checkpoint（复现 benchmark 用）**：

| Checkpoint | 能力 | 总/激活参数 |
|---|---|---|
| Rosetta-3.8B-A1B | LM + MMU + T2I | 3.8B / 0.97B |
| MoE-3.8B-A1B | LM + MMU + T2I | 3.8B / 0.97B |
| MoT-4.5B-A1B | LM + MMU + T2I | 4.5B / 0.97B |

**完整 5 阶段**：`-init`（Upcycling init, iter 0）→ `-stage1-lm`（35K）→ `-stage2-lm-mmu-warmup`（3K）→ `-stage2-lm-mmu`（20K）→ 最终（400K，LM+MMU+T2I）。三套架构的**配置完全一样**（`evaluation/configs/rosetta.yaml` / `moe.yaml` / `mot.yaml`），因此可以**同架构跨阶段**复用 config，这正好用来复现 Figure 1 的 MMLU 训练曲线：

```bash
for EXP in checkpoints/Rosetta-3.8B-A1B-init \
            checkpoints/Rosetta-3.8B-A1B-stage1-lm \
            checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu-warmup \
            checkpoints/Rosetta-3.8B-A1B-stage2-lm-mmu \
            checkpoints/Rosetta-3.8B-A1B; do
    EXP=$EXP CONFIG=evaluation/configs/rosetta.yaml bash scripts/eval/eval_mmlu.sh
done
```

### 5.5 环境与资产

```bash
conda create -n rosetta python=3.12 -y && conda activate rosetta
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# FlashAttention 必装，编译约 20 分钟
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=8 pip install flash-attn==2.8.1 --no-build-isolation --no-binary flash-attn --no-cache-dir
```

资产分两块：**共享资产**（所有模型共用，下载一次）与**模型 checkpoint**：

```bash
hf download tencent/Rosetta-inference public_assets.zip --local-dir . && unzip -o public_assets.zip && rm public_assets.zip
hf download tencent/Rosetta-inference --include "checkpoints/Rosetta-3.8B-A1B/**" --local-dir .
```

目录约定：

```
checkpoints/{Rosetta-3.8B-A1B,MoE-3.8B-A1B,MoT-4.5B-A1B}/hf_weights/
public_assets/
├── image_encoder/      # VAE（flux2-vae 等）
├── vision_encoder/     # ViT（Qwen3-VL-30B-A3B-Instruct）
├── pretrained_llm/     # tokenizer + Qwen3-0.6B-Base
├── generation_configs/
└── evaluation/         # ARC_C, MMLU, BBH, MBPP, MMMU, MMBench, POPE, AI2D, RealWorldQA, T2I-CompBench
```

路径可用环境变量覆盖：`ASSETS_BASE`（默认 `./public_assets`）、`EXP`（默认 `./checkpoints/Rosetta-3.8B-A1B`）。

### 5.6 评测矩阵（11 个脚本，**统一要求 8 GPU**）

| Benchmark | 任务类型 | 脚本 |
|---|---|---|
| ARC-Challenge | Language | `scripts/eval/eval_arc_c.sh` |
| MMLU | Language | `scripts/eval/eval_mmlu.sh` |
| BBH | Language | `scripts/eval/eval_bbh.sh` |
| MBPP | Code | `scripts/eval/eval_mbpp.sh` |
| MMMU | 多模态理解 | `scripts/eval/eval_mmmu.sh` |
| MMBench | 多模态理解 | `scripts/eval/eval_mmbench.sh` |
| POPE | 幻觉 | `scripts/eval/eval_pope.sh` |
| AI2D | 图表理解 | `scripts/eval/eval_ai2d.sh` |
| RealWorldQA | 真实场景 VQA | `scripts/eval/eval_realworldqa.sh` |
| T2I-CompBench | 图像生成 | `scripts/eval/eval_t2i_compbench.sh` |
| COCO (FID/CLIP/HPS) | 图像生成 | `scripts/eval/eval_coco.sh` |

统一切换架构的方式（`EXP` + `CONFIG` 环境变量）：

```bash
bash scripts/eval/eval_arc_c.sh                                                    # Rosetta（默认）
EXP=checkpoints/MoE-3.8B-A1B CONFIG=evaluation/configs/moe.yaml bash scripts/eval/eval_arc_c.sh
EXP=checkpoints/MoT-4.5B-A1B CONFIG=evaluation/configs/mot.yaml bash scripts/eval/eval_arc_c.sh
```

结果写在 `<EXP>/outputs/`。README 提醒：论文结果是 **LM+MMU+T2I 全阶段训练后**在相同数据/超参下评测的；用发布的 HF checkpoint 复现时**可能因数值精度差异有微小波动**。

### 5.7 亮点 / 局限

**亮点**
- **实验设计教科书级**：3 架构 × 5 阶段 × 相同数据超参，把「模态扩展导致遗忘」这个问题的因果链切得很干净；
- 累计发布 **15 个 checkpoint**，可以观察每个阶段的指标演化（而不只是最终分）；
- 评测覆盖**语言 / 多模态理解 / 幻觉 / 图表 / 代码 / 图像生成**六类，且图像生成同时给了 **T2I-CompBench 与 COCO(FID/CLIP/HPS)**；
- Apache 2.0，商用友好。

**局限**
- **评测需要 8 GPU**，门槛偏高，且 README 未说明单卡是否可行；
- 训练代码在本仓库中是**注释掉的状态**（指向未提供的 `TRAIN.md`），意味着**只能复现推理/评测，不能复现训练**；
- 骨干只有 **0.6B**（Qwen3-0.6B-Base），是**方法论验证**规模，不是可直接上线的模型；
- 没有 serving / 量化 / 吞吐相关内容，对推理优化本身无直接贡献。

---

## 6. AuK — 语音生成与编辑基座模型的推理部署

- 仓库：<https://github.com/Tencent-Hunyuan/AuK>
- 项目页：<https://auk-project.github.io/>
- 技术报告：[arXiv:2609.08936](https://arxiv.org/abs/2609.08936)
- 权重：<https://huggingface.co/tencent/AuK> · <https://huggingface.co/tencent/AuK-Flash>
- **License: MIT**

### 6.1 模型概览

**AuK 是 1.5B 的语音生成与编辑基座模型**，在数百万小时多样化音频上训练。通过**统一的自然语言指令接口**支持：零样本与指令式 TTS、内容编辑与声学编辑、副语言（paralinguistic）编辑、语音增强、音源分离。

| 变体 | 说明 |
|---|---|
| **AuK** | 高质量生成基座模型，NFE 与 CFG 可配置 |
| **AuK-Flash** | 蒸馏模型，**固定 4 步推理**、**CFG=0** |

> 注意：AuK 的 checkpoint 只包含 **diffusion transformer 与 layer-fusion 权重**；MLLM encoder 与 VAE 运行时从**独立文件**加载，因此加载时出现 `text_encoder.*` 缺失 key 属**预期行为**。

### 6.2 支持的任务（13 项，全部走同一指令接口）

| 类别 | 任务 |
|---|---|
| 语音生成 | Zero-shot TTS、Instruct TTS（仅凭声音描述，无需参考音频） |
| 内容编辑 | Speech Content Editing（改写「说了什么」：替换/插入/删除文本）、Lyric Editing（改写歌词，保留旋律与音色） |
| 声学编辑 | Pitch Editing（半音升降）、Speed Editing（语速，输出长度随 speed factor 缩放）、Volume Editing（dB 升降） |
| 副语言编辑 | Emotion（改情绪）、Timbre（改音色）、De-accent（去口音）、Nonverbal Editing（增删呼吸/笑声/咳嗽）、Whisper Conversion（正常语音↔耳语互转） |
| 增强与分离 | Speech Enhancement（降噪/去混响）、Speech Separation（按说话顺序保留一人）、Music Separation（提取人声/保留所有人声）、Target Speaker Extraction（按「说了什么」定位目标说话人） |

> 完整的指令模板 + 每任务 CLI/Python 示例在仓库的 `docs/COOKBOOK.md`。

### 6.3 ⭐ 推理后端生态（本仓库最突出的工程价值）

README 的 News 时间线显示 AuK 在开源后**一周内**接入了几乎所有主流语音/多模态推理栈：

| 日期 | 事件 | 链接 |
|---|---|---|
| 2026/09/09 | 🎉 开源 AuK，代码与权重公开 | — |
| 2026/09/09 | **SGLang-Omni Day-0 支持** AuK 与 AuK-Flash | <https://sgl-project.github.io/sglang-omni/cookbook/auk.html> |
| 2026/09/11 | **vLLM-Omni 支持**：单张 H100 上以两阶段 pipeline 服务 AuK 与 AuK-Flash | <https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/AuK-H100.md> |
| 2026/09/13 | **MLX 推理支持 Apple Silicon**（`feat/mlx-apple-silicon` 分支）+ **CUDA CPU offload** | — |
| 2026/09/13 | 🏆 被 **ICASSP 2027 Audio Editing Challenge** 选为 Single Model Track 的**端到端 baseline** | <https://audio-editing-challenge.github.io/> |
| 2026/09/16 | **encoder 显存降低约 7.5 GiB**，消费级 GPU 可本地推理（PR #19） | — |
| 2026/09/20 | **audio.cpp 支持**：以 `--family auk` 通过 **GGUF 离线推理**运行 AuK / AuK-Flash | [audio.cpp doc](https://github.com/0xShug0/audio.cpp/blob/main/docs/community_models/auk.md) |

**推理路径总览**：PyTorch 原生 CLI (`auk-infer`) / Gradio (`auk-gradio`) / Python API (`AukInfer`) / ComfyUI 节点 / **vLLM-Omni** / **SGLang-Omni** / **GGUF (audio.cpp)** / **MLX** / **Xinference**。

### 6.4 安装与权重

```bash
git clone https://github.com/Tencent-Hunyuan/AuK && cd AuK

# uv（推荐）
uv venv --python 3.10 && source .venv/bin/activate
uv pip install -e .                 # 仅核心推理 + CLI
uv pip install -e ".[gradio]"       # + Gradio + Prompt Enhancer + ASR
uv pip install -e ".[comfyui]"      # + ComfyUI 节点
uv pip install -e ".[train]"        # + 微调
uv pip install -e ".[gradio,train]" # 全量

# 或 conda
conda create -n auk python=3.10 -y && conda activate auk
pip install -e ".[gradio]"
```

权重（三个都要下，含 MLLM encoder）：

```bash
hf download tencent/AuK       --local-dir ./ckpts/AuK
hf download tencent/AuK-Flash --local-dir ./ckpts/AuK-Flash
hf download Qwen/Qwen2.5-Omni-3B --local-dir ./ckpts/Qwen2.5-Omni-3B
```

> ⚠️ 当前实现使用 **Qwen2.5-Omni-3B**；**Qwen3-Omni 尚不被 `--qwen_path` 支持**。

### 6.5 ⭐ 显存实测（README 表格，`torch.cuda.max_memory_allocated`）

测试卡：**单张 NVIDIA A800-SXM4-80GB，bf16 推理**。`--cpu_offload` **默认关闭**，两个变体都支持。

| 模型 | 输入 | CPU offload 关 | CPU offload 开 | 节省 |
|---|---|---:|---:|---:|
| AuK | 纯文本，1.5s 输出 | 24.78 GiB | 16.75 GiB | **8.03 GiB (32.4%)** |
| AuK | 5s 参考音频 | 25.00 GiB | 16.98 GiB | **8.02 GiB (32.1%)** |
| AuK-Flash | 纯文本，1.5s 输出 | 24.77 GiB | 16.75 GiB | **8.02 GiB (32.4%)** |
| AuK-Flash | 5s 参考音频 | 24.97 GiB | 16.98 GiB | **7.99 GiB (32.0%)** |

> 换算：开启 CPU offload 后峰值约 **17 GiB**，理论上可落到 **24GB 消费级卡**（如 RTX 4090）区间 —— 这与 News 里「consumer GPUs 可本地推理」相呼应。实际占用依赖输入长度、dtype、硬件与软件版本。

### 6.6 命令行推理

```bash
# 内容编辑
auk-infer --audio assets/demo-input-audio/content-edit/content.wav \
    --instruction "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'." \
    --output out_content_edit.wav --gen_seconds 7.0

# 降噪/增强（中文指令同样可用）
auk-infer --audio assets/demo-input-audio/vocal-extraction/vocal-1-input.wav \
    --instruction "请将这段音频恢复成纯净人声版本：保留原本所有说话人，并去除其中的噪声和混响，输出等长的纯净语音。" \
    --output out_denoise.wav

# 零样本 TTS
auk-infer --audio assets/demo-input-audio/zero-shot-tts/ref.wav \
    --instruction "Say the following with the same voice: 'Ladies and gentlemen, ...'" \
    --output out_tts.wav --gen_seconds 6.0
```

**切到 AuK-Flash**：`--ckpt ckpts/AuK-Flash/auk_flash.safetensors`（4 固定时间步，CFG=0）。
**降显存**：给 `auk-infer` / `auk-gradio` 加 `--cpu_offload`，或 `AukInfer(..., cpu_offload=True)`，或 ComfyUI 的 **AuK Model Loader** 里打开。

**Prompt Enhancer**（推荐用于自由文本需求）：会准备模型指令、目标时长、必要的音频预处理，然后打印一条可直接运行的 `auk-infer` 命令：

```bash
python src/auk/infer/pe.py \
  --audio assets/demo-input-audio/whisper/wh-w2n-zh-input.wav \
  --instruction "Convert this whisper into normal speech while preserving the speaker and content." \
  --asr auto
```

PE 需要 **OpenAI 兼容 LLM**（README 示例用腾讯云 TokenHub，`LLM_MODEL_NAME="hy3"`，`LLM_BASE_URL="https://tokenhub.tencentmaas.com/v1"`）；ASR 可选腾讯云录制文件识别，未配置时**自动下载并懒加载 `iic/SenseVoiceSmall` 作为 CPU 兜底**。

### 6.7 服务 / Gradio 双模型部署

```bash
auk-gradio \
  --base_ckpt ckpts/AuK/auk_base.safetensors \
  --flash_ckpt ckpts/AuK-Flash/auk_flash.safetensors \
  --base_config ckpts/AuK/config.yaml \
  --flash_config ckpts/AuK-Flash/config.yaml \
  --qwen_path ckpts/Qwen2.5-Omni-3B \
  --base_device cuda:0 --flash_device cuda:1 \
  --dtype bf16 --host 0.0.0.0 --port 8080 --preload
```

- `--preload`：启动时加载；否则首次使用时懒加载；
- 默认 `--dtype bf16`、`--host 0.0.0.0`、`--port 7860`；
- 标准目录布局下**最短命令 `auk-gradio` 即可自动探测**两个模型；
- VAE 自动从 checkpoint 旁的 `vae.safetensors` 加载。

### 6.8 Python API 与微调

```python
from auk.infer.infer_auk import AukInfer, save_audio

engine = AukInfer("ckpts/AuK/config.yaml", "ckpts/AuK/auk_base.safetensors")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'."},
        {"type": "audio", "audio": "assets/demo-input-audio/content-edit/content.wav"},
    ]}
]
audio, sample_rate = engine.generate(messages, gen_seconds=7.0)
save_audio(audio, sample_rate, "out_content_edit.wav")
```

微调：`src/auk/train/train.py` + `scripts/train.sh`，数据是 **JSONL**，每行一个 source-target 对（`user` 带指令与可选源/参考音频，`assistant` 带目标音频，`duration` 驱动**动态 batching**）。所有任务共用这一格式，只改 instruction 和音频。详见 `docs/FINETUNING.md`。

### 6.9 亮点 / 局限

**亮点**
- **推理后端覆盖度是这批仓库里最好的**：vLLM-Omni / SGLang-Omni / GGUF(audio.cpp) / MLX / Xinference / ComfyUI，几乎覆盖云-端-Apple Silicon 全谱；
- **Task 统一到自然语言指令**，13 类任务一套接口，指令模板化后极易做产品集成；
- **显存数字是实测的**，且明确给出 `torch.cuda.max_memory_allocated` 口径与测试卡型号 —— 这种可复现的坦诚在开源项目里不多见；
- 有 **AuK-Flash 4 步蒸馏**版本，直接切 `--ckpt` 就能从质量优先切到速度优先；
- **MIT License**，商用最友好；
- 被 ICASSP 2027 挑战赛选为 baseline，说明社区认可度。

**局限**
- **encoder 依赖 Qwen2.5-Omni-3B**，且**不支持 Qwen3-Omni** —— 多一个 3B 模型的内存与下载成本；
- 即使开 CPU offload 峰值仍约 **17 GiB**，且 README 未给出「消费级卡可用」的具体最低显存数字（New 里说 reduced encoder memory ~7.5GiB 使消费级可行，但表格是 A800 80GB 上测的）；
- 未提供**延迟 / RTF / 并发吞吐**数据，选型时缺少在线服务的关键指标；
- ComfyUI 集成有 **30 秒「源+目标」序列长度限制**；
- PE（Prompt Enhancer）依赖**外部 LLM API**，离线场景需要自建或绕过。

---

## 7. 横向对比与技术脉络

### 7.1 五种「省算力」的思路对比

| 库 | 稀疏/加速的**决策依据** | 决策时机 | 粒度 | 是否可学习 | 训推一致 |
|---|---|---|---|---|---|
| **HiLS-Attention** | 压缩 chunk key 估计的 chunk-mass surrogate | 训练中学，推理时用 | chunk（page=64） | ✅ 端到端 | ✅ |
| **flex-block-attn** | 由调用方给定的 block mask（STA/MoBA/SSTA 规则） | 训练/推理前由策略决定 | block（推荐 384） | ❌ 规则/启发式 | ✅ |
| **Prism** | 视频通道方差 + audio→video 交叉注意力范数 | 每 zone 动态计算 | 各向异性 block（形状可变） | 部分（block 形状规则化） | ✅ |
| **AuK-Flash** | 不稀疏 —— 用**蒸馏把步数压到 4 步** | 训练时蒸馏 | 采样步数 | ✅（蒸馏） | ✅ |
| **Rosetta** | 不省算力 —— 靠 **Composable FFN + MAOP** 防遗忘 | 训练时架构设计 | 专家/模态 | ✅ | — |

**脉络观察**：混元在「稀疏注意力」上形成了**两条互补路线** ——

1. **通用 LLM 长上下文**：`HiLS-Attention`，重点是**让稀疏选择可学习**，并直接落到 SGLang 服务；
2. **视频/多模态生成**：`flex-block-attn`（算子层，可插拔 STA/MoBA/SSTA）+ `Prism`（算法层，内容驱动的动态 block 形状）。

两条线共享同一批核心作者（`Jiangfeng Xiong`、`Zhao Zhong` 同时出现在 flex-block-attn 与 Prism）。

### 7.2 工程可用性排序（按「今天就能拿来用」的难度）

| 难度 | 库 | 理由 |
|---|---|---|
| 🟢 低 | **AuK** | MIT + `pip install -e .` + 现成 CLI/Gradio/Python API + 多后端 |
| 🟢 低 | **Rosetta-inference** | Apache 2.0 + 权重已发布 + 11 个评测脚本，但需 8 GPU |
| 🟡 中 | **HiLS-Attention** | 算法代码 + 权重齐全，但服务要装**第三方 fork** 的 SGLang，且需 `tilelang` |
| 🟠 较高 | **flex-block-attn** | 需 Hopper 卡 + CUDA 12.8 + 编译；head dim 锁死 128 |
| 🔴 高 | **Prism** | 推理要 1–4×80GB，训练要 32–64×80GB；脚本式配置 |

---

## 8. 可复用要点

针对工程实践，从这 5 个库里可以**直接抄走**的东西：

1. **把稀疏选择变成可学习模块**（HiLS-Attention）
   不要用「训练完再换稀疏 kernel」的两段式做法 —— 那会引入训推不一致。让 top-k 选择在 LM loss 下可微/可学，并把选择所需的信息压缩（压缩 chunk key）以**避免为选择而付全量 QK 的代价**。这是稀疏注意力落地最容易踩的坑。

2. **块级 mask 表示 + 类 PyTorch 的 mask DSL**（flex-block-attn）
   原始逐元素 mask 在 `block_size=64` 时要展开成 `[128,128]`，显存与生成耗时都不划算。用 `[[0,1],[1,0]]` 的**紧凑块表示** + `create_sparse_mask()` 这类构造函数，把稀疏配置的表达成本降到接近 PyTorch 语义 —— 这对任何自研稀疏 kernel 的库都是通用设计。

3. **两个正交信号决定「该在哪里细看」**（Prism）
   Prism 用「视频通道方差」（内容变化方向）× 「audio→video 交叉注意力范数」（跨模态耦合强度）来**按轴分配 block 粒度**。这个 pattern 可以迁移到任意多模态场景：**用模态间注意力强度作为「对齐热区」的免费探针**，在热区加密、冷区放粗。

4. **混合稀疏度：top-k + top-p CDF**（Prism）
   `bsa_cdf_threshold=0.20` 与 top-k 结合，使**每个 query 有不同稀疏度**。固定稀疏率对「信息分布极不均匀」的输入（如 2K 视频）是浪费；top-p 部分负责自适应，top-k 部分负责兜底上界。

5. **稀疏度/桶采样保证同 batch 同形状**（Prism）
   `DistributedBucketBatchSampler` 让 mini-batch 只从一个桶取样，避免 DP rank 间张量形状不匹配 —— 任何多分辨率/多时长训练都会遇到这个问题，是一个干净的解法。

6. **显存峰值要报口径**（AuK）
   明确写出 `torch.cuda.max_memory_allocated`、测试卡型号（A800-SXM4-80GB）、dtype（bf16）、输入（1.5s 输出 / 5s 参考音频）。**可复现的显存数字比一个漂亮的百分数更有价值**。

7. **蒸馏出「快版」而非只调参**（AuK-Flash）
   与其让用户调 NFE，不如直接发布一个**固定 4 步、CFG=0** 的蒸馏变体，切换只需改 `--ckpt`。产品化时把「质量/速度」二选一做成**模型选择**而非**参数调优**，可以极大降低集成复杂度。

8. **为发布配上「阶段化 checkpoint 矩阵」**（Rosetta）
   3 架构 × 5 阶段 + 统一 config 命名（`rosetta.yaml`/`moe.yaml`/`mot.yaml`，同架构跨阶段复用），让「指标随训练阶段的演化」可复现。这比只发最终权重更能证明方法的有效性。

9. **CPU offload 是性价比最高的一档降显存手段**（AuK）
   约 **32%** 的峰值显存节省（~8 GiB），实现代价只是加一个 `--cpu_offload`。对把 80GB 卡上的模型下沉到 24GB 消费卡的场景，这是第一优先级。

10. **服务后端要主动跟进上游，而不是只发算法代码**（HiLS-Attention 的反面教训）
    HiLS 把 SGLang 后端放在**个人 fork**（`alexzms/SGLang-HiLS`）里，用户必须装 fork 且额外装 `tilelang`，还受 `--page-size == chunk_size`、prefill 单序列、禁用 overlap scheduler 三重约束。对比 AuK 在开源一周内就拿下 SGLang-Omni / vLLM-Omni 的**官方 Day-0 支持** —— 后者的落地摩擦小得多。**做推理优化，最好把 kernel 直接贡献到主流 serving 框架主线里。**

---

## 9. 参考链接

### HiLS-Attention
- 仓库：<https://github.com/Tencent-Hunyuan/HiLS-Attention>
- 论文：<https://arxiv.org/abs/2607.02980>
- 模型权重：<https://huggingface.co/tencent/HiLS-Attention-7B>
- SGLang 后端 fork：<https://github.com/alexzms/SGLang-HiLS>
- SGLang：<https://github.com/sgl-project/sglang>
- OpenCompass：<https://github.com/open-compass/opencompass>
- uv：<https://astral.sh/uv/install.sh>

### flex-block-attn
- 仓库：<https://github.com/Tencent-Hunyuan/flex-block-attn>
- ThunderKittens：<https://github.com/HazyResearch/ThunderKittens>
- STA / Sliding Tile Attention：<https://github.com/hao-ai-lab/FastVideo>
- MoBA：<https://github.com/MoonshotAI/MoBA>
- flex attention (attention-gym)：<https://github.com/meta-pytorch/attention-gym>
- flash-attention：<https://github.com/Dao-AILab/flash-attention>
- MagiAttention：<https://github.com/SandAI-org/MagiAttention>
- SpargeAttn：<https://github.com/thu-ml/SpargeAttn>
- Triton：<https://github.com/triton-lang/triton>
- CUDA Toolkit 12.8：<https://developer.nvidia.com/cuda-12-8-0-download-archive>

### Prism
- 仓库：<https://github.com/Tencent-Hunyuan/Prism>
- 项目页：<https://francis-rings.github.io/Prism>
- 论文：<https://arxiv.org/abs/2610.05416>
- 权重：<https://huggingface.co/FrancisRing/Prism>
- 视频（YouTube）：<https://youtu.be/XEL2IkSOSQw>
- 视频（Bilibili）：<https://www.bilibili.com/video/BV18dHZ6iEM8>

### Rosetta-inference
- 仓库：<https://github.com/Tencent-Hunyuan/Rosetta-inference>
- 项目页：<https://rosetta-lmm.github.io/>
- 论文：<https://arxiv.org/abs/2607.00293>
- 权重：<https://huggingface.co/tencent/Rosetta-inference>
- Qwen3-0.6B-Base：<https://huggingface.co/Qwen/Qwen3-0.6B-Base>

### AuK
- 仓库：<https://github.com/Tencent-Hunyuan/AuK>
- 项目页：<https://auk-project.github.io/>
- 技术报告：<https://arxiv.org/abs/2609.08936>
- 权重：<https://huggingface.co/tencent/AuK> · <https://huggingface.co/tencent/AuK-Flash>
- Demo：<https://huggingface.co/spaces/tencent/AuK> · <https://modelscope.cn/studios/Tencent-Hunyuan/AuK>
- vLLM-Omni recipe：<https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/AuK-H100.md>
- SGLang-Omni cookbook：<https://sgl-project.github.io/sglang-omni/cookbook/auk.html>
- audio.cpp GGUF 文档：<https://github.com/0xShug0/audio.cpp/blob/main/docs/community_models/auk.md>
- ComfyUI 社区节点：<https://github.com/Saganaki22/ComfyUI-AuK>
- 量化 checkpoint：<https://huggingface.co/drbaph/AuK-comfyui>
- ICASSP 2027 Audio Editing Challenge：<https://audio-editing-challenge.github.io/> · <https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-Baseline>
- Xinference：<https://github.com/xorbitsai/inference>
