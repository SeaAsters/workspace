# 混元推理侧增量扫描 · 2026-10-12 → 2026-10-26

> **本次任务**：在既有基线（[`INDEX.md`](../../INDEX.md) + [`infer/2026-10-12/`](../2026-10-12/)）之上，扫描 **2026-10-12 之后**混元在推理方向的新增仓库、代码变更与公开信息，并把可验证的增量落成报告。
>
> **窗口**：`2026-10-12T00:00Z` → `2026-10-26T10:30+08:00`（本报告写作时刻）。
> **基线**：`_research/repos-classification.csv` 的 **85 个仓库**快照（`last_push` 截至 2026-10-11）、`_research/readmes/` 的 README 快照、`_research/licenses.csv`。
> **本次实测范围**：Tencent-Hunyuan 组织现有 **90 个公开仓库**（+5）、其中窗口内被推送过的 **6 个**、HuggingFace `tencent` 组织窗口内新模型 **8 个**、arXiv 论文 **6 篇**（逐一核验过 ID/标题/日期）。

**数据来源标注约定**（沿用 10-12 报告，新增后三类）

| 标记 | 含义 |
|---|---|
| **仓库实测** | 本机浅克隆后直接读文件（`_probe/`），或从 `raw.githubusercontent.com` 拉取原文 |
| **GitHub API / atom feed** | `api.github.com` 仓库元数据、`commits/<branch>.atom` 的提交条目（客观时间戳） |
| **HF API / 模型卡** | `huggingface.co/api/models?author=tencent` 与模型卡 Markdown 原文 |
| **arXiv API** | `export.arxiv.org/api/query` 返回的 `<published>` / `<title>` |
| **公开报道** | 媒体、自媒体、公众号转载页；**公众号原文常被验证码拦截**，凡转述均注明"转载页" |
| `README 未说明` | 原文没有，本报告不做推断补全 |

---

## 0. 一句话结论

窗口内推理侧的主线**不是**再发一个大语言模型，而是三件事：

1. **世界模型被推到"可交互 + 可单卡跑 + 可服务化"** —— 新仓库 `Hy-Nomas`（10-21 建）把交互式视频世界生成做成完整栈：54 层双流 MMDiT + 分段自回归 + 持久 KV cache，**8 步蒸馏推理 / 720×1280 / 24 FPS**，并给出**单张 RTX 4090**（INT8 权重 + INT8 KV + FP8 解码 + CPU offload）与 **CUDA 13 预打包镜像**两条部署路径。
2. **记忆写入被拆成一个可独立服务化的推理负载** —— `Hy-Memory-30B-A3B`（~3B 激活 MoE）在 10-15 上线，10-16 出 **FP8**、10-19 出 **GGUF**，模型卡直接给 **vLLM v0.31.0** 的 OpenAI 兼容部署命令与 `reasoning_effort=no_think` 调用参数。
3. **"少步蒸馏"从成品变成可复现管线** —— `AuK` 在 10-14 开源 **4 步蒸馏训练链**（consistency 初始化 + DMD），`docs/DISTILL.md` 明确 `NFE=4 / CFG=0` 即 **AuK-Flash** 配方，推理命令 `auk-infer --nfe 4 --cfg 0`。
4. **量化从"给个权重"变成"给一份可复算的配方"** —— `Hy-Memory-30B-A3B-GGUF`（10-19）不只给 6 档 GGUF，还给了 **imatrix + MoE 混合精度**的分配表、每档文件大小与 LoCoMo 分数、`llama-server` 全量参数与 commit、SHA256 校验值，并在 **H20** 上公开核验了社区超低比特档 **8/8 提示全失败**——这是本轮"工程可信度"最高的一份材料（详见 §3.1）。

> **与前次判断的关系**：10-12 报告提出的"官方一等公民收敛为 vLLM + SGLang"继续成立（Hy-Memory 走 vLLM、Nomas-PE 走 SGLang）；但**生成类模型（视频/世界模型）的加速路线明显走"蒸馏 + 量化"，而不是 LLM 侧的"投机解码"** —— 这是本轮新增的一条分线结论，详见 §8。

---

## 1. 增量总览

### 1.1 组织级增量（90 = 85 + 5）

| 仓库 | 类型 | created | 最后推送 | stars（现） | 语言 / 体积 | License | 与推理主线的关系 |
|---|---|---|---|---|---|---|---|
| **Hy-Nomas** | 新增 | 2026-10-21 | 2026-10-24 | 160 | Python / 4.9 MB | Apache-2.0（+第三方清单） | ⭐ **核心**：交互式视频世界生成推理栈 |
| **Hy-Memory** | 新增 | 2026-10-15 | 2026-10-26 | **1,282** | Python / 19.5 MB | Apache-2.0（Tencent 前言 + Apache 全文） | ⭐ **核心**：记忆写入模型的落地面（含 FP8/GGUF 推理） |
| **OmniCapBench** | 新增 | 2026-10-12 | 2026-10-22 | 7 | Python / 0.4 MB | Apache-2.0 | 全模态评测；本地模型用 vLLM 起服务 |
| **HYDRA** | 新增（补录） | 2026-10-09 | 2026-10-14 | 9 | Python / 9.1 MB | MIT | 统一多模态 tokenizer + 双头解码（T2I 采样步数敏感） |
| **DiagSafe** | 新增 | 2026-10-15 | 2026-10-22 | 27 | Python / 14.9 MB | CC-BY-4.0（Tencent 前言 + CC 全文） | 安全评测，**非推理主线** |
| AuK | 触及 | 2026-08-19 | 2026-10-22 | 1,546（+59） | Python / 22.1 MB | MIT | ⭐ 4 步蒸馏训练管线开源 |
| Prism | 触及 | 2026-09-21 | 2026-10-15 | **1,145（+971）** | Python / 25.2 MB | MIT | 稀疏注意力（仅 README 改动） |
| UniRL | 触及 | 2026-06-08 | 2026-10-19 | 1,025（+14） | Python / 37.3 MB | Apache-2.0 | 训练框架，**非推理**（见 agent 报告） |
| ExplorationBench | 触及 | 2026-09-30 | 2026-10-14 | 27（+20） | Python / 1.35 MB | Apache-2.0 | Agent 评测（见 agent 报告） |
| CL-bench | 触及 | 2026-01-23 | 2026-10-17 | 589（+6） | Python / 21.0 MB | 无 LICENSE 文件 | 评测，非推理 |
| Precise | 触及 | 2026-05-22 | 2026-10-22 | 18（+2） | Python / 8.0 MB | Apache-2.0 | 流匹配 RL 采样，非推理 |

> **"HYDRA 是补录而非全新"**：其 `created_at=2026-10-09` 早于基线截止日，但基线 CSV 中没有它（应为当时尚未公开），10-14 有一次批量代码上传（`Upload HYDRA hydra/...`、`Upload HYDRA hydra_tok/...`）。为避免与基线口径冲突，本报告把它单列为"补录"。

### 1.2 关注度增量（可量化）

基线 85 个仓库合计 **93,869 stars** → 现在 **95,900 stars**（+2,031）。窗口内变化最大的 8 个：

| 仓库 | 10-12 | 10-26 | Δ | 同期是否有代码变更 |
|---|---:|---:|---:|---|
| Prism | 174 | 1,145 | **+971** | ❌ 仅 1 次 README 提交（改 News 日期） |
| Hy-MT2 | 1,234 | 1,495 | +261 | ❌ 无推送（`last_push` 仍是 2026-08-26） |
| Hunyuan3D-2 | 15,081 | 15,200 | +119 | ❌ 无推送（2025-10-28） |
| Hunyuan3D-2.1 | 4,164 | 4,280 | +116 | ❌ 无推送 |
| HY-WU | 794 | 878 | +84 | ❌ 无推送 |
| AuK | 1,487 | 1,546 | +59 | ✅ 蒸馏管线 |
| HunyuanVideo | 12,613 | 12,667 | +54 | ❌ 无推送 |
| HY-World-2.0 | 2,716 | 2,770 | +54 | ❌ 无推送 |

> **读法**：窗口内 star 增长**基本与代码变更脱钩**（8 个里只有 1 个有推送）。`Prism` 的 5.6× 跃升尤其值得警惕 —— 同期唯一提交是把 News 占位日期 `[2026-x-xx]` 改成 `[2026-10-06]`，**没有任何推理代码或权重变更**。热度来源无法从仓库侧证实（详见 §5.1）。

---

## 2. 核心增量 A：`Hy-Nomas` —— 交互式视频世界生成，从权重到服务栈

- 仓库：<https://github.com/Tencent-Hunyuan/Hy-Nomas>（created 2026-10-21，Apache-2.0）
- 权重：<https://huggingface.co/tencent/Hy-Nomas>（HF 创建 2026-10-21）+ `tencent/Hy-Nomas-Diffusers`（2026-10-24）
- 评测集：<https://huggingface.co/datasets/tencent/Hy-Nomas-Bench>
- 官网 / 演示：<https://hy-nomas.hy.tencent.com/>、<https://hy-nomas.hy.tencent.com/demo>
- 技术报告：**README 与模型卡均写 "Technical Report (Coming Soon)"**，仓库内无 `Hy_Nomas_Tech_Report.pdf`（引用处是待补链接）

### 2.1 它解决什么

把"视频生成"改成"**世界生成 + 多轮指挥**"：模型**一段一段**生成视频，每段可接受新的文本指令或相机轨迹，并通过**持久 KV cache** 把已生成的历史带入下一段，从而在扩展同一个世界时保持一致性。README 的关键特性原文：分段生成、自由形式提示词控制、相机控制、蒸馏推理、交互式演示、LoRA 微调与数据准备。

> **边界要看清**：输出是**视频**，不是可编辑 3D 网格或游戏关卡（这点在公开报道里也被明确指出，见 §7）。

### 2.2 架构参数（模型卡 §Model Architecture，逐项照抄）

| 项 | 配置 |
|---|---|
| 骨干 | 分段自回归 **dense MMDiT**（双流） |
| Transformer blocks | **54** 个双流块、**0** 个单流块 |
| Hidden size | **2,048** |
| Attention | **16 heads × 128 dim** |
| FFN | **4×**，intermediate **8,192** |
| Latent 输入/输出通道 | **32 / 32**（DCAE-V `f32t4c32`：32× 空间压缩、4× 时间压缩） |
| Patch size（t×h×w） | 1 × 1 × 1 |
| 位置编码 | 3D RoPE，维度 `[16, 56, 56]`，theta 256 |
| 文本条件 | **Qwen2.5-VL-7B-Instruct** 的 LM 部分，3,584 维 → 两层 token refiner → 投影到 2,048 |
| 相机条件 | 归一化 4×4 外参 → 按 4 帧分组展平为 **64 维**；`extrinsics_attn`，双流块内做相机 cross-attention |
| 去噪预测 | flow velocity，Euler ODE 采样 |
| 视频解码 | 微调 **PixelUMM**：36 层 Transformer、hidden 4,096、32 query heads / 8 KV heads、4×16×16 tubelet；**1 步解码**，时间窗口 10 latent frames、重叠 5 |
| 蒸馏推理 | **8 步去噪**，720×1280、24 FPS |

### 2.3 推理链路（三件套：分段自回归 + KV cache + 两段式解码）

1. **分段**：每段从潜空间高斯噪声开始去噪；`per_chunk_camera_attn=true` + `camera_chunk_size=1` 使相机注意力对齐到单个 latent frame。
2. **跨段一致性**：segment-wise **causal** joint attention（3D RoPE 编码视频位置），历史由该层 **KV cache** 提供；"注意力在段间是因果的，因此生成当前段不需要未来段"。
3. **解码**：DiT 产出 32 通道潜变量 → PixelUMM **1 步**解码（与 8 步去噪是两件事）。
4. **提示增强**：Web 演示里 **Nomas-PE** 先把用户指令转成结构化提示词；离线推理可直接读准备好的 prompt 文件。

> 一个可复现的规模锚点（模型卡 Diffusers 示例）：`height=704, width=1280, num_frames=240, segment_latent_lengths=[30,30], fps=24, num_inference_steps=4` —— 即**两段各 5 秒**、共 240 帧。注意示例里步数写 4，而蒸馏配置说明是 8 步（`Distilled DiT` 支持 4/6/8 步，README「蒸馏 DiT（FP32）… 用于 4/6/8 步快速推理」）。

### 2.4 单卡 RTX 4090：四个开关叠加（这是本轮最值得抄的工程细节）

先给一个容易漏掉的前提：**标准预设本身是 8 路序列并行**。`configs/experiment/inference.yaml` 里 `sp_size: 8`（模型类型写作 `HY-Nomas-G/2-32324-MMDiT`、注意力 `magi_mp_causal`），即官方标准推理路径按 8 卡规模起；单卡路径是把同一套配置换成 `single_gpu_low_mem_inference.yaml`（`sp_size: 1`）并叠加下面四项优化。**README 没有写"最少几张卡"，但两份配置的 `sp_size` 差异决定了这一点。**

README「低显存 GPU 推理」原文给出四项优化，缺一不可：

| 手段 | 具体做法 | 代价 |
|---|---|---|
| **INT8 权重 + INT8 KV cache** | INT8 存权重与缓存 K/V，**DiT 线性层与注意力仍用 BF16 计算**（W8-stored / BF16-compute） | 需要对应的 INT8 DiT 检查点；**INT8 loader 不支持推理时再挂 LoRA** |
| **滑动窗口历史** | 限制保留的历史上下文，避免 KV cache 随段数无限增长；默认历史预算 **18 latent frames**（只约束历史，不约束当前段长） | 超长世界的远期一致性让位给显存 |
| **CPU offload** | 文本编码器 / DiT / VAE 各自阶段才进 GPU，用完回 CPU | 需要足够系统内存 + 更高的 CPU↔GPU 搬运开销 |
| **单卡内核与并行度** | DiT 用 **FlexAttention**，解码用 **CUTLASS FP8**，并**关闭序列并行**（`sp_size=1`） | 标准化配置下是 SP，单卡时退化 |

配套产物：`scripts/quantization/convert_dit_w8_bf16.py`（"Convert a floating safetensors DiT to validated INT8 stored / BF16 compute"，带 `FORMAT/GROUP_SIZE/MANIFEST/SHA256` 校验与 `load_w8_bf16_model` 回读）、`configs/experiment/single_gpu_low_mem_inference.yaml`、`scripts/inference/benchmark_pixelumm_streaming_decode.py`。

> **口径提醒**：24 FPS 是**输出视频播放帧率**，不是生成速度。模型卡自己写明"生成速度取决于硬件、分辨率、段长、历史与配置"。因此"单卡 4090 可跑"≠"可实时交互"。

### 2.5 权重与资产矩阵（全部来自模型卡 §Model Assets）

| 组件 | 资产 | 用途 | 仓库内路径 |
|---|---|---|---|
| DiT | Distilled（FP32） | 8 步推理与 Web 演示主检查点（标准配置以 BF16 加载） | `DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors` |
| DiT | Distilled **INT8** | 单卡低显存 | `DiT/distilled_int8/..._int8.safetensors` |
| DiT | Base autoregressive（FP32） | **未蒸馏**，用于 segment-forcing LoRA 微调 | `DiT/base_ar/diffusion_pytorch_model_fp32.safetensors` |
| Adapter | Cinematic LoRA | 视觉质量 | `DiT/lora_weights/cinematic/` |
| Adapter | Robotics LoRA | 真实机械臂数据 [RealSource-World](https://huggingface.co/datasets/RealmanRobot/RealSource-World) 上训练，增强机器人仿真世界 | `DiT/lora_weights/dual_arm_robotics/` |
| VAE | DCAE-V | 32 通道潜变量编解码 | `VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt` |
| 解码器 | PixelUMM_FT | 默认解码 | `VAE/PixelUMM_FT/...` |
| 解码器 | PixelUMM_FT **FP8** | 低显存解码 | `VAE/PixelUMM_FT_FP8/..._fp8.safetensors` |
| PE | **Nomas-PE** | 在线提示增强（独立服务；**Web 演示必需，离线/基准推理不需要**） | `PE/` |

### 2.6 服务化拓扑（仓库里有完整的多进程/多机脚手架）

`apps/` + `scripts/inference/` 暴露的不是一个 `demo.py`，而是一套小型推理平台：

```
apps/
  seg_ar_gateway.py            # 网关（对外）
  seg_ar_dispatcher.py         # 调度
  seg_ar_resource_manager.py   # GPU 资源管理
  seg_ar_resource_monitor.py   # 资源监控
  seg_ar_job_store.py          # 任务持久化（+ durable_file.py）
  seg_ar_attempt_archive.py    # 尝试归档 / replay
  gradio_app.py + frontend/    # 前端（Node 22.x / npm）
  kimi_k3_prompt_enhancer.py   # 提示增强客户端
scripts/inference/
  run_seg_ar_gateway.sh / run_seg_ar_dispatcher.sh
  run_seg_ar_gpu_pool.sh / run_seg_ar_gpu_worker.sh
  run_seg_ar_multinode_worker.sh        # 多机 worker
  start_seg_ar_demo.sh / stop_seg_ar_demo.sh
  run_nomas_http_proxy.mjs / run_nomas_auth_proxy.py
  benchmark_pixelumm_streaming_decode.py
  DEMO_REPLAY.md / build_demo_replay_manifest.py
```

**PE 服务**（`apps/README.md`）用 **SGLang** 官方镜像跑 Qwen3.8 推理、默认单机 8 卡、`PE_TP_SIZE` / `PE_PORT` 可调，网关通过 `PE_API_URL` / `PE_MODEL` / `PE_API_KEY` 连接；"PE 使用自己的 GPU 资源，需与会话生成 worker 分开计"。

### 2.7 环境与依赖（照抄，便于复现）

- **推荐 Docker**：`hynomasoss/hynomasoss:cu130`，`docker run --gpus all --ipc=host --network=host`；镜像内已备好 Python、PyTorch 与 CUDA 注意力内核。
- **手动**：Linux + Python 3.12+ + `uv` + CUDA **13.0** Toolkit（`nvcc`）+ C++ 编译器；`uv sync --extra all`。
- `all` 依赖组含 **PyTorch 2.10.0 (CUDA 13.0)**、**FlashAttention 2**、**FlashAttention 3**、**MagiAttention**；`pyproject.toml` 已配 CUDA 包源、FA3 源码地址，并为注意力内核关闭构建隔离（要用环境里的 PyTorch 编译，"内核编译可能需要一些时间"）。
- **Diffusers 路线**：`pip install -e "diffusers[hy-nomas] @ git+https://github.com/jiaxiangc/diffusers.git@hy-nomas"`，用 `HyNomasModularPipeline.from_pretrained("tencent/Hy-Nomas-Diffusers")`；需显式 `pipe.transformer.enable_reference_attention(text_backend="flash_varlen", deterministic=True)`。
- **前端**：Node.js **22.x（≥22.12）**、npm；`npm --prefix apps/frontend ci`；FFmpeg 用 `imageio-ffmpeg` 自带二进制（部分验收测试另需 `ffprobe`）。
- 文本编码器默认从 `Qwen/Qwen2.5-VL-7B-Instruct` 自动下载，可用 `model.text_encoder.model_name=` 指向本地。

### 2.8 训练 / 微调（LoRA 与数据标注也是开源的）

| 项 | 值（README / 模型卡） |
|---|---|
| LoRA | rank/alpha **256**、dropout 0、**Muon** 优化器、恒定 lr **1e-4**、**2,000 步**、只训 LoRA |
| 并行 | 训练配置用 **SP8**；128 卡对应 **DP16 × SP8** |
| 序列打包 | `python scripts/data_processing/build_sft_packing_plan.py --max-seq-len 360000 --num-groups 16`，组数须为 DP 的整数倍 |
| 训练基座 | **未蒸馏**检查点训练 → 导出 LoRA → 合并进**蒸馏**检查点（`export_lora_checkpoint` + `merge_lora`）；两者架构与参数形状必须匹配 |
| 数据 | `captioning_harness/`（K3 服务标注 + DCAE-V 编码 + Arrow/Parquet 打包 + 相机轨迹） |

### 2.9 评测：`Hy-Nomas-Bench`（数据集卡比仓库 README 细得多）

模型卡只说"4 基准 / 280 用例"；**数据集卡（10-22 上线）给了完整 split 结构**：

| Benchmark | 用例数 | 主评维度 | 轮/段数 | 平均时长 |
|---|---:|---|---:|---:|
| Bench-1 多轮文本 | 60 | 提示遵循、物理合理性 | 2–6 | 37.88 s |
| Bench-2 相机导航 | 100 | 相机控制 | 4–20 | 27.05 s |
| Bench-3 文本×相机交错 | 60 | 提示遵循 + 物理合理性 + 相机控制 | 2–7 | 30.59 s |
| Bench-4 空间记忆 | 60 | **重访一致性** | 8 | 18.00 s |

- 附 **220 条参考相机轨迹**；请求时长范围 **8–86 秒**；提示以英文为主，Bench-1 含少量中文/中英混合。
- **Bench-2** 覆盖 **49 个场景、8 种基本相机动作、24 种双键组合**，每个固定动作 2–4 秒并配等长反向返回；**Bench-3** 含 **125 个 TEXT 段 + 68 个 CAMERA 段**；**Bench-4** 每个第一人称场景绕一个显著静物做 4 组"出去/返回"。
- **不含视频、参考帧与任何评测分数**；四个 split 全部是 `test`。
- 离线推理**不需要**在线提示增强服务（PE）：直接 `scripts/inference/inference_from_config.sh` + `inference.prompt=<bench json>`，只需改模型/输入/输出路径。

> **口径提醒**：官方**至今没有公布任何模型在该 benchmark 上的分数**；"280 用例"是**题目数量**，不是成绩。

### 2.10 许可

- 代码：`LICENSE` 正文首行 **"Hy-Nomas is licensed under Apache-2.0, except for the third-party components listed below"**，随后是 105 KB 的第三方组件清单（体量大是因为逐条附了第三方许可全文）。
- 权重：HF 模型卡 `license: apache-2.0` + `license_link` 指向仓库 `LICENSE`。
- ⚠️ 与 10-12 报告踩到的坑一致：**以 LICENSE 与模型卡为准，不看 README 徽章**（本轮 4 个新仓库中 2 个的 README 徽章与 LICENSE 文件形式不同：Hy-Memory/Hy-Nomas 的 LICENSE 都是 "Tencent is pleased to support…" 前言 + Apache 正文，GitHub API 因而判为 `NOASSERTION`）。

---

## 3. 核心增量 B：`Hy-Memory-30B-A3B` —— 记忆写入作为独立推理服务

> Agent 侧的框架/产品视角见 [`../../agent/2026-10-26/05-agent-memory-and-delta.md`](../../agent/2026-10-26/05-agent-memory-and-delta.md)；本节只写**推理与部署事实**。

| 项 | 事实 | 来源 |
|---|---|---|
| 模型 | `tencent/Hy-Memory-30B-A3B`，**30B 总参 / 约 3B 激活**的紧凑 MoE（"compact long-term-memory writer model (~3B active parameters)"） | HF 模型卡 / 论文 |
| 定位 | **直接替换** Mem0 / MemOS / Hy-Memory 里的通用 LLM，只做"记忆写入"这一段：决定保留什么、怎么精确表示、新证据如何更新记忆库 | HF 模型卡 |
| 训练 | 两阶段：**Memory On-Policy Distillation (OPD)** → **Attributed Memory RL** | HF 模型卡 / arXiv 2610.22196 |
| 泛化 | 可迁移到训练时未见的框架 **EverMemOS**（held out from training） | HF 模型卡 |
| 部署 | **vLLM**（"tested with v0.31.0"），`--served-model-name hy-memory` + OpenAI 兼容 API | HF 模型卡 Quickstart |
| 推理参数 | `temperature=0.0`、`max_tokens=512`、`extra_body.chat_template_kwargs.reasoning_effort="no_think"` | HF 模型卡 |
| 量化版 | **FP8**（2026-10-16，`tencent/Hy-Memory-30B-A3B-FP8`，"optimized for efficient vLLM deployment"）、**GGUF**（2026-10-19） | HF API + 模型卡 |
| 许可 | Apache-2.0（模型卡 `license: apache-2.0`） | HF 模型卡 |
| 下载量（写作时） | 主模型 4,208 / GGUF 18,814 / FP8 635 | HF API `downloads` |

**跨 harness 结果**（模型卡 Table 1，固定 20% 子采样，六个指标等权：LoCoMo Judge / LongMemEval / PersonaMem-v2 / Travel SPS / Shopping PS / CMB）：

| Memory writer | Mem0 | Hy-Mem. | MemOS | EverMem.（held-out） | 平均 |
|---|---:|---:|---:|---:|---:|
| **Hy-Memory-30B-A3B** | **54.02** | **60.55** | **50.51** | **56.89** | **55.49** |
| Claude Opus 5 | 53.43 | 60.53 | 50.50 | 54.04 | 54.62 |
| DeepSeek V4.1 Flash | 49.73 | 60.20 | 49.17 | 55.47 | 53.64 |
| Hy3 | 45.99 | 58.06 | 50.19 | 52.30 | 51.63 |
| Qwen3.6-35B-A3B | 47.51 | 55.87 | 47.36 | 50.75 | 50.37 |
| Hy3-A3B Base | 40.20 | 50.39 | 49.05 | 50.98 | 47.66 |
| Qwen3-30B-A3B | 39.81 | 45.67 | 45.44 | 46.57 | 44.37 |

> **推理侧含义**：这是一个**低温度、结构化 JSON 输出、短输出、高并发**的负载形态（模型卡示例直接要求"Return only a JSON object with a memories list"），与 chat 负载的优化目标不同 —— 更适合 FP8/GGUF + 大 batch、且几乎没有长上下文解码压力。厂商自报"超过 Claude Opus 5"的口径是**固定 20% 子采样 + 六指标等权 + 四 harness 等权**，跨 harness 均值差 0.87 分（55.49 vs 54.62），**不宜当作通用能力超越**。

### 3.1 GGUF：6 档量化 + 官方 `llama-server` 配方 + 一次"能加载 ≠ 能用"的公开核验

`tencent/Hy-Memory-30B-A3B-GGUF`（HF 创建 **2026-10-19**）把量化做成了**可复算的工程文档**，不是"上传一个 Q4 就完事"。

**配方**：imatrix 校准 + **MoE 感知混合精度** —— 路由专家（占参数大头）可以压得很狠，**共享专家、注意力、输入/输出层保高精度**。

| 量化 | 文件大小 | Mem0 LoCoMo Judge |
|---|---:|---:|
| `Q6_K` imatrix | 24.71 GB | **58.78%**（154/262） |
| `Q8_0`（无 imatrix 基线） | 31.99 GB | 57.63%（151/262） |
| `Q3_K_M` imatrix | 14.45 GB | 57.25%（150/262） |
| **`Q8_0-expsIQ1_M`**（路由专家 IQ1_M，其余 Q8_0） | **8.03 GB** | 57.25%（150/262） |
| `Q4_K_M` imatrix | 18.24 GB | 56.49%（148/262） |
| `Q8_0-expsIQ2_XXS`（路由专家 IQ2_XXS，其余 Q8_0） | 9.14 GB | 55.73%（146/262） |

- 官方结论原话：**~8 GB 的专家混合档与最强 Q6_K 只差约 1.5 分，体积约为 Q8_0 的四分之一**。
- **混合档的精度分配是可查的**：141 个路由专家张量走 IQ 类型、338 个其余矩阵张量走 Q8_0、287 个张量保 F32 —— 明确"它既不是整模型 1/2-bit，也不是专家剪枝"。
- 6 个文件都通过了 `--check-tensors`、真实 no-think 生成、以及完整的固定 20% Mem0 LoCoMo 流程；官方同时声明**这是单次运行分数，不保证低比特一定劣于高比特**。
- 评测环境（照抄）：Mem0 **2.0.19**、固定 20% LoCoMo（2 users / 262 questions / 59 chunks）、`top_k=10`、每 chunk 10 turns、记忆温度 0.7、no-think、embedding `qwen3-embedding-8b-for-online`(4096d)、答案与裁判 `gpt-5.6-sol`、llama.cpp commit `5aeac37…`、KV cache F16。

**服务化命令（官方给的是"论文口径"那一套）**：

```bash
llama-server --model ./Hy-Memory-30B-A3B-Q6_K.gguf --alias Hy-Memory-30B-A3B-Q6_K \
  --host 127.0.0.1 --port 8000 --n-gpu-layers all --flash-attn on \
  --ctx-size 65536 --parallel 1 --batch-size 2048 --ubatch-size 512 \
  --cache-type-k f16 --cache-type-v f16 --temp 0.7 --top-k 0 --top-p 1.0 --repeat-penalty 1.0 \
  --jinja --chat-template-kwargs '{"reasoning_effort":"no_think"}' --check-tensors
```

> `--ctx-size` 是**服务总上下文**：单槽 64K；`--parallel 2` 时要设 `131072` 才能保持每请求 64K。

**最有价值的一段：官方公开纠偏社区超低比特**。2026-10-21，官方在 **H20** 上核验 `vinnipukh/Hy-Memory-30B-A3B-GGUF`（commit `aa6b9155…`，SHA256 匹配、llama.cpp 同 commit、no-think、F16 KV、不强制 JSON 解码）：

| 社区档 | 能加载 | 8 条基础提示通过 | 现象 |
|---|---|---|---|
| `Q1_0` | 是 | **0/8** | 无关词循环直到输出上限 |
| `Q2_0` | 是 | **0/8** | 混杂字符/词 + 重复 |
| `TQ1_0` | 是 | **0/8** | 非句子内容 + 重复 |
| `TQ2_0` | 是 | **0/8** | 非句子内容 + 重复 |

- 同一批提示下，官方 `Q6_K` **8/8** 返回合法 JSON。
- 官方明确边界：这只是**基础输出检查，不是 LoCoMo Judge 分**，也不能区分是量化误差、转换问题还是后端问题；结论**只适用于上述文件与设置**。
- 结论一句话：**"能加载 ≠ 能用"** —— 这条对任何打算把 30B 级 MoE 压到 1–2 bit 上端侧的方案都成立。

---

## 4. 核心增量 C：`AuK` —— 4 步蒸馏从成品变成可训练管线

10-14 的 `PR #22` + 10-22 的 `docs/DISTILL.md` 把 AuK-Flash 的配方完整落地（README News 新增一条 `[2026/10/14] The distillation training pipeline is released!`）。

| 阶段 | 入口 | 启动脚本 | 默认预算 |
|---|---|---|---|
| 1. **Consistency 初始化** | `src/auk/train/distill/initializer/train.py` | `scripts/train_init.sh` | `max_updates=500`、lr `1e-5`（AdamW、恒定、无 warmup）、`teacher_cfg_scale=3.0`、EMA `0.999`、`save_per_updates=500`/`last_per_updates=100` |
| 2. **DMD（分布匹配蒸馏）** | `src/auk/train/distill/dmd/train.py` | `scripts/train_dmd.sh` | `max_student_updates=2500`、`fake_score_updates_per_student=5`、`teacher_cfg_scale=4.0`、**`nfe=4`**、`sway_sampling_coef=-1.0`、`simulation_free=true`、`task_routing=false` |

零散但要紧的默认值：

- **`nfe=4` + `cfg=0` 就是部署配方**（"Keep these, they are the deployed recipe"）；推理侧对应 `auk-infer --ckpt ckpts/auk_dmd/model_2500.pt --nfe 4 --cfg 0`。
- **`task_routing`**（默认关）开启后，`--regression_only_tasks`（默认 `ss,music,fisher`）的样本只贡献回归损失，其余只贡献 DMD 损失 —— 即**按任务类型分流梯度**，需要每条记录带顶层 `task` 字段。
- **动态批**预算：`frames_threshold=10000` / `max_samples=24`；显存不够可 `--checkpoint_every_n_layers 1`（10-22 新增的覆盖参数）。
- **监控口径**：`dmd_loss` 与 `student_loss` **即使健康也不会单调下降**（student 与 critic 互相追逐），应看 `fake_score_loss` 与 `reg_loss` 是否下降，并以生成音频为准 —— 这是一条很实用的排障提示。
- **`attn_backend` 默认 `torch`**，`flash_attn` 需匹配 wheel。

---

## 5. 其他推理相关增量

### 5.1 `Prism`：星标 5.6× 但代码未动

- 窗口内唯一提交：`2026-10-15 Update README.md`（`e5ef67b`），diff 只有一处 —— News 里的占位日期 `[2026-x-xx]` 改为 `[2026-10-06]`。
- 同期 stars 从 **174 → 1,145（+971）**。
- README 现有信息与 10-12 快照一致：4/6/8 步？**不适用** —— Prism 是"2K 联合视频-音频动态稀疏注意力"训练/推理框架，preview checkpoint 在 `huggingface.co/FrancisRing/Prism`（非 `tencent` 组织），推理 720p 需 1×80 GB、1080p/2K 需 ≥4×80 GB。
- **结论**：热度变化**无仓库侧支撑**，不能据此判断"推理能力更新"。若要跟进，应看论文/项目页与第三方复现。

### 5.2 `HYDRA`（补录，NeurIPS 2026，MIT）：统一 tokenizer 的推理代价在"采样步数"

- 结构：`HYDRA-TOK` 单一纯 ViT tokenizer、无量化 —— **Gen-ViT → GSB（压到 64 通道）→ Sem-ViT**；`HYDRA` 在共享 Qwen 主干上双头（AR 文本 + rectified-flow 图像）。
- 权重：`HYDRA-1.5B` / `HYDRA-7B`（Qwen2.5-1.5B/7B-Instruct 基座）+ 两个 tokenizer 变体（`hydra_tok_gan` 用于生成、`hydra_tok_base` 用于重建/训练初始化）+ `hydra_tok_latent_stats.pt`（**不可省**，flow head 消费归一化潜变量）。
- 采样（模型卡表，加粗为非默认）：

| 设置 | HYDRA-1.5B | HYDRA-7B |
|---|---|---|
| `transport.guidance_scale` | 5.0 | 5.0 |
| `transport.time_shifting_factor` | 5.0 | **3.0** |
| `transport.num_inference_steps`（GenEval / WISE） | 50 | **200** |
| `transport.num_inference_steps`（DPG-Bench） | **100** | **100** |

- 工程约束：`transformers` 必须 **4.x（>=4.45,<5）** —— v5 移除了 Qwen2 主干 import 的 `cache_utils.SlidingWindowCache`；T2I 用 `torchrun --nproc_per_node=8` 分片出图。
- arXiv **2603.15228**（v3 更新于 2026-10-09），作者 12 人（Xuerui Qiu 等）；NeurIPS 2026 poster 编号 **149164**（据 NeurIPS 虚拟会场页）。**注意模型卡/仓库的权重在 `huggingface.co/Xuerui123/HYdra`（非 tencent 组织）**。
- ⚠️ **标题两处不一致**：arXiv 版写作《HYDRA: Unifying Multi-modal Generation and Understanding via Representation-Harmonized Tokenization》，仓库描述写作《HYDRA: Representation Harmonized Tokenization for Multimodal Generation and Understanding》。引用前需确认统一口径。
- **窗口内没有新论文/新权重**，发生的是"放代码 + 放权重 + News 补登"（10-14 一次批量 `Upload HYDRA …` 提交）。

### 5.3 `OmniCapBench`（NeurIPS 2026，Apache-2.0）：评测框架自带的"本地推理"要求

- 规模（README）：**786 个密集标注视频**、12.8 小时、20 个一级 / 125 个二级类别；产出 **5,818 引用 / 11,419 镜头 / 39,160 子镜头 / 6,537 音事件**。
- 评测范式：拆成"**确定性验证**（schema/作用域/时间边界/IoU 对齐）→ **局部语义检查**（LLM 只比较一对已对齐字段）"，**不输出聚合总分**。
- 与推理直接相关：本地模型走 **vLLM**（`bash scripts/serve_vllm.sh`，支持 Qwen3-Omni-30B-A3B 的 Instruct/Thinking/Captioner 三版、Qwen2.5-Omni-7B）；judge 端也接受**自托管 vLLM**（任何 OpenAI 兼容端点，含 DashScope/OpenRouter）。脚本支持 `--start/--end` 多机分片与断点续跑。
- arXiv **2610.12458**（2026-10-08）。

### 5.4 组织外但同属腾讯、且直接落在推理主题上（**不计入混元仓库数**）

| 项目 | 事实 | 为什么写进本报告 |
|---|---|---|
| **DSpine**（`tencent/Qwen3-8B-DSpine-b16`，HF 创建 **2026-10-16**；`github.com/tencent/dspine`） | 并行起草的**投机解码** drafter：每个起草层的每个位置接收前一个位置的预测特征，让因果链沿**网络深度**展开而所有位置仍并行起草 | 本窗口内**唯一**新增的投机解码推理工作；已进 **SGLang**（`--speculative-algorithm DSPINE --speculative-num-draft-tokens 16`）。Qwen3-8B 上平均接受长度 **7.83（T=0）/ 6.98（T=1）**，对比 DFlash2 **7.55 / 6.53**、DSpark **7.44 / 6.65**；推理代码 2026-10 发布，arXiv **2609.36173**（2026-09-28，v2 10-10），Apache-2.0；roadmap 未做：训练代码、Qwen3.8-27B 版 |
| **WeVisOne-VL**（`tencent/WeVisOne-VL-4B` / `-2B`，HF 创建 **2026-10-22**） | **微信视觉**团队的统一视觉感知 VLM（Qwen3-VL 基座 + **9.6M 参数**像素解码器，靠特殊 `<OBJ>` token 输出实例级像素结果），统一物体级（开放词表检测/指代定位/计数/pointing/OCR）与像素级（指代分割/alpha matting）；三阶段课程 SFT + 混合任务 RL；论文称 **30+ 基准**领先；推理侧限制：**不做显式 CoT**、**只支持单轮图像**（视频网格会被像素解码器拒绝）。arXiv **2610.31329**（2026-10-22） | 说明同窗口内腾讯其他团队也在发可部署 VLM；**非混元团队**，勿混算 |

> ⚠️ 这两项的团队归属以模型卡/仓库自述为准；DSpine 作者列表（Haohui Zhang 等 7 人）**未在任何一处自述"混元"**，本报告按"腾讯其他团队"处理。

---

## 6. HuggingFace `tencent` 组织窗口内新模型（8 个）

| 日期 | repo id | pipeline | 归组 | 备注 |
|---|---|---|---|---|
| 2026-10-15 | `tencent/Hy-Memory-30B-A3B` | text-generation | 混元 | 记忆写入模型主权重 + 技术报告 |
| 2026-10-16 | `tencent/Hy-Memory-30B-A3B-FP8` | text-generation | 混元 | 面向 vLLM 高效部署 |
| 2026-10-16 | `tencent/Qwen3-8B-DSpine-b16` | — | 腾讯其他团队 | 投机解码 drafter（SGLang） |
| 2026-10-19 | `tencent/Hy-Memory-30B-A3B-GGUF` | text-generation | 混元 | 本地/端侧路线 |
| 2026-10-21 | `tencent/Hy-Nomas` | — | 混元 | 世界模型全部资产（DiT/VAE/PE/LoRA） |
| 2026-10-22 | `tencent/WeVisOne-VL-4B` / `-2B` | image-text-to-text | 微信视觉 | 统一视觉感知 VLM |
| 2026-10-24 | `tencent/Hy-Nomas-Diffusers` | — | 混元 | Diffusers 版打包 |

> 对照：10-12 基线时的最新混元权重是 `Hy4-preview`(-FP8) / `Hy3` / `Hunyuan3D-2.1` 等；**窗口内没有新的旗舰 LLM**（Hy3/Hy4 两个仓库 `pushed_at` 仍停在 2026-07 / 2026-08）。

---

## 7. 公开信息时间线（推理/模型发布侧）

| 日期 | 事件 | 来源类型 | 链接 |
|---|---|---|---|
| 2026-10-14 | `AuK` README News 宣布**蒸馏训练管线开源**（PR #22） | 仓库自述 | <https://github.com/Tencent-Hunyuan/AuK> |
| 2026-10-15 | `Hy-Memory-30B-A3B` 上线 HF；技术报告 arXiv 同日 | HF / arXiv | <https://huggingface.co/tencent/Hy-Memory-30B-A3B> · <https://arxiv.org/abs/2610.22196> |
| 2026-10-16 | 混元宣布 **Hy Memory 2.0**："模型、框架、应用三层全开源"，已在金融/资讯/知识管理/办公 Agent/社交娱乐/游戏等内部业务使用 | 媒体（证券时报转发） | <https://finance.sina.com.cn/jjxw/2026-10-16/doc-inivmiif7708191.shtml> |
| 2026-10-16 | 微信公众号原文《Hy Memory 2.0 发布：模型、框架、应用三层全开源》 | **公众号** | <https://mp.weixin.qq.com/s/TT70I3zzMSLAae5LSPeMzQ>（⚠️ 正文被验证码拦截，仅标题/URL 可证） |
| 2026-10-16 / 10-19 | FP8 / GGUF 量化版上线 | HF | 同上组织页 |
| 2026-10-19 | **UniRL `v0.2.0`** 发布（~350 PR）：rollout 底座升到 **SGLang 0.5.19 / vLLM-Omni 0.28 / torch 2.13 + CUDA 13**，新增直连 vLLM TP 引擎与 FastVideo 引擎 | Release atom | <https://github.com/Tencent-Hunyuan/UniRL/releases/tag/v0.2.0> |
| 2026-10-21 | `Hy-Nomas` 仓库创建、HF 权重上线 | GitHub / HF | <https://github.com/Tencent-Hunyuan/Hy-Nomas> |
| 2026-10-22 | `WeVisOne-VL` 2B/4B 上线（微信视觉）+ 论文 arXiv 2610.31329 | HF / arXiv | <https://huggingface.co/tencent/WeVisOne-VL-4B> · <https://arxiv.org/abs/2610.31329> |
| 2026-10-23 | 中文媒体《腾讯混元发布 Hy-Nomas：支持自由交互与镜头控制的视频世界模型》，转述架构与单卡 4090 方案 | 媒体（中文编述，标注源为 "腾讯混元 模型发布(HF)"） | <https://www.readaitime.com/news/2026-10-23/1m4fpfgc> |
| 2026-10-23 | 腾讯云社区（**社区作者稿，非官方公告**）：Token Plan 比价 —— 通用版 Lite/Standard/Pro/Max **39/99/299/599 元**，**Hy 版 28/78/238/468 元**（只含 Hy3、Hy4 preview）；一账号可同时持有 1 通用 + 1 Hy，共用一个 API Key 与接入地址；**不支持退款、只能升配**；Hy 版暂不支持图片/视频多模态 | 媒体（社区稿） | <https://cloud.tencent.cn/developer/article/2767796> |
| 2026-10-24 | `Hy-Nomas-Diffusers` 上线；Diffusers 集成 PR 合并 | HF / GitHub | <https://huggingface.co/tencent/Hy-Nomas-Diffusers> |
| 2026-10-24 | 日文报道称 Hy-Memory-30B-A3B "長期記憶でOpus 5超え"、無償公開 | 媒体（日） | <https://pc.watch.impress.co.jp/docs/news/2148974.html> |
| 2026-10-25 | 中文自媒体《混元开源新模型，视频可以接着指挥》—— 指出 24 FPS 是播放帧率、单卡能跑≠可顺畅交互、输出是视频而非可编辑 3D | 媒体（自媒体转载） | <https://www.sohu.com/a/1092512937_122990733> |
| — | 英文/土耳其报道《Tencent Unveils Hy-Nomas…》 | 媒体（境外） | <https://en.shiftdelete.net/tencent-unveils-hy-nomas-an-interactive-generative-ai-video-model/> |

> **验证边界**：README 徽章里链到混元的 X 帖（`x.com/TencentHunyuan/status/2113676827114053892`）本机无法取回正文，**不作为证据**；`readaitime`/`sohu` 均为二手转述，架构数字与 HF 模型卡一致，但**不代表官方确认**。

---

## 8. 对照 10-12 报告的三条推理侧判断

| 10-12 判断 | 本轮（10-26）证据 | 结论 |
|---|---|---|
| **① 官方镜像化交付成为新标准，但覆盖不均** | `Hy-Nomas` 直接给预打包镜像 `hynomasoss/hynomasoss:cu130`（含 CUDA 注意力内核），仍保留 `uv sync` 手动路线；`Hy-Memory` 走 PyPI/`pip install hy-memory==2.0.x` + vLLM，无官方镜像 | **成立并扩展** —— 镜像化从 LLM 扩散到**生成模型**；但"镜像 or 手动"二元结构没变，记忆模型侧交付形态是 **PyPI 包**（第三种形态） |
| **② 投机解码（MTP）默认开启** | 窗口内混元**没有**更新 Hy3/Hy4 的部署文档；`Hy-Nomas` 的加速路线是**蒸馏 8 步 + INT8/FP8**，不是投机解码；`Hy-Memory` 用 `no_think` 短输出而非 MTP | **需分线修订** —— "LLM 侧投机解码默认开"继续成立；但**生成模型（视频/世界模型）的默认加速是少步蒸馏 + 量化**，与 LLM 完全不同 |
| **③ 量化工具链统一收敛到 AngelSlim** | 窗口内出现**两套更强的自研配方**：`Hy-Nomas` 的 `convert_dit_w8_bf16.py`（INT8 权重 / BF16 计算 + manifest/SHA256 校验）、`PixelUMM_FT_FP8`；`Hy-Memory` 的 **GGUF imatrix + MoE 混合精度**（6 档、体积最省 8.03 GB 只掉约 1.5 分）与 FP8 版 | **收敛度被削弱** —— 至少三轨：LLM 走 AngelSlim（FP8-static/GPTQ/1.25-bit）、扩散/世界模型走**仓库内自研 W8/FP8**、记忆类模型走**llama.cpp imatrix + MoE 混合精度** |

**本轮新增的一条判断**：

> **④ 世界模型正在复制 LLM 推理栈的成熟套路，但"交互延迟"缺口被显式留白。**
> 分段自回归 + KV cache ≙ 长上下文缓存；8 步蒸馏 ≙ 少步采样；INT8/FP8 + CPU offload ≙ 显存换速度；gateway/dispatcher/GPU pool ≙ 服务化。但官方**不给吞吐/延迟数字**，评测集只覆盖可控性与一致性，第三方报道也把"连续交互的等待时间"列为未验证项。若要在生产里评估，必须自建"指令→下一段可见"的端到端时延基线。

---

## 9. 风险与待核实

| # | 事项 | 现状 | 建议动作 |
|---|---|---|---|
| 1 | `Hy-Nomas` 技术报告 | README/模型卡/徽章三处都写 "Coming Soon"，仓库无 PDF | 等报告；在此之前**不要引用任何质量/速度倍数** |
| 2 | `Hy-Nomas-Bench` 分数 | 只发布评测集与 280 用例，**无模型得分** | 自行跑 `datasets/tencent/Hy-Nomas-Bench` 或等报告 |
| 3 | 单卡 4090 的可用性 | 需 INT8 DiT + FP8 解码 + `sp_size=1` + 18 latent frames 历史预算 + 系统内存兜底；**INT8 loader 不支持推理时挂 LoRA** | 先在 4090 上量"每段生成耗时 vs 段长/分辨率/历史长度"再谈交互 |
| 4 | 24 FPS 的误读 | 模型卡明确是**输出播放帧率**；生成速度依赖硬件/分辨率/段长/历史 | 对外沟通务必区分 fps 与 throughput |
| 5 | `Prism` 星标异常 | +971 但只改了一行 README；preview 权重在个人命名空间 | 视为**传播事件**而非技术更新；跟踪论文与 FrankRing/Prism 权重 |
| 6 | DSpine 团队归属 | 作者与仓库均未自述混元 | 若报告需要归属口径，先向官方确认；本报告按"腾讯其他团队"处理 |
| 7 | 公众号原文 | `mp.weixin.qq.com/s/TT70I3zzMSLAae5LSPeMzQ` 触发验证码；`r.jina.ai` 代理亦失败 | 仅把标题+URL 作为"存在公众号发布"的证据，内容以证券时报/新浪转载为准 |
| 8 | NPU / 国产卡适配 | 仓库侧：Hy-Nomas、Hy-Memory 的 README **均未提及** Ascend/昇腾或任一国产卡。外部侧：vLLM-Ascend 文档站有 `Hy4-preview (Experimental)` 与 `Hunyuan-A13B-Instruct` 教程页，且 `Hy3-preview` 页标注 **"discontinued since vLLM Ascend v0.30.0"** —— **但取不到该变更的日期，不能断言发生在窗口内** | 结论写"仓库未声明 + 外部文档有 experimental 路径"，别把"有教程页"等同于"官方支持"；查 `<https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Hy3-preview.html>` |
| 9 | GitHub API 限流 | 未认证 60 次/小时，本轮中途 403；`ConvertFrom-Json` 在 PS 5.1 亦因深度失败 | 已改走 atom feed + raw + HF API；如需全量元数据建议带 token |
| 10 | ⚠️ **时间错位陷阱**：网传"混元世界模型 1.1（WorldMirror）于 2026-10-22/23 开源" | **实为 2025-10 旧闻**：IT之家原文标注 **2025-10-22**（<https://m.ithome.com/html/891505.htm>），HF `tencent/HunyuanWorld-Mirror` 的 `createdAt` 为 **2025-10-16**；误判源于 2026-10-25 一篇内容农场稿 | **不要写进 2026-10 时间线**；引用前先核 HF `createdAt` 与原文日期 |
| 11 | **HunyuanOCR 版本口径冲突** | 一种说法称存在 `HunyuanOCR-1.5` 仓库；另一种核对显示 `tencent` 组织只有 `tencent/HunyuanOCR`（HF 创建 2025-11-18），社区 `prithivMLmods/HunyuanOCR-1.5-GGUF-Updated` 实测 `createdAt 2026-07-22` 且 `base_model: tencent/HunyuanOCR`——"1.5"只在仓库名里。**窗口内 OCR 线无新版本** | 引用 OCR 版本前先看 HF 仓库创建时间与 base_model 字段 |
| 12 | **没有任何官方吞吐/延迟数字** | 窗口内官方材料**没有** tokens/s、P99、并发数、qps；Hy-Nomas 模型卡明确"架构与采样参数来自配置，**不代表性能评测结果**"，DSpine 只给 acceptance length | 性能结论只能自测；对外沟通区分 fps / acceptance length / throughput |

---

## 10. 证据来源与采集脚本

| 产物 | 位置 | 说明 |
|---|---|---|
| 组织全量快照（90 仓库） | [`../../_research/2026-10-26/org-repos-2026-10-26.csv`](../../_research/2026-10-26/org-repos-2026-10-26.csv) | 含 created/pushed/stars/forks/language/size/license/default_branch |
| 增量表（11 行） | [`../../_research/2026-10-26/delta-2026-10-26.csv`](../../_research/2026-10-26/delta-2026-10-26.csv) | kind = NEW / TOUCHED / MISSED-IN-BASELINE |
| 提交 feed | [`../../_research/2026-10-26/commits/`](../../_research/2026-10-26/commits/) | 11 个仓库的 `commits/<branch>.atom` 原文 |
| HF 模型清单 | [`../../_research/2026-10-26/hf-tencent-models.json`](../../_research/2026-10-26/hf-tencent-models.json) | `author=tencent` 前 100 个模型 |
| 模型卡原文 | [`../../_research/2026-10-26/hf-cards/`](../../_research/2026-10-26/hf-cards/) | Hy-Memory(-FP8)、Hy-Nomas、WeVisOne-VL、DSpine |
| arXiv 元数据 | [`../../_research/2026-10-26/arxiv-batch.xml`](../../_research/2026-10-26/arxiv-batch.xml) | 6 篇论文的 API 原文 |
| README 快照（窗口末） | [`../../_research/2026-10-26/readmes/`](../../_research/2026-10-26/readmes/) | 17 个仓库当前 README |
| README 差异 | [`../../_research/2026-10-26/readme-diff.txt`](../../_research/2026-10-26/readme-diff.txt) | 与 10-12 快照逐行 diff |
| 采集脚本 | `scan_org.py` / `scan_delta_detail.py` / `diff_readmes.py` / `star_delta.py` / `parse_arxiv.py` | 位于同目录，可重跑 |

**关键仓库链接**：<https://github.com/Tencent-Hunyuan/Hy-Nomas> · <https://github.com/Tencent-Hunyuan/Hy-Memory> · <https://github.com/Tencent-Hunyuan/AuK> · <https://github.com/Tencent-Hunyuan/HYDRA> · <https://github.com/Tencent-Hunyuan/OmniCapBench> · <https://huggingface.co/tencent/Hy-Nomas> · <https://huggingface.co/tencent/Hy-Memory-30B-A3B>
