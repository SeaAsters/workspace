# 腾讯混元（Tencent Hunyuan）模型发布库与推理部署路径研究报告

> 范围：10 个已开源仓库 —— `Hy3`、`Hy4-preview`、`Hy3-preview`、`Hy-MT`、`Hy-MT2`、`Hunyuan-A13B`、`Hunyuan-7B`、`Hunyuan-TurboS`、`HunyuanOCR`、`HunyuanVision`。
>
> 核心问题：**每个模型具体怎么跑起来（部署/推理）**。

## 0. 证据来源与阅读约定

本报告严格区分三类证据，正文中对每条关键事实标注来源：

| 标记 | 含义 |
|:---|:---|
| `[README]` | 直接来自仓库 `README.md` 原文（本次已完整读取的本地副本） |
| `[repo-doc]` | 来自同一仓库的子文档，如 `docs/inference/inference.md`、`docs/llama_cpp.md`、`inference/vLLM/serve.sh`、`finetune/README.md`、`rl/README.md` |
| `[model-card]` | 来自 Hugging Face 模型仓库的 `config.json` / `README.md` frontmatter / HF 模型 API 元数据 |

三条硬性约定：

1. **命令与参数一律照抄 README 原文**，包括其中的拼写错误（例如 `Hy-MT` README 中的 `hunyuan/huanyuan_7B`，原文如此，已在正文标注）。
2. **README 未出现的内容写 `README 未说明`**，绝不推测版本号、GPU 数量、吞吐数字或显存收益。
3. 不引用任何 README 之外的 benchmark 分数。

---

## 1. 总览表

| 仓库 | 模型代次 | 参数量（总/激活） | 上下文 | 支持的推理框架 | 量化 | License | 最近更新 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **Hy3** | Hy3（新旗舰 MoE，295B-A21B） | 295B / 21B，另含 **3.8B MTP 层参数**<br>`[model-card]` HF 权重实测 298.79B | 256K | vLLM（源码编译）、SGLang（源码编译） | 原生 BF16；官方另发 **Hy3-FP8**；压缩工具 **AngelSlim** | **Apache License 2.0** | README News 未标日期；仓库最近提交 2026-07-17；HF 权重 2026-07-20 |
| **Hy4-preview** | Hy4 preview（770B-A49B 旗舰 MoE） | 770B / 49B，另含 **1 层原生 MTP**（10B 总参 / 0.7B 激活）<br>`[model-card]` HF 权重实测 779.96B | **1M** | vLLM（官方预构建镜像 `vllm/vllm-openai:hy4-preview`）、SGLang（官方预构建镜像 `lmsysorg/sglang:hy4-preview`） | 原生 BF16；官方另发 **Hy4 preview-FP8**；AngelSlim | **Apache License 2.0** | README News 未标日期；仓库最近提交 2026-08-28；HF 权重 2026-08-28 |
| **Hy3-preview** | Hy3 preview（295B-A21B） | 295B / 21B，另含 **3.8B MTP 层参数**<br>`[model-card]` HF 权重实测 298.79B | 256K | vLLM（源码编译）、SGLang（源码编译） | 原生 BF16；AngelSlim | **Tencent Hy Community License Agreement** | README News **[2026-04-23]**；仓库最近提交 2026-04-23；HF 权重 2026-04-23 |
| **Hy-MT**（HY-MT1.5） | HY-MT1.5 翻译模型族 | 1.8B dense / 7B dense<br>`[model-card]` HF 实测 2.04B / 8.03B | README 未说明（正文只声明"1.8B / 7B 翻译模型"） | **TensorRT-LLM**、**vLLM**、**SGLang**（三者均提供 Docker） | **FP8-static**、**GPTQ Int4**、**GGUF**（llama.cpp）；工具链 **AngelSlim** | **README 未说明**；仓库根目录无 `LICENSE`，HF 模型卡片无 `license` 字段 → **未见任何 License 声明** | README 未标日期；仓库最近提交 2026-06-01；HF 7B 2026-06-01 / 1.8B 2026-01-01 |
| **Hy-MT2** | Hy-MT2 翻译模型族（"fast-thinking"） | 1.8B / 7B dense / **30B-A3B (MoE)**<br>`[model-card]` HF 实测 2.04B / 8.03B / 30.06B | **`max_context: 8192`**（`[README]` "Inference and Deployment" 章节明写） | transformers ≥5.6.0、vLLM（源码编译）、SGLang（源码编译）、**llama.cpp**（依赖 STQ kernel，PR #22836） | **FP8**、**GGUF**、**2bit GGUF**、**1.25bit GGUF**（均由 AngelSlim 产出） | **Apache License 2.0**（仓库 `LICENSE.txt`；HF 卡片 `license: apache-2.0`） | README News **2026.5.21**；仓库最近提交 2026-08-26；HF 30B-A3B 2026-07-21 |
| **Hunyuan-A13B** | Hunyuan-A13B（80B-A13B 细粒度 MoE） | 80B / 13B<br>`[model-card]` HF 权重实测 80.39B | **256K**（natively supports a 256K context window） | **TensorRT-LLM**、**vLLM**（镜像内置 vLLM 0.8.5）、**SGLang**（均有 Docker 镜像） | **FP8-static (W8A8C8)**、**GPTQ Int4 (W4A16)**、INT8 weight-only；工具链 **AngelSlim** | **Tencent Hunyuan Community License Agreement**（`LICENSE` 首行注明 Release Date: June 27, 2025） | README 未标日期；仓库最近提交 2025-07-08；HF 权重 2025-08-21 |
| **Hunyuan-7B** | Hunyuan dense 系列 0.5B / 1.8B / 4B / 7B（含 Pretrain + Instruct） | 0.5B/1.8B/4B/7B dense<br>`[model-card]` Hunyuan-7B-Instruct 实测 7.50B | **256K**（natively supports a 256K context window） | **TensorRT-LLM**、**vLLM**（要求 ≥ v0.10.0）、**SGLang**（均有 Docker 镜像） | **FP8-static (W8A8C8)**、**GPTQ Int4 (W4A16)**、**AWQ Int4 (W4A16)**、INT8 weight-only；工具链 AngelSlim | **Tencent Hunyuan Community License Agreement**（`LICENSE.txt` 首行注明 Tencent Hunyuan 7B Release Date: August 4, 2025） | README 未标日期；仓库最近提交 2025-08-11；HF 权重 2025-09-02 |
| **Hunyuan-TurboS** | Hunyuan-TurboS（Mamba-Transformer 混合 MoE，560B-A56B） | **560B / 56B**，128 层（Mamba2 + Attention + FFN，AMF/MF block pattern） | **256K** | **README 未说明** —— 该仓库是纯技术报告型 README，**没有 Deployment / Quickstart / Inference 章节，也没有权重下载链接** | **README 未说明** | **README 未说明** | README 未标日期；仓库最近提交 2025-05-23 |
| **HunyuanOCR** | HunyuanOCR-1.5（轻量端到端 OCR VLM） | README 未给出参数量<br>`[model-card]` HF 权重实测 **1.12B** | **128K**（训练侧 context window extended to 128K；推理侧 vLLM `--max-model-len` 默认 `131072`） | **vLLM ≥0.25.1**（AR / DFlash 两条路径）、**llama.cpp**（上游 `master`，build **b11103** 起）、**transformers**（原生多卡对齐用） | **GGUF** f16 / bf16（`convert_hf_to_gguf.py`）；README **未提** AngelSlim | **Tencent Hunyuan Community License Agreement**（HF 卡片 `license_name: tencent-hunyuan-community`） | README News **2026/07/24**、**2026/07/13**、**2026/07/07**（1.5 发布）；仓库最近提交 2026-09-24；HF 权重 2026-08-28 |
| **HunyuanVision** | Hunyuan-Vision-1.5（mamba-transformer 混合 VLM） | **开源计划中**：Checkpoints (A56B, 4B) —— 复选框**未勾选** | README 未说明 | **README 未说明** —— Open-source Plan 中 "TRT Inference Support" 与 "VLLM Support" 两项**均未勾选**；当前唯一可用路径是**腾讯云 API** | README 未说明 | README 未说明 | README 未标日期；仓库最近提交 2025-10-21 |

> **关于"最近更新"**：仓库最近提交时间取自 `https://github.com/Tencent-Hunyuan/<repo>/commits/main.atom`；HF 权重时间为 HF 模型 API 的 `lastModified`。二者与 README 内标注的 News 日期并不同步，故三者并列展示，避免混淆。

---

## 2. 推理框架支持矩阵（核心交付物）

图例：✅ = README 明确支持；— = README 未提及或明确不支持；指令与参数**逐字照抄 README**。

| 模型 | **vLLM** | **SGLang** | **TensorRT-LLM** | **llama.cpp** | **其他 / 自研** |
|:---|:---|:---|:---|:---|:---|
| **Hy3** | ✅ 源码编译（`uv pip install --editable . --torch-backend=auto`）<br>`vllm serve tencent/Hy3 --tensor-parallel-size 8 --speculative-config.method mtp --speculative-config.num_speculative_tokens 2 --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice --port 8000 --served-model-name hy3` | ✅ 源码编译（`pip3 install "transformers>=5.6.0"` + `pip3 install -e "python"`）<br>`python3 -m sglang.launch_server --model tencent/Hy3 --tp-size 8 --tool-call-parser hunyuan --reasoning-parser hunyuan --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3 --speculative-algorithm EAGLE --port 8000 --served-model-name hy3` | — | — | transformers 直跑：README 未给代码，只给 OpenAI 兼容 API 调用<br>量化工具链：**AngelSlim**<br>官方 recipe 外链：vLLM recipes / SGLang cookbook<br>RL rollout 引擎：vLLM（verl + Megatron-LM） |
| **Hy4 preview** | ✅ **官方预构建镜像** `vllm/vllm-openai:hy4-preview`<br>`docker run --gpus all -p 8000:8000 --ipc=host -v ~/.cache/huggingface:/root/.cache/huggingface vllm/vllm-openai:hy4-preview tencent/Hy4-preview-FP8 --tensor-parallel-size 8 --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' --attention-backend FLASHMLA_SPARSE --tool-call-parser hy_v4 --reasoning-parser hy_v4 --enable-auto-tool-choice --port 8000 --served-model-name hy4-preview` | ✅ **官方预构建镜像** `lmsysorg/sglang:hy4-preview`（multi-arch, x86 and Arm）<br>`docker run --gpus all --ipc=host -p 8000:8000 lmsysorg/sglang:hy4-preview python3 -m sglang.launch_server --model tencent/Hy4-preview-FP8 --tp-size 8 --reasoning-parser auto --tool-call-parser auto --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --port 8000 --served-model-name hy4-preview` | — | — | transformers 直跑：README 未给代码<br>量化工具链：**AngelSlim** |
| **Hy3 preview** | ✅ 源码编译<br>`vllm serve tencent/Hy3-preview --tensor-parallel-size 8 --speculative-config.method mtp --speculative-config.num_speculative_tokens 1 --tool-call-parser hy_v3 --reasoning-parser hy_v3 --enable-auto-tool-choice --served-model-name hy3-preview` | ✅ 源码编译<br>`python3 -m sglang.launch_server --model tencent/Hy3-preview --tp 8 --tool-call-parser hunyuan --reasoning-parser hunyuan --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --speculative-algorithm EAGLE --served-model-name hy3-preview` | — | — | transformers 直跑：README 未给代码<br>训练：DeepSpeed ZeRO + LLaMA-Factory（见 `train/README.md`）<br>量化工具链：**AngelSlim** |
| **HY-MT1.5**（Hy-MT） | ✅ **要求 vLLM ≥ v0.10.0**<br>`python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --trust-remote-code --model ${MODEL_PATH} --tensor-parallel-size 1 --dtype bfloat16 --quantization experts_int8 --served-model-name hunyuan` | ✅ Docker 镜像 `lmsysorg/sglang:latest`<br>`docker run --entrypoint="python3" --gpus all --shm-size 32g -p 30000:30000 --ulimit nproc=10000 --privileged --ipc=host lmsysorg/sglang:latest -m sglang.launch_server --model-path hunyuan/huanyuan_7B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000`（原文如此，`huanyuan_7B` 为 README 拼写） | ✅ Docker 镜像 `docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm` / `hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm`<br>`trtllm-serve /path/to/HunYuan-7b --host localhost --port 8000 --backend pytorch --max_batch_size 32 --max_num_tokens 16384 --tp_size 2 --kv_cache_free_gpu_memory_fraction 0.6 --trust_remote_code --extra_llm_api_options /path/to/extra-llm-api-config.yml` | ✅ **权重层面支持**：官方发布 `HY-MT1.5-1.8B-GGUF`、`HY-MT1.5-7B-GGUF`（描述为 "llama.cpp"）<br>**但 README 未给任何 llama.cpp 构建或启动命令** | transformers：**推荐 v4.56.0**（`pip install transformers==4.56.0`）<br>FP8 权重用 transformers 加载需改 `config.json` 中 `ignored_layers`→`ignore` 并升级 `compressed-tensors-0.11.0`<br>训练：LLaMA-Factory |
| **Hy-MT2** | ✅ 源码编译<br>`vllm serve tencent/Hy-MT2-30B-A3B --tensor-parallel-size 1` | ✅ 源码编译<br>`python3 -m sglang.launch_server --model tencent/Hy-MT2-30B-A3B --tp 1` | — | ✅ **需自研 STQ kernel**（README 原文：**"This gguf depends on our STQ kernel, which is released at PR #22836"**）<br>构建：`cmake -B build` + `cmake --build build --config Release`<br>推理：`./build/bin/llama-completion --model model.gguf -p "Translate the following segment into Chinese, without additional explanation：Hello" --jinja -ngl 0 -n 64 -st`<br>压测：`./build/bin/llama-bench -m model_zoo/model.gguf -ngl 0` | transformers **≥5.6.0**<br>`AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)`<br>量化工具链：**AngelSlim** |
| **Hunyuan-A13B** | ✅ 官方镜像内置 **vLLM 0.8.5**（`docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-vllm` 或 `hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm`，**要求 CUDA 12.8**）<br>`-m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --tensor-parallel-size 4 --model tencent/Hunyuan-A13B-Instruct --trust-remote-code` | ✅ 镜像 `docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang` / `hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-sglang`<br>`-m sglang.launch_server --model-path hunyuan/huanyuan_A13B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000`（原文如此） | ✅ 镜像 `hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm`<br>`trtllm-serve /path/to/HunYuan-moe-A13B --host localhost --port 8000 --backend pytorch --max_batch_size 32 --max_num_tokens 16384 --tp_size 2 --kv_cache_free_gpu_memory_fraction 0.6 --trust_remote_code --extra_llm_api_options /path/to/extra-llm-api-config.yml` | — | transformers 直跑（`MODEL_PATH` 环境变量 + `trust_remote_code=True`）<br>量化部署：`sh run_server_int8.sh` / `run_server_int4.sh` / `run_server_fp8.sh`，请求脚本 `sh openapi.sh`<br>量化工具链：**AngelSlim** |
| **Hunyuan-7B** | ✅ **要求 vLLM ≥ v0.10.0**<br>`python3 -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 --trust-remote-code --model ${MODEL_PATH} --tensor-parallel-size 1 --dtype bfloat16 --quantization experts_int8 --served-model-name hunyuan` | ✅ 镜像 `lmsysorg/sglang:latest`<br>`docker run --entrypoint="python3" --gpus all --shm-size 32g -p 30000:30000 --ulimit nproc=10000 --privileged --ipc=host lmsysorg/sglang:latest -m sglang.launch_server --model-path hunyuan/huanyuan_7B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000` | ✅ 镜像 `docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm` / `hunyuaninfer/hunyuan-7B:hunyuan-7b-trtllm`<br>`trtllm-serve /path/to/HunYuan-7b --host localhost --port 8000 --backend pytorch --max_batch_size 32 --max_num_tokens 16384 --tp_size 2 --kv_cache_free_gpu_memory_fraction 0.6 --trust_remote_code --extra_llm_api_options /path/to/extra-llm-api-config.yml` | — | transformers（指定 commit `4970b23cedaf745f963779b4eae68da281e8c6ca`）<br>量化部署：`--quantization experts_int8` / `--quantization gptq_marlin` / `--kv-cache-dtype fp8`<br>量化工具链：**AngelSlim** |
| **Hunyuan-TurboS** | — | — | — | — | — （README 无任何部署章节；仅 arXiv 论文与 LMSYS Arena 结果） |
| **HunyuanOCR-1.5** | ✅ **`vllm>=0.25.1`，统一推理环境要求 CUDA 13**<br>AR：`MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh` → 内部为 `vllm serve "${MODEL_PATH}" --served-model-name "${SERVED_NAME}" -tp 1 --limit-mm-per-prompt '{"image":4,"video":0}' --trust-remote-code --port "${PORT}" --gpu-memory-utilization "${GPU_MEM_UTIL}" --max-model-len "${MAX_MODEL_LEN}" --max-num-batched-tokens "${MAX_MODEL_LEN}"`<br>DFlash：`bash inference/DFlash/serve_DFlash.sh` → 追加 `--speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_PATH}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"` | — | — | ✅ **上游已合并，无需 fork**：PR **#28890**，build **b11103** 及以后<br>`cmake -B build -DLLAMA_BUILD_EXAMPLES=ON` + `cmake --build ./build --config Release -j`<br>`build/bin/llama-server --model ./HunyuanOCR/hyocr-f16.gguf --mmproj ./HunyuanOCR/mmproj-hyocr-f16.gguf --host 0.0.0.0 --port 8080 --alias HYVL --ctx-size 10240 --n-predict 4096 -fa on --jinja`<br>DFlash 追加：`--spec-draft-model ./HunyuanOCR/hyocr-dflash-bf16.gguf --spec-type draft-dflash --spec-draft-n-max 15 --parallel 1` | transformers（原生多卡对齐）：`python inference/transformers/infer_hf_8gpu.py --model ./HunyuanOCR --attn-implementation flash_attention_2 --input ./input.jsonl --output ./results/hf_out --gpu-ids 0,1,2,3,4,5,6,7 --max-new-tokens 32768 --merge`<br>旧独立环境版本：vLLM 0.18.1 (release) / CUDA 12.x；vLLM nightly / CUDA 13；transformers 5.13.0 |
| **Hunyuan-Vision-1.5** | ❌ **未支持**：Open-source Plan 中 "VLLM Support" 复选框未勾选 | — | ❌ **未支持**：Open-source Plan 中 "TRT Inference Support" 复选框未勾选 | — | 唯一可用路径：**腾讯云 API**（`https://api.hunyuan.cloud.tencent.com/v1`，模型名 `hunyuan-t1-vision-20250916`），以及 LMArena Direct Chat<br>权重（A56B / 4B）与 checkpoint 均**尚未发布** |

### 2.1 矩阵读数摘要

- **官方一等公民是 vLLM + SGLang**：10 个仓库中 7 个同时给出两者配方，这是混元开源的默认双栈。
- **给 docker 镜像的只有 4 条线**：`Hy4 preview`（vLLM + SGLang 双官方镜像）、`Hunyuan-A13B`、`Hunyuan-7B`、`HY-MT1.5`（后三者均为 TRT-LLM + vLLM + SGLang 三镜像）。
- **TensorRT-LLM 只覆盖 3 条老产品线**：`Hunyuan-A13B`、`Hunyuan-7B`、`HY-MT1.5`。四代旗舰（Hy3 / Hy3 preview / Hy4 preview）与 Hy-MT2、HunyuanOCR **均无 TRT-LLM 路径**。
- **Hy3 与 Hy3 preview 是"源码编译"路线**：README 明确要求 `git clone` + `uv pip install --editable .`（vLLM）或 `pip3 install -e "python"`（SGLang），没有官方镜像。
- **`llama.cpp` 出现在 3 条线上**，但成熟度差异极大：`HunyuanOCR-1.5` 已**上游合并**（b11103+，可用 master）；`Hy-MT2` **依赖尚未合并的自研 STQ kernel**（PR #22836）；`HY-MT1.5` **只发了 GGUF 权重，没给任何命令**。
- **`Hy4 preview` 是唯一带专用 attention backend 参数的模型**：`--attention-backend FLASHMLA_SPARSE`，对应其 Gated DSA 稀疏注意力架构。
- **AngelSlim 是混元事实上的量化标准工具链**：Hy3 / Hy4 preview / Hy3 preview / HY-MT1.5 / Hy-MT2 / Hunyuan-A13B / Hunyuan-7B 共 7 个仓库在量化章节点名 AngelSlim；HunyuanOCR 与 Hunyuan-TurboS 未提。

---

## 3. 逐个模型的部署指南摘录

### 3.1 Hy3（`Tencent-Hunyuan/Hy3`）

#### 模型规格 `[README]`

| Property | Value |
|:---|:---|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | 295B |
| Activated Parameters | 21B |
| MTP Layer Parameters | 3.8B |
| Number of Layers (excluding MTP layer) | 80 |
| Number of MTP Layers | 1 |
| Attention Heads | 64 (GQA, 8 KV heads, head dim 128) |
| Hidden Size | 4096 |
| Intermediate Size | 13312 |
| Context Length | 256K |
| Vocabulary Size | 120832 |
| Number of Experts | 192 experts, top-8 activated |
| Supported Precisions | BF16 |

注意力类型：**GQA**（64 heads / 8 KV heads / head dim 128）。

#### vLLM 启动命令（原文照抄）`[README]`

先编译 vLLM：

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install --editable . --torch-backend=auto
```

再启动服务（**开启 MTP**）：

```bash
# Switch to trtllm backend to work-around mnnvl workspace size issue.
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
vllm serve tencent/Hy3 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --port 8000 \
  --served-model-name hy3
```

#### SGLang 启动命令（原文照抄）`[README]`

```bash
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3 \
  --tp-size 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --speculative-algorithm EAGLE \
  --port 8000 \
  --served-model-name hy3
```

#### 量化 `[README]`

README 未给出量化启动参数或显存收益数字，只声明：

> We provide [AngelSlim](https://github.com/tencent/AngelSlim), a more accessible, comprehensive, and efficient toolkit for large model compression. AngelSlim supports a comprehensive suite of compression tools for large-scale multimodal models, including common quantization algorithms, low-bit quantization, and speculative sampling.

官方同时发布 **Hy3-FP8** 权重（FP8 quantized instruct model）。

#### 硬件需求与吞吐 `[README]` `[repo-doc]`

- `[README]`：**"Hy3 has 295B parameters in total. To serve it on 8 GPUs, we recommend using H20-3e or other GPUs with larger memory capacity."** → 最小推荐配置 = **8 卡**（H20-3e 或显存更大的 GPU），对应 `--tensor-parallel-size 8`。
- `[repo-doc]` `rl/README.md`：**"on H20, rollout (vLLM inference) requires at least TP=16 (the script defaults to `ROLLOUT_TP=16`) — the weights of a single instance only fit when sharded across 16 GPUs."** → 在 H20 上做 RL rollout 时，**vLLM 推理需要 TP=16**。
- 吞吐数字（TPS / latency）：**README 未说明**。

#### 调用方式 `[README]`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="hy3",
    messages=[
        {"role": "user", "content": "Hello! Can you briefly introduce yourself?"},
    ],
    temperature=0.9,
    top_p=1.0,
    # reasoning_effort: "no_think" (default, direct response), "low", "high" (deep chain-of-thought)
    extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}},
)
print(response.choices[0].message.content)
```

推荐参数：`temperature=0.9`、`top_p=1.0`；`reasoning_effort` 支持 `"no_think"`（默认）/ `"low"` / `"high"`。

#### 生态链 `[README]`

- Finetuning 指南：`./finetune/README.md`
- RL 后训练：GRPO + [verl](https://github.com/volcengine/verl)，训练于 Megatron-LM（模型转换走 NVIDIA Megatron-Bridge），**rollout 用 vLLM**
- 官方 recipe 外链：[vLLM recipes](https://recipes.vllm.ai/tencent/Hy3)、[SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Tencent/Hy3)

#### 权重下载 `[README]`

| Model Name | Description | Hugging Face | ModelScope | GitCode | CNB |
|:---|:---|:---:|:---:|:---:|:---:|
| Hy3 | Instruct model | [Model](https://huggingface.co/tencent/Hy3) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy3) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy3) | [Model](https://cnb.cool/ai-models/tencent/Hy3) |
| Hy3-FP8 | FP8 quantized instruct model | [Model](https://huggingface.co/tencent/Hy3-FP8) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy3-FP8) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy3-FP8) | [Model](https://cnb.cool/ai-models/tencent/Hy3-FP8) |

#### License `[README]`

> Hy3 is released under the **Apache License 2.0**.

---

### 3.2 Hy4 preview（`Tencent-Hunyuan/Hy4-preview`）

#### 模型规格 `[README]`

README 明确注明：**下表只列 backbone 参数，不含 MTP 层**。

| Property | Value |
|:---|:---|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | 770B |
| Activated Parameters | 49B |
| Layers | 78 |
| Hidden Size | 6144 |
| Attention Type | **Gated DSA** |
| Attention Heads | 64 |
| Query Compression Dimension | 2048 |
| Key-Value Compression Dimension | 512 |
| Indexer Heads / Head Dimension | 32 / 128 |
| Indexer top-k | 2048 |
| Residual Streams | 4 |
| Routed Experts | 256 |
| Shared Experts | 1 |
| Activated Routed Experts per Token | 8 |
| MoE Intermediate Size | 2048 |
| FFN Intermediate Size | 18432 |
| Context Length | 1M |
| Vocabulary Size | 120832 |

README 正文补充（原文）：

> The backbone consists of 78 layers, where the first layer uses a standard dense FFN and the remaining 77 layers replace it with MoE, each containing 256 routed experts and 1 shared expert; every token activates the top-8 routed experts along with the shared expert. In addition to the backbone, **1 native MTP layer (10B total parameters, 0.7B activated) is built in for speculative decoding**.

架构灵感来源（原文）：`Gated [DeepSeek Sparse Attention](https://arxiv.org/abs/2512.02556) (Gated DSA) with [IndexCache](https://arxiv.org/abs/2603.12201) for cross-layer sparse index reuse`；残差路径使用 `iHC (identity Hyper-Connections)`。

#### vLLM 启动命令（原文照抄）`[README]`

> Use official prebuilt image `vllm/vllm-openai:hy4-preview`:

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:hy4-preview tencent/Hy4-preview-FP8 \
    --tensor-parallel-size 8 \
    --speculative-config '{"num_speculative_tokens":3,"method":"mtp"}' \
    --attention-backend FLASHMLA_SPARSE \
    --tool-call-parser hy_v4 \
    --reasoning-parser hy_v4 \
    --enable-auto-tool-choice \
    --port 8000 \
    --served-model-name hy4-preview
```

注意 README 的示例权重是 **`tencent/Hy4-preview-FP8`**（FP8 版），不是 BF16 主权重。

#### SGLang 启动命令（原文照抄）`[README]`

> Use the official prebuilt image `lmsysorg/sglang:hy4-preview` (multi-arch, x86 and Arm):

```bash
docker pull lmsysorg/sglang:hy4-preview

docker run --gpus all --ipc=host -p 8000:8000 lmsysorg/sglang:hy4-preview \
  python3 -m sglang.launch_server \
    --model tencent/Hy4-preview-FP8 \
    --tp-size 8 \
    --reasoning-parser auto \
    --tool-call-parser auto \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --port 8000 \
    --served-model-name hy4-preview
```

#### 量化 `[README]`

- 官方发布 **Hy4 preview-FP8**（FP8 quantized instruct model）。
- 工具链：**AngelSlim**（与 Hy3 同一段描述文字）。
- 具体量化配方 / 显存收益数字：**README 未说明**。

#### 硬件需求与吞吐 `[README]` `[repo-doc]`

- 推理：`[README]` **未说明**最小 GPU 数或显存；从命令可读出 `--tensor-parallel-size 8` / `--tp-size 8` = **8 卡**。
- 吞吐数字：**README 未说明**。
- 微调硬件（`[repo-doc]` `finetune/README.md`，可与推理规模作数量级参照）：
  - **LoRA Fine-tuning**：At least 8 machines with 64 GPUs (at least 96GB GPU memory per GPU, at least 2TB CPU memory per machine).
  - **Full Fine-tuning**：At least 16 machines with 128 GPUs (at least 96GB GPU memory per GPU, at least 2TB CPU memory per machine).

#### 调用方式 `[README]`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="hy4-preview",
    messages=[
        {"role": "user", "content": "Hello! Can you briefly introduce yourself?"},
    ],
    temperature=0.9,
    top_p=1.0,
)
print(response.choices[0].message.content)
```

推荐参数：`temperature=0.9`、`top_p=1.0`。**推理模式默认 `"high"`**（deep chain-of-thought），若需直答需显式传 `extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}}`。

#### 已知限制 `[README]`

> This is an early version of Hy4. There is real headroom left in both pre-training and post-training, and we are shipping with known issues — among them, **spending longer than necessary reasoning through complex tasks**, and **a tendency to over-verify its own work**.

#### 权重下载 `[README]`

| Model Name | Description | Hugging Face | ModelScope | GitCode | CNB |
|:---|:---|:---:|:---:|:---:|:---:|
| Hy4 preview | Instruct model | [Model](https://huggingface.co/tencent/Hy4-preview) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview) | [Model](https://cnb.cool/ai-models/tencent/Hy4-preview) |
| Hy4 preview-FP8 | FP8 quantized instruct model | [Model](https://huggingface.co/tencent/Hy4-preview-FP8) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy4-preview-FP8) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy4-preview-FP8) | [Model](https://cnb.cool/ai-models/tencent/Hy4-preview-FP8) |

#### License `[README]`

> Hy4 preview is released under the **Apache License 2.0**.

---

### 3.3 Hy3 preview（`Tencent-Hunyuan/Hy3-preview`）

#### 模型规格 `[README]`

与 Hy3 完全一致的架构表（295B / 21B / 3.8B MTP / 80 层 / 1 MTP 层 / 64 heads GQA 8 KV heads head dim 128 / hidden 4096 / intermediate 13312 / context 256K / vocab 120832 / 192 experts top-8 activated / Supported Precisions BF16）。

#### vLLM 启动命令（原文照抄）`[README]`

```bash
# 编译
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install --editable . --torch-backend=auto
```

```bash
vllm serve tencent/Hy3-preview \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 1 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --served-model-name hy3-preview
```

#### SGLang 启动命令（原文照抄）`[README]`

```bash
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

```bash
python3 -m sglang.launch_server \
  --model tencent/Hy3-preview \
  --tp 8 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --speculative-algorithm EAGLE \
  --served-model-name hy3-preview
```

#### 量化 `[README]`

AngelSlim 段落与 Hy3 相同；具体配方与显存数字 **README 未说明**。

#### 硬件需求与吞吐 `[README]`

> Hy3-preview has 295B parameters in total. **To serve it on 8 GPUs, we recommend using H20-3e or other GPUs with larger memory capacity.**

并发/吞吐/Latency：**README 未说明**。

#### 权重下载 `[README]`

| Model Name | Description | Hugging Face | ModelScope | GitCode |
|:---|:---|:---:|:---:|:---:|
| Hy3 preview | Instruct model | [Model](https://huggingface.co/tencent/Hy3-preview) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy3-preview) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy3-preview) |
| Hy3 preview-Base | Pre-trained base model | [Model](https://huggingface.co/tencent/Hy3-preview-Base) | [Model](https://modelscope.cn/models/Tencent-Hunyuan/Hy3-preview-Base) | [Model](https://ai.gitcode.com/tencent_hunyuan/Hy3-preview-Base) |

（Hy3 preview 未提供 CNB 通道。）

#### License `[README]`

> Hy3 preview is released under the **Tencent Hy Community License Agreement**.

---

### 3.4 HY-MT1.5（`Tencent-Hunyuan/HY-MT`）

#### 模型规格 `[README]`

- **HY-MT1.5-1.8B**：1.8B 翻译模型
- **HY-MT1.5-7B**：7B 翻译模型，`WMT25 championship model` 的升级版
- 支持 **33 种语言互译 + 5 种民族/方言变体**（README 语言表共 38 行，含 `zh-Hant`、`bo` 藏语、`kk` 哈萨克语、`mn` 蒙古语、`ug` 维吾尔语、`yue` 粤语）
- 新能力：terminology intervention（术语干预）、contextual translation（上下文翻译）、formatted translation（格式翻译）
- `[model-card]` `config.json`：两模型均 `model_type = hunyuan_v1_dense`，32 层，`max_position_embeddings = 262144`；1.8B → heads 16 / KV heads 4 / hidden 2048 / FFN 6144；7B → heads 32 / KV heads 8 / hidden 4096 / FFN 14336。**注意力为 GQA**（KV heads < Q heads）
- 层数 / 注意力类型 / MTP：**README 未说明**（上一条来自 model-card，非 README）
- 上下文：**README 未说明**（README 未声明 context length）

#### 部署前情 `[README]`

> For deployment, you can use frameworks such as **TensorRT-LLM**, **vLLM**, or **SGLang** to serve the model and create an OpenAI-compatible API endpoint.

镜像页：`https://hub.docker.com/r/hunyuaninfer/hunyuan-7B/tags`

#### TensorRT-LLM 启动命令（原文照抄）`[README]`

```bash
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
```

```bash
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7b:hunyuan-7b-trtllm
```

配置文件：

```bash
cat >/path/to/extra-llm-api-config.yml <<EOF
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
- 1
- 2
- 4
- 8
- 16
- 32
print_iter_log: true
EOF
```

启动：

```bash
trtllm-serve \
  /path/to/HunYuan-7b \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --max_num_tokens 16384 \
  --tp_size 2 \
  --kv_cache_free_gpu_memory_fraction 0.6 \
  --trust_remote_code \
  --extra_llm_api_options /path/to/extra-llm-api-config.yml
```

#### vLLM 启动命令（原文照抄）`[README]`

> **Please use vLLM version v0.10.0 or higher for inference.**

依赖（README 原文说明：*We will merge it into the main branch later*）：

```SHELL
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

模型路径：

```shell
export MODEL_PATH=tencent/Hunyuan-7B-MT
```

```shell
export MODEL_PATH=/root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-MT/
```

启动 API server：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --quantization experts_int8 \
    --served-model-name hunyuan \
    2>&1 | tee log_server.txt
```

请求示例：

```shell
curl http://0.0.0.0:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
"model": "hunyuan",
"messages": [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "请按面积大小对四大洋进行排序，并给出面积最小的洋是哪一个？直接输出结果。"}]
    }
],
"max_tokens": 2048,
"temperature":0.7,
"top_p": 0.6,
"top_k": 20,
"repetition_penalty": 1.05,
"stop_token_ids": [127960]
}'
```

#### SGLang 启动命令（原文照抄）`[README]`

```bash
docker pull lmsysorg/sglang:latest
```

```bash
docker run --entrypoint="python3" --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ulimit nproc=10000 \
    --privileged \
    --ipc=host \
     lmsysorg/sglang:latest \
    -m sglang.launch_server --model-path hunyuan/huanyuan_7B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000
```

> ⚠️ 注：`hunyuan/huanyuan_7B` 与 `--tp 4` 均为 README 原文（含拼写），本报告照抄不改。

#### 量化 `[README]`

README 的 "Quantization Compression" 章节原文：

> We used our own [AngelSlim](https://github.com/tencent/AngelSlim) compression tool to produce FP8 and INT4 quantization models.

**FP8 Quantization** 原文要点：**FP8-static** 量化，8-bit 浮点格式，通过少量 calibration data（**without training**）预先确定 quantization scale，权重与激活值都转为 FP8。

README 同时给出三档量化模型的 vLLM 启动参数：

**Int8 weight-only**：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --quantization experts_int8 \
    2>&1 | tee log_server.txt
```

**Int4（GPTQ）** ：

```shell
export MODEL_PATH=PATH_TO_INT4_MODEL
```

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --quantization gptq_marlin \
    2>&1 | tee log_server.txt
```

**FP8（W8A8C8）** ：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --kv-cache-dtype fp8 \
    2>&1 | tee log_server.txt
```

**显存收益数字：README 未说明**（README 只做定性表述 "reduce the deployment threshold"）。

transformers 加载 FP8 权重的注意事项（原文）：需要把 `config.json` 中的 `"ignored_layers"` 改名为 `"ignore"`，并把 `compressed-tensors` 升级到 `compressed-tensors-0.11.0`。

#### 硬件需求与吞吐 `[README]`

- 最小 GPU 数 / 显存：**README 未说明**（命令中出现的 `--tp_size 2`、`--tp 4`、`--tensor-parallel-size 1` 是示例值，README 未声明其为最小配置要求）。
- 端侧：README 声明 **HY-MT1.5-1.8B 量化后可在边缘设备部署并支持实时翻译**，原文：*"After quantization, the 1.8B model can be deployed on edge devices and support real-time translation scenarios"*。但**未给出具体显存/存储数字**。
- 吞吐 / latency / TPS：**README 未说明**。

#### 推荐推理参数 `[README]`

```json
{
  "top_k": 20,
  "top_p": 0.6,
  "repetition_penalty": 1.05,
  "temperature": 0.7
}
```

> 注意：*our model does not have the default system_prompt*。

#### 权重下载 `[README]`

| Model Name | Description | Download |
|:---|:---|:---|
| HY-MT1.5-1.8B | Hunyuan 1.8B translation model | [Model](https://huggingface.co/tencent/HY-MT1.5-1.8B) |
| HY-MT1.5-1.8B-FP8 | Hunyuan 1.8B translation model, fp8 quant | [Model](https://huggingface.co/tencent/HY-MT1.5-1.8B-FP8) |
| HY-MT1.5-1.8B-GPTQ-Int4 | Hunyuan 1.8B translation model, int4 quant | [Model](https://huggingface.co/tencent/HY-MT1.5-1.8B-GPTQ-Int4) |
| HY-MT1.5-1.8B-GGUF | Hunyuan 1.8B translation model, llama.cpp | [Model](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF) |
| HY-MT1.5-7B | Hunyuan 7B translation model | [Model](https://huggingface.co/tencent/HY-MT1.5-7B) |
| HY-MT1.5-7B-FP8 | Hunyuan 7B translation model, fp8 quant | [Model](https://huggingface.co/tencent/HY-MT1.5-7B-FP8) |
| HY-MT1.5-7B-GGUF | Hunyuan 7B translation model, llama.cpp | [Model](https://huggingface.co/tencent/HY-MT1.5-7B-GGUF) |

ModelScope 通道：`modelscope download --model Tencent-Hunyuan/Hunyuan-7B-MT`（README 中 vLLM 示例所用）；集合页 `https://modelscope.cn/collections/Tencent-Hunyuan/HY-MT15`。

#### License

**未见任何 License 声明**。已验证：
- `[README]`：HY-MT README **没有 License 章节**；
- `[model-card]`：HF 上 `tencent/HY-MT1.5-1.8B` / `tencent/HY-MT1.5-7B` 的卡片 frontmatter **无 `license` 字段**，HF 模型 API 的 tags 中也**不含 `license:*`**；
- 仓库 `main` / `master` 分支的 `LICENSE`、`LICENSE.txt` 路径均返回 404。

→ 结论：**HY-MT1.5 系列的 License 在公开可见范围内未说明，商用前需向 `hunyuan_opensource@tencent.com` 确认。**

---

### 3.5 Hy-MT2（`Tencent-Hunyuan/Hy-MT2`）

#### 模型规格 `[README]` `[model-card]`

- 三个尺寸：**1.8B / 7B / 30B-A3B (MoE)**
- 支持 **33 种语言**互译，且能跟随多语种翻译指令
- **`max_context: 8192`**（README "Inference and Deployment" 章节明写）
- `[model-card]` config：
  - `Hy-MT2-1.8B`：`hunyuan_v1_dense`，32 层，heads 16 / KV 4，hidden 2048，FFN 6144，`max_position_embeddings 262144`
  - `Hy-MT2-7B`：`hunyuan_v1_dense`，32 层，heads 32 / KV 8，hidden 4096，FFN 14336，`max_position_embeddings 262144`
  - `Hy-MT2-30B-A3B`：`HYV3ForCausalLM` / `model_type: hy_v3`，**48 层**，heads 32 / KV 4，head_dim 128，hidden 2048，**128 experts / top-8 activated + 1 shared expert**，`moe_intermediate_size 768`，`first_k_dense_replace 1`，`max_position_embeddings 262144`，`qk_norm: true`，`rope_theta 11158840.0`
- MTP / speculative decoding 层：**README 未说明**
- 注意力类型：README 未说明；由 config 可读为 GQA（KV heads < Q heads），但**README 未声明**

#### 推荐推理参数 `[README]`

1.8B / 7B：

```json
{
  "temperature": 0.7,
  "top_p": 0.6,
  "top_k": 20,
  "repetition_penalty": 1.05,
  "max_tokens": 4096
}
```

30B-A3B：

```json
{
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": -1,
  "repetition_penalty": 1.0,
  "max_tokens": 4096
}
```

> 两个模型族均 *do not have a default system_prompt*。

#### transformers `[README]`

> transformers>=5.6.0

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "tencent/Hy-MT2-30B-A3B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

# Example inference
prompt = "将以下文本翻译成英语,注意只需要输出翻译后的结果,不要额外解释:\n\n今天天气真好。"
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
    )
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

#### vLLM 启动命令（原文照抄）`[README]`

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install --editable . --torch-backend=auto
```

```bash
vllm serve tencent/Hy-MT2-30B-A3B --tensor-parallel-size 1
```

#### SGLang 启动命令（原文照抄）`[README]`

```bash
git clone https://github.com/sgl-project/sglang
cd sglang
pip3 install pip --upgrade
pip3 install "transformers>=5.6.0"
pip3 install -e "python"
```

```bash
python3 -m sglang.launch_server --model tencent/Hy-MT2-30B-A3B --tp 1
```

#### llama.cpp（原文照抄）`[README]`

> **❕❕ This gguf depends on our STQ kernel, which is released at [PR #22836](https://github.com/ggml-org/llama.cpp/pull/22836).**

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

```bash
cd llama.cpp
```

```bash
cmake -B build
cmake --build build --config Release
```

```bash
./build/bin/llama-completion \
  --model model.gguf  \
  -p "Translate the following segment into Chinese, without additional explanation：Hello" \
  --jinja \
  -ngl 0 \
  -n 64 -st 
```

benchmark：

```bash
./build/bin/llama-bench -m model_zoo/model.gguf  -ngl 0
```

> 注意 `-ngl 0` 表示 **0 层 offload 到 GPU**（即纯 CPU 路径），这是 README 原文给出的示例参数。

#### 量化 `[README]`

> We provide [AngelSlim](https://github.com/tencent/AngelSlim), an easy-to-use, comprehensive, and efficient large model compression toolkit covering common quantization algorithms, low-bit quantization, speculative sampling, and more.

**显存/存储收益数字（README 首页原文）**：

> For on-device deployment, **AngelSlim 1.25-bit extreme quantization reduces the storage requirement of the 1.8B model to only 440 MB and improves inference speed by 1.5x.**

这是本次调研范围内**唯一给出明确端侧量化收益数字**的模型族。

#### 硬件需求与吞吐 `[README]`

- 最小 GPU 数 / 显存：**README 未说明**（`--tensor-parallel-size 1` / `--tp 1` 是示例值）。
- 吞吐 / latency / TPS：**README 未说明**（仅有 1.25-bit 量化的 "improves inference speed by 1.5x" 相对值）。
- 端侧：1.25-bit 量化后 **440 MB**，面向 on-device 场景。

#### 权重下载 `[README]`

| Model Name | Description | Download Link |
|:---|:---|:---|
| Hy-MT2-1.8B | Hy 1.8B translation model | [Model](https://huggingface.co/tencent/Hy-MT2-1.8B) |
| Hy-MT2-1.8B-FP8 | Hy 1.8B translation model, FP8 quantization | [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-FP8) |
| Hy-MT2-1.8B-GGUF | Hy 1.8B translation model, llama.cpp | [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) |
| Hy-MT2-1.8B-2bit-GGUF | Hy 1.8B translation model, llama.cpp, 2bit | [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-2bit-GGUF) |
| Hy-MT2-1.8B-1.25bit-GGUF | Hy 1.8B translation model, llama.cpp, 1.25bit | [Model](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25bit-GGUF) |
| Hy-MT2-7B | Hy 7B translation model | [Model](https://huggingface.co/tencent/Hy-MT2-7B) |
| Hy-MT2-7B-FP8 | Hy 7B translation model, FP8 quantization | [Model](https://huggingface.co/tencent/Hy-MT2-7B-FP8) |
| Hy-MT2-7B-GGUF | Hy 7B translation model, llama.cpp | [Model](https://huggingface.co/tencent/Hy-MT2-7B-GGUF) |
| Hy-MT2-30B-A3B | Hy 30B-A3B translation model | [Model](https://huggingface.co/tencent/Hy-MT2-30B-A3B) |
| Hy-MT2-30B-A3B-FP8 | Hy 30B-A3B translation model, FP8 quantization | [Model](https://huggingface.co/tencent/Hy-MT2-30B-A3B-FP8) |

ModelScope 集合：`https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2`。

#### License

> `[repo-doc]` 仓库 `LICENSE.txt`：**"Hy-MT2 is licensed under the Apache License, Version 2.0."**（`Copyright (C) 2026 Tencent. All rights reserved.`）
>
> `[model-card]` HF 卡片 frontmatter：`license: apache-2.0`（`tencent/Hy-MT2-1.8B`、`Hy-MT2-1.8B-1.25bit-GGUF` 均已核实）。
>
> → **Hy-MT2 全系 Apache 2.0。**

#### 配套发布物 `[README]`

- **IFMTBench**：翻译指令跟随能力评测基准（`./IFMTBench/README.md`）
- **Hy-MT2-Translator Skill**：可集成到 agent 的翻译 Skill，下载于 [ClawHub](https://clawhub.ai/tencent-adm/hy-mt2-translator-skill) 与 [SkillHub](https://skillhub.cn/skills/hy-mt2-translator)
- 训练：全参微调 + LoRA + 多套 DeepSpeed ZeRO + LLaMA-Factory 集成（`./train/README.md`）

---

### 3.6 Hunyuan-A13B（`Tencent-Hunyuan/Hunyuan-A13B`）

#### 模型规格 `[README]`

- **80B 总参数 / 13B 激活参数**（fine-grained MoE）
- **Natively supports a 256K context window**
- **Grouped Query Attention (GQA)**
- 混合推理：fast / slow thinking 双模式
- Agent 能力：BFCL-v3、τ-Bench、C3-Bench
- 层数 / MTP 层：**README 未说明**
- `[model-card]` HF 权重实测 80.39B

#### 推理模式控制 `[README]`

> Our model defaults to using slow-thinking reasoning, and there are two ways to disable CoT reasoning.
> 1. Pass `"enable_thinking=False"` when calling apply_chat_template.
> 2. Adding `"/no_think"` before the prompt will force the model not to use perform CoT reasoning. Similarly, adding `"/think"` before the prompt will force the model to perform CoT reasoning.

#### vLLM 启动命令（原文照抄）`[README]`

> We provide a pre-built Docker image containing **vLLM 0.8.5** with full support for this model. The official vllm release is currently under development， **note: cuda 12.8 is require for this docker**.

```bash
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-vllm
or
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm
```

HF 权重路径：

```bash
docker run  --privileged --user root  --net=host --ipc=host \
        -v ~/.cache:/root/.cache/ \
        --gpus=all -it --entrypoint python  hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm
 \
         -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8000 \
         --tensor-parallel-size 4 --model tencent/Hunyuan-A13B-Instruct --trust-remote-code

```

ModelScope 权重路径：

```bash
docker run  --privileged --user root  --net=host --ipc=host \
        -v ~/.cache/modelscope:/root/.cache/modelscope \
        --gpus=all -it --entrypoint python   hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-vllm \
         -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --tensor-parallel-size 4 --port 8000 \
         --model /root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct/ --trust_remote_code
```

ModelScope 下载命令（README）：`modelscope download --model Tencent-Hunyuan/Hunyuan-A13B-Instruct`

#### 量化部署（vLLM，原文照抄）`[README]`

> Image: Deploy image as in BF16.

**Int8**：

```SHELL
export MODEL_PATH=PATH_TO_BF16_MODEL
```

```shell
sh run_server_int8.sh
```

```shell
sh openapi.sh
```

**Int4（GPTQ）** ：

```SHELL
export MODEL_PATH=PATH_TO_INT4_MODEL
```

```shell
sh run_server_int4.sh
```

```shell
sh openapi.sh
```

**FP8（W8A8C8）** ：

```SHELL
export MODEL_PATH=PATH_TO_FP8_MODEL
```

```shell
sh run_server_fp8.sh
```

```shell
sh openapi.sh
```

> ⚠️ README 原文小瑕疵：FP8 段落写的是 "setting the environment variables in `run_server_int8.sh`"，但启动命令用的是 `run_server_fp8.sh`。本报告照抄不改。

#### TensorRT-LLM 启动命令（原文照抄）`[README]`

```bash
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm
```

```bash
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-trtllm
```

```bash
cat >/path/to/extra-llm-api-config.yml <<EOF
use_cuda_graph: true
cuda_graph_padding_enabled: true
cuda_graph_batch_sizes:
- 1
- 2
- 4
- 8
- 16
- 32
print_iter_log: true
EOF
```

```bash
trtllm-serve \
  /path/to/HunYuan-moe-A13B \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --max_num_tokens 16384 \
  --tp_size 2 \
  --kv_cache_free_gpu_memory_fraction 0.6 \
  --trust_remote_code \
  --extra_llm_api_options /path/to/extra-llm-api-config.yml
```

#### SGLang 启动命令（原文照抄）`[README]`

```bash
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang
or
docker pull hunyuaninfer/hunyuan-a13b:hunyuan-moe-A13B-sglang
```

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ipc=host \
    docker.cnb.cool/tencent/hunyuan/hunyuan-a13b:hunyuan-moe-A13B-sglang \
    -m sglang.launch_server --model-path hunyuan/huanyuan_A13B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000
```

#### 量化方案细节 `[README]`

- **FP8-static**：8-bit 浮点，少量 calibration data（without training）预定 quantization scale，权重与激活都转 FP8。
- **Int4**：**GPTQ 算法实现 W4A16**，逐层处理权重，用近似 Hessian 逆矩阵的优化过程逐层调权重，**无需重训练**，只需少量 calibration data。
- 工具链：**AngelSlim**（README 原文写作 `AngleSlim`，属原文拼写）。

**量化前后 benchmark（README 表格，原文数字）**：

| Bench | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-FP8 |
|:---:|:---:|:---:|
| AIME 2024 | 87.3 | 86.7 |
| Gsm8k | 94.39 | 94.01 |
| BBH | 89.1 | 88.34 |
| DROP | 91.1 | 91.1 |

| Bench | Hunyuan-A13B-Instruct | Hunyuan-A13B-Instruct-GPTQ-Int4 |
|:---:|:---:|:---:|
| OlympiadBench | 82.7 | 84.0 |
| AIME 2024 | 87.3 | 86.7 |
| Gsm8k | 94.39 | 94.24 |
| BBH | 89.1 | 87.91 |
| DROP | 91.1 | 91.05 |

**显存收益数字：README 未说明。**

#### 硬件需求与吞吐 `[README]`

- 最小 GPU 数 / 显存：**README 未说明**（`--tensor-parallel-size 4`、`--tp 4`、`--tp_size 2` 均为示例值）。
- 吞吐 / latency / TPS：**README 未说明**。
- Benchmark 说明（原文）：`Note: The following benchmarks are evaluated by TRT-LLM-backend on several base models.` —— 即 benchmark 跑在 **TensorRT-LLM backend** 上。

#### 权重下载 `[README]`

- HuggingFace：`https://huggingface.co/tencent/Hunyuan-A13B-Instruct`（另有 Pretrain / Instruct-FP8 / Instruct-GPTQ-Int4 四份权重，见 News 2025.6.27）
- ModelScope：`https://modelscope.cn/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct`
- cnb.cool：`https://cnb.cool/tencent/hunyuan/Hunyuan-A13B`
- Docker Hub：`https://hub.docker.com/r/hunyuaninfer/hunyuan-a13b/tags`

#### License `[README]` `[model-card]`

> `LICENSE` 文件首行：**TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT**，`Tencent Hunyuan A13B Release Date: June 27, 2025`。
>
> 关键限制（原文）：**"THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA AND IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW."**

---

### 3.7 Hunyuan-7B（`Tencent-Hunyuan/Hunyuan-7B`）

#### 模型规格 `[README]`

- 定位（原文）：*"designed for versatile deployment across diverse computational environments. **From edge devices to high-concurrency production systems**"*
- dense 模型族，参数规模 **0.5B / 1.8B / 4B / 7B**，各有 Pretrain + Instruct 变体
- **Natively supports a 256K context window**（原文小标题写作 "Ultra-Long Context Understanding"，正文为 256K）
- **Grouped Query Attention (GQA)**
- 混合推理：fast / slow thinking 双模式
- 层数 / MTP 层：**README 未说明**
- `[model-card]` Hunyuan-7B-Instruct 权重实测 7.50B

#### 推理模式控制 `[README]`

与 A13B 相同：`enable_thinking=False` 或 prompt 前置 `/no_think` / `/think`。

#### 推荐推理参数 `[README]`

```json
{
  "do_sample": true,
  "top_k": 20,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "temperature": 0.7
}
```

#### vLLM 启动命令（原文照抄）`[README]`

> **Please use vLLM version v0.10.0 or higher for inference.**

```SHELL
pip install git+https://github.com/huggingface/transformers@4970b23cedaf745f963779b4eae68da281e8c6ca
```

```shell
export MODEL_PATH=tencent/Hunyuan-7B-Instruct
```

```shell
export MODEL_PATH=/root/.cache/modelscope/hub/models/Tencent-Hunyuan/Hunyuan-7B-Instruct/
```

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --quantization experts_int8 \
    --served-model-name hunyuan \
    2>&1 | tee log_server.txt
```

请求示例（含 `"stop_token_ids": [127960]`，`temperature 0.7` / `top_p 0.6` / `top_k 20` / `repetition_penalty 1.05`）与 A13B 章节同构，见 README 原文。

**量化三档的 vLLM 参数（原文照抄）**：

Int8：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --quantization experts_int8 \
    2>&1 | tee log_server.txt
```

Int4（GPTQ）：

```shell
export MODEL_PATH=PATH_TO_INT4_MODEL
```

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --quantization gptq_marlin \
    2>&1 | tee log_server.txt
```

FP8（W8A8C8）：

```shell
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --model ${MODEL_PATH} \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --served-model-name hunyuan \
    --kv-cache-dtype fp8 \
    2>&1 | tee log_server.txt
```

#### TensorRT-LLM 启动命令（原文照抄）`[README]`

```bash
docker pull docker.cnb.cool/tencent/hunyuan/hunyuan-7b:hunyuan-7b-trtllm
```

```bash
docker run --privileged --user root --name hunyuanLLM_infer --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all hunyuaninfer/hunyuan-7B:hunyuan-7b-trtllm
```

```bash
trtllm-serve \
  /path/to/HunYuan-7b \
  --host localhost \
  --port 8000 \
  --backend pytorch \
  --max_batch_size 32 \
  --max_num_tokens 16384 \
  --tp_size 2 \
  --kv_cache_free_gpu_memory_fraction 0.6 \
  --trust_remote_code \
  --extra_llm_api_options /path/to/extra-llm-api-config.yml
```

（`extra-llm-api-config.yml` 内容与 A13B 章节完全相同。）

#### SGLang 启动命令（原文照抄）`[README]`

```bash
docker pull lmsysorg/sglang:latest
```

```bash
docker run --entrypoint="python3" --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    --ulimit nproc=10000 \
    --privileged \
    --ipc=host \
     lmsysorg/sglang:latest \
    -m sglang.launch_server --model-path hunyuan/huanyuan_7B --tp 4 --trust-remote-code --host 0.0.0.0 --port 30000
```

#### 量化 `[README]`

- **FP8-static**：同 A13B 描述。
- **Int4**：README 明确使用 **GPTQ 和 AWQ 两种算法实现 W4A16**：
  - GPTQ：逐层处理权重，少量 calibration data 最小化重构误差，用近似 Hessian 逆矩阵的优化过程逐层调权重，无需重训练。
  - AWQ：用少量 calibration data（无需训练）统计激活值幅度，对每个权重通道计算缩放系数 `s`，扩大重要权重的数值范围，使量化时保留更多信息。
- 工具链：AngelSlim（README 原文写作 `AngleSlim`）。

**量化 benchmark（README 表格，覆盖 0.5B / 1.8B / 4B / 7B 四个尺寸 × B16 / FP8 / Int4GPTQ / Int4AWQ 四种精度）**：README 原表见 `Hunyuan-7B.md` 第 283–288 行，此处不重复摘录以避免转录误差。

**显存收益数字：README 未说明。**

#### 硬件需求与吞吐 `[README]`

- 最小 GPU 数 / 显存：**README 未说明**。
- 吞吐 / latency：**README 未说明**。
- 端侧定位（原文）：模型族覆盖 *"from resource-constrained edge computing with smaller variants to high-throughput production environments with larger models"* —— 即 **0.5B / 1.8B / 4B 面向端侧，7B 面向高吞吐生产**。
- Benchmark 说明（原文）：`Note: The following benchmarks are evaluated by TRT-LLM-backend on several base models.`

#### 权重下载 `[README]`

- HuggingFace 组织页：`https://huggingface.co/tencent/`
- ModelScope（README 链接指向 A13B，属原文笔误）：`https://modelscope.cn/models/Tencent-Hunyuan/Hunyuan-A13B-Instruct`
- 具体模型：`tencent/Hunyuan-7B-Instruct`、`tencent/Hunyuan-0.5B/1.8B/4B-Instruct` 等（News 2025.7.30 列出 8 个权重）
- cnb.cool：`https://cnb.cool/tencent/hunyuan/Hunyuan-7B`
- CNB 快速上手：`https://cnb.cool/tencent/hunyuan/examples/Hunyuan-7B-Instruct-quick-start`

#### License `[README]` `[model-card]`

> README 的 LICENSE 链接指向 `blob/main/LICENSE`；实际文件为仓库根目录 `LICENSE.txt`，首行：**TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT**，`Tencent Hunyuan 7B Release Date: August 4, 2025`。包含与 A13B 相同的欧盟/英国/韩国排除条款。

---

### 3.8 Hunyuan-TurboS（`Tencent-Hunyuan/Hunyuan-TurboS`）

#### 模型规格 `[README]`

README 是一篇**论文摘要 + Arena 结果**型文档，规格信息全部来自 Abstract 正文：

> **Hunyuan-TurboS**, a novel large hybrid **Transformer-Mamba Mixture of Experts (MoE)** model. […] This **56B activated (560B total)** parameter model employs **128 layers (Mamba2, Attention, FFN)** with an innovative **AMF/MF block pattern**. Faster Mamba2 ensures linear complexity, **Grouped-Query Attention** minimizes KV cache, and FFNs use an MoE structure. Pre-trained on **16T high-quality tokens**, it supports a **256K context length** and is the first industry-deployed large-scale Mamba model.

其他要点：`adaptive long-short chain-of-thought (CoT) mechanism`；后训练含 SFT（3M instructions）、Adaptive Long-short CoT Fusion、Multi-round Deliberation Learning、two-stage Large-scale RL。

#### 部署 / 推理

**README 未说明。** 该 README 的结构是：

1. Abstract
2. Technical Details（指向 `Hunyuan-Turbos_Report.pdf`）
3. Human Evaluation: LMSYS Chatbot Arena
4. Full Evaluation Results
5. Citation（arXiv:2505.15431）

**没有** Quickstart、没有 Deployment、没有 Inference、没有 Model Links / 权重下载地址、没有量化章节、没有 License 章节。因此：

- 支持的推理框架：**README 未说明**
- 量化：**README 未说明**
- 硬件需求 / 吞吐：**README 未说明**
- 权重下载：**README 未说明**
- License：**README 未说明**

README 中仅有的性能数字（原文，非本报告推算）：

- LMSYS Chatbot Arena overall score **1356**，top 7 / 239 models，outperforming `o4-mini-2025-04-16`
- Chinese / French / Spanish **top 1**，Korean **top 2**
- 23 个自动化 benchmark 平均 **77.9%**

> 结论：**Hunyuan-TurboS 不可本地部署**——它是"论文发布型"仓库，模型权重从未开源。

---

### 3.9 HunyuanOCR-1.5（`Tencent-Hunyuan/HunyuanOCR`）

#### 模型规格 `[README]` `[model-card]`

- 定位：**lightweight, end-to-end OCR-specialized vision-language model**，统一 **document parsing / text spotting / information extraction / text-image translation** 四类任务
- README **未给出总参数量**；`[model-card]` HF 权重实测 **1.12B**（BF16）
- 训练侧升级：pretraining Stage-3 重规划，**maximum image resolution extended to 4K**，**context window extended to 128K**
- `[model-card]` `config.json`：
  - `architectures: HunYuanVLForConditionalGeneration`，`model_type: hunyuan_vl`
  - text 侧：hidden 1024，**24 层**，heads 16 / KV heads 8，head_dim 128，FFN 3584，`max_position_embeddings 131072`，`use_qk_norm: true`，`rope_type: xdrope`（`xdrope_section: [16,16,16,16]`），vocab 120818
  - vision 侧：hidden 1152，**27 层**，heads 16 / KV heads 16，patch_size 16，`max_image_size 2048`，`min_image_size 512`，`img_max_token_num 16384`，`max_vit_seq_len 65536`，`spatial_merge_size 2`
- MTP 层：无；加速机制是 **DFlash 块扩散 draft 模型**（另见第 5 节）
- 推理框架支持总览（README 原文）：`vLLM AR` / `DFlash` / `transformers` / `llama.cpp` 四条路径

#### 环境搭建（原文照抄）`[README]`

> Inference now uses a **single unified environment** (built on `uv`, requires **CUDA 13**) that runs all three configurations from the same install: **vLLM AR, DFlash speculative decoding, and native transformers**. Accuracy alignment across the three has been verified.

```bash
pip install uv
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "vllm>=0.25.1"
uv pip install --no-build-isolation --no-cache-dir "flash-attn==2.8.3"
```

`[repo-doc]` `docs/inference/inference.md` 版本多装一个包：

```bash
uv pip install "vllm>=0.25.1" runai-model-streamer
```

**无 CUDA 13 时的备选（`[repo-doc]` 表格原文）**：

| Configuration | Old standalone env | CUDA | Guide |
|:---|:---|:---|:---|
| vLLM AR only | vLLM 0.18.1 (release) | 12.x | `archive/vLLM.md` |
| vLLM AR + DFlash | vLLM nightly | 13 | `archive/DFlash.md` |
| Native transformers | transformers 5.13.0 | host driver | `archive/transformers.md` |

#### vLLM AR 启动（原文照抄）`[README]` `[repo-doc]`

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/vLLM/serve.sh
curl -sf http://127.0.0.1:8000/v1/models     # readiness check
```

`serve.sh` 展开后的实际命令（`[repo-doc]` `inference/vLLM/serve.sh`）：

```bash
CUDA_VISIBLE_DEVICES=${GPU} nohup vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    -tp 1 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --trust-remote-code \
    --port "${PORT}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_MODEL_LEN}" \
    > "${LOG}" 2>&1 &
```

脚本默认值（`[repo-doc]`）：`GPU=0`、`PORT=8000`、`GPU_MEM_UTIL=0.9`、`MAX_MODEL_LEN=131072`、`SERVED_NAME=tencent/HunyuanOCR`。

单图推理（**prompt 锁定到官方任务类型**，`--list-tasks` 可见全部 12 种）：

```bash
python inference/vLLM/infer_vllm_client.py \
    --image /path/to/document.png --task-type doc_parse \
    --model tencent/HunyuanOCR --port 8000 --max-tokens 32768
```

批量推理（多端点并发、可续跑）：

```bash
python inference/vLLM/batch_infer.py \
    --image-dir /path/to/images --out-dir /path/to/output \
    --ports 8000 --task-type doc_parse --max-tokens 32768 --concurrency 16
```

内置采样参数（`[README]` 原文）：`temperature=0.0`, `top_p=1.0`, `top_k=-1`, `repetition_penalty=1.08`；并内置 tail-repetition early-stop / cleanup。

#### DFlash 启动（原文照抄）`[repo-doc]`

```bash
MODEL_PATH=./HunyuanOCR GPU=0 PORT=8000 bash inference/DFlash/serve_DFlash.sh
```

展开后的关键差异是追加 `--speculative-config`（`[repo-doc]` `inference/DFlash/serve_DFlash.sh`）：

```bash
--speculative-config "{\"method\":\"dflash\",\"model\":\"${DFLASH_PATH}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}"
```

脚本默认值（`[repo-doc]`）：`DFLASH_PATH=${MODEL_PATH}/dflash`、`GPU_MEM_UTIL=0.85`（注释说明 *draft adds ~0.7GB*）、`MAX_MODEL_LEN=131072`、`NUM_SPEC_TOKENS=15`（注释：*official recommendation*）。

> 启动脚本的提示：`Readiness check (first load includes torch.compile, ~3-5 min)`。

#### llama.cpp 部署（原文照抄）`[README]` `[repo-doc]`

> HunyuanOCR DFlash speculative decoding is now **merged upstream** ([PR #28890](https://github.com/ggml-org/llama.cpp/pull/28890), build [`b11103`](https://github.com/ggml-org/llama.cpp/releases/tag/b11103) and later), so no fork is needed: upstream `master` serves both the base model and DFlash.

```bash
# 1. Build
git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
cmake -B build -DLLAMA_BUILD_EXAMPLES=ON     # add -DGGML_CUDA=ON for NVIDIA GPU
cmake --build ./build --config Release -j

# 2. Convert HunyuanOCR to GGUF (base + mmproj)
hf download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/hyocr-f16.gguf        --outtype f16 ./HunyuanOCR
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/mmproj-hyocr-f16.gguf --outtype f16 --mmproj ./HunyuanOCR

# 3. Serve (OpenAI-compatible)
build/bin/llama-server \
    --model  ./HunyuanOCR/hyocr-f16.gguf \
    --mmproj ./HunyuanOCR/mmproj-hyocr-f16.gguf \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    -fa on --jinja
```

启用 DFlash：

```bash
python3 convert_hf_to_gguf.py --outfile ./HunyuanOCR/hyocr-dflash-bf16.gguf --outtype bf16 \
    --target-model-dir ./HunyuanOCR ./HunyuanOCR/dflash

# ... same llama-server command, plus:
#     --spec-draft-model ./HunyuanOCR/hyocr-dflash-bf16.gguf \
#     --spec-type draft-dflash --spec-draft-n-max 15 --parallel 1
```

`[repo-doc]` `docs/llama_cpp.md` 补充的完整 DFlash 启动命令：

```bash
build/bin/llama-server \
    --model       "./HunyuanOCR/hyocr-f16.gguf" \
    --mmproj      "./HunyuanOCR/mmproj-hyocr-f16.gguf" \
    --spec-draft-model "./HunyuanOCR/hyocr-dflash-bf16.gguf" \
    --spec-type draft-dflash --spec-draft-n-max 15 \
    --host 0.0.0.0 --port 8080 --alias HYVL \
    --ctx-size 10240 --n-predict 4096 \
    --parallel 1 \
    -fa on --jinja
```

高分辨率输入可额外提升 `--batch-size / --ubatch-size`（`[repo-doc]` 原文：*we used `8192` on our own boxes*）。

#### 量化 `[README]`

- 官方路径是 **GGUF 转换**（f16 / bf16），**README 未提 AngelSlim**，也**未提供任何 INT4 / FP8 量化权重或显存收益数字**。
- 旧的 v1.0 分支另有历史说明（README 归档 News）：`[2025/11/28] We fixed vLLM inference bugs and hyperparameter configuration issues such as system prompt. … Currently, there is still a certain accuracy difference between Transformers and the vLLM framework (we are working on fixing this).`

#### 硬件需求与吞吐 `[README]` `[repo-doc]`

**硬件（`[repo-doc]` `docs/benchmark.md`）**：

- **Hardware: single NVIDIA H20 (80 GB) per server**
- **Config: concurrency=1 (single-request latency), `max_tokens=8000`, `temperature=0.0`**
- 部署形态：vLLM 路径默认 **单 GPU**（`-tp 1`，`GPU=0`）；llama.cpp 路径面向 **CPU / consumer-GPU / laptop**

**吞吐（`[repo-doc]` `docs/benchmark.md`，H20 80GB 实测数字，原文照抄）**：

| Config | wall time | avg latency | page/s | speedup |
|:---|---:|---:|---:|---:|
| **HunyuanOCR base (AR)** | 2821.9 s | 3.032 s | 0.330 | 1.00× |
| **HunyuanOCR + DFlash (14.7k packs, finetune)** | **1316.5 s** | **1.408 s** | **0.706** | **2.14×** |

- **DFlash achieves 2.14× end-to-end speedup vs AR baseline**
- **Speculative decoding is essentially lossless: total-tokens difference < 0.15% vs AR**

**跨模型对比（同条件 H20 80GB，c=1，`max_tokens=8000`）**：

| Rank | Model | Latency (s/img) | Page/s | vs HunyuanOCR AR | Notes |
|:--:|:---|---:|---:|---:|:---|
| 1 | **HunyuanOCR + DFlash** | **1.41** | **0.706** | **2.14×** | Speculative decoding |
| 2 | GLM-OCR (SDK page pipeline) | 1.65 | 0.604 | 1.83× | Layout + region OCR concurrent |
| 3 | PaddleOCR-VL 1.6 (0.9B) | 1.74 | 0.562 | 1.71× | 0.9B small model + two-stage |
| 4 | HunyuanOCR base (AR) | 3.03 | 0.330 | 1.00× | Baseline |
| 5 | Unlimited-OCR (SGLang) | 3.66 | 0.255 | 0.77× | image tiling (gundam mode) |
| 6 | DeepSeek-OCR-2 | 5.46 | 0.179 | 0.54× | Large model + grounding |
| 7 | dots.ocr | 7.15 | 0.136 | 0.41× | Slow but structured |

> `[repo-doc]` 的 Fairness Notes 明确：**`wall time` / `avg_latency` / `page_per_s` 是公平可比指标；`token_per_s` / `total_tokens` 不公平**（不同 tokenizer 计数不同）。

#### 权重下载 `[README]`

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"
```

> The download contains both the base model and the `dflash/` draft model.

HuggingFace：`https://huggingface.co/tencent/HunyuanOCR`

#### License `[README]` `[model-card]`

> HunyuanOCR-1.5 is released under the same license as HunyuanOCR-1.0 — the **Tencent Hunyuan Community License Agreement**. See [`LICENSE`](LICENSE) for the full terms.
>
> `[model-card]` HF frontmatter：`license: other` / `license_name: tencent-hunyuan-community` / `license_link: https://huggingface.co/tencent/HunyuanOCR/blob/main/LICENSE`

#### 致谢（技术来源）`[README]`

> We would like to thank [Qwen](https://github.com/QwenLM/Qwen3.6) and [DFlash](https://github.com/z-lab/dflash) for their valuable models and ideas.

---

### 3.10 HunyuanVision（`Tencent-Hunyuan/HunyuanVision`）

#### 模型规格 `[README]`

- **Hunyuan-Vision-1.5**，a **mamba-transformer hybrid architecture** vision-language model
- 能力：多语言多模态理解与推理、image & video understanding、visual reasoning、3D spatial comprehension、"Thinking-on-Image"（含 crop/zoom-in、画点/线/框、web search 等工具调用）
- 规划中的 checkpoint 尺寸：**A56B、4B**
- 上下文 / 层数 / 注意力类型 / MTP：**README 未说明**

#### 部署状态 —— **尚不可本地部署** `[README]`

README 的 **📑 Open-source Plan** 是一张**全部未勾选**的清单（原文照抄）：

```
- Hunyuan-Vision-1.5 (vision language model)
  - [ ] Hunyuan-Vision-1.5 Technical Report
  - [ ] Hunyuan-Vision-1.5 Checkpoints (A56B, 4B)
  - [ ] Hunyuan-ViT-V1 Checkpoints
  - [ ] TRT Inference Support
  - [ ] "Thinking on Images" Support
  - [ ] VLLM Support
```

因此：

- **TRT Inference Support：未勾选 → 不支持**
- **VLLM Support：未勾选 → 不支持**
- **Checkpoints：未发布 → 无权重可下载**
- 结论：**当前唯一可用路径是腾讯云 API**。

#### 当前可用路径（腾讯云 API）`[README]`

```python
from openai import OpenAI

# set your Tencent Cloud API key here

API_KEY = ""

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.hunyuan.cloud.tencent.com/v1",
)

MODEL_NAME = 'hunyuan-t1-vision-20250916'

completion = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": [{"type": "image_url","image_url": {"url": "https://dscache.tencent-cloud.cn/upload/uploader/hunyuan-64b418fd052c033b228e04bc77bbc4b54fd7f5bc.png"}},{"type": "text", "text": "What is it?"},]}]
)

print(completion.choices[0])
```

其他入口：LMArena Direct Chat（选择 `hunyuan-vision-1.5-thinking`）；腾讯云文档 `https://cloud.tencent.com/document/product/1729/104753`。

#### 量化 / 硬件 / 吞吐 / 权重下载 / License

全部为 **README 未说明**。

README 中唯一的榜单信息（原文）：`October 6, 2025: hunyuan-vision-1.5-thinking ranked 3rd on LmArena, the best performing model in China.`

---

## 4. 端侧 / 轻量部署专题

### 4.1 全景：四条端侧路径

| 路径 | 代表模型 | 交付物 | 运行时 | 是否有官方命令 |
|:---|:---|:---|:---|:---|
| **AngelSlim 极低比特 GGUF** | Hy-MT2-1.8B-1.25bit-GGUF | 1.25-bit GGUF，**440 MB** | llama.cpp（需 STQ kernel PR #22836） | ✅ README 给了构建 + `llama-completion` 命令 |
| **常规 GGUF** | HY-MT1.5-1.8B/7B-GGUF、Hy-MT2-1.8B/7B-GGUF | f16 / 2bit GGUF 权重 | llama.cpp | ⚠️ HY-MT1.5 只发权重无命令；Hy-MT2 有命令 |
| **VLM + mmproj GGUF** | HunyuanOCR-1.5 | `hyocr-f16.gguf` + `mmproj-hyocr-f16.gguf` + `hyocr-dflash-bf16.gguf` | llama.cpp（上游 master，**b11103+**） | ✅ 完整命令 + 转换脚本 + smoke test 客户端 |
| **INT4 权重量化（服务端/边缘服务器）** | HY-MT1.5-1.8B-GPTQ-Int4、Hunyuan-A13B-Instruct-GPTQ-Int4、Hunyuan-0.5B/1.8B/4B/7B-Instruct Int4(GPTQ/AWQ) | GPTQ / AWQ 权重 | vLLM（`--quantization gptq_marlin` 等） | ✅ 命令齐全 |

### 4.2 Hy-MT2 1.25-bit 量化（本报告最强的端侧证据）

**收益（README 原文，唯一带数字的端侧量化声明）**：

> For on-device deployment, **AngelSlim 1.25-bit extreme quantization reduces the storage requirement of the 1.8B model to only 440 MB and improves inference speed by 1.5x.**

拆解：

- **量化方案名**：`1.25-bit`（AngelSlim 极低比特量化）
- **工具链**：AngelSlim（`https://github.com/Tencent/AngelSlim`）
- **存储收益**：1.8B 模型 → **440 MB**（README 未给 BF16 基线大小，故不推算压缩率）
- **速度收益**：**1.5x**
- **运行时依赖（关键风险）**：`This gguf depends on our STQ kernel, which is released at PR #22836.`
  - PR 号：**#22836**（`ggml-org/llama.cpp`）
  - README 未说明该 PR 是否已合并 → **需自行确认该 PR 的合并状态后再使用 1.25-bit GGUF**
- **同族其他低比特档**：`Hy-MT2-1.8B-2bit-GGUF`（2bit）、`Hy-MT2-1.8B-GGUF`（常规）

**完整端侧运行流程（照抄 README）**：

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

```bash
./build/bin/llama-completion \
  --model model.gguf  \
  -p "Translate the following segment into Chinese, without additional explanation：Hello" \
  --jinja \
  -ngl 0 \
  -n 64 -st 
```

```bash
./build/bin/llama-bench -m model_zoo/model.gguf  -ngl 0
```

> `-ngl 0` = 不向 GPU offload 任何层；这是 README 原文示例，契合端侧/CPU 场景。

### 4.3 HunyuanOCR-1.5 PC 端 llama.cpp（含实测加速数字）

**为什么值得单独列**：这是本次调研中**唯一把 DFlash 推进 llama.cpp 上游**的模型，且有 macOS 上的真实实测数字。

`[repo-doc]` `docs/llama_cpp.md` 的 **Measured DFlash speedup** 表格（原文照抄）：

> From the upstream PR, on an **Apple M5 Pro (6 performance + 12 efficiency cores, 48 GB unified memory), macOS 26.4.1, Metal backend, Release build**, over the 26 document-image OCR requests shipped in this repo:

| Metric | Value |
|:---|:---|
| draft acceptance | ~0.5 |
| mean accepted length | 8.6 |
| output | byte-identical to the non-speculative run |
| decode | 84.6 → 167.1 t/s (~2×) |
| prefill | 232 → 192 t/s (~17% slower) |

原文对取舍的说明：*Prefill regresses because the draft also encodes the whole prompt, so end-to-end is roughly break-even on short answers and clearly faster on long structured outputs (dense documents, tables, formulas).*

**DFlash 关键 flag 释义（`[repo-doc]` 原文表格）**：

| Flag | Meaning |
|:---|:---|
| `--spec-draft-model <path>` (`-md`) | GGUF path of the DFlash draft model |
| `--spec-type draft-dflash` | Select DFlash speculative decoding. Auto-detected from the draft GGUF metadata; passing it explicitly keeps the choice unambiguous. |
| `--spec-draft-n-max 15` | Draft tokens per speculative step. The released draft has `block_size = 16`, so 15 is the maximum; larger values are clamped. **Default is only 3, so set this explicitly.** |
| `--parallel 1` | Single serial slot (recommended for DFlash on PC) |
| `-fa on` | Flash attention on both target and draft |
| `--jinja` | Use the chat template shipped with the checkpoint |

**从旧 fork 迁移的 flag 改名（`[repo-doc]` 原文表格）**：

| Old (fork) | New (upstream) |
|:---|:---|
| `--model-draft <path>` | `--spec-draft-model <path>` (`-md`, `--model-draft` still accepted) |
| `--dflash` | `--spec-type draft-dflash` |
| `--draft-max 16` | `--spec-draft-n-max 15` |

**部署自检方法（`[repo-doc]` 原文）**：

```bash
build/bin/llama-server --help | grep -A2 -- --spec-type
# the listed types must include draft-dflash
```

以及：*When DFlash is enabled, the server log also reports the draft acceptance rate, which is the quickest way to confirm speculative decoding is actually running.*

### 4.4 HY-MT1.5 的 GGUF（只有权重，没有命令）

`HY-MT1.5-1.8B-GGUF` 与 `HY-MT1.5-7B-GGUF` 在 Model Links 表中的描述只有两个字：**"llama.cpp"**。README **没有**给出：

- llama.cpp 的版本 / commit 要求
- `convert_hf_to_gguf.py` 转换步骤
- `llama-server` / `llama-cli` 启动命令
- 量化档位（Q4_K_M 等）

→ 如需使用，只能按 llama.cpp 通用流程自行处理。**这是 HY-MT1.5 与 Hy-MT2 / HunyuanOCR 在端侧支持成熟度上的主要差距。**

### 4.5 Hunyuan-7B 的端侧定位

README 明确把 dense 系列定位为覆盖端到端：

> This comprehensive model family enables flexible deployment optimization - **from resource-constrained edge computing with smaller variants to high-throughput production environments with larger models**.

且量化支持面最宽：**FP8 / GPTQ Int4 / AWQ Int4 / INT8** 四档。但 README **未给出任何端侧显存或存储数字**。

### 4.6 MLX

**在所有 10 个仓库的 README、子文档与 HF 模型卡片中，`MLX` 均未出现。** 混元官方未提供 MLX 路径，亦未提供 Apple Silicon 专用量化权重（唯一的 Apple 生态证据是 HunyuanOCR 的 llama.cpp Metal backend 实测数据，`[repo-doc]`）。

### 4.7 端侧选型速查

| 需求 | 推荐 | 依据 |
|:---|:---|:---|
| 极致小体积翻译（< 500MB） | **Hy-MT2-1.8B-1.25bit-GGUF** | README：440 MB / 1.5x；但需 STQ kernel PR #22836 |
| 稳妥的 CPU 翻译 | **Hy-MT2-1.8B-GGUF** 或 **Hy-MT2-7B-GGUF** + llama.cpp | README 给了完整构建与运行命令 |
| 本地文档 OCR（笔记本/消费级 GPU） | **HunyuanOCR-1.5** + llama.cpp（b11103+）+ DFlash | README 给了 GGUF 转换 + DFlash flag 全流程 |
| 有 GPU 的边缘服务器 | **Hunyuan-7B-Instruct** GPTQ/AWQ Int4 或 **Hunyuan-A13B-Instruct-GPTQ-Int4** | README 给了 vLLM `--quantization gptq_marlin` 路径 |
| Apple Silicon | **HunyuanOCR-1.5 via llama.cpp**（唯一有 Apple 实测数据的路径） | `[repo-doc]` M5 Pro 数据；MLX 全无支持 |

---

## 5. Speculative decoding 专题

混元系共出现**两套**加速机制：**MTP**（Multi-Token Prediction，用于 Hy3 / Hy3 preview / Hy4 preview）与 **DFlash**（块扩散 draft，用于 HunyuanOCR-1.5；Hy-MT2 的 STQ 属于量化 kernel，不属于此列）。

### 5.1 MTP 层配置总表

| 模型 | MTP 层数 | MTP 参数量 | vLLM 参数 | SGLang 参数 |
|:---|:---|:---|:---|:---|
| **Hy3** | **1** | **3.8B** | `--speculative-config.method mtp`<br>`--speculative-config.num_speculative_tokens 2` | `--speculative-algorithm EAGLE`<br>`--speculative-num-steps 2`<br>`--speculative-eagle-topk 1`<br>`--speculative-num-draft-tokens 3` |
| **Hy3 preview** | **1** | **3.8B** | `--speculative-config.method mtp`<br>`--speculative-config.num_speculative_tokens 1` | `--speculative-algorithm EAGLE`<br>`--speculative-num-steps 1`<br>`--speculative-eagle-topk 1`<br>`--speculative-num-draft-tokens 2` |
| **Hy4 preview** | **1**（native） | **10B 总参 / 0.7B 激活** | `--speculative-config '{"num_speculative_tokens":3,"method":"mtp"}'` | `--speculative-algorithm NEXTN`<br>`--speculative-num-steps 3`<br>`--speculative-eagle-topk 1`<br>`--speculative-num-draft-tokens 4` |

关键读数：

1. **Hy3 / Hy3 preview 的 MTP 层参数（3.8B）不计入 21B 激活参数**——README 表格把 `MTP Layer Parameters` 与 `Activated Parameters` 并列，且注明 `Number of Layers (excluding MTP layer) | 80`。
2. **Hy4 preview 的 MTP 层更"重"但更稀疏**：10B 总参 / 0.7B 激活，而 backbone 是 770B / 49B。
3. **SGLang 侧算法名不统一**：Hy3 / Hy3 preview 用 `EAGLE`，Hy4 preview 用 `NEXTN`。这是两个版本 SGLang 的命名差异，**照抄即可，不要互换**。
4. **Hy3 唯一需要额外环境变量**：`export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`，README 注释原因：*Switch to trtllm backend to work-around mnnvl workspace size issue.*
5. **Hy4 preview 唯一需要专用 attention backend**：`--attention-backend FLASHMLA_SPARSE`（对应 Gated DSA 稀疏注意力）。

**加速收益数字**：Hy3 / Hy3 preview / Hy4 preview 的 README **均未给出** speculative decoding 的加速倍数或吞吐数字。

### 5.2 DFlash（HunyuanOCR-1.5）—— 唯一有量化收益数据的加速机制

**机制说明（README 原文）**：

> End-to-end OCR is often accompanied by long autoregressive decoding, which becomes the major bottleneck for dense documents, tables, formulas, and other long structured outputs. HunyuanOCR-1.5 adapts a speculative-decoding framework based on **DFlash**: a lightweight **block-diffusion draft model** drafts multiple candidate tokens **in parallel**, which are then verified by the target model in a single pass. This significantly reduces the decoding latency of long structured outputs while **preserving the output distribution** of the target model.

技术来源致谢：`[DFlash](https://github.com/z-lab/dflash)`。

**draft 模型形态**：

- 权重随主模型一起发布在 HF 仓库的 `dflash/` 子目录（`huggingface-cli download tencent/HunyuanOCR --local-dir ./HunyuanOCR --exclude "v1.0/*"` 会一并拉取）
- 目录需包含 `config.json` + `dflash.py` + `model.safetensors`（`[repo-doc]` `serve_DFlash.sh` 的完整性检查逻辑）
- `[repo-doc]` 说明 draft 额外占用约 **0.7GB** 显存（`GPU_MEM_UTIL` 因此从 0.9 降到 0.85）
- draft 的 `block_size = 16`，故 `--spec-draft-n-max` 上限为 **15**

**可训练性（README 训练章节）**：

- `scripts/sft_dflash.sh`：从零训练 DFlash draft。默认 profile：`lr=1e-4`、`epochs=2`、`num_mask_tokens=16`、`sample_block_num=8`，入口 `train/train_draft.py`
- `scripts/sft_dflash_finetune.sh`：从已有 DFlash checkpoint 继续微调。推荐 profile：`lr=2e-5`、`epochs=10`、`warmup_ratio=0.05`、`save_steps=500`，入口 `train/train_draft_from_dflash.py`

**两条推理路径的 DFlash 参数对照**：

| 运行时 | 参数 | 值 |
|:---|:---|:---|
| vLLM | `--speculative-config` | `{"method":"dflash","model":"<draft_dir>","num_speculative_tokens":15}` |
| llama.cpp | `--spec-draft-model` | `./HunyuanOCR/hyocr-dflash-bf16.gguf` |
| llama.cpp | `--spec-type` | `draft-dflash` |
| llama.cpp | `--spec-draft-n-max` | `15` |
| llama.cpp | `--parallel` | `1` |

**实测收益（两套独立数据，务必区分）**：

| 环境 | 指标 | 数值 | 来源 |
|:---|:---|:---|:---|
| **NVIDIA H20 80GB**，OmnidocBench，c=1，`max_tokens=8000` | 端到端 wall time | 2821.9 s → **1316.5 s** | `[repo-doc]` `docs/benchmark.md` |
| 同上 | avg latency | 3.032 s → **1.408 s** | 同上 |
| 同上 | page/s | 0.330 → **0.706** | 同上 |
| 同上 | speedup | **2.14×** | 同上 |
| 同上 | 保真度 | total-tokens difference **< 0.15%** vs AR | 同上 |
| **Apple M5 Pro / 48GB / Metal**，26 张文档图 | draft acceptance | **~0.5** | `[repo-doc]` `docs/llama_cpp.md` |
| 同上 | mean accepted length | **8.6** | 同上 |
| 同上 | decode | **84.6 → 167.1 t/s（~2×）** | 同上 |
| 同上 | prefill | **232 → 192 t/s（~17% slower）** | 同上 |
| 同上 | 输出保真 | **byte-identical to the non-speculative run** | 同上 |

### 5.3 无 speculative decoding 的模型

| 模型 | 状态 |
|:---|:---|
| HY-MT1.5 | README 未说明任何投机解码 / MTP 机制 |
| Hy-MT2 | README 未说明；仅有 STQ kernel（量化 kernel，非投机解码） |
| Hunyuan-A13B | README 未说明 |
| Hunyuan-7B | README 未说明 |
| Hunyuan-TurboS | README 未说明 |
| HunyuanVision | README 未说明 |

> 值得注意：Hy3 / Hy3 preview / Hy4 preview 三个旗舰都**原生内置 MTP 层**且 README **默认开启**（`--speculative-config` 直接写在推荐启动命令里），而 HunyuanOCR-1.5 的 DFlash 是**可选路径**（AR 与 DFlash 是两条独立脚本，输出可直接对比）。

---

## 6. 结论：选型建议表

### 6.1 按场景选型

| 场景 | 推荐模型 | 推荐框架 | 理由（全部有 README 依据） |
|:---|:---|:---|:---|
| **旗舰级通用 Agent / Coding，有 8 卡 H20 级集群** | **Hy3** | **vLLM**（`--tensor-parallel-size 8`，MTP num_speculative_tokens 2） | README 明确推荐 vLLM/SGLang 生产服务；vLLM 命令最完整（含 tool/reasoning parser + auto tool choice）；Apache 2.0 商用最省心 |
| **同上但要官方镜像、想省编译** | **Hy4 preview-FP8** | **vLLM 官方镜像 `vllm/vllm-openai:hy4-preview`** | 唯一提供官方预构建 vLLM + SGLang 双镜像的旗舰线；`--attention-backend FLASHMLA_SPARSE` 已内置适配 |
| **1M 上下文长文档 / 超大代码库理解** | **Hy4 preview** | vLLM 或 SGLang | 唯一声明 **Context Length 1M** 的模型（Hy3/Hy3 preview 为 256K）；Gated DSA + IndexCache 为长上下文稀疏化设计 |
| **想要旗舰能力但需要宽松 License** | **Hy3** 或 **Hy4 preview** | vLLM / SGLang | 两者均为 **Apache 2.0**，是四代旗舰中唯二的 Apache 许可 |
| **多语言翻译，服务端高并发** | **Hy-MT2-30B-A3B** | **vLLM**（`--tensor-parallel-size 1`） | MoE 30B-A3B，单卡即可起（README 示例 tp=1）；Apache 2.0；家族内性能最强档 |
| **多语言翻译，单卡/小显存** | **Hy-MT2-7B** 或 **Hy-MT2-1.8B** | vLLM / SGLang | README 给了 tp=1 的启动命令；1.8B 还有 1.25-bit 端侧档 |
| **翻译 + 术语干预/上下文/格式保持** | **Hy-MT2**（任意尺寸） | 任意上述框架 | README 的指令模板表覆盖 Terminology / Style / Personalization / Delimiters / Structured Data 1&2 |
| **纯 CPU / 笔记本端侧翻译** | **Hy-MT2-1.8B-1.25bit-GGUF** | **llama.cpp**（需 STQ kernel PR #22836） | README：**440 MB / 1.5x 提速**，全报告中唯一量化收益数字 |
| **本地文档 OCR（PC / 消费级 GPU）** | **HunyuanOCR-1.5** | **llama.cpp**（上游 `master`，**b11103+**） | DFlash 已合并上游，**无需 fork**；README 给了完整 GGUF 转换 + `llama-server` 命令 + 26 张测试图 |
| **服务端高吞吐文档 OCR** | **HunyuanOCR-1.5 + DFlash** | **vLLM ≥0.25.1**（CUDA 13），单卡 `-tp 1` | `[repo-doc]` H20 80GB 实测 **2.14× 端到端加速**、latency 1.408 s/img、0.706 page/s，且 token 差异 < 0.15% |
| **需要 TensorRT-LLM 极致低延迟** | **Hunyuan-A13B** / **Hunyuan-7B** / **HY-MT1.5** | **TensorRT-LLM**（官方镜像 + `trtllm-serve`） | 只有这三条线提供 TRT-LLM 镜像与 `extra-llm-api-config.yml` 配置模板（CUDA Graph、batch size 1/2/4/8/16/32） |
| **资源受限但需较强通用能力（80B-A13B 层）** | **Hunyuan-A13B-Instruct** | **vLLM**（官方镜像内置 vLLM 0.8.5，`--tensor-parallel-size 4`） | 13B 激活参数 + 256K 上下文；量化档最全（BF16 / INT8 / FP8 / GPTQ-Int4） |
| **边缘设备到生产全覆盖、尺寸可伸缩** | **Hunyuan-7B 系列（0.5B/1.8B/4B/7B）** | vLLM ≥v0.10.0 / TRT-LLM / SGLang | README 明确覆盖 *resource-constrained edge computing* 到 *high-throughput production*；量化四档（FP8 / GPTQ Int4 / AWQ Int4 / INT8） |
| **翻译业务但要求明确商业许可** | **Hy-MT2**（Apache 2.0） | 任意 | **HY-MT1.5 未声明 License**，有合规风险；Hy-MT2 已明确 Apache 2.0 |
| **仅做研究/论文复现（Mamba 架构）** | **Hunyuan-TurboS** | **无法本地部署** | README 无权重、无部署章节，仅论文 arXiv:2505.15431 |
| **多模态理解（当前可用）** | **Hunyuan-Vision-1.5** | **仅腾讯云 API** | Open-source Plan 的 Checkpoints / VLLM / TRT 三项均未勾选 |

### 6.2 框架选型速查

| 你的约束 | 选它 | 不要选 |
|:---|:---|:---|
| 要最完整的启动参数与官方推荐 | **vLLM**（10 个仓库中 7 个首选） | — |
| 要官方预构建镜像、免编译 | `vllm/vllm-openai:hy4-preview` / `lmsysorg/sglang:hy4-preview`（Hy4 preview）、`hunyuaninfer/*`（A13B / 7B / MT1.5） | Hy3 / Hy3 preview / Hy-MT2（**无官方镜像，必须源码编译**） |
| 要 TensorRT-LLM | Hunyuan-A13B / Hunyuan-7B / HY-MT1.5 | Hy3、Hy4 preview、Hy-MT2、HunyuanOCR（**均无 TRT 路径**） |
| 要 CPU / 端侧 | HunyuanOCR-1.5（上游 llama.cpp，最省事）、Hy-MT2（需 STQ kernel） | HY-MT1.5（只有 GGUF 权重，无命令）；全部旗舰 MoE（无 GGUF） |
| 要 Apple Silicon | HunyuanOCR-1.5 via llama.cpp | MLX：**全系无支持** |
| 要 1M 上下文 | Hy4 preview | 其余全部为 256K 或更低（Hy-MT2 仅 **8192**） |

### 6.3 部署前三项检查清单（基于本次调研暴露的真实坑）

1. **License 有雷**：`Hy3 preview` / `Hunyuan-A13B` / `Hunyuan-7B` / `HunyuanOCR` 用 **Tencent Hunyuan Community License**，明文排除 **EU / UK / South Korea**；`HY-MT1.5` **完全没有 License 声明**。
2. **CUDA 版本有硬门槛**：`Hunyuan-A13B` 的 vLLM 镜像**要求 CUDA 12.8**；`HunyuanOCR-1.5` 的统一推理环境**要求 CUDA 13**（否则只能退到旧的三套独立环境）。
3. **llama.cpp 的合并状态不一致**：`HunyuanOCR` 的 DFlash **已上游合并**（b11103+，用 master 即可）；`Hy-MT2` 的 STQ kernel **仍是 PR #22836**，合并状态 README 未说明。

---

## 7. 参考链接

### 7.1 模型仓库（GitHub）

- Hy3 — https://github.com/Tencent-Hunyuan/Hy3
- Hy4-preview — https://github.com/Tencent-Hunyuan/Hy4-preview
- Hy3-preview — https://github.com/Tencent-Hunyuan/Hy3-preview
- Hy-MT（HY-MT1.5） — https://github.com/Tencent-Hunyuan/HY-MT
- Hy-MT2 — https://github.com/Tencent-Hunyuan/Hy-MT2
- Hunyuan-A13B — https://github.com/Tencent-Hunyuan/Hunyuan-A13B
- Hunyuan-7B — https://github.com/Tencent-Hunyuan/Hunyuan-7B
- Hunyuan-TurboS — https://github.com/Tencent-Hunyuan/Hunyuan-TurboS
- HunyuanOCR — https://github.com/Tencent-Hunyuan/HunyuanOCR
- HunyuanVision — https://github.com/Tencent-Hunyuan/HunyuanVision

### 7.2 仓库内子文档（本次实际读取）

- Hy3 Finetuning Guide — `finetune/README.md`
- Hy3 RL Training Guide — `rl/README.md`
- Hy4 preview Finetuning Guide — `finetune/README.md`
- HunyuanOCR 推理与部署 — `docs/inference/inference.md`
- HunyuanOCR PC 端 llama.cpp — `docs/llama_cpp.md`
- HunyuanOCR 速度 benchmark — `docs/benchmark.md`
- HunyuanOCR vLLM serve 脚本 — `inference/vLLM/serve.sh`
- HunyuanOCR DFlash serve 脚本 — `inference/DFlash/serve_DFlash.sh`

### 7.3 权重与分发渠道

| 渠道 | 地址 |
|:---|:---|
| Hugging Face（混元组织） | https://huggingface.co/tencent |
| Hy-MT1.5 集合 | https://huggingface.co/collections/tencent/hy-mt15 |
| Hy-MT2 集合 | https://huggingface.co/collections/tencent/hy-mt2 |
| ModelScope Hy-MT1.5 集合 | https://modelscope.cn/collections/Tencent-Hunyuan/HY-MT15 |
| ModelScope Hy-MT2 集合 | https://modelscope.cn/collections/Tencent-Hunyuan/Hy-MT2 |
| cnb.cool（Hy3 / Hy4 preview / A13B / 7B） | https://cnb.cool/ai-models/tencent/Hy3 等 |
| GitCode（Hy3 / Hy4 preview / Hy3 preview） | https://ai.gitcode.com/tencent_hunyuan/Hy3 等 |
| Docker Hub（Hunyuan 推理镜像） | https://hub.docker.com/r/hunyuaninfer/hunyuan-7B/tags · https://hub.docker.com/r/hunyuaninfer/hunyuan-a13b/tags · https://hub.docker.com/r/hunyuaninfer/hunyuan-large/tags |

### 7.4 推理框架与工具链

- vLLM — https://github.com/vllm-project/vllm
- vLLM recipes: Hy3 — https://recipes.vllm.ai/tencent/Hy3
- vLLM recipes: Hy4-preview — https://recipes.vllm.ai/tencent/Hy4-preview
- SGLang — https://docs.sglang.io/
- SGLang cookbook: Hy3 — https://lmsysorg.mintlify.app/cookbook/autoregressive/Tencent/Hy3
- SGLang cookbook: Hy4-Preview — https://lmsysorg.mintlify.app/cookbook/autoregressive/Tencent/Hy4-Preview
- llama.cpp — https://github.com/ggml-org/llama.cpp
- llama.cpp PR #22836（Hy-MT2 STQ kernel） — https://github.com/ggml-org/llama.cpp/pull/22836
- llama.cpp PR #28890（HunyuanOCR DFlash 上游合并） — https://github.com/ggml-org/llama.cpp/pull/28890
- llama.cpp build b11103 — https://github.com/ggml-org/llama.cpp/releases/tag/b11103
- **AngelSlim**（混元官方压缩工具链） — https://github.com/Tencent/AngelSlim
- AngelSlim 文档 — https://angelslim.readthedocs.io/
- DFlash（draft 模型技术来源） — https://github.com/z-lab/dflash
- verl（Hy3 RL 训练） — https://github.com/volcengine/verl
- LLaMA-Factory — https://github.com/hiyouga/LLaMA-Factory

### 7.5 论文 / 技术报告

- Hy-MT1.5 Technical Report — arXiv:2512.24092 — https://arxiv.org/abs/2512.24092
- Hy-MT2 Report — arXiv:2605.22064 — https://arxiv.org/pdf/2605.22064
- Hunyuan-A13B Technical Report — 仓库内 `report/Hunyuan_A13B_Technical_Report.pdf`
- Hunyuan-TurboS — arXiv:2505.15431 — https://arxiv.org/abs/2505.15431
- HunyuanOCR-1.5 — arXiv:2607.04884 — https://arxiv.org/pdf/2607.04884
- HunyuanOCR Technical Report — arXiv:2511.19575
- DeepSeek Sparse Attention（Hy4 preview 的 DSA 灵感来源） — https://arxiv.org/abs/2512.02556
- IndexCache（Hy4 preview） — https://arxiv.org/abs/2603.12201

### 7.6 其他

- 腾讯混元官网 — https://hunyuan.tencent.com
- AI Studio — https://aistudio.tencent.com/
- HunyuanVision 云 API 文档 — https://cloud.tencent.com/document/product/1729/104753
- IFMTBench（Hy-MT2 配套基准） — `./IFMTBench/README.md`
- Hy-MT2-Translator Skill — https://clawhub.ai/tencent-adm/hy-mt2-translator-skill · https://skillhub.cn/skills/hy-mt2-translator
- 联系邮箱 — hunyuan_opensource@tencent.com

---

## 8. 证据缺口与未采信项声明

为遵守"绝不编造"的硬性要求，以下内容在本报告中被明确标记为缺失，**请勿在后续引用中补齐为推测值**：

| 缺口类别 | 涉及模型 | 状态 |
|:---|:---|:---|
| **吞吐 / TPS / Latency 数字** | Hy3、Hy4 preview、Hy3 preview、HY-MT1.5、Hy-MT2、Hunyuan-A13B、Hunyuan-7B、Hunyuan-TurboS、HunyuanVision | **全部 README 未说明**。唯一有真实吞吐数字的是 HunyuanOCR-1.5（见第 3.9 / 5.2 节） |
| **最小 GPU 数与显存** | Hy4 preview、HY-MT1.5、Hy-MT2、Hunyuan-A13B、Hunyuan-7B、Hunyuan-TurboS | README 未声明"最小配置"，命令中的 tp 值仅为示例。**唯一有明确最小推荐的是 Hy3 / Hy3 preview（8 卡 H20-3e 或更大显存）** |
| **量化显存收益数字** | Hy3、Hy4 preview、Hy3 preview、HY-MT1.5、Hunyuan-A13B、Hunyuan-7B | README 只有定性描述（"reduce the deployment threshold"），**无任何 MB/GB 数字**。唯一有数字的是 Hy-MT2（440 MB / 1.5x） |
| **speculative decoding 加速倍数** | Hy3、Hy4 preview、Hy3 preview | README 给了参数但**没给加速数字**。唯一有数字的是 HunyuanOCR-1.5 的 DFlash |
| **License 缺失** | **HY-MT1.5** | README 无 License 章节；仓库无 `LICENSE` / `LICENSE.txt`；HF 卡片无 `license` 字段；HF API tags 无 `license:*` → **完全未声明** |
| **部署能力缺失** | **Hunyuan-TurboS** | 无权重、无部署章节、无 License、无量化 |
| **部署能力未就绪** | **HunyuanVision** | Checkpoints / VLLM Support / TRT Inference Support 三项在 Open-source Plan 中**均未勾选**；当前仅云 API |
| **端侧支持不完整** | **HY-MT1.5** | 发了 GGUF 权重（1.8B / 7B）但**无任何 llama.cpp 命令、无版本要求、无量化档位说明** |
| **MLX** | 全部 10 个仓库 | **完全未出现** |

### 数据来源的方法学说明

1. **本地 README 副本**（`D:\WorkSpace\workspace\Hunyuan\_research\readmes\`）是本报告的第一手证据，10 份 README 均已完整读取。
2. **子文档与脚本**通过 `https://raw.githubusercontent.com/Tencent-Hunyuan/<REPO>/main/<path>` 获取，保存于 `D:\WorkSpace\workspace\Hunyuan\_research\extra\`。
3. **模型规格补充**来自 Hugging Face 模型仓库的 `config.json` 与 HF 模型 API（`https://huggingface.co/api/models/<id>`）—— 用于参数量实测值、`safetensors` dtype 分布、`lastModified`、license tag。
4. **"仓库最近提交"** 来自 GitHub 的 Atom feed（`https://github.com/Tencent-Hunyuan/<repo>/commits/main.atom`），**未使用** 受限的 GitHub REST API。
5. 本报告**未**引用任何第三方转述、博客或社区帖；所有命令、参数、版本号、数字均可回溯到上述四类来源之一。
