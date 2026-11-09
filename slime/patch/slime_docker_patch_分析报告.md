# slime docker Patch 梳理报告

> 分析对象：
> - `slime/docker/npu_patch/`（Ascend NPU 适配补丁集，5 个 patch + README）
> - `slime/docker/patch/latest/`（GPU 侧最新补丁集，7 个 patch）
> - 追加第二部分：上述 patch 内容向 sglang 迁移的可行性梳理
>
> 分析日期：2026-11-03（patch 梳理）/ 2026-11-05（sglang 迁移梳理）
> 本地仓库：slime `D:\WorkSpace\slime`（v0.2.2 基线）；sglang 核对基准 `D:\WorkSpace\sglang-prwork`（SeaAsters fork，2026-11-03）

---

## 1. 目录背景与依赖关系

slime 的 `docker/patch/` 目录按 sglang 版本滚动维护（v0.5.0 ~ v0.5.21、gb10、latest），每个版本目录只含 `megatron.patch` + `sglang.patch` 两个**基础 patch**；`latest` 是最新版 GPU 补丁集，在基础 patch 之上新增了 5 个 RL 专用补丁。

`docker/npu_patch/` 是 Ascend 适配层。其 README 指明的应用顺序为：

```
slime v0.2.2              ← slime.patch（NPU）
sglang  (dce8b06)
  ├─ docker/patch/v0.5.7/sglang.patch   ← GPU 基础 patch 先打
  └─ npu_patch/sglang.patch             ← NPU patch 再叠加
Megatron-LM (3714d81)
  ├─ docker/patch/v0.5.7/megatron.patch
  └─ npu_patch/megatron.patch
Megatron-Bridge (dev_rl)  ← megatron-bridge.patch
MindSpeed (fc63de5)       ← mindspeed.patch
```

即：**NPU 侧复用 GPU 的基础 patch，再叠加硬件适配层**。NPU 路线钉死一整套版本组合（slime v0.2.2、sglang dce8b06、Megatron-LM 3714d81、Megatron-Bridge dev_rl、MindSpeed fc63de5、sgl-kernel-npu 2026.02.01、CANN 8.5.0、torch-npu 2.8.0），兼容性脆弱、依赖强锁定。

---

## 2. NPU 侧补丁梳理（npu_patch）

### 2.1 mindspeed.patch（MindSpeed 适配，3 处小改）

| 文件 | 改动 |
|---|---|
| `mindspeed/core/fusions/fused_rope.py` | `apply_rotary_pos_emb` 增加 `cp_group` 入参传递（支持上下文并行 CP 下 RoPE） |
| `mindspeed/megatron_adaptor.py` | `repatch()` 中将 `args` 转为 `vars(args)` 字典（接口兼容修复） |
| `mindspeed/te/pytorch/attention/dot_product_attention.py` | `DotProductAttention.forward` 增加 `local_cp_size` / `cp_group` 参数；**强制使用 2048×2048 bool 因果 mask**（NPU 注意力 mask workaround） |

### 2.2 megatron-bridge.patch（让权重转换认识 MindSpeed 层）

| 文件 | 改动 |
|---|---|
| `src/megatron/bridge/models/conversion/param_mapping.py` | 将 MindSpeed 的 TE 替身模块注册进 AutoMapping 的 column/row/replicated 分类：`MindSpeedTELayerNormColumnParallelLinear`、`MindSpeedTEColumnParallelGroupedLinear`、`MindSpeedTEDotProductAttention`、`MindSpeedTERowParallelGroupedLinear`、`MindSpeedTELinear`。否则 HF↔Megatron 权重转换无法识别这些层 |
| `src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/transformer_block.py` | Qwen3-VL 的 `deepstack_visual_embeds` 从 list 参数改为**解包 varargs**（`*deepstack_visual_embeds_args`），绕过 activation recompute/checkpoint 对 list 类参数的处理问题 |

### 2.3 megatron.patch（Megatron-LM 的 NPU 兼容）

| 类别 | 改动 |
|---|---|
| 删除 `@jit_fuser` | **批量删除 40+ 处 `@jit_fuser`（TorchScript 融合）**，覆盖：`activations.py`（squared_relu/quick_gelu/fast_gelu）、`fused_bias_swiglu/geglu/gelu/dropout.py`、`fused_cross_entropy.py`、`fused_pad_routing_map.py`、`fused_weighted_squared_relu.py`、`transformer/utils.py`、`torch_norm.py`、`moe/experts.py`、`moe/router.py`、`moe/token_dispatcher.py`、`ssm/gated_delta_net.py`、`transformer/attention.py`、`legacy/model/*` 等。原因：torch-npu 环境下 TorchScript jit 融合不可用/无收益 |
| dtype 支持 | `transformer/module.py`：`_FLOAT_TYPES` / `_HALF_TYPES` / `_BF16_TYPES` 增加 `torch.npu.FloatTensor/HalfTensor/BFloat16Tensor`，使 dtype 判断覆盖 NPU 张量 |
| import 顺序 | `models/gpt/gpt_layer_specs.py`：把 `TESpecProvider` 的 import 挪进函数体内，避免 NPU 环境顶层导入失败 |

### 2.4 sglang.patch（SGLang NPU 后端）

| 文件 | 改动 |
|---|---|
| `hardware_backend/npu/attention/ascend_backend.py` | 非 MLA 注意力路径**统一为 `npu_fused_infer_attention_score`（TND 布局、`actual_seq_lengths`、`sparse_mode=0`）**；删除 bs<tp 时的 `_npu_paged_attention` 特例与 BSH 布局特例，简化为一条路径 |
| `npu/graph_runner/npu_graph_runner.py` | Graph 重放属性固定为 MLA（删除 bs<tp / target_verify / use_fia 的分支判断）；profiler 输出目录增加 `..` 路径穿越检查 |
| `layers/rotary_embedding.py` | NPU 不再走 sgl-kernel fused RoPE 路径；新增 bf16 query + fp32 cos_sin_cache 时走 `forward_native` 的兜底 |
| `layers/vocab_parallel_embedding.py` | NPU 上禁用 `torch.compile`（`get_masked_input_and_mask`） |
| `models/llama.py` | NPU 使用 `sgl_kernel_npu.norm.split_qkv_rmsnorm_rope` **融合算子**（QKV 切分 + RMSNorm + RoPE 一次完成），保留 native 兜底路径 |
| `models/qwen3.py` / `qwen3_moe.py` | 适配新版 `split_qkv_rmsnorm_rope` 签名（`q_weight/k_weight/eps` 关键字参数）；qwen3 的 lm_head 增加 `use_attn_tp_group`（配合 `enable_dp_lm_head`） |
| `models/qwen3_vl.py` | ① `fast_pos_embed_interpolate` 改用 **numpy** 计算插值索引/权重（顺带修复 idx_list 少乘 t、split 尺寸两个 bug）；② deepstack 处理简化为 decoder layer 后直接相加（去掉 `post_residual_addition` 全链路透传）；③ 禁用 DP sharded 视觉编码器路径；④ 仅 `tie_word_embeddings` 时才把 embed_tokens 加载进 lm_head |
| `scripts/ci/npu_ci_install_dependency.sh` | sgl-kernel-npu 版本 tag 升至 2025.12.31 |

### 2.5 slime.patch（slime 本体的 NPU 化，最大）

**Ray 调度全面 NPU 化**：
- `ray/actor_group.py`、`ray/placement_group.py`、`ray/rollout.py`、`ray/train_actor.py`：所有 `num_gpus` 改为自定义 resource `{"NPU": n}`（依赖 Ray NPU device plugin）；placement group bundle、InfoActor、锁 actor、RolloutManager 全部适配；`get_local_gpu_id()` 改用 `ASCEND_RT_VISIBLE_DEVICES` / ray accelerator ids。

**通信后端**：
- `backends/megatron_utils/actor.py`、`update_weight_from_distributed.py`、`fsdp_utils/update_weight_utils.py`：所有跨进程通信组 `nccl` → **`hccl`**；tcp `init_method` 使用 `_wrap_ipv6`（IPv6 地址兼容）。

**MindSpeed 接入**：
- `train.py`、`backends/megatron_utils/actor.py`：NPU 时 import `mindspeed.megatron_adaptor` 并在初始化后调用 `repatch(args)`。

**Qwen3-VL 全链路支持**：
- 新增 `backends/megatron_utils/megatron_to_hf/qwen3_vl.py`（133 行）：Qwen3-VL 的 Megatron→HF 参数名/权重转换（含 PP offset、语言模型 attention/MLP/LayerNorm、视觉模型 blocks、deepstack merger），接入 `convert_to_hf`——**训练→rollout 权重同步必需**。
- `backends/megatron_utils/__init__.py`：Qwen3VLTextRotaryEmbedding / Qwen3VLMoETextRotaryEmbedding 的 forward 猴补丁（吞掉 `packed_seq_params`）；`Qwen3VLModel.forward` 吞掉 `loss_mask`（megatron.bridge 与 mbridge 两个来源都补）。
- `backends/megatron_utils/model_provider.py`：新增 Qwen3-VL MoE critic 的 `provide_wrapper`（用 MoE layer spec 组装语言模型）；`LinearForLastLayer` 的 bias 也设置 `sequence_parallel`；将 `LinearForLastLayer` 注册进 AutoMapping（replicated）。

**权重同步修复**：
- `update_weight/common.py`：MoE `linear_fc1.weight`（gate/up 融合权重）在 NPU 上 gather 后强制 `partition_dim = 0`（修复 megatron grouped moe 的 bug）。
- 新增 `slime_plugins/patch/mbridge_patch.py`：`build_conversion_tasks` 过滤 None 任务，防止后续 `None.task.xxx` 崩溃。
- 新增 `slime_plugins/patch/critic_patch.py`：包装 `load_weights_hf_to_megatron`，critic 的 `output_layer`（被替换为 1 维 value head）形状不匹配时跳过。

**环境与工具**：
- 新增 `slime/utils/common.py`：`is_npu()` 判断工具。
- `utils/external_utils/command_utils.py`：新增 `execute_train_npu`（注入 `ASCEND_TOOLKIT_HOME/OPP_PATH/AICPU_PATH`、HCCL 端口段、`RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` 等环境变量）。
- `utils/memory_utils.py`：NPU 时使用 `torch.npu.empty_cache()` / `torch.npu.mem_get_info()`。
- `backends/sglang_utils/sglang_engine.py`：设备重映射环境变量 `CUDA_VISIBLE_DEVICES` → `ASCEND_RT_VISIBLE_DEVICES`。

---

## 3. GPU 侧补丁梳理（patch/latest）

### 3.1 megatron.patch（checkpoint 兼容 + kernel 修复 + 优化器修复）

| 主题 | 改动 |
|---|---|
| **checkpoint 部分加载** | `dist_checkpointing`：`torch.load` 加 `weights_only=False`；缺失 key 由 raise 改为 warning+skip；`allow_partial_load=True`。允许加载与当前模型不完全一致的 checkpoint（RL 场景常见：值头增删、结构微调） |
| **MLA YaRN RoPE Triton 修复** | `fused_mla_yarn_rope_apply.py`：fwd/bwd kernel 增加 `k_dim_ceil`（非二次幂 k_dim 补齐+掩码）、head 越界掩码、`v_dim=0` 分支、`o_value`/`dv` 的 view 形状修正 |
| torch_memory_saver | `dynamic_context.py`、`training.py`：适配新版 API（`TorchMemorySaver()` 实例化替代模块级单例） |
| **MTP loss 修复** | `gpt_model.py`：forward 增加 `mtp_kwargs`；MTP loss 仅在有 labels/mtp_labels 时计算；`process_mtp_loss` 用 `mtp_labels` 作为 input_ids——支持 RL 侧自定义 MTP label |
| 分布式优化器 | `distrib_optimizer.py`：param_group 的 `step=None` 删除；checkpoint bucket state 去除 `step`/`padding` 冗余（消除对 save 时优化器布局的依赖） |
| ckpt_step | `checkpointing.py`：`ckpt_step=0` 生效（`is not None`），强制非 release，跳过 non-persistent 查找 |
| **hybrid optimizer 关键修复** | `hybrid_optimizer.py`：`update_fp32_param_by_new_param` 改为刷新**所有** inner fp32 副本（原实现只刷新 `param_to_fp32_param` 里的，已 FP32 但有独立 CPU 副本的参数会被旧副本在第一次 optimizer step 时覆盖掉刚同步的新权重——RL 权重同步关键 bug） |
| 其他 | `argument_utils.py`：help 文本转义 `%`（argparse 格式化崩溃） |

### 3.2 megatron-sglang-aligned.patch（训练/推理数值对齐）

env `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS=1` 启用。改动 `transformer_layer.py`：

- 残差加（attention 后、MLP 后）不再走 bias_dropout_add，而是**保留 FP32 精确和**（`_sglang_residual_sum_fp32`），跨层通过 tensor 属性传递；
- RMSNorm 直接在**未取整的 FP32 残差和**上计算（复刻 SGLang fused residual+RMSNorm 的数值行为）；
- 要求 dropout=0 且 attention/MLP 无 bias（否则 RuntimeError）。

目的：消除训练侧（Megatron）与 rollout 侧（SGLang）在 residual+norm 边界的数值漂移，使 on-policy logprob / 优势估计一致，避免 RL 训练因 off-policy 噪声退化。

### 3.3 sglang.patch（GPU 基础增强：权重更新、PD 部署、观测）

| 主题 | 改动 |
|---|---|
| **权重更新** | 新增 `/post_process_weights` 端点 + `model_runner.post_process_weights`（`restore_weights_before_load` + `post_process_quantization`：量化模型权重更新前 restore 原始权重、更新后重新量化打包）；新增 `/get_weight_version`；`update_weights_from_distributed`/`from_tensor` 兼容旧客户端（不再强制 begin/end weight session、支持 `disable_draft_model` 只更新 target）；CUDA graph 下 expert/indexer capturer 的 `cuda_graph_batch` 改为 num_tokens 口径 |
| **PD 分离部署** | transfer/bootstrap 超时（`SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT`，默认 600s）与请求 abort；失败会话处理调整；mooncake 传输 `dst_addr` bug 修复；decode 侧 `release/resume_memory_occupation` 时清理在途传输并释放注册 buffer；retract 请求在 decode disagg 下允许清空 |
| **PD 观测指标** | `req_time_stats.py`：新增 `prefill_retry_count` 及成套阶段耗时指标进 `meta_info`（`pd_prefill_bootstrap/alloc_wait/forward/transfer_queue_duration`、`pd_decode_*` 系列、`pd_transfer_speed_gb_s` 等） |
| **DSA（DeepSeek 稀疏注意力）** | indexer head-gate q_scale 对齐（`_match_head_gate_q_scale`）、query head 数 pad 到 32（小头数模型修复）、`INDEXER_ROPE_NEOX_STYLE` 环境变量；DSA KV pool 的压缩尾 buffer 纳入 memory_saver region（可随 release/resume 释放恢复） |
| radix cache | `cache_finished_req` 只缓存 committed KV 长度（避免把 speculative/未提交部分写入前缀树）；`dec_lock_ref` 由 assert 改为 break（允许持有跨树节点） |
| 其他 | draft worker 不再 set offloader；glm4v_moe 支持 `encoder_only`/`language_only`（VIT/LM 拆分部署）+ 权重加载跳过 + mrope rope_parameters 兼容；glm4v processor 支持预计算 embedding 的 `get_mm_data`；qwen_vl processor 回退 `legacy_load_mm_data`；EAGLE draft cuda graph 的 `topk_p`/`topk_index` clamp 防越界；TTFT/ITL 统计修正；fastapi route path 匹配修复 |

### 3.4 sglang-deterministic.patch（确定性推理 / batch-invariant）

核心目标：**rollout 输出与 batch 组成无关（batch-invariant），且可与训练 bit 级对齐**。

| 主题 | 改动 |
|---|---|
| **FP8 量化 kernel 对齐** | `per_token_group_quant.cuh`、`per_token_group_quant_8bit_v2.cuh`：fused silu_and_mul+quant 与非 fused 路径 bit 级一致——silu 改用 `expf`（匹配 sgl_kernel）、scale 用 `__fdiv_rn(amax, MAX)`、FP8 逐元素转换替代打包转换（避免 tie 方向差异）；masked layout × fuse_silu × scale 组合的调度分支补全。测试注释明确写着 "Hopper train-rollout alignment path" / "Slime's training alignment path" |
| **Router GEMM** | 新增 `kernels/jit/glm5_router_gemm.py`（BF16 [M,6144]×[256,6144]ᵀ→FP32）+ 精度/批不变性测试；`batch_invariant_ops.router_gemm_batch_invariant`：按 16 行分块保证固定归约顺序；`matmul_persistent` 对 M≤16 用 BLOCK_SIZE_M=16；DSV3/GLM5 router 形状走快速 kernel，其余走 FP32 matmul_persistent |
| **DeepGEMM** | `deep_gemm_wrapper`：`configure_deep_gemm_batch_invariant`（`set_batch_invariant` + `use_deterministic_algorithms`，需 GLM-5 alignment 版 DeepGEMM）；MoE grouped GEMM 的 expert M **pad 到 128 tile**（消除 tile 内归约顺序随 M 变化）；DeepEP low-latency dispatch 传 `align_fp8_quantization`；新增 `SGLANG_DEEPEP_LL_PREFILL_STAGING`（prefill 超过 LL 容量时按窗口分批 dispatch，各 rank 同步窗口）；`biased_grouped_topk_gpu` 确定性模式下回退非融合实现 |
| **MoE router** | 新参数 `enable_fp32_moe_router`：确定性模式下 DeepSeek-style router 用 `router_gemm_batch_invariant`（FP32 logits） |
| **DSA 确定化** | indexer head gate 在确定性模式下走 DeepGEMM bf16bf16f32（代替 `torch.mm`）；topk 非融合路径的无效 key 排序确定化（invalid key 排到末尾）；FP8 KV cache 为 `flashmla_sparse` 提供 gather+dequant BF16 路径 |
| CUDA IPC | `weight_updater.py`：`load_weights` 后 `torch.cuda.synchronize()`（trainer 在 RPC 返回前持有 IPC buffer 所有权，防 race） |
| 调试工具 | `tensor_dump_forward_hook.py` 大幅增强：layer-outputs-only 模式、分块落盘、模块后缀过滤、ForwardBatch 更多字段（req_pool_indices/rids/forward_mode 等）——用于训练/推理逐层对齐调试 |

### 3.5 sglang-top_p.patch（top-p 采样重放与 logprob 重归一）

解决的问题：top-p 采样时，"标准 logprob 分布"与"实际采样分布（截断后重归一）"不一致，策略梯度 / GSPO 的 ratio 计算语义错误。

| 主题 | 改动 |
|---|---|
| 核心算法 | 新增 `layers/utils/top_p.py`：nucleus keep-mask（支持 top-k/top-p/min-p 组合）、**`renorm_logprob_over_top_p`**（在 top-p 支持集 ∪ {已采样 token} 上重归一化 logprob，边界 token 不会被截成 -inf）、`get_top_p_token_ids_from_probs` |
| 采样路径 | greedy / probs sampling / Ascend backend 等路径挂载 top-p token ids；需要时返回按 top-p 截断重归一的 logprobs |
| 全链路 plumbing | `LogitsProcessorOutput` → io_struct → detokenizer / multi_tokenizer → `ReqLogprob` → batch_result_processor / logprob_result_processor → output_streamer → tokenizer_manager（`meta_info` 中 base64 的 `top_p_token_ids` / `top_p_token_offsets` / `top_p_log_probs`） |
| Spec decode | EAGLE / ngram 按 accept_lens 计算 spec logprobs 的 top-p 重归一 |
| PD 分离 | prefill/decode 之间传输 top-p 元数据 buffer（`output_top_p_token_ids_len/_ids/_log_probs`） |

### 3.6 sglang-pull_weights.patch（共享存储版本化权重下发）

NCCL 权重同步之外、**基于共享存储的替代通道**，适合大模型 / 多机 / 对象存储场景。

| 主题 | 改动 |
|---|---|
| 端点 | 新增 `/pull_weights`（HTTP）+ `PullWeightsReqInput/Output` + scheduler weight_updater + tokenizer fan-out communicator 全链路；新参数 `custom_pull_weights_pre_read_hook` |
| 核心实现 | 新增 `weight_sync/local_checkpoint.py`（358 行）：trainer 把每次权重同步发布为共享目录下的 `weight_v{N:06d}/`——完整 HF checkpoint 或 **zstd 压缩的 XOR / overwrite delta**（含逐 tensor 校验和：xxh3-128 / blake3 / adler32）；每个 host 拉取并物化本地 checkpoint：文件锁（同机多 rank 合并为一次 pull）、applied-version 标记、mmap 原地打 delta、线程池并行、page cache 淘汰/预热、对象存储一致性的 `pre_read_hook`；引擎随后走常规 `update_weights_from_disk` 加载 |
| 语义 | 版本 0 = 引擎自身 base checkpoint；delta 链乱序/校验失败立即 raise（绝不 serve 错误权重） |

### 3.7 sglang-release_hicache.patch（释放 HiCache host 内存）

| 主题 | 改动 |
|---|---|
| 参数 | 新增 `--release-hicache`：`release_memory_occupation` 时同时释放分层 KV cache（HiCache）的 host 侧缓冲 |
| 权重更新器 | `weight_updater.py`：offload/resume 的 tag 去重 + 空集短路；KV_CACHE offload 时调用 `tree_cache.release_memory_occupation()`，resume 时对称恢复 |
| HostKVCache 重构 | `pool_host/base.py`：init 拆分为 `_check_host_memory_available` / `_init_host_buffers` / `_post_init_host_buffers` 钩子；新增 `release_memory_occupation` / `resume_memory_occupation`——释放时 unregister pinned memory、置空派生引用、收缩 free_slots/mem_state、`malloc_trim` 归还 glibc 堆；恢复时重新检查 host 内存预算并重建 |
| 各 pool 适配 | mha / mla / dsa / mamba / unified 各自声明 buffer 属性名与 init 钩子；unified 的共享 backing（full+swa 共用）做**引用计数式 release/resume**；`unified_radix_cache` 暴露钩子；带两组单测 |

用途：colocated RL 部署中，训练阶段 rollout 引擎释放 GPU 内存时把 host 内存也腾出来（避免 host OOM / 供训练进程使用）。

---

## 4. NPU 与 GPU 侧工作对比

| 维度 | NPU 侧（npu_patch） | GPU 侧（patch/latest） |
|---|---|---|
| **核心目标** | 让 slime 全栈在 Ascend 上"**能跑起来**" | 把 RL 训练做"**对、快、省**" |
| **改动性质** | 兼容性适配：算子替换、API 对齐、调度/通信后端替换、环境变量注入、bug workaround | 能力增强：确定性推理、训练/推理数值对齐、采样语义修正、权重下发协议、内存管理 |
| **覆盖范围** | 必须改 5 个组件：slime 本体 + sglang + Megatron-LM + Megatron-Bridge + MindSpeed | 只改 sglang（5/7）和 Megatron（2/7），**slime 本体零改动** |
| **硬件耦合** | 强耦合：hccl、ASCEND_RT_VISIBLE_DEVICES、torch_npu 算子（`npu_fused_infer_attention_score`）、Ray NPU device plugin、sgl_kernel_npu 融合算子、CANN 环境变量 | 无硬件适配层，工作集中在 NVIDIA 栈的精度/性能语义（FP8 量化、DeepGEMM、FlashMLA、CUDA IPC、Triton kernel） |
| **典型问题** | TorchScript 不可用→删 jit_fuser；attention 接口不匹配→统一融合算子；Ray 不认 NPU→自定义 resource；mbridge 不认 MindSpeed 层→注册映射；Qwen3-VL 缺转换→补全链路 | 训练/rollout logprob 不一致→residual+RMSNorm FP32 对齐；非确定性→batch-invariant 算子族；top-p 采样语义→重放重归一；权重同步带宽→共享存储 delta 下发；host 显存挤占→HiCache 释放 |
| **版本策略** | 钉死整套版本组合（README 的版本映射表），适配脆弱 | 按 sglang 版本滚动维护（v0.5.0…v0.5.21 + latest），功能分层：基础 patch + 5 个可独立启用的 RL 增强 patch |
| **新增面** | Qwen3-VL 全链路（NPU 上从训练到 rollout 完整跑通 VLM RL） | 面向规模化 RL 的系统能力（确定性、对齐、下发、内存），均可通过开关/env 独立启用 |

**一句话总结**：

- **NPU 侧做的是"平台移植"（make it work）**——解决 MindSpeed/TorchScript/HCCL/Ray/算子层面的所有不兼容，让同一套 RL 流程在异构硬件上跑通，顺带补齐 Qwen3-VL 全链路；
- **GPU 侧做的是"RL 训练正确性与系统效率"（make it right & efficient）**——训练与推理数值对齐、确定性推理保证可复现、top-p 采样重放保证策略梯度语义正确、两种权重同步通道（NCCL / 共享存储 delta）、以及训练阶段的显存与 host 内存腾挪。

两者偶有交叉（NPU patch 中也有正确性修复，GPU 的 top_p/deterministic patch 也覆盖了 Ascend 路径），但分工边界清晰，且 NPU 侧以 GPU 基础 patch（v0.5.7）为底座叠加，说明官方的维护模式是：**GPU 基础 patch 之下是通用功能，NPU 层只做硬件适配，不做功能分叉**。

---
---

# 第二部分：哪些 patch 内容可以梳理进 sglang

> 核对基准：本地 sglang fork `D:\WorkSpace\sglang-prwork`（SeaAsters/sglang，2026-11-03，最新 checkout）。
> 注意：sglang **官方**上游（sgl-project/sglang）状态可能与本地 fork 有差异，正式提 PR 前需再对官方 main 核对。

## 5. 核对方法与分类总览

用每个 patch 的标志性符号（新增函数名、环境变量、参数名）在本地 sglang 中全文检索：命中 = 已在上游；未命中 = 上游缺失。

| 类别 | 含义 | 涉及 patch |
|---|---|---|
| **A. 应梳理进 sglang** | sglang 职责、上游缺失、通用性高 | top_p、pull_weights、release_hicache、deterministic 大部分、sglang.patch 通用部分 |
| **B. 上游已有，可剔除** | patch 内容已被 sglang 吸收 | NPU sglang.patch 大部分、部分通用设施 |
| **C. 与上游方向冲突** | 不能直接搬，需要重验决策 | NPU qwen3_vl deepstack |
| **D. 不属于 sglang** | Megatron / MindSpeed / slime 训练侧 | megatron ×2、mindspeed、megatron-bridge、slime.patch 全部、协议兼容层 |

## 6. A 类：建议梳理进 sglang（上游 PR 候选）

### 6.1 sglang-top_p.patch —— 整体迁移（价值最高、边界最干净）

- 核对结果：`renorm_logprob_over_top_p` / `top_p_token_ids` 在本地 sglang **完全不存在**；`layers/utils/` 下也没有 top_p.py。
- 迁移理由：纯推理侧能力——nucleus keep-mask、在 top-p 支持集 ∪ {已采样 token} 上重归一化 logprob、meta_info 返回 top-p 元数据。触发走 sampling_params 级开关，**不依赖 slime 的任何训练侧概念**。
- 迁移注意：plumbing 链条长（sampler → logits_processor → io_struct → detokenizer → ReqLogprob → tokenizer_manager → PD 分离传输），建议拆 2~3 个 PR：① 核心算法 + sampler 挂载；② logprob/meta_info 全链路；③ spec decode + PD 分离。上游已有 `rl_on_policy_target`，PR 可关联"RL 采样语义正确性"主题。

### 6.2 sglang-pull_weights.patch —— 整体迁移（框架顺位自然）

- 核对结果：`/pull_weights` 端点、`weight_sync/local_checkpoint.py` **不存在**；但 `srt/weight_sync/` 目录已存在（tensor_bucket.py、utils.py），pull_weights 是该子系统的自然扩展。
- 迁移理由：共享存储版本化下发（full checkpoint / zstd XOR/overwrite delta + 逐 tensor 校验和）对多机、对象存储、PD 分离部署通用。
- 边界：发布端（trainer 侧写 delta）留在 slime，sglang 只收"读端"，正好是干净的一半；`custom_pull_weights_pre_read_hook` 做成可选参数。

### 6.3 sglang-release_hicache.patch —— 整体迁移

- 核对结果：`release_hicache`、`release_memory_occupation` 在 `mem_cache/pool_host/base.py` **不存在**（本地 pool_host 已大幅演进，但 host 缓冲释放/恢复没有做）。
- 迁移理由：`release/resume_memory_occupation` 是 sglang 已有内存管理面，把 HiCache host 缓冲纳入释放范围是纯通用增强。
- 迁移注意：本地 pool_host 已新增 mha_mxfp8/npu_memfabric/qsa 等池，patch 基于 old base.py 的钩子重构需 rebase；unified 共享 backing 的引用计数逻辑要适配新目录。

### 6.4 sglang-deterministic.patch —— 大部分可迁移（拆"通用 kernel/算法"与"slime 触发"两层）

| 改动 | 上游状态 | 迁移建议 |
|---|---|---|
| `batch_invariant_ops` 基础设施、`enable_deterministic_inference` | **已存在**（mm/log_softmax/mean/rms_norm 等） | 剔除，作为基线 |
| `router_gemm_batch_invariant`（16 行分块固定归约） | 不存在 | **迁**：纯通用 batch-invariant 算子 |
| `glm5_router_gemm` JIT kernel + 测试 | 不存在 | **迁**：新模型算子，sglang kernels/jit 体系内 |
| `enable_fp32_moe_router` + MoEGate 接入 | 不存在 | **迁**：router 精度开关，通用 |
| DeepGEMM `set_batch_invariant`/`use_deterministic_algorithms` 配置 | 不存在 | **迁**：deep_gemm_wrapper 一处接入 |
| MoE grouped GEMM expert M pad 到 128 tile | 不存在 | **迁**：确定性归约正确性修复 |
| DeepEP `align_fp8_quantization` + LL prefill staging | 不存在 | **迁**：dispatch 数值一致性与分窗执行 |
| FP8 per-token-group quant 两个 .cuh 一致性修复（silu expf、`__fdiv_rn` scale、逐元素 FP8 转换） | 不存在 | **迁**：本质是"fused 与非 fused 路径 bit 级一致"的 kernel bug fix；mask 调度分支补全同理 |
| DSA：indexer head-gate DeepGEMM 化、topk 排序确定化、FP8 KV cache dequant 路径 | 不存在 | **迁**：`dsa` 加入确定性 backend 列表一并提 |
| tensor dump 调试工具增强 | 不存在 | **迁**：纯调试工具 |
| `weight_updater.py` 的 `torch.cuda.synchronize()`（IPC buffer 所有权） | 不存在 | **迁**：位于 sglang 侧，IPC 权重更新 race 修复 |
| 注释/测试中为"训练对齐"服务的 slime 语境 | — | 迁移时改写为通用动机（kernel 一致性） |

### 6.5 sglang.patch —— 通用部分迁移，协议部分留下

可迁入 sglang（上游缺失、通用）：

| 改动 | 核对结果 | 建议 |
|---|---|---|
| `/post_process_weights` 通用机制（restore + 重量化） | 上游仅 bailing_moe_v3 有单模型 `post_process_weights_if_quant`，**无通用机制** | **迁**：通用化为 model_runner + endpoint |
| PD transfer/bootstrap 超时（`SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT`）+ 在途请求 abort + release/resume 清理在途传输 | 不存在 | **迁**：PD 稳定性 |
| DSA indexer `_match_head_gate_q_scale`、head pad 32、INDEXER_ROPE_NEOX_STYLE | 不存在 | **迁**：模型正确性修复 |
| EAGLE draft cuda graph topk clamp | 不存在 | **迁**：崩溃修复 |
| radix cache 只缓存 committed KV + dec_lock_ref break | 不存在 | **迁**：前缀缓存正确性 |
| mooncake dst_addr、fastapi route path、TTFT/ITL 统计修正 | 不存在 | **迁**：散装 bug fix |
| glm4v_moe encoder_only / language_only | 不存在 | **迁**：VIT/LM 拆分部署 |
| PD 阶段耗时指标（pd_* 系列 meta_info） | 不存在 | **迁**：可观测性 |

留在 slime 侧（slime↔sglang 协议层，sglang 官方大概率不收）：

- `update_weights_from_distributed/tensor` 免 begin/end session、`disable_draft_model`——旧版 slime 客户端兼容，slime 升级到新版协议即可删除。

## 7. B 类：上游已有，patch 可剔除

| patch 内容 | 本地 sglang 现状 |
|---|---|
| llama.py 接入 `split_qkv_rmsnorm_rope`（NPU sglang.patch） | **已上游**，且推广到 qwen3.py、glm4_moe.py、minimax_m3.py、dflash.py、llada2.py 共 6 个模型 |
| `vocab_parallel_embedding.py` 的 `disable=_is_npu`（禁 torch.compile） | **已上游**（逐字一致） |
| rotary_embedding 的 `_is_npu` fallback | **已上游**（条件一致） |
| ascend_backend 统一 TND / `npu_fused_infer_attention_score` | **已上游**（TND 出现 13+ 处，文件演进到 3800+ 行） |
| qwen3_vl `fast_pos_embed_interpolate` numpy 化 | **部分上游**（`np.linspace` 已在用） |
| `enable_deterministic_inference` + `batch_invariant_ops` 框架 | **已上游** |
| `rl_on_policy_target` 参数 | **已上游** |

结论：**NPU 侧 sglang.patch 的绝大部分已被上游吸收，需基于新版 sglang 重新生成（预计大幅缩水）**。

## 8. C 类：与上游方向冲突，需决策后处理

**qwen3_vl deepstack 处理顺序**：

- 上游：保留 `post_residual_addition` 机制（与 HF 行为对齐）；
- NPU patch：删除该机制，改为 decoder layer 返回后直接 `hidden_states += deepstack_embeds`。

这不是"上游化"而是**分叉**：要么 NPU patch 基于旧上游、现已过时；要么 NPU 上 post_residual_addition 有实际 bug。处置：在最新上游上重跑 NPU Qwen3-VL 验证，无问题则删掉该 NPU 改动；仍有问题则定位清楚后作为 NPU backend 内部 workaround 提给 sglang，不改模型主逻辑。

## 9. D 类：不属于 sglang，留在 slime patch

| Patch | 原因 |
|---|---|
| npu_patch/megatron.patch | Megatron-LM 的 NPU 兼容（删 jit_fuser、npu dtype） |
| npu_patch/megatron-bridge.patch | Megatron-Bridge 的 MindSpeed 层映射 |
| npu_patch/mindspeed.patch | MindSpeed 自身适配 |
| npu_patch/slime.patch | slime 训练侧（Ray NPU 调度、hccl、repatch、megatron_to_hf、critic/mbridge plugin） |
| sglang.patch 的协议兼容层 | slime↔sglang 客户端协议（见 6.5） |

## 10. 行动建议（优先级排序）

1. **第一批（低风险、高价值）**：top_p 重放、release_hicache、`/post_process_weights` 通用化、PD 超时、EAGLE clamp、radix committed-len——sglang 职责内的独立改进，先在本地 fork 落地再考虑官方 PR。
2. **第二批（需拆分改写）**：deterministic patch——把 kernel/算法通用部分与 RL 对齐动机解耦后迁入，注释与测试去 slime 化。
3. **第三批（框架类）**：pull_weights——作为 `weight_sync/` 子系统扩展迁入。
4. **清理动作**：NPU sglang.patch 按新版 sglang 重新生成（预计大部分消失）；qwen3_vl deepstack 重验后决去留；slime 协议兼容层随 slime 升级评估删除。
5. **官方 PR 前置**：本地基准是 SeaAsters fork，向 sgl-project/sglang 提 PR 前需对官方 main 重新核对 B 类结论。
