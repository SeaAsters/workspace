# slime sglang patch 排查 md

> 2026-11-09 创建，同日更新（官方 main 全量复核 ＋ slime 上游动态同步）。排查对象：[slime/docker/npu_patch](https://github.com/THUDM/slime/tree/main/docker/npu_patch)（sglang 部分）与 [slime/docker/patch/latest](https://github.com/THUDM/slime/tree/main/docker/patch/latest)（slime 已同步至 `79aa41a8`，2026-11-06）。
>
> **核对基准**：sglang 官方 main `ab84df61dd`（2026-11-09，本地 fetch 后 git grep 全量复核，不受 API 限流）＋ 本地 SeaAsters fork（`sglang-prwork`，2026-11-03）。
> **口径**：NPU 侧 `sglang.patch` 打在 sglang `dce8b06`（sglang-slime 分支）+ `docker/patch/v0.5.7` 基础 patch 之上；`patch/latest` 基于 arg_groups/msgspec 结构的新版 sglang。
> 判定用语与 verl 排查文档一致：**已修复/已吸收**（上游有等价实现，patch 变空操作）、**未修复**（上游缺失）、**不应下沉**（slime/训练侧职责，sglang 不收）、**NA**（不适用）。

11-09

---

## 一、NPU 侧（npu_patch/sglang.patch 逐项拆解）

| patch                               | 作用                                                                                             | 官方 main      | 备注                                                                                                                                                                                                                     |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| _patch_ascend_attn_tnd_unify        | 非 MLA 注意力统一 `npu_fused_infer_attention_score`（TND 布局），删掉 bs<tp 的 `_npu_paged_attention`/BSH 特例 | 已吸收          | TND 遍布 `ascend_backend.py`（13+ 处），文件演进到 3800+ 行，patch 项过时                                                                                                                                                              |
| _patch_npu_graph_runner_mla_attr    | NPU graph 重放属性固定 MLA，删 bs<tp / target_verify / use_fia 分支                                      | NA           | `npu_graph_runner.py` 在新版被整体重构（仅剩 vit_npu_graph_runner 等），无对应物，上游重构吸收                                                                                                                                                  |
| _patch_rotary_npu_fallback          | NPU 不走 fused RoPE；bf16 q + fp32 cos_sin 走 forward_native                                       | 已吸收          | `layers/rotary_embedding/base.py:126` 条件逐字一致                                                                                                                                                                           |
| _patch_vocab_embed_disable_compile  | NPU 上禁用 `torch.compile`（get_masked_input_and_mask）                                             | 已吸收          | `vocab_parallel_embedding.py:138` `disable=_is_npu` 逐字一致                                                                                                                                                               |
| _patch_llama_split_qkv_rmsnorm_rope | llama 接入 sgl_kernel_npu 融合算子（QKV 切分+RMSNorm+RoPE）                                              | 已吸收，且更强      | 官方 main 的 [llama.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)/qwen3.py/llada2.py 均已接入，llada2 还用了 `split_qkv_rmsnorm_rope_pos_cache_half_npu` 变体；本地 fork 推广到 6 个模型。patch 变空操作 |
| _patch_qwen3_op_signature           | qwen3/qwen3_moe 适配算子新签名（q_weight/k_weight/eps）＋ lm_head `use_attn_tp_group`                    | 已吸收          | 随算子一起进上游（qwen3.py 已用）                                                                                                                                                                                                  |
| _patch_qwen3vl_pos_embed_numpy      | pos embed 插值改 numpy（顺带修 idx 少乘 t、split 尺寸两个 bug）                                               | 基本吸收         | `qwen3_vl.py:522` 已用 `np.linspace`，torch 分支并存                                                                                                                                                                          |
| _patch_qwen3vl_deepstack_simplify   | 删除 `post_residual_addition` 机制，deepstack 改为层返回后直接相加                                            | **未吸收，方向相反** | main 保留该机制并抽象为 `layers/layer_boundary/prepare.py`（gigachat35 等共用）；`qwen3_vl.py:1235-1279` 仍在用。**不应下沉**，见 §4.2                                                                                                          |
| _patch_qwen3vl_lm_head_tie_guard    | 仅 `tie_word_embeddings` 时才把 embed_tokens 载入 lm_head                                            | 已被演进替代       | 新版 `qwen3_vl.py:1366-1371` 有更精细的 tie 配置传播处理，patch 项过时                                                                                                                                                                  |
| _patch_profiler_dir_guard           | profiler 输出目录 `..` 路径穿越检查                                                                      | 未修复          | 可下沉（通用小修，低风险）                                                                                                                                                                                                          |
| CI：sgl-kernel-npu tag 升 2025.12.31  | CI 依赖版本                                                                                        | —            | CI 维护项，NA                                                                                                                                                                                                              |

**小结**：NPU 侧 sglang.patch **绝大部分已被官方 main 吸收且走得更远**（融合算子、TND、compile 禁用、rope fallback 均逐字或更强等价）。剩余两项：deepstack 简化与 main **方向冲突**不应下沉；profiler 目录检查可下沉。**该 patch 需基于新版 sglang 重新生成，预计大幅缩水。**

---

## 二、GPU 侧（docker/patch/latest）

### 2.1 sglang.patch（基础增强）

| 项 | 作用 | 官方 main | 备注 |
|---|---|---|---|
| /post_process_weights | 权重更新后 restore 原始权重并重新量化打包（量化模型 RL 更新） | **部分**：[loader.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py) 已有 `restore_weights_before_loading`（加载时机）；无运行时端点 | 应下沉：复用 loader 函数做运行时端点，见 §4.4。bailing_moe_v3 仅有单模型版 `post_process_weights_if_quant` |
| PD transfer/bootstrap 超时 | `SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT`（默认 600s）+ 请求 abort | 未修复 | 应下沉 |
| PD release/resume 清理在途传输 | decode 侧释放内存时清理 in-flight 传输并释放注册 buffer | 未修复 | 应下沉 |
| DSA indexer q_scale 对齐、head pad 32 | 修复小头数/精度问题 | 未修复 | 应下沉（正确性） |
| EAGLE draft topk clamp | draft 输出 topk_p/topk_index 钳到合法范围防崩溃 | 未修复 | 应下沉 |
| radix cache committed-len、dec_lock_ref break | 只缓存已提交 KV；允许跨树节点解锁 | 未修复 | 应下沉（前缀缓存正确性） |
| mooncake dst_addr、fastapi path、TTFT/ITL | 散装修复 | 未修复 | 应下沉 |
| glm4v encoder_only / language_only | VIT/LM 拆分部署 | 未修复 | 应下沉 |
| PD 阶段耗时指标（pd_* meta_info） | PD 分阶段可观测性 | 未修复 | 应下沉 |
| 免 begin/end session、disable_draft_model | 旧版 slime 客户端协议兼容 | 未修复 | **不应下沉**：slime↔sglang 协议层，slime 职责；slime 升级新协议后可删 |

### 2.2 sglang-deterministic.patch（确定性推理 / 训推对齐，12 件事）

| # | 改动 | 目的 | 官方 main | 结论 |
|---|---|---|---|---|
| 0 | `batch_invariant_ops` 框架、`enable_deterministic_inference` | 批不变推理基座 | **已存在**（mm/log_softmax/mean/rms_norm 等） | 基线，剔除 |
| 1 | FP8 per-token-group quant 两个 .cuh 一致性修复（silu 换 expf、scale 用 `__fdiv_rn`、FP8 逐元素转换） | fused 与非 fused 路径 bit 级一致 | 未修复 | 应下沉：本质是 kernel 数值一致性 bug fix |
| 2 | `router_gemm_batch_invariant`（16 行分块固定归约） | router logits 批不变 | 未修复 | 应下沉 |
| 3 | `glm5_router_gemm` JIT kernel（BF16 [M,6144]×[256,6144]ᵀ→FP32） | GLM5 router 快速通道 | 未修复 | 应下沉 |
| 4 | `enable_fp32_moe_router` 参数 | DeepSeek 式 router FP32 logits | 未修复（upstream/main `ab84df61dd` git grep 已复核） | 应下沉 |
| 5 | DeepGEMM `set_batch_invariant`/`use_deterministic_algorithms` 配置 | 训推同选不变 kernel 变体 | 未修复 | 应下沉 |
| 6 | MoE grouped GEMM expert M pad 到 128 tile | 消除 tile 内归约随 M 变化 | 未修复 | 应下沉 |
| 7 | DeepEP `align_fp8_quantization` ＋ LL prefill staging（`SGLANG_DEEPEP_LL_PREFILL_STAGING`） | dispatch 数值一致；prefill 超容量分窗 | 未修复 | 应下沉 |
| 8 | DSA indexer head-gate DeepGEMM 化、topk 排序确定化、FP8 KV cache dequant 路径 | DSA 确定性 ＋ flashmla_sparse 吃 FP8 cache | 未修复 | 应下沉 |
| 9 | `dsa` 加入确定性 attention backend 列表 | 后端注册 | 未修复 | 应下沉（随 #8 一并） |
| 10 | tensor dump 增强（layer-outputs-only、分块、模块过滤） | 训推逐层对齐调试 | 未修复 | 应下沉（纯调试工具） |
| 11 | weight_updater `torch.cuda.synchronize()`（IPC buffer 所有权） | 修 CUDA IPC 权重更新 race | 未修复 | 应下沉（sglang 侧修复） |

> 注：#1~#3、#5~#8 的动机是训推一致（训练对齐），但实现本身是通用 kernel/算法修复，下沉时应把注释与测试里的 slime 语境改写为通用动机。

### 2.3 其余整体 patch

| patch | 作用 | 官方 main | 备注 |
|---|---|---|---|
| sglang-top_p.patch | top-p 采样重放：nucleus keep-mask ＋ 在 top-p 支持集 ∪ {已采样 token} 上重归一化 logprob，meta_info 返回 top-p 元数据 | **未修复**（`renorm_logprob_over_top_p`/`top_p_token_ids` 无任何命中） | 应整体下沉：纯推理侧能力，sampling_params 级开关，不依赖 slime。链条长，建议拆 2~3 个 PR |
| sglang-pull_weights.patch | `/pull_weights` 端点 ＋ `weight_sync/local_checkpoint.py`：共享存储版本化下发（full / zstd XOR/overwrite delta ＋ 逐 tensor 校验和） | **未修复**（[weight_sync/](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/weight_sync) 仅 tensor_bucket.py、utils.py） | 应下沉读端，见 §4.3；发布端（trainer 写 delta）留 slime |
| sglang-release_hicache.patch | `--release-hicache`：release_memory_occupation 时释放 HiCache host 缓冲（unregister pinned、malloc_trim），resume 重建 | **未修复**（`pool_host/base.py` 无 release/resume） | 应下沉：sglang 已有内存管理面的通用扩展；需 rebase 到新版 pool 家族（mha_mxfp8/npu_memfabric/qsa 等） |
| sglang-nemotron-multi-image.patch（2026-11-06 新增） | NanoNemotron VL 多图占位符按数量展开并校验（原实现 `replace(..., 1)` 只替换第一个占位符，多图错位） | **未修复**（`expand_image_placeholders` 无命中） | 应下沉：patch 内注释明写 `TODO(shifangx): upstream to SGLang, then drop this patch` |
| sglang-radio_layerscale.patch（2026-11-06 新增） | C-RADIO 视觉模型无 LayerScale，ls1/ls2 保持常量 1.0 不建 Parameter（否则 colocated release/resume 会把不存在的参数清零） | **未修复**（internvl.py 无 radio 分支） | 应下沉：model_type 门控，真 InternViT 分支不受影响 |

---

## 三、不涉及 sglang 的 patch（不下沉，各自归属）

| patch | 归属/原因 |
|---|---|
| npu_patch/megatron.patch | Megatron-LM 职责（删 jit_fuser、npu dtype），提给 NVIDIA/Megatron-LM |
| npu_patch/megatron-bridge.patch | Megatron-Bridge 职责（MindSpeed 层映射、deepstack varargs） |
| npu_patch/mindspeed.patch | MindSpeed 自身（CP 参数、2048 因果 mask workaround） |
| npu_patch/slime.patch | slime 训练框架职责（Ray NPU 调度、hccl、megatron_to_hf、critic/mbridge plugin） |
| patch/latest/megatron.patch | Megatron-LM 职责（checkpoint 部分加载、MLA YaRN RoPE 修复、hybrid optimizer fp32 刷新） |
| patch/latest/megatron-sglang-aligned.patch | 训推一致相关：**是 Megatron 模仿 sglang** 的 residual+RMSNorm FP32 边界（`MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS`），sglang 侧的 fused 行为本就存在，无需改动 |

---

## 四、深挖

### 4.1 NPU 已吸收项的代码级核对

| patch 项 | 官方 main / 本地 fork 证据 |
|---|---|
| split_qkv_rmsnorm_rope | 官方 main（ab84df61dd）：**9 个模型**接入（llama / qwen3 / qwen3_moe / qwen3_next / qwen3_5 / glm4_moe / minimax_m3 / dflash / llada2）；`llada2.py` 还用了 `split_qkv_rmsnorm_rope_pos_cache_half_npu`（带位置缓存的更强变体，slime patch 没有）。本地 fork（2026-11-03）：`llama.py:82/253`，6 模型。**补丁变成数值等价的空操作** |
| disable=_is_npu | `vocab_parallel_embedding.py:138` `@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)` 逐字一致 |
| rotary fallback | `layers/rotary_embedding/base.py:126` `and not (_is_npu)` 条件一致 |
| TND 统一 | `ascend_backend.py` 全文 TND 13+ 处；`_npu_paged_attention` 仅剩 MLA/graph 专属路径（:3767/:3898），语义与 patch 的"非 MLA 统一"不冲突 |

### 4.2 _patch_qwen3vl_deepstack_simplify：与 main 的方向冲突

| 维度 | main 当前实现 | slime NPU patch |
|---|---|---|
| deepstack 注入 | `post_residual_addition` 随 layer 调用传入，加在 residual 之后（`qwen3_vl.py:1243`），与 HF 逐字对齐（注释 `:1235` 明说） | 删除该参数，layer 返回后直接 `hidden_states += deepstack_embeds` |
| 基础设施 | 已抽象为 `layers/layer_boundary/prepare.py`（gigachat35 等模型共用），是正式演进方向 | 无 |
| 最后一层 | `post_residual_addition=last_deepstack`（`:1279`） | 与 norm 一起简化 |

这是**分叉**而非滞后的一种可能：要么 patch 基于旧上游、现已过时；要么 NPU 上 `post_residual_addition` 路径有实际 bug。**处置**：在最新官方 main 上重跑 NPU Qwen3-VL 验证；无问题则删除该项；有问题则定位后在 NPU backend 内 workaround，不改模型主逻辑。**不应直接下沉。**

### 4.3 patch_pull_weights vs sglang 主线权重同步

| 维度 | slime patch | sglang 主线 |
|---|---|---|
| 通道 | 共享存储版本化：trainer 发布 `weight_v{N}/`（full 或 zstd XOR/overwrite delta），各 host 拉取物化本地 checkpoint | NCCL 进程组（update_weights_from_distributed）/ CUDA IPC / update_weights_from_disk |
| 时机 | 训练侧发布 → host 拉取 → 复用 `update_weights_from_disk` 装载 | 训练侧直推在线更新 |
| 数据布局 | HF checkpoint 原生布局 ＋ 逐 tensor 校验和（xxh3-128/blake3/adler32），mmap 原地打 delta | 模型参数布局 |
| 适用 | 多机、对象存储、带宽受限、发布/消费解耦 | 单机、低延迟 |
| 与主线关系 | 最后一步复用主线 disk 路径；`weight_sync/` 已有 tensor_bucket，pull 是该子系统的自然扩展 | — |

**结论**：读端（端点 + local_checkpoint.py）应下沉 sglang；`custom_pull_weights_pre_read_hook`（对象存储一致性 hook）做成可选参数。发布端（写 delta）是 trainer 职责，留 slime。

### 4.4 /post_process_weights 与官方 loader 的关系

官方 main 的 `model_loader/loader.py` 已提供模块级 `restore_weights_before_loading(model, target_device)`，`quantization/base_config.py` 定义了 `restore_weights_before_loading` 钩子——即 patch 中 "restore_weights_before_load" 这半件事在**加载时机**上已有官方实现。patch 的增量在于：把它与 `process_weights_after_loading` 组装成**运行时权重更新端点**（RL 每轮更新量化模型后重新打包）。下沉时应改为复用官方 loader 函数，而不是自带一份遍历逻辑。

### 4.5 slime 上游动态（2026-11-06，`79aa41a8`）：基线升级的验证

- **npu_patch 零变化**：仍钉在 sglang `dce8b06` / CANN 8.5.0 组合，rebase 债务持续积累（同期官方 main 的 NPU 支持高频演进：dspark kernel、DSA+PD 分离、Kimi K3 attention、qwen3.6 spec 测试、dsv4_flash 128K 等）。
- **patch/latest/megatron.patch 重写**（Megatron-LM 基线 `4b4acac9` → `ae76c60e`），删掉两块——① YaRN RoPE 修复：已验证新基线上游用 `K_BLOCK/V_BLOCK/ROT_BLOCK = next_power_of_2(...)` ＋"padded dims 才建掩码"的方式**另法修复**（blob hash 对比 ＋ kernel 内容双重确认，非丢修复）；② `param_and_grad_buffer` 的 `_allocate_buffer` 等价重构（无功能损失）。新增 `multi_token_prediction.py` 的 position_ids 断言放宽（Nemotron VL 场景）。
- **megatron-sglang-aligned.patch 纯 rebase**：TransformerLayer 上游演进为 TwoStage 结构、加入 fine-grained offloading，hunk 挪位但 FP32 residual 逻辑一字未动。
- **新增两个 sglang patch**（服务 Nemotron 3.5 Super VL，主仓库 PR #2471）：见 §2.3。

这个维护模式（基线升级时主动剔除上游已修复的部分）是 GPU 侧 patch 的健康形态；**NPU 侧应对齐同一节奏**——当前 npu_patch 与官方 main 的差距只会越拉越大。

---

## 五、行动建议与参考

**下沉优先级**：

1. 第一批（低风险独立改进）：top_p 重放、release_hicache、post_process_weights 通用化、PD 超时、EAGLE clamp、radix committed-len、profiler dir guard、nemotron-multi-image、radio_layerscale（后两个为 11-06 新增小 patch，前者官方已标 TODO upstream）；
2. 第二批（需拆分改写）：deterministic 的 #1~#11（去 slime 语境）；
3. 第三批（框架类）：pull_weights 读端；
4. 清理：NPU sglang.patch 基于新版重新生成（大部分消失）；qwen3_vl deepstack 重验后决去留；协议兼容层随 slime 升级评估删除。

**参考**：

- 排查对象：[slime/docker/npu_patch](https://github.com/THUDM/slime/tree/main/docker/npu_patch)、[slime/docker/patch/latest](https://github.com/THUDM/slime/tree/main/docker/patch/latest)（slime `79aa41a8`，2026-11-06）
- 官方 main 证据文件：[llama.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama.py)、[weight_sync/](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/weight_sync)、[layer_boundary/prepare.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/layer_boundary/prepare.py)、[model_loader/loader.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py)
- 本地 fork：`D:\WorkSpace\sglang-prwork`（SeaAsters/sglang，2026-11-03；upstream/main 已 fetch 至 `ab84df61dd`，2026-11-09）
- 核对方式（2026-11-09 全量）：`git fetch upstream` 后 `git grep <签名> upstream/main -- python/`（本地全量、不受 API 限流）；辅助 `gh api contents/<path>?ref=<sha> --jq .sha` 取 blob hash 与 patch index 行比对，判定"基线已修 vs 丢修复"
