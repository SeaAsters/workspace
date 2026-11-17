# slime-ascend 下沉 sglang：完整分析

> 对象：`gitcode.com/Ascend/slime-ascend` `main@e58ab12`（本地 `D:\WorkSpace\slime-ascend`），补丁清单以 `patches/MANIFEST.toml` 为准。
> 上游基准（**2026-11-17 拉取最新后复核**）：`sgl-project/sglang` `upstream/main@c67db7c762`、`sgl-project/sgl-kernel-npu` `origin/main@3f4d922`；`THUDM/slime` `origin/main@79aa41a8`（无新提交）、`slime-ascend` `origin/main@e58ab12`（无新提交）。补丁自身的基线是 sglang `v0.5.21`（`e00930c5`）。
> 方法：逐补丁、逐文件、逐 hunk 阅读 → 对每个可识别条目在上游用 `git grep` 找签名 → 有命中=上游已有/已吸收，无命中=仍然缺 → 再按职责边界判定去向；另用 `patch --dry-run` 量化补丁在新 main 上的漂移（见 §9）。全文结论可由文末第 11 节的命令复现。

---

## 1. 问题背景与判定口径

slime 是 RL post-training 框架：Megatron 训练 + SGLang 推理 rollout。推理侧的能力（采样、权重热更新、内存睡眠、PD 分离、HiCache）**全部由 SGLang 提供**，slime 只负责编排。所以每当 slime 需要 SGLang 多暴露一点东西，就产生一个补丁；补丁挂着不 upstream，就变成长期的 rebase 债务。

slime-ascend 是同一框架的昇腾适配版，它的做法是**纯 patch 管理**：上游 slime 以 submodule 引入，NPU 侧改动一律写成 patch 由 `build.sh` 重放；SGLang 则按 `scripts/quick_install.sh` 从 `v0.5.21` tag 拉源码、打 5 个补丁、再叠加 1 个 NPU 小补丁；`sgl-kernel-npu` 从 release 装 wheel，再按 3 个补丁重编译 `torch_memory_saver`。

本文的"下沉"就是把 patch 里的代码搬进上游仓库，让 patch 栈变薄。去向有四类：

| 判定 | 含义 |
|---|---|
| **可下沉 / sglang 主线** | 与硬件无关的推理能力或缺损修复，sglang 该收 |
| **可下沉 / sglang NPU backend** | Ascend 特有的推理侧行为，落 `hardware_backend/npu` 或 `utils/torch_memory_saver_adapter.py` |
| **可下沉 / sgl-kernel-npu** | NPU 版 `torch_memory_saver`、算子的缺陷，落 Ascend 自己的 kernel 仓 |
| **无法下沉** | 训练框架、Megatron/Bridge、协议兼容层、CANN/分配器——各归其主，不该由 sglang 承接 |
| **应删除** | 死代码与噪音，删比沉更对 |

---

## 2. 扫描结果：补丁栈全景

MANIFEST 登记 22 条；目录内另有 2 个文件未登记、不参与构建。

| 目标仓 | 条数 | 内容 | 是否推理侧 |
|---|---|---|---|
| slime | 6 | npu-compat 4（设备/Ray/运维）+ upstream-fixes 2 | 否 |
| Megatron-LM | 2 | NPU 训练接口 / 异步 ckpt | 否 |
| Megatron-Bridge | 3 | VL、Transformers 5.x、Qwen | 否 |
| mbridge | 1 | FP8 反量化设备选择 | 否 |
| Megatron-Adaptor | 1 | 异步 preload 序列化 | 否 |
| **sglang** | **6** | 1 个综合 + 4 个功能件 + 1 个 NPU 小补丁 | **是** |
| **sgl-kernel-npu** | **3** | TMS 分配生命周期 / 图接口 / 对齐与嵌套 | **是** |

推理侧补丁体量：

| 补丁 | 文件 | +/− | 性质 |
|---|---|---|---|
| `sglang.patch` | 29 | +636 / −95 | 混合：PD 超时、权重端点、指标、若干正确性修复、协议兼容、CLI 别名 |
| `sglang-top_p.patch` | 18 | +551 / −6 | 单一功能：top-p 采样重放 |
| `sglang-release_hicache.patch` | 13 | +276 / −32 | 单一功能：HiCache host 内存释放 |
| `sglang-pull_weights.patch` | 8 | +384 / −0 | 单一功能：共享存储权重拉取（读端） |
| `sglang-deterministic.patch` | 27 | +807 / −81 | 混合：确定性推理，CUDA 与通用混杂 |
| `0001-tms-forward-graph-args.patch` | 1 | +4 / −4 | NPU 专属：签名透传 |
| `sgl-kernel-npu/0001..0003` | 2/2/4 | +71 / −17 | NPU 专属：TMS 缺陷修复 |

**三条总体判断**

1. 22 条里只有 9 条与推理引擎相关。其中 **7 条可整体下沉**（`top_p`、`release_hicache`、`pull_weights`、TMS adapter、TMS 三项），另 **2 条是混合体**——`sglang.patch` 与 `deterministic`：主体可沉，但各自含有局部不可沉（协议兼容层、CUDA 专属内核）与应删除（inflight 段、等价改写）的成分。
2. ascend 在 sglang 侧**没有自造改动**——5 个功能补丁与上游 slime 的 GPU 版本同源，NPU 真正独有的只有 4 处（1 个 sglang 小补丁 + 3 个 TMS 修复）。这说明"sglang 吸收 NPU 适配"这条路在本仓上已经走通了绝大部分。
3. 有 4 类内容**不该沉、该删**：为旧启动脚本保留的 CLI 别名、`begin/end session` 协议兼容、`/v1/loads` 的 inflight 段、以及一条把新多模态加载路径回退成 legacy 的改动（后者需先查根因）。

### 2.1 拉最新 main 后的复核（2026-11-17）

sglang 一天内进了 **81 个提交**，sgl-kernel-npu 进了 16 个（并打出 `2026.11.0.post4/5/6` 三个新 tag）。逐项重跑签名后，**15 个待沉条目全部仍然零命中**，`radix_cache.py:619` 的 `assert node is self.root_node` 原样在，adapter 三处仍是 `def cuda_graph(self, **kwargs)`。四条值得写进结论的变化：

1. **NPU 图捕获路径被重构，但 N1 的 bug 还在。** `#49999 用 npugraph_ex 编译 decode graph` 重写了 `npu_cudagraph_backend.py` 的捕获段（去掉了 `skip_guard_context`、`torch.compile` 改 `fullgraph=False`），**但 `graph_ctx(graph, pool=…, stream=…, auto_dispatch_capture=True)` 的位置传参一行没动**，adapter 侧也没变——N1 依旧是"改签名即可"的状态。
2. **但触发条件变严了**：图内存回收现在由 `enable_memory_saver and SGLANG_MEMORY_SAVER_CUDA_GRAPH` 门控，而该 env 是 `EnvBool(False)`。也就是说 N1 不是"天天炸"，而是**一旦为了省显存开启 `SGLANG_MEMORY_SAVER_CUDA_GRAPH=1` 就炸**。这让它从"最高优先级崩溃"降为"潜伏的坑"，但仍必修（colocate 想回收图内存就得开）。
3. **NPU HiCache host pool 换了承载方式**（`#50676 hicache: back hugepage host pools with the mmap allocator`）：`AclHostMemoryBackend` 从 pageable `torch.empty` 改成 mmap 块（`SGLANG_HUGEPAGE_SIZE` 支持 2MB/1GB hugetlb，失败回落 4KB），`MemfabricHostMemoryBackend` 加了 `executor=AICPU` + `OFFLOAD_FLAG_GIANT_PAGE`。这把 S2 的 NPU 结论钉得更死：**release_hicache 在 Ascend 上要释放的是 mmap 映射，不是 pinned buffer**，而 `storage/mmap/mmap_allocator.py` 目前只有 `alloc_mmap` / `alloc_shm`，**没有任何 free/munmap 接口**——NPU 侧要拿收益，得先给分配器补释放路径。
4. **权重更新的 stream 序问题上游已修**（`#50717 wait_for_launched_forward()`，已插入 disk/distributed/tensor/ipc 与 begin/end session 全部路径）。这与 slime-ascend 在训练 actor 里加的 `torch.npu.synchronize()` 是同一族问题的上游解，但不等价：上游解决的是"forward 还在读权重"，TMS 需要的是"设备级 unmap 前所有在途 capture 收敛"。所以该 workaround 的**动机收窄但不消失**，根因仍在 TMS（K1）。

另外：4 个 NPU 补丁（sglang 的 `0001` + sgl-kernel-npu 的三条）在新 main 上 `patch --dry-run` **0 失败**，可原样提；5 个功能补丁则全部有 15%~38% 的 hunk 打不上（见 §9）。

---

## 3. 可下沉 A：sglang 主线（通用能力）

### 3.1 三个大件

**S1 · top-p 采样重放**（`top_p`）—— 可下沉，建议拆 3 个 PR

- 新增 `layers/utils/top_p.py`：nucleus keep-mask、`renorm_logprob_over_top_p`（在 top-p 支持集 ∪ {已采样 token} 上重归一）、`get_top_p_token_ids_from_probs`、`attach_top_p_log_probs`。
- 贯通链路：`sampling_params.return_top_p_token_ids` → `sampling_batch_info` → sampler（含 **`use_ascend_backend` 分支**）→ `logits_processor` / `logprob_processor` → `batch_result_processor` / `logprob_result_processor` / `output_streamer` → `tokenizer_manager` 以 base64 组装 `meta_info`（`top_p_token_ids` / `top_p_token_offsets` / `top_p_log_probs`）；spec（eagle/ngram）按 accept_lens 计算；PD 在 `disaggregation/utils.py` 的 `MetadataBuffers` 里加 4096 长度上限的三块 buffer 传递。
- 上游状态：`layers/utils/top_p.py` 不存在，`renorm_logprob_over_top_p` 零命中。上游 `sampler.py:138` 有 `use_ascend_backend`（`sampling_backend == "ascend"`），所以补丁里的 Ascend 分支是打在真实存在的能力上——**这条对 NPU 直接有效**。
- 风险：链条最长（18 文件），必须拆：core 数学 / meta_info 链路 / spec+PD。数学验收点很干净：支持集内概率和为 1、已采样 token 必在支持集内。

**S2 · HiCache host 内存释放**（`release_hicache`）—— 可下沉，但 NPU 收益需要额外做

- `pool_host/base.py` 把 `__init__` 拆成 `_check_host_memory_available` / `_init_host_buffers` / `_post_init_host_buffers` 三个钩子，新增 `release_memory_occupation`（unregister pinned、置空派生视图、缩 `free_slots`/`mem_state`、`malloc_trim`）与 `resume_memory_occupation`；各池实现 `_host_buffer_attr_names`；`unified.py` 的共享 backing 用引用计数（full/swa 两池都释放才真 free）；`scheduler_components/weight_updater.py` 在 offload KV_CACHE 时按 `--release-hicache` 触发；新参数 `--release-hicache`。
- 上游状态：`pool_host/` 下无 `release_memory_occupation`，参数不存在 → 未上游。
- **必须注意的两点**：
  - 上游池家族已扩到 12 个（`dsa`/`group`/`mamba`/`mha`/`mha_mxfp8`/`minimax_m3_nvfp4`/`mla`/`page_interleave`/`page_unified`/`qsa`/`hisparse`/`unified`），每个都要实现钩子，否则 release 漏池。
  - **NPU 侧语义不同**：`pool_host/npu_host_memory.py` 里的 host 池不是 torch pinned buffer——`memfabric` 后端由 MemFabric `Offload` 实例持有，`acl` 后端自 11-17 的 `#50676` 起改为 **mmap 块**（`alloc_mmap`，`SGLANG_HUGEPAGE_SIZE` 支持 2MB/1GB hugetlb）再 `aclrtHostRegister(MAPPED)`。补丁的 "unregister pinned + `malloc_trim`" 在 Ascend 上对不上；而 `storage/mmap/mmap_allocator.py` 只有 `alloc_mmap` / `alloc_shm`、**没有 free/munmap**，所以要拿 NPU 收益，得先给分配器补释放路径，再做 host pool 的 release/resume。**结论：可下沉（GPU 语义正确），但不能指望它顺带解决 NPU 的 host 内存占用。**

**S3 · `/pull_weights` 读端**（`pull_weights`）—— 可下沉（只沉读端）

- 新文件 `weight_sync/local_checkpoint.py`（约 358 行）：版本目录 `weight_v{N}`、full 与 zstd XOR/overwrite 两种 delta、逐 tensor 校验（xxh3-128 / blake3 / adler32）、文件锁、mmap 原地打 delta、线程池、page cache 处理；`/pull_weights` 端点全链路（`io_struct` → `tokenizer_control_mixin` → `scheduler` → `weight_updater`），可选 `--custom-pull-weights-pre-read-hook`（对象存储一致性回调）。
- 上游状态：`pull_weights` 零命中；`weight_sync/` 只有 tensor_bucket 等，没有共享存储通道 → 未上游。
- 边界：**发布端（trainer 写 delta）留在 slime**；读端天然属于 sglang 的 weight_sync 子系统，最后一步还会复用 `update_weights_from_disk`。
- NPU 价值：多机 Ascend 上 HCCL 全量同步昂贵，共享存储 delta 通道收益比 GPU 更明显。

### 3.2 中件（可独立提 PR）

| 编号 | 项 | 改动要点 | 上游状态 |
|---|---|---|---|
| S5 | PD bootstrap/transfer 超时 + 在途清理 | 新 env `SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT`（默认 600s）；decode prealloc/transfer、prefill bootstrap/transfer 四处按 `time_stats.*_queue_entry_time` 判超时 → `prepare_abort` + 503/504；`kv_receiver.abort()`、`disagg_kv_sender.clear()`；`release/resume_memory_occupation` 清空队列并 `deregister/register_buffer_to_engine`；超时计入 `increment_bootstrap_failed_reqs` / `increment_transfer_failed_reqs`；scheduler abort 时同时 abort prefill 侧 sender | env 名零命中；`increment_*_failed_reqs` helper 上游已有（说明指标骨架就位，只是超时路径没接） |
| S6 | `/post_process_weights` 运行时端点 | `restore_weights_before_load` / `post_process_quantization` 两个开关，`model_runner` 用 `device_loading_context` 逐模块调用 `quant_method.restore_weights_before_loading` 与 `process_weights_after_loading`；`http_server` 注册 `ADMIN_OPTIONAL` 端点；engine 侧同步暴露 | `post_process_weights` 零命中；但 `quantization/base_config.restore_weights_before_loading`（:49）与 compressed_tensors 版（:1519）上游已有 → 缺的只是运行时编排，**PR 应复用官方 loader 的 `device_loading_context`，不要自带遍历** |
| S6b | compressed_tensors 恢复钩子守卫 | `layer.scheme.restore_weights_before_loading` 前加 `hasattr` 判断（两处） | 上游两处均裸调用 → 某些 scheme 无此方法时崩，这是真 bugfix，可单独小 PR |
| S7 | `GET /get_weight_version`（+`/weight_version`） | 轻量返回当前权重版本，内部转 `/model_info` | 上游只有 `/model_info` 里带 `weight_version` 与 `POST /update_weight_version`，无独立 GET → RL 每轮轮询版本不该拉全量 model_info |
| S8 | `/server_info` 超时兜底 | `SGLANG_SERVER_INFO_TIMEOUT`（默认 2s），超时记 warning 并返回空 `internal_states` | 零命中 |
| S9 | PD 分阶段耗时指标 | `req_time_stats` 的 `meta_data` 新增 `pd_prefill_bootstrap_queue_duration` / `pd_prefill_bootstrap_duration` / `pd_prefill_alloc_wait_duration` / `pd_prefill_forward_duration` / `pd_prefill_transfer_queue_duration`、decode 侧 prealloc/transfer/retract 时长、`pd_transfer_speed_gb_s` / `pd_transfer_total_mb`、`prefill_retry_count`；顺带把 `state[key]` 判 0 改成 `> 0.0` 避免 None 比较 | 零命中 |
| S10 | EAGLE draft topk clamp | 图捕获输入 `topk_p.clamp(0,1)`、`topk_index.clamp(0, vocab-1)`，防 draft 越界访存 | `topk_p.clamp` 零命中 |
| S11 | radix cache 两条正确性 | ①`cache_finished_req` 只缓存到 `kv_committed_len`（speculative 未提交的 KV 不进前缀树）②`dec_lock_ref` 父节点为 None 时 `break`，允许跨树节点解锁 | `radix_cache.py:619` 仍是 `assert node is self.root_node`；`get_fill_ids` 在该文件零命中 |
| S12 | DSATokenToKVPool 漏的 memory saver 区 | `compress_tail_k` 分配只包了 `torch.cuda.use_mem_pool`，缺 `memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)` → sleep/release 时这块不释放 | 上游该处仍无 region |
| S13 | retract_decode 语义 | 统一模式下至少留 1 个请求；**PD decode 模式允许 retract 到 0**（`num_minimum_reqs = 0 if disagg_mode=="decode" else 1`），并把 `len(sorted_indices) <= 1` 的判断改成 `sorted_indices and len(...) <= num_minimum_reqs` | 上游仍是 `== 1` 硬编码 |
| S14 | TTFT/ITL 指标修正 | prefill-only 模式不再观察 TTFT（但 `ttft_observed` 仍置位）；ITL 用 `num_new_tokens > 0` 判空并**总是**更新 `last_completion_tokens`（原来 token 未增长时不同步，导致下一轮间隔算错） | 未修 |
| S15 | FastAPI 路由 path 兜底 | `_get_fastapi_request_path` 在 `route.path` 缺失时退到 `route.path_format`，再退到 `child_scope["route"]` | 未修（metrics 中间件在挂载子路由上有概率拿不到 path） |
| S16 | GLM4V 拆分部署 + 预计算 embedding | `glm4v_moe` 支持 `encoder_only` / `language_only`（encoder_only 时不建语言模型与 lm_head）；rope 配置兼容 `rope_parameters`（Transformers 5.x）；processor 新增 `get_mm_data(prompt, embeddings, img_grid_thw)` 直接吃外部 embedding 并算 mrope | `encoder_only` 只存在于 embedding 配置域，模型侧零命中 |
| S17 | DSA indexer 对齐 | `_match_head_gate_q_scale`（head-gate 权重按 q_scale 的 head 数 repeat_interleave 对齐）、`_maybe_repeat_query_heads`（head 数 <32 时补到 32）、`INDEXER_ROPE_NEOX_STYLE` 环境变量覆盖 rope 布局、prefill 图里 tile 形状改用 `q_scale.shape[1]` | 三个签名零命中。**注意**：head 补齐的落点要看算子约束来源（NPU fused attention vs flashmla），沉之前先确定是通用 kernel 约束还是单后端约束 |
| S18 | profiler 步进 | decode 分支同时接受 `forward_mode.is_prebuilt()`（上游确有 prebuilt 模式，但 profiler 没覆盖） | `is_prebuilt` 上游存在，profiler_manager 未覆盖 |
| S20 | `/v1/loads` inflight 段 | `LoadSnapshot` 增 `inflight` section，`load_inquirer.get_loads(include_inflight=)` 按 disagg 模式枚举 running/waiting/bootstrap/prealloc/transfer/retracted 队列并给出每请求 stage 与入队时间；`tokenizer_control_mixin` 在 `include` 含 inflight 时走 DP fan-out | 上游 `load_snapshot.py` 的 section 集合无 inflight，未知 section 直接 `raise ValueError` |

> S20 单独说明：**能够沉，但不该按原样沉**。slime 客户端请求的是 `?include=core,disagg,queues`（`server_control.py`），inflight 段没有任何调用方，属死代码。要么作为纯可观测性特性重写（含文档与测试），要么直接从 sglang 侧删掉。

### 3.3 大件 S4：deterministic 拆分

`deterministic` 是"训推数值一致"的补丁，27 个文件、+807/−81，里面同时装着 CUDA 内核和通用逻辑。逐项拆开：

| 子项 | 内容 | 硬件 | 下沉判断 |
|---|---|---|---|
| D1 | `per_token_group_quant*.cuh` 一致性（silu 换 expf、scale 用 `__fdiv_rn`、逐元素 FP8 转换） | CUDA kernel | 只能由 sglang GPU 线沉；Ascend 无收益 |
| D2 | `matmul_persistent` 的 `BLOCK_SIZE_M = 16 if M<=16 else 128` | Triton（设备无关） | 可沉，NPU 上理论可用（triton-ascend 需实测） |
| D3 | `router_gemm_batch_invariant`（固定 K 归约、FP32 logits；CUDA、sm90+、4 个 shape 白名单走 JIT 快核，否则回落 fp32 persistent） | 混合 | 可沉；声明清楚 fallback 语义后对 NPU 友好 |
| D4 | `kernels/jit/glm5_router_gemm.py`（新文件）+ JIT 路径/编译规格改动 | CUDA | GPU 线 |
| D5 | `enable_fp32_moe_router`（`MoEGate` 走 FP32 router logits） | torch 层 | 可沉，**但要改写法**：补丁把 `_is_cuda or _is_hip` 的 `torch.mm(out_dtype=fp32)` 换成了"flag 否则 `F.linear`"，等于在不开 flag 时改变了 CUDA 默认数值。上游化时必须保留原默认路径 |
| D6 | DeepGEMM `set_batch_invariant` / `use_deterministic_algorithms` 配置 | CUDA | GPU 线 |
| D7 | MoE grouped GEMM 的 M pad 到 128 tile | CUDA | GPU 线 |
| D8 | DeepEP `align_fp8_quantization` + LL prefill staging（`SGLANG_DEEPEP_LL_PREFILL_STAGING`） | CUDA/DeepEP | GPU 线 |
| D9 | DSA indexer head-gate 改 DeepGEMM + topk 排序确定化 + FP8 KV cache dequant 路径 | 前半 CUDA，后半通用 | 拆开沉：dequant 路径（`quantize_k_cache`/`dequantize_k_cache_paged` + 索引重映射）与硬件无关，可单独 PR |
| D10 | `dsa` 加入确定性 attention backend 列表 | 注册项 | 可沉 |
| D11 | tensor dump 增强（层级过滤、分块写出、module 过滤、注入 metadata 端点 `SetInjectDumpMetadataReqInput`） | 纯调试 | 可沉，独立 PR |
| D12 | `weight_updater` 的 `torch.cuda.synchronize()`（CUDA IPC buffer 所有权） | CUDA | GPU 线；NPU 需要的是 hccl/stream 语义，另行处理 |

**Ascend 视角的结论**：这个补丁在 ascend 仓里是"整体背负"——`quick_install.sh` 无条件 `git apply`，但 NPU 上真正执行的只有 D2/D5/D10/D11 这类通用项，CUDA 子项是纯粹 rebase 成本。建议按上表拆成两个方向：通用子集沉 sglang，CUDA 子集交给 GPU 线，NPU 侧只保留必要的确定性开关。

---

## 4. 可下沉 B：sglang NPU backend

**N1 · TMS 适配器不收位置参数（必修，2 行）**

- 事实链（两边都可查）：
  - `hardware_backend/npu/graph_runner/npu_cudagraph_backend.py` 开启 memory saver 时构造 `partial(self._memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)`，随后**位置传参**调用 `graph_ctx(graph, pool=..., stream=..., auto_dispatch_capture=True)`；
  - `utils/torch_memory_saver_adapter.py` 三个实现类的签名都是 `def cuda_graph(self, **kwargs)`。
  - ⇒ `TypeError: cuda_graph() takes 1 positional argument but 2 were given`。v0.5.21 与今日 main（`c67db7c762`）都是同一写法，11-17 的 `#49999 npugraph_ex` 重构也没碰到这两处。
- **触发条件**：memory saver 的图内存回收由 `enable_memory_saver and SGLANG_MEMORY_SAVER_CUDA_GRAPH` 门控，后者默认 `False`。所以不是"必然天天崩"，而是**一开这个 env 就崩**——而 colocate 想回收图显存恰恰要开它。
- 修法就是补丁本身：签名改 `(self, *args, **kwargs)`，透传 `*args`。`torch.npu.graph` / `torch.cuda.graph` 首参都是位置参数，适配器本来就该照抄。补丁在新 main 上 `patch --dry-run` 零失败，可直接提。
- 附带问题：`auto_dispatch_capture` 是 NPU 图选项，适配器把它当 kwargs 转给 `torch_memory_saver.cuda_graph`，而 GPU 版 TMS 不接受这个 kwarg —— 所以这条与 §5 的 K2 是一条链的两端：**sglang 传得出、TMS 收得下，缺一不可**。
- 优先级：改动最小、无任何 slime 语境，是最该先提的一条。

**N2 · DSA head 补齐的落点待定**（见 S17）：若 `<32 heads` 的补齐确实是 Ascend fused attention 的约束，这段应放进 NPU backend 而不是通用路径；若 GPU 的 flashmla_sparse 同样要求，则留在通用代码。沉之前先确认，别把单后端约束写进通用层。

---

## 5. 可下沉 C：sgl-kernel-npu（TMS）

这三个补丁改的是 `contrib/torch_memory_saver`——NPU 版 TMS（Ascend ACL VMM 实现）。`quick_install.sh` 的流程是：装 release wheel → 拉 `9765e275` 源码 → `git am` 三个补丁 → 只重编 TMS wheel。**`origin/main`（2026-11-17 `3f4d922`）里 `is_paused` / `auto_dispatch_capture` / `GetAllocationGranularity` 依旧零命中，即均未上游；三条补丁对该版本 `patch --dry-run` 零失败。**（`TMS_HOOK_MODE` 这个 env 名在代码里有 3 处引用，但 `configure_subprocess()` 仍只设 `LD_PRELOAD`——K2 的后半段依然缺。）

| 编号 | 内容 | 为什么是缺陷 | 备注 |
|---|---|---|---|
| K1 | `AllocationMetadata` 增 `is_paused`；`pause`/`resume`/`free` 各状态转换幂等；`aclrtMapMem` 返回值检查 | 重复 `pause` 会从未映射地址重拷、把有效 `cpu_backup` 覆盖成垃圾；重复 `resume` 会把新物理 handle 映射到仍在使用中的地址 | **等价 PR 提过并被关闭**：`sgl-kernel-npu#867`「torch_memory_saver: make pause/resume idempotent and sync before backup」（+104/−10，2026-10-20 closed）。通道验证过，可以重开，但要把"为什么被关"搞清楚，避免第三次重复 |
| K2 | TMS 的 `cuda_graph` 接受 `auto_dispatch_capture` 并传给 `torch.npu.graph`；`configure_subprocess` 同时注入 `TMS_HOOK_MODE=preload` | TMS 默认 `hook_mode` 取 `TMS_HOOK_MODE`，缺省 `"torch"`，而 `cuda_graph` 里硬 `assert self._hook_mode == "preload"`。不注入 env 就是 AssertionError；不接 `auto_dispatch_capture` 就是 TypeError | 与 N1 成对 |
| K3 | 不满足物理内存粒度的分配回退普通 ACL 分配器（新增 `cann_mem_granularity`）；嵌套 region 退出时恢复 tag/region/cpu_backup 而非断言 | 小块分配走 VMM 直接失败；sglang 存在嵌套 region 用法，`AssertionError` 会打穿 | 新增 `tms_get_current_tag` / `tms_get_enable_cpu_backup` 两个 getter，是 K1 的配套 |

**根因侧建议**：`patches/npu-compat/slime-main-20261026-adapter.patch` 在训练 actor 里加了 `torch.npu.synchronize()`，注释直说"Ascend's allocator rejects a TMS transition while an asynchronous graph capture is still winding down"。这是在**调用方**绕 TMS 的缺陷——正确位置是 TMS：`pause/resume/disable` 应自行等待在途 capture 收敛（K1 里"sync before backup"就是这个思路的弱化版）。TMS 修好后，这条 workaround 应当删除，而不是继续往 slime 里加。

需要补充的是：sglang 侧 11-17 合入的 `#50717 wait_for_launched_forward()` 已在 disk/distributed/tensor/ipc 与 begin/end session **全部**权重更新路径插入 `schedule_stream.wait_stream(forward_stream)`，motivation 正是"an in_place pause can leave one reading the weights"。这解决的是**流序**，而 TMS 需要的是**设备级 unmap 前所有在途 capture 收敛**——两者相关但不等价。所以该 workaround 不能因为上游这条 PR 就删，但它的一半理由已经被上游吃掉了，写 TMS 的 PR 时可以直接引用 #50717 说明方向。

---

## 6. 无法下沉

| 项 | 内容 | 为什么 sglang 接不了 | 该去哪儿 |
|---|---|---|---|
| npu-compat/adapter | 新增 `NPUAccelerator`（`ray_resource_name="NPU"`、`post_import_torch` 加载 Megatron shim）、`ray_remote_options`、placement group 用 `{"NPU":1}` bundle、`get_accelerator_ids()` 取卡、`ASCEND_RT_VISIBLE_DEVICES` / `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES` / `HCCL_*` 等环境变量、`PYTHONPATH` 指向 Megatron 栈 | 全是训练编排与集群资源语义，sglang 不感知 Ray placement group，也不负责拉起训练进程 | **THUDM/slime**（上游已有 `slime/utils/accelerator/` 抽象与 cuda 后端，NPU 后端是天然补齐）；环境变量属部署配置，最好外置成 env 文件而非硬编码 |
| npu-compat/empty-cache-captures | `empty_cache()` 时吞掉 `captures_underway.empty()` 这一种 RuntimeError（其余照抛） | 现象来自 Ascend 分配器跨 stream 跟踪图捕获，`is_current_stream_capturing()` 为假但捕获仍在收尾 | torch_npu/CANN；TMS 可缓解（见 K1），sglang 层无法承接 |
| npu-compat/megatron-root-override | 允许隔离的 Megatron root；Ray GCS 超时后重启 | 运维/进程管理 | THUDM/slime 或运维脚本 |
| npu-compat/ray-cpu-cap | 共享主机上给测试 head 限制 CPU worker 数 | 测试基础设施 | THUDM/slime |
| upstream-fixes/qwen2-hf-converter | 从 Qwen2 HF config 推断 rope/RMSNorm/GQA/QKV bias | slime 自己的 `convert_hf_to_torch_dist` bug | THUDM/slime |
| upstream-fixes/critic-mla-test-mock | 补 Megatron MLA import 桩 | slime 测试 | THUDM/slime |
| Megatron ×2 / Bridge ×3 / mbridge / adaptor | 训练栈各自的 NPU 适配 | 各仓职责 | Megatron-LM / Megatron-Bridge / mbridge |
| `sglang.patch` 内：`begin/end session` 可省略、`disable_draft_model` | 让旧客户端可以不带 session 直接流式更新权重；draft 模型不参与更新 | 这是 slime↔sglang 的**协议兼容层**，sglang 不会为单个 RL 框架永久保留废弃协议 | 留 slime，随客户端升级删除（补丁注释自己也写着 "Legacy clients…"） |
| `sglang.patch` 内：CLI 别名（`--fp8-gemm-runner-backend`、`--nsa-prefill-backend`、`--nsa-decode-backend`、`--mamba-scheduler-strategy`、`--cuda-graph-max-bs`、`--piecewise-cuda-graph-max-tokens`、`--prefill-round-robin-balance`、`--enable-breakable-cuda-graph`、`--disable-piecewise-cuda-graph`、`--enforce-piecewise-cuda-graph`） | 给固定启动脚本兜底的旧名映射。**这套机制是上游自己的**：`Arg(aliases=)` 与 `arg_groups/argparse_actions.py` 的 `DeprecatedAction` / `DeprecatedStoreConstAction` 都在 main 上，且上游持续用它们保留旧名，所以别名本身符合惯例、并非不可沉 | 两种解法择一：**改 slime launcher 用新名**（零 PR，推荐）；或作为 deprecated alias 单独小 PR。不要混在功能补丁里跟着一起 rebase |
| `sglang.patch` 内：mooncake "一次失败即标记 session failed" 的删除 | 容错策略改动 | 语义变更需要上游单独评估（改动后失败会话不再被快速隔离） | 单独讨论，不随大补丁走 |

---

## 7. 应删除 / 需复核

| 项 | 处置 |
|---|---|
| `/v1/loads` inflight 段（S20） | 无调用方，从补丁里删；如要作为特性保留则重写为独立 PR |
| mooncake `dst_addr = dst_aux_ptrs[i] + …` → `dst_aux_ptr + …` | 循环变量与列表元素同值，**等价改写**，纯噪音 |
| `server_control.py` 里新增的三行注释 hunk | 只加注释，无功能改动，删 |
| `qwen_vl.py`：`load_mm_data` → `legacy_load_mm_data` | **反向改动**：把新多模态加载路径退回 legacy。上游两个方法都在，这里必须查清是缺能力还是时序问题——不能当"修复"沉，也不能长期挂着 |
| `accelerator_runtime_latest.patch`、`upstream-fixes/thd_packed_context_parallel.patch` | 目录里存在但不在 MANIFEST、不参与构建 → 删除或补登记，避免"看起来在生效其实没打"的误判 |

---

## 8. 建议路线

| 批次 | 内容 | 理由 |
|---|---|---|
| 0 | N1（TMS adapter 位置参数）；重开 sgl-kernel-npu #867 合并 K1+K3，K2 独立一个 PR | 最小改动、修真实崩溃；容器阻塞点先解 |
| 1 | S10、S11、S12、S5、S6b、S8、S15 | 都是十几到几十行的正确性/稳定性修复，风险低、评审快 |
| 2 | S6、S7、S9、S13、S14、S16、S17 | 需要一点设计讨论（端点语义、指标命名、retract 策略） |
| 3 | S2（先 base 钩子，再逐池）、S3、S1（三段） | 大件，按依赖顺序；S2 需同步决定 NPU host backend 是否一并做 |
| 4 | deterministic 拆分（通用子集 → sglang，CUDA 子集 → GPU 线） | 一次性把"背负"减掉 |
| 5 | 清理：inflight 段、CLI 别名、注释噪音、孤儿补丁文件；复核 qwen_vl 回退 | 降低 rebase 面积 |

---

## 9. 风险与注意事项

1. **PR 必须基于 `main`，不是 `v0.5.21`**。补丁基线 `e00930c5` 已落后主线约三个月。实测（2026-11-17 的 main，`patch --dry-run`）：

   | 补丁 | 总 hunk | 打不上的 hunk | 比例 |
   |---|---|---|---|
   | `sglang.patch` | 100 | 38 | 38% |
   | `sglang-top_p.patch` | 56 | 13 | 23% |
   | `sglang-release_hicache.patch` | 33 | 10 | 30% |
   | `sglang-pull_weights.patch` | 13 | 2 | 15% |
   | `sglang-deterministic.patch` | 59 | 11 | 19% |
   | `0001-tms-forward-graph-args` | 1 | 0 | 0 |
   | `sgl-kernel-npu/0001..0003` | 5 | 0 | 0 |

   结论：**4 个 NPU 补丁可以原样提**（对应文件自 v0.5.21 起没动过）；5 个功能补丁必须按"新写"的心态移植，不要试图 `git am`。
2. **S2 的池覆盖面**是这条最容易踩的坑：12 个 host pool，尤其 NPU 的 `npu_host_memory` 已改为 mmap 承载（`#50676`），而 `mmap_allocator` 尚无 free/munmap 接口。
3. **S4/D5 会改变默认数值**：上游化时保留 `_is_cuda` 的 FP32 路径，只把新行为挂在 flag 后面。
4. **N1 与 K2 必须同时在线**：只改 sglang 不改 TMS，会在 `auto_dispatch_capture` 处继续炸；只改 TMS 不改 sglang，位置参数照样 TypeError。
5. **不要把 slime 语境写进 PR**：补丁注释里的 "trainer"、"Slime"、"rollout alignment" 要改写成通用动机（数值一致性 / 采样语义 / 内存管理 / 可观测性），否则会被当成下游定制拒收。
6. **先查再沉**：S17 的 head 补齐、qwen_vl 的 legacy 回退、S13 的 retract 语义，都还需要在最新 main 上实测确认根因，避免把 workaround 沉成规范。

---

## 10. 复核命令

```bash
# sglang：逐项确认是否已上游（命中=已有）
cd <sglang>; git fetch upstream
for s in renorm_logprob_over_top_p pull_weights SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT \
         SGLANG_SERVER_INFO_TIMEOUT post_process_weights enable_fp32_moe_router \
         router_gemm_batch_invariant INDEXER_ROPE_NEOX_STYLE _maybe_repeat_query_heads \
         SetInjectDumpMetadataReqInput disable_draft_model num_minimum_reqs; do
  printf "%-42s %s\n" "$s" "$(git grep -c "$s" upstream/main -- python/ | wc -l)"
done
git grep -n "def cuda_graph" upstream/main -- python/sglang/srt/utils/torch_memory_saver_adapter.py
git grep -n "def release_memory_occupation" upstream/main -- python/sglang/srt/mem_cache/pool_host/
git grep -n "assert node is self.root_node" upstream/main -- python/sglang/srt/mem_cache/radix_cache.py

# sgl-kernel-npu：TMS 三项是否已合
cd <sgl-kernel-npu>; git fetch origin
git grep -n "is_paused\|auto_dispatch_capture\|GetAllocationGranularity" origin/main -- contrib/torch_memory_saver

# 历史 PR（判断是否有人提过）
gh pr list -R sgl-project/sgl-kernel-npu --state all --search "memory saver in:title"

# 漂移量化：在 upstream/main 的 worktree 上干跑（--dry-run 不落盘）
cd <sglang-worktree-at-upstream-main>
for p in sglang sglang-top_p sglang-release_hicache sglang-pull_weights sglang-deterministic; do
  f=<slime-ascend>/patches/third_party/sglang/$p.patch
  out=$(patch -p1 --dry-run -f < $f 2>&1)
  printf "%-28s 总hunk=%s 失败=%s\n" "$p" "$(grep -c '^@@' $f)" \
    "$(echo "$out" | grep -cE '^Hunk #[0-9]+ FAILED')"
done
```

判定：**有命中 = 已吸收/已有（删补丁或缩小补丁）；无命中 = 仍需下沉**。本文上游基准：sglang `c67db7c762`、sgl-kernel-npu `3f4d922`，均为 2026-11-17 拉取后复核。
