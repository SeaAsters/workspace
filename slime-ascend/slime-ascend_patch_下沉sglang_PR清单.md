# slime-ascend patch → sglang 下沉 PR 清单

> 用途：按此清单把 slime-ascend 的推理侧补丁逐个沉淀为上游 PR（sglang / sgl-kernel-npu），每合入一项就删掉对应 patch 段。
> **核对基准（2026-11-17 拉取最新后复核）**：`sgl-project/sglang` `upstream/main@c67db7c762`、`sgl-project/sgl-kernel-npu` `origin/main@3f4d922`，均为本地 fetch 后 `git grep` 全量复核；15 个待沉签名仍全部零命中。
> **来源**：`gitcode.com/Ascend/slime-ascend` `main@e58ab12`，其 sglang 基线是 `v0.5.21`（`e00930c5`），TMS 基线是 `sgl-kernel-npu@9765e275`。
> 每次提 PR 前请重新核对当日 main（方法见文末「实操提示」）。
> 逐项判定依据（含代码级证据与"为什么不能沉"的理由）见同目录《slime-ascend_下沉sglang_完整版.md》。

---

## 0. 总览（按建议动手顺序）

| # | PR 主题 | 来源 patch | 预估规模 | 优先级 | 与 NPU 的关系 |
|---|---|---|---|---|---|
| 1 | TMS adapter 接受位置参数 | sglang/0001-tms-forward-graph-args | 8 行 | ★★★ | 直接相关：开 `SGLANG_MEMORY_SAVER_CUDA_GRAPH` 就崩（默认关） |
| 2 | TMS pause/resume 幂等 + 对齐/嵌套回退 | sgl-kernel-npu/0001+0003 | ~80 行 | ★★★ | NPU 专属，直接相关 |
| 3 | TMS 接受 `auto_dispatch_capture` + 子进程注入 `TMS_HOOK_MODE` | sgl-kernel-npu/0002 | ~30 行 | ★★★ | NPU 专属，与 #1 成对 |
| 4 | 小修合批：EAGLE clamp / radix×2 / DSA KV 池 region / FastAPI path / server_info 超时 / compressed_tensors 守卫 | sglang.patch | 共 ~60 行 | ★★☆ | 通用（DSA 池那条 NPU 直接受益） |
| 5 | PD bootstrap/transfer 超时 + 在途清理 | sglang.patch（disagg 段） | ~130 行 | ★★☆ | 通用稳定性，NPU 多机 PD 同样受益 |
| 6 | `/post_process_weights` 运行时端点 | sglang.patch（权重段） | ~150 行 | ★★☆ | 量化模型 RL 更新；NPU 量化路径同样需要 |
| 7 | `/get_weight_version` GET + `/server_info` 超时 | sglang.patch（http 段） | ~25 行 | ★★☆ | 通用，客户端轮询成本 |
| 8 | PD 分阶段耗时指标 | sglang.patch（req_time_stats） | ~120 行 | ★★☆ | 通用可观测性 |
| 9 | TTFT/ITL 指标修正 | sglang.patch（tokenizer_manager） | ~25 行 | ★★☆ | 通用（prefill-only 部署必踩） |
| 10 | retract_decode 在 PD decode 下允许退到 0 | sglang.patch（schedule_batch） | ~10 行 | ★☆☆ | 通用，需上游讨论 |
| 11 | HiCache host 内存释放（`--release-hicache`） | sglang-release_hicache | ~310 行（拆 2） | ★★☆ | 通用；**NPU 方向需另做**（见 §11） |
| 12 | `/pull_weights` 读端 | sglang-pull_weights | ~390 行 | ★☆☆ | NPU 多机收益更大 |
| 13 | top-p 采样重放 | sglang-top_p | ~560 行（拆 3） | ★☆☆ | 含 Ascend sampler 分支，NPU 直接受益 |
| 14 | deterministic 拆分：通用子集 / CUDA 子集分流 | sglang-deterministic | 按子集 | ★☆☆ | NPU 只吃通用子集 |
| 15 | VL 两项：GLM4V encoder_only 拆分、DSA head 对齐 | sglang.patch 内 | ~90 行 | 待定 | 需先查根因（见 §15） |

**与 GPU 版清单的口径差异**：本清单的补丁基线是 sglang `v0.5.21`，比 GPU 版（基于 main）更旧，冲突面更大；另外 ascend 多了 4 处 NPU 独有改动（#1~#3 + §11 的 NPU host 语义），GPU 版没有这部分。

---

## 1. TMS adapter 接受位置参数 ★★★

- **PR 标题**：`[Fix][NPU] torch_memory_saver adapter: accept the positional graph argument`
- **来源**：`patches/third_party/sglang/0001-v0.5.21-tms-forward-graph-args.patch`（+4/−4）
- **改动要点**：`python/sglang/srt/utils/torch_memory_saver_adapter.py` 三个实现类的 `def cuda_graph(self, **kwargs)` 改成 `(self, *args, **kwargs)`，`_TorchMemorySaverAdapterReal` 把 `*args` 原样透传给 `_memory_saver.cuda_graph(*args, **kwargs)`；Noop 版同样改签名（否则两行都不对齐）。
- **现状**：`hardware_backend/npu/graph_runner/npu_cudagraph_backend.py` 用 `partial(self._memory_saver_adapter.cuda_graph, tag=...)` 构造后**位置传参**调用 `graph_ctx(graph, pool=..., stream=..., auto_dispatch_capture=True)` → `TypeError`。11-17 的 `#49999 npugraph_ex` 重构过这段捕获代码（去掉 `skip_guard_context`、`torch.compile` 改 `fullgraph=False`），**但位置传参和 adapter 签名都没动**。
- **触发条件**：图内存回收由 `enable_memory_saver and SGLANG_MEMORY_SAVER_CUDA_GRAPH` 门控，该 env 默认 `False` → 不是天天炸，而是**开了就炸**（colocate 想回收图显存就得开）。
- **交付性**：补丁对今日 main `patch --dry-run` 零失败，可原样提。
- **注意**：XPU 分支的 `_noop_context` 语义不变；这是签名对齐，不是功能新增，`torch.cuda.graph` / `torch.npu.graph` 首参本就位置。
- **测试**：NPU 上 `--enable-memory-saver` + `SGLANG_MEMORY_SAVER_CUDA_GRAPH=1` + decode 图捕获跑通一次；CUDA 上 memory saver 回归。
- **依赖**：必须与 #3 同时在线，否则会从 TypeError 变成 kwarg 不识别。

## 2. TMS pause/resume 幂等 + 对齐/嵌套回退 ★★★

- **PR 标题**：`[Bugfix] torch_memory_saver: make pause/resume idempotent; fall back for unaligned allocations`
- **来源**：`sgl-kernel-npu/0001`（+18/−6）+ `0003`（+39/−5），目标仓 `contrib/torch_memory_saver`
- **改动要点**：
  - `AllocationMetadata` 增 `is_paused`；`pause()` 跳过已暂停项、`resume()` 跳过未暂停项、`free()` 在已暂停时不再 unmap/free physical；补 `aclrtMapMem` 返回值检查。
  - 新增 `cann_mem_granularity()`；`aclrtMallocAlign32` 判断 `size % granularity != 0` 时回退普通 ACL 分配（小块不必走 VMM）。
  - 嵌套 region：退出时恢复 `(tag, interesting_region, enable_cpu_backup)` 三元组，替代 `assert not tms_get_interesting_region()`；配套新增 `tms_get_current_tag` / `tms_get_enable_cpu_backup` 及 `binary_wrapper` 签名。
- **历史（重要）**：等价 PR 已提过并被关闭——`sgl-kernel-npu#867`「make pause/resume idempotent and sync before backup」（+104/−10，2026-10-20 closed）。重开前先确认关闭原因，避免第三次重复劳动；可复用其 PR 描述里的复现图。
- **测试**：`test/cpu_backup.py` 连续两次 pause/resume、pause 后直接 free、小块（非粒度对齐）分配、嵌套 region 进出。
- **边界**：只改 Ascend 的 contrib 版，不牵扯上游 CUDA 版 TMS。
- **交付性**：两条补丁对 `origin/main@3f4d922` `patch --dry-run` 零失败，可原样提。

## 3. TMS 接受 `auto_dispatch_capture` + 子进程注入 hook mode ★★★

- **PR 标题**：`[Bugfix] torch_memory_saver: pass auto_dispatch_capture through and propagate TMS_HOOK_MODE`
- **来源**：`sgl-kernel-npu/0002`（+14/−6）
- **改动要点**：`TorchMemorySaver.cuda_graph` / `_TorchMemorySaverImpl.cuda_graph` 增加 `auto_dispatch_capture` 参数并传给 `torch.npu.graph(...)`；`hooks/mode_preload.configure_subprocess()` 在设 `LD_PRELOAD` 的同时设 `TMS_HOOK_MODE=preload`。
- **为什么必须**：`_hook_mode` 取 `os.environ.get("TMS_HOOK_MODE", "torch")`，而 `cuda_graph` 里硬 `assert self._hook_mode == "preload"`；不注入 env 就是 AssertionError。`configure_subprocess()` 至今仍只设 `LD_PRELOAD`（env 名在代码里有 3 处引用，但没人注入）。
- **测试**：NPU 子进程（scheduler 子进程）里图捕获 + pause/resume 一轮。
- **交付性**：对 `origin/main@3f4d922` `patch --dry-run` 零失败。
- **依赖**：#1 的前置。

## 4. 小修合批 ★★☆

同一批低风险修复，建议合到 2~3 个 PR（按文件域合并），便于一次评审过。

| 项 | 文件 | 改动 | 规模 |
|---|---|---|---|
| EAGLE draft topk clamp | `speculative/eagle_draft_cuda_graph_runner.py` | 图捕获输入 `topk_p.clamp(0,1)`、`topk_index.clamp(0, vocab-1)` | 6 行 |
| radix 只缓存已提交 KV | `mem_cache/radix_cache.py` `cache_finished_req` | 取 `get_fill_ids()[: kv.kv_committed_len]` | ~6 行 |
| radix 跨树 unlock | 同上 `dec_lock_ref` | 父节点为 None 时 `break`，替代 `assert node is self.root_node` | ~8 行 |
| DSA KV 池补 memory saver 区 | `mem_cache/memory_pool.py` `DSATokenToKVPool` | `compress_tail_k` 分配外包 `memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)` | 5 行 |
| FastAPI 路由 path 兜底 | `utils/common.py` `_get_fastapi_request_path` | `route.path` 缺失时退 `path_format`、再退 `child_scope["route"]` | ~8 行 |
| `/server_info` 超时 | `entrypoints/http_server.py` | 新 env `SGLANG_SERVER_INFO_TIMEOUT`（默认 2s），超时 warning + 空 `internal_states` | ~15 行 |
| compressed_tensors 恢复守卫 | `layers/quantization/compressed_tensors/compressed_tensors.py` | `layer.scheme.restore_weights_before_loading` 前加 `hasattr`（两处裸调用） | 6 行 |

- **测试**：radix 两条造 spec 请求 finish + 跨树解锁；DSA 池开 sleep/release 后确认 `compress_tail_k` 被释放；`/server_info` 人为挂住一个 DP rank 验证 2s 兜底；compressed_tensors 用无 `restore_weights_before_loading` 的 scheme 加载。
- **备注**：`DSATokenToKVPool` 那条是**真实的 sleep 漏洞**（开 memory saver 时这块 KV 不会被释放），NPU 上用 DSV3.2/DSA 模型时直接相关。

## 5. PD bootstrap/transfer 超时 + 在途清理 ★★☆

- **PR 标题**：`[Feat] Configurable PD bootstrap/transfer timeout with request abort and in-flight cleanup`
- **来源**：`sglang.patch` 的 disaggregation 段（decode.py / prefill.py / scheduler.py，约 +130）
- **改动要点**：
  - 新 env `SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT`（默认 600s），四处独立判定：decode prealloc、decode transfer、prefill bootstrap、prefill transfer，均以 `req.time_stats.*_queue_entry_time` 为起点。
  - 超时 `prepare_abort` + `HTTPStatus.GATEWAY_TIMEOUT`，prefill 侧额外 `release_kv_cache` + `disagg_kv_sender.clear()`，decode 侧 `kv_receiver.abort()`；计入 `increment_bootstrap_failed_reqs` / `increment_transfer_failed_reqs`（**这两个 helper 上游已有**，说明指标骨架就位、只缺超时路径）。
  - `release_memory_occupation` / `resume_memory_occupation` 清空 prealloc/transfer/retracted 队列并 `deregister/register_buffer_to_engine`；scheduler 在超时 abort 时同时 abort prefill sender。
- **测试**：人为延迟 mooncake/queue 传输，验证超时路径 abort（503/504）而非挂死；release 后确认无在途 buffer 泄漏。
- **注意**：env 命名建议先与上游讨论（是否并入 `srt/environ.py` 的 Envs 机制），避免新 PR 又造一个裸 `os.environ`。

## 6. `/post_process_weights` 运行时端点 ★★☆

- **PR 标题**：`[Feat] post_process_weights endpoint: restore + re-quantize after runtime weight updates`
- **来源**：`sglang.patch` 权重段（io_struct / engine / http_server / tokenizer_control_mixin / tp_worker / model_runner / weight_updater，约 +150）
- **改动要点**：`PostProcessWeightsReqInput{restore_weights_before_load, post_process_quantization}` 两条链路；`model_runner.post_process_weights` 用官方 `model_loader.loader.device_loading_context` 逐模块调 `quant_method.restore_weights_before_loading` 与 `process_weights_after_loading`；http 端点用 `ADMIN_OPTIONAL`。
- **关键约束**：上游已有 `quantization/base_config.restore_weights_before_loading`（:49）与 compressed_tensors 实现（:1519），**缺的只是运行时编排**——PR 必须复用官方 loader 工具，不要自带模块遍历逻辑。
- **测试**：量化模型（fp8 marlin repack）`update_weights_from_distributed` → `post_process_weights` → 输出与重启加载一致。
- **备注**：#4 里的 compressed_tensors `hasattr` 守卫是这条的前置修复，可先合或并入本 PR。

## 7. 权重版本 GET + server_info 超时 ★★☆

- **PR 标题**：`[Feat] Add GET /get_weight_version for cheap version polling`
- **来源**：`sglang.patch` http_server 段（约 +12；`/server_info` 超时见 #4）
- **改动要点**：新增 `@app.get("/get_weight_version")`（别名 `/weight_version`），内部转 `model_info()` 只取 `weight_version`。
- **理由**：上游只有 `/model_info` 里带该字段 + `POST /update_weight_version`。RL 每轮都要确认版本，拉全量 model_info 代价过高。
- **测试**：版本更新后 GET 是否立即反映；与 `/model_info` 的 `weight_version` 一致。

## 8. PD 分阶段耗时指标 ★★☆

- **PR 标题**：`[Feat] PD stage durations in request meta_info`
- **来源**：`sglang.patch` 的 `observability/req_time_stats.py`（约 +120）+ `schedule_batch.py` 一处（`time_stats.disagg_mode` 传递）
- **改动要点**：`meta_data` 增 `pd_prefill_bootstrap_queue_duration` / `pd_prefill_bootstrap_duration` / `pd_prefill_alloc_wait_duration` / `pd_prefill_forward_duration` / `pd_prefill_transfer_queue_duration`、decode 侧 prealloc/transfer/retract 时长、`pd_transfer_speed_gb_s` / `pd_transfer_total_mb`、`pd_prefill_retry_count`；`add_duration` 只在两端都 >0 时写入；顺带把 `state[key]` 判空从真值改成 `> 0.0`（避免 None 比较）。
- **测试**：PD 单请求跑通，检查 meta_info 里各段时长单调、和用不重叠。

## 9. TTFT/ITL 指标修正 ★★☆

- **PR 标题**：`[Bugfix] metrics: don't emit TTFT on prefill-only workers; fix ITL accumulation`
- **来源**：`sglang.patch` 的 `managers/tokenizer_manager.py`（约 +25）
- **改动要点**：
  - prefill-only 模式不再 `observe_time_to_first_token`，但 `ttft_observed` 仍置位（否则每个 token 都进 TTFT 分支）。
  - ITL 判空改 `num_new_tokens > 0`，且 `state.last_completion_tokens` **无条件**更新（原实现 token 未增长时不更新，导致下一轮间隔被算长）。
- **测试**：PD 分离 prefill worker 不出 TTFT；流式请求里出现空增量后 ITL 仍准确。

## 10. retract_decode 在 PD decode 下允许退到 0 ★☆☆

- **PR 标题**：`[Fix] retract_decode: allow retracting the last request in decode disaggregation`
- **来源**：`sglang.patch` 的 `managers/schedule_batch.py`（约 +10）
- **改动**：`num_minimum_reqs = 0 if disaggregation_mode == "decode" else 1`；循环与末尾判断都用它，并把 `len(sorted_indices) <= 1` 改成 `sorted_indices and len(...) <= num_minimum_reqs`。
- **风险**：语义变更（decode 侧可能把请求全部 retract），需上游讨论后再提；建议 PR 里给出"内存不足时宁可 retract 也不要 abort"的动机与压测数据。
- **测试**：PD decode 侧内存压到极限，验证 retract 全部后请求存活、无泄漏。

## 11. HiCache host 内存释放（`--release-hicache`）★★☆

- **PR 标题（1）**：`[Feat] pool_host: release/resume host KV buffers (base hooks)`
- **PR 标题（2）**：`[Feat] pool_host: implement release/resume for each host pool + --release-hicache`
- **来源**：`sglang-release_hicache.patch`（+276/−32，13 文件）
- **改动要点**：
  - `base.py`：`__init__` 拆 `_check_host_memory_available` / `_init_host_buffers` / `_post_init_host_buffers`；新增 `release_memory_occupation`（unregister pinned、置空派生视图、缩 `free_slots`/`mem_state`、`malloc_trim`）与 `resume_memory_occupation`；`_host_buffer_attr_names` / `_host_derived_attr_names` 钩子。
  - 各池实现钩子：dsa / mamba / mha（含 KOnly）/ mla / group / unified；`unified.py` 的共享 backing 用**引用计数**（full+swa 两池都释放才真 free）。
  - `weight_updater.py` 在 offload KV_CACHE 时按 `--release-hicache` 调用 tree_cache 钩子；新参数登记到 `arg_groups/fields/memory.py` + `field_order.py`。
- **rebase 注意（最容易踩）**：上游池家族已扩到 12 个（`dsa`/`group`/`mamba`/`mha`/`mha_mxfp8`/`minimax_m3_nvfp4`/`mla`/`page_interleave`/`page_unified`/`qsa`/`hisparse`/`unified`），每个都要实现钩子或明确默认行为，漏一个就"释放不全"。
- **NPU 语义差异（必读）**：`pool_host/npu_host_memory.py` 的 host 池不是 torch pinned buffer——`memfabric` 后端由 MemFabric `Offload` 持有，`acl` 后端自 11-17 的 `#50676` 起改为 **mmap 块**（`alloc_mmap`，`SGLANG_HUGEPAGE_SIZE` 支持 2MB/1GB hugetlb）再 `aclrtHostRegister(MAPPED)`。而 `storage/mmap/mmap_allocator.py` 目前只有 `alloc_mmap` / `alloc_shm`、**没有 free/munmap 接口**。所以本 PR 对 Ascend 拿不到收益；要拿需先给 mmap 分配器补释放路径，再做 host pool 的 release/resume。**不要在 PR 里承诺 NPU 收益。**
- **测试**：patch 自带两个单测可搬（weakref 验证释放、unified 共享 backing 引用计数）；另加 release→resume→前向数值一致。

## 12. `/pull_weights` 读端 ★☆☆

- **PR 标题**：`[Feat] /pull_weights: pull versioned weight deltas from shared storage`
- **来源**：`sglang-pull_weights.patch`（+384/−0，8 文件）
- **改动要点**：新文件 `srt/weight_sync/local_checkpoint.py`（~358 行）：`weight_v{N}` 版本目录、full 与 zstd XOR/overwrite 两种 delta、逐 tensor 校验（xxh3-128/blake3/adler32）、文件锁 + `applied_version` 记录、mmap 原地 apply、线程池、`_drop_page_cache`；端点全链路（io_struct → tokenizer_control_mixin → scheduler → weight_updater），可选 `--custom-pull-weights-pre-read-hook`（对象存储一致性回调）。
- **边界**：**只沉读端**；发布端（trainer 写 delta）留 slime。最后一步复用 `update_weights_from_disk`。
- **NPU 价值**：多机 Ascend 上 HCCL 全量同步昂贵，共享存储 delta 收益比 GPU 更明显。
- **测试**：版本跳跃（v0→v3 只打 delta）、乱序 delta 报错、校验失败报错、并发 rank 单 pull。

## 13. top-p 采样重放（拆 3 个 PR）★☆☆

- **PR-13a** `[Feat] Top-p nucleus token ids and renormalized logprobs (core)`：新增 `layers/utils/top_p.py`（keep-mask、`renorm_logprob_over_top_p`、`get_top_p_token_ids_from_probs`、`attach_top_p_log_probs`）+ sampler 挂载，**含 `use_ascend_backend` 分支**（上游 sampler 已有该属性）+ `LogitsProcessorOutput` 字段。
- **PR-13b** `[Feat] Top-p replay: meta_info plumbing`：`sampling_params.return_top_p_token_ids` → io_struct → 结果处理链（batch_result_processor / logprob_result_processor / output_streamer）→ tokenizer_manager 组装 base64（`top_p_token_ids` / `top_p_token_offsets` / `top_p_log_probs`）。
- **PR-13c** `[Feat] Top-p replay: spec decode and PD disaggregation`：eagle/ngram 按 accept_lens 计算；PD 在 `MetadataBuffers` 增三块 buffer（`MAX_PD_TOP_P_TOKEN_IDS = 4096`）随 KV 传输。
- **来源**：`sglang-top_p.patch`（+551/−6，18 文件）
- **核心数学（可直接写单测）**：重归一后支持集内概率和为 1；已采样 token 必在支持集内（不被截成 -inf）。
- **注意**：13a 里的 `_forward_ascend_backend` 改动是 NPU 唯一直接受益点，单独 review 时不要被当成"GPU 特性"。

## 14. deterministic 拆分（通用子集 / CUDA 子集分流）★☆☆

- **来源**：`sglang-deterministic.patch`（+807/−81，27 文件）
- **可沉（通用或 Triton，设备无关）**：
  - `matmul_persistent` 的 `BLOCK_SIZE_M = 16 if M<=16 else 128`
  - `router_gemm_batch_invariant`（CUDA 快核 + fp32 persistent 兜底，需在 docstring 写明 fallback 语义）
  - `enable_fp32_moe_router` ⚠️ **改写法**：补丁把 `_is_cuda or _is_hip` 的 `torch.mm(out_dtype=fp32)` 换成"flag 否则 `F.linear`"，等于不开 flag 时改变 CUDA 默认数值；上游化必须保留原默认分支
  - DSA FP8 KV cache dequant 路径（`quantize_k_cache` / `dequantize_k_cache_paged` + 索引重映射）
  - `dsa` 加入确定性 attention backend 列表
  - tensor dump 增强（层级过滤、分块、module 过滤、`SetInjectDumpMetadataReqInput`）
- **不沉（CUDA 专属，交 GPU 线）**：`per_token_group_quant*.cuh` 一致性修复、`glm5_router_gemm` JIT 内核、DeepGEMM 配置、MoE grouped GEMM M pad 128、DeepEP `align_fp8_quantization` + LL prefill staging、`weight_updater` 的 `torch.cuda.synchronize()`。
- **落点判断**：ascend 仓目前无条件 `git apply` 整个补丁，但 NPU 上真正执行的只有通用子集 → 拆完可以把 CUDA 子集从 ascend 的 patch 栈里摘掉，直接减掉 27 文件里的绝大部分 rebase 面。

## 15. VL / DSA 两项（待定，先查根因）

| 项 | 改动 | 为什么不能直接提 |
|---|---|---|
| GLM4V `encoder_only` / `language_only` 拆分 + processor `get_mm_data(prompt, embeddings, img_grid_thw)` | 模型侧不建语言模型/lm_head；processor 直接吃外部 embedding 并算 mrope；rope 配置兼容 `rope_parameters`（Transformers 5.x） | 上游模型侧无 `encoder_only`（只有 embedding 配置域有同名概念）。需确认这是通用能力（VIT/LM 分部署）还是 slime 侧特需；若是后者，应做成 slime 的自定义模型注册，不进主线 |
| DSA `_match_head_gate_q_scale` / `_maybe_repeat_query_heads`（<32 head 补到 32）/ `INDEXER_ROPE_NEOX_STYLE` env | head 对齐 + rope 布局可覆盖 | 三个签名上游全无。但 head 补齐究竟是 NPU fused attention 的约束还是 flashmla 的约束未确认——**若只是单后端约束，就不该写进通用路径**，应落 NPU backend |

---

## 不直接下沉的项（防止误搬）

| 项 | 处置 |
|---|---|
| `begin/end session` 可省略、`disable_draft_model` | slime↔sglang 协议兼容层（补丁注释自称 "Legacy clients…"），sglang 不收；留 slime，随客户端升级删除 |
| CLI 别名 10 个（`--fp8-gemm-runner-backend`、`--nsa-prefill-backend`、`--nsa-decode-backend`、`--mamba-scheduler-strategy`、`--cuda-graph-max-bs`、`--piecewise-cuda-graph-max-tokens`、`--prefill-round-robin-balance`、`--enable-breakable-cuda-graph`、`--disable-piecewise-cuda-graph`、`--enforce-piecewise-cuda-graph`） | 机制是上游自己的（`Arg(aliases=)` 与 `DeprecatedAction`/`DeprecatedStoreConstAction`），所以"可以沉"，但**更省的做法是改 slime launcher 用新名**（零 PR）。不要混在功能补丁里跟着 rebase |
| mooncake「一次失败即标记 session failed」的删除 | 容错策略语义变更，需单独提案评估；不随大补丁走 |
| npu-compat 4 条 | Ray NPU 资源 bundle、`ASCEND_RT_VISIBLE_DEVICES`/`HCCL_*`/`RAY_EXPERIMENTAL_*` env、Megatron root 隔离、Ray CPU 封顶 —— 训练编排与集群语义，sglang 不感知。**建议提 THUDM/slime**（上游已有 `slime/utils/accelerator/` 抽象与 cuda 后端，NPU 后端是天然补齐） |
| `captures_underway.empty()` 吞错 | 根因在 Ascend 分配器跨 stream 跟踪图捕获；TMS 侧可缓解（#2 的"transition 等待捕获收敛"），sglang 层无法承接 |
| upstream-fixes 2 条（qwen2-hf-converter / critic-mla-test-mock） | slime 自身 bug 与测试桩 → THUDM/slime |
| Megatron ×2 / Bridge ×3 / mbridge / adaptor | 各训练栈职责 → 对应上游仓 |

## 应删除 / 需复核（不是下沉）

| 项 | 处置 |
|---|---|
| `/v1/loads` 的 `inflight` 段（LoadSnapshot section + `load_inquirer.get_loads(include_inflight=)` + tokenizer fan-out） | **删**。slime 只请求 `?include=core,disagg,queues`，无调用方；上游对未知 section 直接 `raise ValueError`。若要作特性保留，需重写为独立 PR（含文档与测试） |
| mooncake `dst_addr = dst_aux_ptrs[i] + …` → `dst_aux_ptr + …` | **删**。循环变量与列表元素同值，等价改写，纯噪音 |
| `server_control.py` 的三行注释 hunk | **删**。只加注释，无功能改动 |
| `qwen_vl.py`：`load_mm_data` → `legacy_load_mm_data` | **先查后定**。这是把新多模态加载路径退回 legacy，属反向改动；上游两方法都在。查明是缺能力还是时序问题，再决定做成修复还是删掉 |
| `accelerator_runtime_latest.patch`、`upstream-fixes/thd_packed_context_parallel.patch` | **删或登记**。目录里有、MANIFEST 里没有、不参与构建，容易误判"已在生效" |
| 训练 actor 里的 `torch.npu.synchronize()`（npu-compat/adapter 内） | **暂时保留，等 #2 落地后删**。上游 11-17 的 `#50717 wait_for_launched_forward()` 已在全部权重更新路径插入 `schedule_stream.wait_stream(forward_stream)`，解决的是"forward 还在读权重"的流序问题；而 TMS 需要的是"设备级 unmap 前所有在途 capture 收敛"，两者相关但不等价。写 #2 的 PR 时可直接引用 #50717 说明方向 |

---

## 实操提示

1. **提 PR 前复核（一分钟）**：
   ```bash
   cd <sglang> && git fetch upstream
   for s in renorm_logprob_over_top_p pull_weights SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT \
            SGLANG_SERVER_INFO_TIMEOUT post_process_weights enable_fp32_moe_router \
            INDEXER_ROPE_NEOX_STYLE _maybe_repeat_query_heads SetInjectDumpMetadataReqInput; do
     printf "%-42s %s\n" "$s" "$(git grep -c "$s" upstream/main -- python/ | wc -l)"
   done
   git grep -n "def cuda_graph" upstream/main -- python/sglang/srt/utils/torch_memory_saver_adapter.py
   git grep -n "def release_memory_occupation" upstream/main -- python/sglang/srt/mem_cache/pool_host/
   ```
   命中 = 已吸收（删补丁）；无命中 = 仍需下沉。
2. **PR 必须基于 `main`，不是 `v0.5.21`**。实测（2026-11-17 的 main，`patch -p1 --dry-run`）：`sglang.patch` 100 个 hunk 里 38 个打不上（38%）、`release_hicache` 10/33（30%）、`top_p` 13/56（23%）、`deterministic` 11/59（19%）、`pull_weights` 2/13（15%）；而 **4 个 NPU 补丁（sglang `0001` + sgl-kernel-npu 三条）全部 0 失败**。结论：NPU 四条可以先提，功能补丁一律按"新写一遍"处理，别 `git am`。
3. **语境改写**：补丁注释里的 "trainer"、"Slime"、"rollout alignment" 改写为通用动机（数值一致性 / 采样语义 / 内存管理 / 可观测性），否则容易被当成下游定制拒收。
4. **PR 粒度**：一 PR 一主题。`release_hicache` 与 `top_p` 各拆 2~3 个；`deterministic` 按硬件分流；小修可按文件域合并（见 #4）。
5. **下沉后的回归**：每合入一项，删 ascend 仓对应 patch 段 → 更新 `MANIFEST.toml` 的 sha256 → 跑 `scripts/run-qwen3-8B-npu-colocate.sh`（覆盖 release/resume + 图捕获，正是 #1~#3 的场景）。
6. **签名速查（复核用）**：`cuda_graph(*args` / `is_paused` / `auto_dispatch_capture` / `GetAllocationGranularity` / `renorm_logprob_over_top_p` / `release_hicache` / `pull_weights` / `SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT` / `pd_prefill_bootstrap_queue_duration` / `num_minimum_reqs` / `kv_committed_len` / `topk_p.clamp` / `_maybe_repeat_query_heads`。
