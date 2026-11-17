# slime-ascend sglang patch 排查

> 2026-11-17 创建。排查对象：slime-ascend 的推理侧补丁 —— [patches/third_party/sglang](https://gitcode.com/Ascend/slime-ascend/tree/main/patches/third_party/sglang)（6 个）与 [patches/third_party/sgl-kernel-npu](https://gitcode.com/Ascend/slime-ascend/tree/main/patches/third_party/sgl-kernel-npu)（3 个），源仓 `main@e58ab12`。
>
> **核对基准**：sglang 官方 `upstream/main@c67db7c762`（2026-11-17 fetch 后 `git grep` 全量复核，不受 API 限流）、`sgl-kernel-npu origin/main@3f4d922`（同日）；补丁自身基线是 sglang `v0.5.21`（`e00930c5`）、sgl-kernel-npu `9765e275`。
> **口径**：判定用语沿用 GPU 版排查文档 —— **已吸收/已修复**（上游有等价实现，patch 变空操作）、**未修复**（上游缺失）、**不应下沉**（slime/训练/协议层职责，sglang 不收）、**NA**（Ascend 不适用）。另加 **落点** 列标注可下沉的目标仓：`sg` = sglang 主线，`sg-npu` = sglang NPU backend，`kern` = sgl-kernel-npu。

---

## 一、总览

22 条补丁里，属于推理引擎侧的只有 9 条，其余 13 条是 slime / Megatron / Bridge / mbridge 的适配。

| 目标仓 | 条数 | 推理侧 | 处理 |
|---|---|---|---|
| slime | 6（npu-compat 4 + upstream-fixes 2） | 否 | 应提 THUDM/slime，不进 sglang |
| Megatron-LM | 2 | 否 | 训练栈各归其主 |
| Megatron-Bridge | 3 | 否 | 同上 |
| mbridge / Megatron-Adaptor | 1 / 1 | 否 | 同上 |
| **sglang** | **6** | **是** | 5 个功能件可沉、1 个 NPU 小补丁必修；综合补丁内另有协议层与死代码需拆 |
| **sgl-kernel-npu** | **3** | **是** | 全部可沉，其中 1 条有被 closed 的历史 PR |

**一句话**：**0 项已吸收**——与 GPU 版不同，ascend 栈里没有一条因为上游演进变成空操作（因为它不含 THUDM 的 `npu_patch` 那批老适配，那批早被 sglang NPU backend 吃掉了）。所以这里的每一条都还"活着"，要么沉、要么删。

---

## 二、sglang 侧（6 条）

### 2.1 sglang.patch（29 文件 / +636−95）

按文件域拆成三张表。这张补丁是混合体，**必须先拆再沉**。

**A. 权重与端点**

| 项 | 作用 | 官方 main | 判定 | 落点/备注 |
|---|---|---|---|---|
| `/post_process_weights` | 权重热更后 restore + 重新量化打包 | 部分：`quantization/base_config.py:49` 与 `compressed_tensors.py:1519` 已有 `restore_weights_before_loading`，无运行时端点 | 未修复 | `sg`。复用官方 `device_loading_context`，别自带遍历 |
| compressed_tensors 守卫 | 调 `scheme.restore_weights_before_loading` 前加 `hasattr`（两处裸调用） | 未修复 | 未修复 | `sg`。上述端点的前置，可并入或单提 |
| `GET /get_weight_version`（+`/weight_version`） | 轻量返回权重版本，内部转 `/model_info` | 仅有 `/model_info` 里的字段与 `POST /update_weight_version` | 未修复 | `sg`。RL 每轮轮询不该拉全量 |
| `begin/end session` 可省略 | 旧客户端可直接流式更新权重 | — | **不应下沉** | 协议兼容层，补丁注释自称 "Legacy clients…"，留 slime |
| `disable_draft_model` | 更新权重时只打 target、跳过 draft | 零命中 | **不应下沉** | 同上，协议层 |
| `/server_info` 超时 | `SGLANG_SERVER_INFO_TIMEOUT`（默认 2s）超时返回空 `internal_states` | 零命中 | 未修复 | `sg` |

**B. PD 分离、调度与指标**

| 项 | 作用 | 官方 main | 判定 | 落点/备注 |
|---|---|---|---|---|
| PD bootstrap/transfer 超时 | `SGLANG_DISAGGREGATION_TRANSFER_TIMEOUT`（600s），四处判定 + `prepare_abort`(504) + `kv_receiver.abort()` / `sender.clear()` | 零命中 | 未修复 | `sg`。`increment_bootstrap_failed_reqs` 等 helper 上游已有，说明只缺超时路径 |
| PD 在途清理 | release/resume occupation 时清 prealloc/transfer/retracted 队列 + deregister/register buffer | 未修复 | 未修复 | `sg`，随上一条 |
| scheduler abort 联动 | 超时 abort 时同时 abort prefill 侧 `disagg_kv_sender` | 未修复 | 未修复 | `sg`，随上一条 |
| retract_decode 语义 | `num_minimum_reqs = 0 if decode 分离 else 1`，允许 retract 到 0 | 仍硬编码 `== 1` | 未修复 | `sg`，语义有争议，需上游讨论 |
| PD 阶段耗时指标 | `pd_prefill_bootstrap_queue_duration` / `bootstrap_duration` / `alloc_wait_duration` / `forward_duration` / `transfer_queue_duration`、decode 侧 prealloc/transfer/retract 时长、`pd_transfer_speed_gb_s` | 零命中 | 未修复 | `sg`（可观测性） |
| TTFT/ITL 修正 | prefill-only 不发 TTFT；ITL 判空改 `num_new_tokens > 0` 且无条件更新 `last_completion_tokens` | 未修复 | 未修复 | `sg`。prefill-only 部署必踩 |
| `/v1/loads` inflight 段 | LoadSnapshot 增 `inflight` section + 逐队列 describe_req | 上游 `VALID_SECTIONS` 无 inflight，未知 section 直接 `raise ValueError` | 未修复 | **应删**：slime 只请求 `?include=core,disagg,queues`，无调用方 |
| `weight_updater` 会话标记 | 无 session 时跳过 `loaded_weights` 更新 | — | **不应下沉** | 与 A 组协议层同源 |

**C. 模型、多模态与杂项**

| 项 | 作用 | 官方 main | 判定 | 落点/备注 |
|---|---|---|---|---|
| EAGLE draft clamp | `topk_p.clamp(0,1)` / `topk_index.clamp(0, vocab-1)` | 零命中 | 未修复 | `sg`。6 行 |
| radix 只缓存已提交 KV | `get_fill_ids()[: kv.kv_committed_len]` | 未修复 | 未修复 | `sg` |
| radix 跨树 unlock | 父节点 None 时 `break` | `radix_cache.py:619` 仍是 `assert node is self.root_node` | 未修复 | `sg` |
| DSATokenToKVPool 内存区 | `compress_tail_k` 分配缺 `memory_saver_adapter.region(KV_CACHE)` | 未修复 | 未修复 | `sg`。**真实的 sleep 漏洞**，NPU 用 DSA 模型直接相关 |
| DSA head/q_scale 对齐 | `_match_head_gate_q_scale`、`_maybe_repeat_query_heads`（<32 head 补到 32）、`INDEXER_ROPE_NEOX_STYLE` | 三个签名零命中；tile 形状 `(x.shape[0], weight.shape[0], …)` 未修，且代码已移到 `dsa/head_gate.py:46` | 未修复 | `sg` 或 `sg-npu`。**先确认约束来源**：若只是 Ascend fused attention 要求，落 `sg-npu` |
| GLM4V 拆分部署 | `encoder_only` / `language_only`；rope 兼容 `rope_parameters`（TF 5.x） | 模型侧无 `encoder_only` | 未修复 | 待定。先确认是通用能力还是 slime 特需 |
| GLM4V `get_mm_data` | processor 直接吃外部 embedding 并算 mrope | 零命中 | 未修复 | 同上 |
| qwen_vl 回退 | `load_mm_data` → `legacy_load_mm_data` | 上游两方法都在 | **反向改动** | **先查根因**，不能当修复沉 |
| FastAPI path 兜底 | `route.path` 缺失退 `path_format`、再退 `child_scope["route"]` | 未修复 | 未修复 | `sg`。metrics 中间件取不到 path |
| profiler prebuilt | decode 分支同时接 `forward_mode.is_prebuilt()` | `is_prebuilt` 存在但 profiler 未覆盖 | 未修复 | `sg` |
| CLI 别名 ×10 | `--fp8-gemm-runner-backend`、`--nsa-*-backend`、`--cuda-graph-max-bs`、`--prefill-round-robin-balance` 等 | 机制是上游自己的（`Arg(aliases=)`、`argparse_actions.py` 的 `DeprecatedAction`/`DeprecatedStoreConstAction`），这些具体映射未收 | 未修复 | **改 slime launcher 用新名** 更省；或单独小 PR，别混在功能补丁里 |
| mooncake `dst_addr` | `dst_aux_ptrs[i]` → `dst_aux_ptr` | — | **等价改写** | **删**（循环变量同值，纯噪音） |
| mooncake 失败策略 | 删掉"一次失败即标记 session failed" | — | 语义变更 | 单独评估，不随大补丁走 |

### 2.2 三个功能件

| patch | 功能 | 官方 main | 判定 | 落点/备注 |
|---|---|---|---|---|
| `sglang-top_p.patch`（18 文件 / +551−6） | top-p nucleus 掩码 + 支持集内重归一 logprob + meta_info 回传（含 `use_ascend_backend` 分支） | `layers/utils/top_p.py` 不存在；`renorm_logprob_over_top_p` 零命中 | 未修复 | `sg`。拆 3 个 PR（core / meta 链路 / spec+PD）。sampler 的 `use_ascend_backend` 上游存在（`sampler.py:138`），**NPU 直接受益** |
| `sglang-release_hicache.patch`（13 文件 / +276−32） | `--release-hicache`：base 三钩子 + 各池 release/resume + unified 引用计数 + offload 挂钩 | `pool_host/` 无 `release_memory_occupation`，参数不存在 | 未修复 | `sg`，拆 2 个 PR。**NPU 拿不到收益**，见 §5.2 |
| `sglang-pull_weights.patch`（8 文件 / +384−0） | `/pull_weights` 读端 + `weight_sync/local_checkpoint.py`（版本目录、zstd XOR/overwrite delta、校验和、锁、mmap 原地打） | `pull_weights` 零命中；`weight_sync/` 只有 tensor_bucket | 未修复 | `sg`。只沉读端，发布端留 slime。多机 HCCL 慢 → NPU 收益更大 |

### 2.3 sglang-deterministic.patch（27 文件 / +807−81）

| # | 改动 | 目的 | 官方 main | 判定 |
|---|---|---|---|---|
| 1 | `per_token_group_quant*.cuh` 一致性（expf / `__fdiv_rn` / 逐元素 FP8） | fused 与非 fused 位级一致 | 未修复 | 未修复，但 **CUDA 专属 → 交 GPU 线**，Ascend 不适用 |
| 2 | `matmul_persistent` 的 `BLOCK_SIZE_M = 16 if M<=16 else 128` | 小 M 归约固定 | 未修复 | 未修复，`sg`（Triton，设备无关） |
| 3 | `router_gemm_batch_invariant` | router logits 批不变 | 未修复 | 未修复，`sg`（含 fp32 persistent 兜底） |
| 4 | `glm5_router_gemm` JIT 内核（新文件） | GLM5 router 快核 | 未修复 | 未修复，CUDA 专属 → GPU 线 |
| 5 | `enable_fp32_moe_router` | DeepSeek 式 FP32 router | 未修复 | 未修复，`sg`，**但要改写法**（见 §5.3） |
| 6 | DeepGEMM `set_batch_invariant` / deterministic 配置 | 训推同选 kernel | 未修复 | CUDA 专属 → GPU 线 |
| 7 | MoE grouped GEMM M pad 128 | 消除 tile 内归约抖动 | 未修复 | CUDA 专属 → GPU 线 |
| 8 | DeepEP `align_fp8_quantization` + LL prefill staging | dispatch 数值一致 / 分窗 | 未修复 | CUDA/DeepEP 专属 → GPU 线 |
| 9 | DSA head-gate DeepGEMM 化 + topk 确定化 + FP8 KV dequant | 确定性 + 吃 FP8 cache | 未修复 | 拆：DeepGEMM 部分 GPU 线；**dequant 路径设备无关 → `sg`** |
| 10 | `dsa` 加入确定性 backend 列表 | 注册 | 未修复 | 未修复，`sg` |
| 11 | tensor dump 增强 + `SetInjectDumpMetadataReqInput` | 逐层对齐调试 | 零命中 | 未修复，`sg`（纯调试工具） |
| 12 | `weight_updater` 的 `torch.cuda.synchronize()` | CUDA IPC buffer 所有权 | 未修复 | CUDA 专属；NPU 需 hccl/stream 语义另行处理 |

> 这个补丁在 ascend 仓里是**整体背负**：`quick_install.sh` 无条件 `git apply`，但 NPU 上真正执行的只有 2/3/5/10/11 这类通用项。拆完可以把 27 个文件里的大半从 patch 栈摘掉。

### 2.4 0001-tms-forward-graph-args（NPU 专属，+4−4）

| 项 | 作用 | 官方 main | 判定 | 落点 |
|---|---|---|---|---|
| TMS adapter 收位置参数 | 三处 `def cuda_graph(self, **kwargs)` → `*args, **kwargs` | 三处仍是 `**kwargs`（`torch_memory_saver_adapter.py:67/103/149`） | 未修复 | `sg-npu`。**必修**，见 §5.1 |

---

## 三、sgl-kernel-npu 侧（TMS 3 条）

改动对象是 `contrib/torch_memory_saver`（Ascend ACL VMM 版）。`origin/main@3f4d922` 逐项复核：`is_paused` / `auto_dispatch_capture` / `GetAllocationGranularity` **全部零命中**。

| patch | 作用 | origin/main | 判定 | 备注 |
|---|---|---|---|---|
| 0001 allocation lifecycle（+18−6） | `is_paused` 状态位；pause/resume/free 幂等；`aclrtMapMem` 返回值检查 | 零命中 | 未修复 | **等价 PR 被 closed**：`sgl-kernel-npu#867`（+104/−10，2026-10-20）。重开前先问清关闭原因 |
| 0002 graph api + preload（+14−6） | `cuda_graph` 接受 `auto_dispatch_capture` 并传给 `torch.npu.graph`；`configure_subprocess` 注入 `TMS_HOOK_MODE=preload` | 零命中（`TMS_HOOK_MODE` 名字有 3 处引用，但没人注入） | 未修复 | 与 §2.4 成对；不注入 env 就 `assert hook_mode == "preload"` 失败 |
| 0003 unaligned + nested（+39−5） | 非粒度对齐分配回退普通 ACL；嵌套 region 恢复 tag/region/cpu_backup | 零命中 | 未修复 | 0001 的配套（新增 `cann_mem_granularity` + 两个 getter） |

三条对 `origin/main` `patch --dry-run` **零失败**，可原样提。

---

## 四、不涉及 sglang 的补丁（13 条）

| patch | 归属/原因 |
|---|---|
| npu-compat/adapter | slime 职责：`NPUAccelerator`、Ray `{"NPU":1}` bundle、`ASCEND_RT_VISIBLE_DEVICES` / `HCCL_*` / `RAY_EXPERIMENTAL_*`、`PYTHONPATH` 指向 Megatron 栈。**建议提 THUDM/slime**（上游已有 `slime/utils/accelerator/` 抽象与 cuda 后端） |
| npu-compat/empty-cache-captures | Ascend 分配器跨 stream 跟踪图捕获 → torch_npu/CANN，或 TMS 侧缓解；sglang 层无法承接 |
| npu-compat/megatron-root-override、ray-cpu-cap | 运维/测试基础设施 → THUDM/slime |
| upstream-fixes/qwen2-hf-converter | slime 自己的 `convert_hf_to_torch_dist` bug → THUDM/slime |
| upstream-fixes/critic-mla-test-mock | slime 测试桩 → THUDM/slime |
| Megatron ×2 | Megatron-LM / MindSpeed |
| Megatron-Bridge ×3 | Megatron-Bridge |
| mbridge / Megatron-Adaptor | 各仓 |

**应删除/需复核（非下沉）**

| 项 | 处置 |
|---|---|
| `/v1/loads` inflight 段 | 删（无调用方）；要保留则重写为独立可观测性 PR |
| mooncake `dst_addr` 等价改写、`server_control.py` 的纯注释 hunk | 删 |
| `qwen_vl` 的 `legacy_load_mm_data` 回退 | 先查根因 |
| `accelerator_runtime_latest.patch`、`upstream-fixes/thd_packed_context_parallel.patch` | 目录里有、MANIFEST 里没有、不参与构建 → 删或登记 |
| 训练 actor 的 `torch.npu.synchronize()` | 暂留，等 TMS 三条合了再删（见 §5.4） |

---

## 五、深挖

### 5.1 §2.4 那条为什么必修：代码级证据链

| 环节 | 证据 |
|---|---|
| 调用方 | `hardware_backend/npu/graph_runner/npu_cudagraph_backend.py:442` 用 `partial(self._memory_saver_adapter.cuda_graph, tag=…)` 构造，随后**位置传参** `graph_ctx(graph, pool=…, stream=…, auto_dispatch_capture=True)`（:449） |
| 被调方 | `utils/torch_memory_saver_adapter.py:67/103/149` 三处均为 `def cuda_graph(self, **kwargs)` |
| 结果 | `TypeError: cuda_graph() takes 1 positional argument but 2 were given` |
| 触发条件 | 图内存回收由 `enable_memory_saver and SGLANG_MEMORY_SAVER_CUDA_GRAPH` 门控，`environ.py:1977` 为 `EnvBool(False)` → **不是天天炸，是开了就炸**（colocate 想省图显存就得开） |
| 是否被近期重构带走 | 否。11-17 的 `#49999 npugraph_ex` 重写了捕获段（去掉 `skip_guard_context`、`torch.compile` 改 `fullgraph=False`），**位置传参与 adapter 签名都没动** |

### 5.2 release_hicache 与 NPU host pool 的语义错位

补丁释放的是 torch pinned buffer（unregister + `malloc_trim`）。NPU 不是这套：

- `memfabric` 后端：host 池由 MemFabric `Offload` 实例持有；
- `acl` 后端：自 11-17 `#50676` 起改为 **mmap 块**（`alloc_mmap`，`SGLANG_HUGEPAGE_SIZE` 支持 2MB/1GB hugetlb）再 `aclrtHostRegister(MAPPED)`；
- `storage/mmap/mmap_allocator.py` 只有 `alloc_mmap:276` / `alloc_shm`，**没有 free/munmap**。

⇒ Ascend 要拿收益的正确顺序是"先给 mmap 分配器补释放路径，再做 host pool release/resume"，不是 unregister pinned。**PR 里别承诺 NPU 收益。**

### 5.3 §2.3 #5 的隐性回归

补丁把 `MoEGate` 原来的 `if _is_cuda or _is_hip: torch.mm(..., out_dtype=fp32)` 换成了"flag 开路，否则 `F.linear`"。也就是说**不开 `enable_fp32_moe_router` 时，CUDA 的 router logits 从 fp32 变成 bf16**。上游化必须保留原默认分支，新行为只挂 flag。

### 5.4 上游 11-16→11-17 的三条相关动静

| 提交 | 与本文判断的关系 |
|---|---|
| `#50717 wait_for_launched_forward()` | 已在 disk/distributed/tensor/ipc 与 begin/end session **全部**权重更新路径插入 `schedule_stream.wait_stream(forward_stream)`，动机是"in_place pause 可能让 forward 还在读权重"。它解决**流序**，TMS 要的是**设备级 unmap 前在途 capture 收敛**——相关不等价。所以 slime 的 `torch.npu.synchronize()` 动机收窄但不消失；写 TMS 的 PR 可引它说明方向 |
| `#50676 hicache mmap allocator` | 见 §5.2，把 release_hicache 的 NPU 落点钉死 |
| `#49999 npugraph_ex` | 见 §5.1，重构了捕获路径但没碰坑 |

### 5.5 漂移量化（2026-11-17 main 上 `patch -p1 --dry-run`）

| 补丁 | 总 hunk | 打不上 | 比例 |
|---|---|---|---|
| `sglang.patch` | 100 | 38 | 38% |
| `release_hicache` | 33 | 10 | 30% |
| `top_p` | 56 | 13 | 23% |
| `deterministic` | 59 | 11 | 19% |
| `pull_weights` | 13 | 2 | 15% |
| **4 个 NPU 补丁** | 6 | **0** | **0%** |

结论：**NPU 四条可原样提，功能补丁一律"新写一遍"**，别 `git am`。另有一类"文件搬家"要当心：DSA head-gate 的 tile 形状修复在补丁里打的是 `dsa/dsa_prefill_cuda_graph.py`，上游已把它挪到 `dsa/head_gate.py:46`，定位文件时不能按旧路径找。

---

## 六、行动建议与参考

**下沉优先级**

1. 第一批（本周，最小改动）：§2.4 TMS adapter → sgl-kernel-npu 0001+0003 合并重开 #867 → 0002 独立。
2. 第二批（小修，一次 4~5 个 PR）：EAGLE clamp、radix ×2、DSATokenToKVPool 内存区、FastAPI path、server_info 超时、compressed_tensors 守卫。
3. 第三批（端点/指标）：`/post_process_weights`、`/get_weight_version`、PD 阶段指标、TTFT/ITL、PD 超时与在途清理、retract 语义（需讨论）。
4. 第四批（大件）：release_hicache（拆 2）→ pull_weights → top_p（拆 3）。
5. 清理：deterministic 按硬件分流（通用沉 sg、CUDA 交 GPU 线）；删 inflight 段 / 等价改写 / 注释噪音 / 2 个孤儿补丁文件；复核 qwen_vl 回退与 GLM4V 拆分。

**参考**

- 排查对象：[slime-ascend patches/third_party](https://gitcode.com/Ascend/slime-ascend/tree/main/patches/third_party)、[patches/README.md](https://gitcode.com/Ascend/slime-ascend/blob/main/patches/README.md)（补丁分类与维护流程）、`patches/MANIFEST.toml`（逐 patch 元数据与上游跟踪号）
- 上游证据文件：[torch_memory_saver_adapter.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/utils/torch_memory_saver_adapter.py)、[npu_cudagraph_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py)、[pool_host/](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/mem_cache/pool_host)、[dsa/head_gate.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa/head_gate.py)、[sgl-kernel-npu contrib/torch_memory_saver](https://github.com/sgl-project/sgl-kernel-npu/tree/main/contrib/torch_memory_saver)
- 本地仓：`D:\WorkSpace\slime-ascend`（源）、`D:\WorkSpace\sglang`（upstream 已 fetch 至 `c67db7c762`）、`D:\WorkSpace\sgl-kernel-npu`（`origin/main@3f4d922`）
- 核对方式（2026-11-17 全量）：`git fetch upstream` 后 `git grep -c <签名> upstream/main -- python/`，命中=已吸收、零命中=未修复；漂移用 `patch -p1 --dry-run` 在 upstream/main 的 worktree 上干跑（不落盘）
