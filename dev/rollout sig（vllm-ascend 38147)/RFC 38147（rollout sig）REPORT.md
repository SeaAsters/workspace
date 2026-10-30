# vLLM Issue #38147 在 vllm-ascend 上的生效性验证

验证日期：2026-10-27
目标机：`192.168.13.188`（hostname `node-zj-110-132`），容器 `lzy-vllm-ascned`
被测 issue：<https://github.com/vllm-project/vllm/issues/38147>

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| issue 描述的功能在 vllm-ascend 上能生效吗？ | **能生效。** 3 个 endpoint 全部返回 `200` 且数据正确 |
| 按 issue/README 里给的命令能直接跑起来吗？ | **不能。** 会 `RuntimeError: Failed to infer device type` 启动失败 |
| 需要改什么？ | `VLLM_PLUGINS` 白名单里必须**加上 `ascend`**（否则 Ascend 平台插件被一起过滤掉） |
| 有功能缺口吗？ | 有 1 处：`kv-cache` 的 `groups` 恒为空，原因是 vLLM 侧 #48121 未落地，**与 Ascend 无关** |

一句话：**功能可用，但启用方式是"vLLM 通用 + Ascend 特有白名单"的组合，issue 里的一行式启用命令在 NPU 上不成立。**

---

## 1. 验证环境

| 项 | 值 |
|---|---|
| 容器镜像 | `quay.nju.edu.cn/ascend/vllm-ascend:nightly-main-a3` |
| vLLM | `0.31.0+empty`（源码树 `/vllm-workspace/vllm`） |
| vllm-ascend | `0.19.1rc2.dev3033+g4e6eac4e4`（源码树 `/vllm-workspace/vllm-ascend`） |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post7` |
| CANN | `9.2.0` |
| Python | `3.12.13` |
| 硬件 | 8 颗物理 NPU × 2 die = **16 个 vLLM 逻辑设备**（A3 按 16 算） |
| 本次使用 | `ASCEND_RT_VISIBLE_DEVICES=8,9`（NPU4 的两颗 die，验证时为空闲） |
| 权重 | `/mnt/weight/Qwen3-30B-A3B`（`Qwen3MoeForCausalLM`，48 层，128 experts，top-8） |
| 并行 | `--tensor-parallel-size 2` |
| 插件包 | `llm-d-api-extensions-vllm 0.1.0`（从 `llm-d/llm-d-api-extensions` 源码 `pip install --no-deps .`） |

> 注意：该 issue 的交付物**不是 vLLM 内核代码**，而是外部 pip 包 `llm-d-api-extensions-vllm`，通过 vLLM 的 endpoint plugin 框架接入。所以"能否生效"实际上取决于两件事：① 容器里的 vLLM 有没有 endpoint plugin 框架；② Ascend 平台能否被这套框架容纳。

---

## 2. issue 要求的能力拆解

issue 最终形态（不是最初的核心 `/v1/server/*` 方案，那个已被 superseded）要求：

1. **endpoint plugin 框架**：`vllm.endpoint_plugins` entry point group + `VLLM_PLUGINS` 白名单，默认关闭。
2. **worker 扩展机制**：`--worker-extension-cls` 把 `DeviceInfoWorkerExtension` 混入具体 Worker 类，从而通过 `collective_rpc("get_device_properties")` 拿到每 rank 硬件信息。
3. **三个只读 GET endpoint**：
   - `/plugins/llm-d-server-introspection/config`
   - `/plugins/llm-d-server-introspection/devices`
   - `/plugins/llm-d-server-introspection/kv-cache`

在容器里逐项核对源码，三条机制在 vLLM 0.31.0 里**都在**：

- `vllm/plugins/endpoint_plugins/interface.py`（`EndpointPlugin` 协议 + `attach_endpoint_plugins`）
- `vllm/v1/worker/worker_base.py:289` `Injected ... for extended collective_rpc calls`
- `EngineClient` 侧 `get_kv_cache_group_metadata` **只存在于 `vllm/v1/engine/core.py:460`**，没上 `EngineClient` / `AsyncLLM` → 这就是 `groups` 为空的原因（对应 PR #48121 未合入）。

vllm-ascend 侧**没有**自己的 `entrypoints/`，也**没有** patch `build_app` / `init_worker` / `api_server`，所以 API app 与 worker 扩展都是 vLLM 内核原路径 —— 这是兼容性成立的根本原因。

---

## 3. 关键发现 ①：`VLLM_PLUGINS` 是全局白名单，会连带关掉 Ascend 平台

这是本次验证**最重要**的结论。

`vllm/plugins/__init__.py: load_plugins_by_group()` 对所有 plugin group 都用同一个 `envs.VLLM_PLUGINS` 做过滤；而 `vllm/platforms/__init__.py:250` 的 `resolve_current_platform_cls_qualname()` 正是走 `load_plugins_by_group(PLATFORM_PLUGINS_GROUP)`。

容器里安装的 plugin entry points：

```
vllm.platform_plugins
    vllm_ascend | ascend -> vllm_ascend:register          ← 少了这个就没有 NPU
vllm.general_plugins
    vllm_ascend | ascend_kv_connector -> vllm_ascend:register_connector
    vllm_ascend | ascend_model        -> vllm_ascend:register_model
    vllm_ascend | ascend_model_loader -> vllm_ascend:register_model_loader
    vllm_ascend | ascend_service_profiling -> vllm_ascend:register_service_profiling
vllm.endpoint_plugins
    llm_d_server_introspection_config  -> ...config_plugin:ServerConfigPlugin
    llm_d_server_introspection_devices -> ...devices_plugin:ServerDevicesPlugin
    llm_d_server_introspection_kv_cache-> ...kv_cache_plugin:ServerKVCachePlugin
```

平台探测 A/B（同一容器，只改 `VLLM_PLUGINS`）：

| `VLLM_PLUGINS` | 探测到的 platform | `device_type` |
|---|---|---|
| `llm_d_server_introspection_config,llm_d_server_introspection_devices,llm_d_server_introspection_kv_cache`（**issue/README 原文**） | `vllm.platforms.interface.UnspecifiedPlatform` | `""` |
| `ascend,llm_d_server_introspection_config,...` | `vllm_ascend.platform.NPUPlatform` | `npu` |
| 不设置（基线） | `vllm_ascend.platform.NPUPlatform` | `npu` |

端到端复现 issue/README 原文命令：

```
RuntimeError: Failed to infer device type, please set the environment variable
`VLLM_LOGGING_LEVEL=DEBUG` ...
  File "/vllm-workspace/vllm/vllm/config/device.py", line 55, in __post_init__
[ERROR] ... ERR99999 UNKNOWN applicaiton exception
exit status: 1
```

即：**照抄 issue 的启用命令，vllm-ascend 上服务根本起不来**（报错还很不直观，指向 device 推断而不是插件）。

---

## 4. 关键发现 ②：`kv-cache` 的 `groups` 恒为空 —— 与 Ascend 无关

实测返回：

```json
{"kv_cache_size_tokens": 593664, "max_concurrency": 72.46875,
 "num_gpu_blocks": 4638, "num_cpu_blocks": null, "groups": []}
```

`kv_cache_plugin.py` 里 `groups` 来自 `engine_client.get_kv_cache_group_metadata()`，用 `getattr` 特性探测；这个方法在 vLLM 0.31.0 里**只在 `EngineCore` 上**，没暴露到 `EngineClient`，于是回落到"capacity only"。

这是 vLLM 版本水位问题（PR #48121），在 CUDA 上跑同样会得到 `groups: []`，**不是 vllm-ascend 的适配缺口**。

另外 `num_cpu_blocks: null` 也是本版 vLLM 的 `cache_config.num_cpu_blocks` 为 `None` 所致。

**capacity 数值可信**：与 `/metrics` 的 `vllm:cache_config_info` 完全一致，正好印证 issue 的动机（"现在只能从 telemetry metric 里抠这些数据"）：

```
block_size="128" kv_cache_size_tokens="593664" num_gpu_blocks="4638" kv_cache_max_concurrency="72.46875"
```

`593664 = 4638 × 128` ✓

---

## 5. 实测三个 endpoint 的返回（TP=2，2 die）

### `GET /plugins/llm-d-server-introspection/config` → HTTP 200

```json
{
  "model": {"name": "qwen3-30b-a3b", "served_names": ["qwen3-30b-a3b"],
            "dtype": "bfloat16", "quantization": null, "max_model_len": 8192},
  "kv_cache": {"gpu_memory_utilization": 0.92, "dtype": "bfloat16",
               "enable_prefix_caching": true},
  "scheduler": {"max_num_seqs": 256, "max_num_batched_tokens": 2048,
                "enable_chunked_prefill": true, "policy": "fcfs"},
  "parallelism": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1,
                  "data_parallel_size": 1, "data_parallel_rank": 0},
  "features": {"speculative_decoding": false, "lora": false, "hma": true},
  "kv_transfer": null
}
```

值得注意：`gpu_memory_utilization` 报的是 **0.92**，而我没传这个参数 —— 说明 endpoint 读到的是 vllm-ascend `check_and_update_config` **覆盖后的生效值**，而不是 argparse 默认值。这正是 issue 想要的"operator 意图 + 平台实际生效配置"。

### `GET /plugins/llm-d-server-introspection/devices` → HTTP 200

```json
{"devices": [
  {"rank": 0, "name": "Ascend910_9382", "total_memory_bytes": 65787658240,
   "compute_capability": null, "num_compute_units": 24},
  {"rank": 1, "name": "Ascend910_9382", "total_memory_bytes": 65796046848,
   "compute_capability": null, "num_compute_units": 24}
]}
```

- 每 rank 数据来自真实 NPU，`collective_rpc` 通路打通。
- `compute_capability: null` —— 预期行为：`NPUPlatform.get_device_capability()` 直接 `return None`（NPU 无 CUDA compute capability 概念），插件的 `_safe()` 把它降级为 `null`，符合 README "nullable" 契约。
- `num_compute_units: 24` —— 实际是 `torch.npu.get_device_properties().cube_core_num`（vllm-ascend 已把它实现成 CUDA SM 的等价物）。
- `total_memory_bytes ≈ 61.27 GiB`，比 `npu-smi` 标称的 65536 MB 略小（CANN 上报可用总量口径），做容量规划时以这个为准更稳。

### `GET /plugins/llm-d-server-introspection/kv-cache` → HTTP 200

见第 4 节（`groups` 为空）。

### 附带确认

- 旧的核心路径 `/v1/server/{config,devices,kv-cache}` → **HTTP 404**，确认已被 superseded 的 plugin 方案取代。
- worker 扩展注入日志（证明混入的是 Ascend 的 Worker 类）：
  ```
  INFO [worker_base.py:310] Injected <class 'DeviceInfoWorkerExtension'> into
       <class 'vllm_ascend.worker.worker.NPUWorker'> for extended collective_rpc calls ['get_device_properties']
  ```
- 服务真能推理：`POST /v1/chat/completions` 正常返回，`system_fingerprint: vllm-0.31.0-tp2-1854ca76`。
- 路由注册日志：3 条 `Route: /plugins/llm-d-server-introspection/{config,devices,kv-cache}, Methods: GET`。
- 启动耗时 ~325–370 s（Qwen3-30B-A3B / TP2 / 图模式）。

---

## 6. 在 vllm-ascend 上的正确启用方式

最小可用（实测通过，服务正常推理）：

```bash
export VLLM_PLUGINS=ascend,llm_d_server_introspection_config,llm_d_server_introspection_devices,llm_d_server_introspection_kv_cache
export ASCEND_RT_VISIBLE_DEVICES=8,9

vllm serve /mnt/weight/Qwen3-30B-A3B \
  --served-model-name qwen3-30b-a3b \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --port 8123 \
  --worker-extension-cls llm_d_api_extensions_vllm.server_introspection.device_worker_ext.DeviceInfoWorkerExtension
```

推荐写法（把 vllm-ascend 自己的 4 个 general_plugin 也保留，避免 model / loader / connector / profiling 注册缺失；实测同样通过）：

```bash
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,ascend_service_profiling,llm_d_server_introspection_config,llm_d_server_introspection_devices,llm_d_server_introspection_kv_cache
```

注意事项：

1. **`ascend` 必须显式列入白名单。** 只要 `VLLM_PLUGINS` 被设置，就等于给所有 plugin group 开了严格白名单，Ascend 平台插件会被静默过滤。
2. `--worker-extension-cls` 只影响 `devices`。不带它服务照常启动，但 `devices` 会返回 `503`（upstream README 的声明；本次验证的是"带上后可用"这一侧）。
3. `ascend_model` / `ascend_model_loader` / `ascend_kv_connector` / `ascend_service_profiling` 对 Qwen3-30B-A3B 这种 vLLM 原生架构模型不是必须的（缺失也能起、也能推理），但换到需要 Ascend 自定义模型注册 / 量化 loader / AscendStoreConnector 的场景就会踩坑，建议一律带上。
4. 想确认白名单没写漏，最快的自检是把启动日志里的 `INFO Loading plugin ...` 和 `Injected ... NPUWorker` 两行核对一遍。

---

## 7. 遗留物与现场清理

- 服务已停止（`pkill -INT`），容器内无我的残留 vllm 进程；`ASCEND_RT_VISIBLE_DEVICES=8,9` 的设备已交还（验证结束时 die 8 上跑的是**另一位用户**的 bare-metal pytest（`x00886548/.../test_batch_invariant.py`），非本次遗留）。
- 容器内新装了 `llm-d-api-extensions-vllm 0.1.0`（`pip install --no-deps`，wheel 版）。不需要时：`pip uninstall -y llm-d-api-extensions-vllm`。
- 日志：`/tmp/introspect.log`（完整白名单）、`/tmp/minimal.log`（最小白名单）、`/tmp/rfcform.log`（issue 原文命令的失败现场）、插件源码 `/tmp/llmdsrc`。

---

## 8. 一句话给上游的结论

> 该 RFC 的 endpoint plugin 机制在 vllm-ascend 上**完全可用**（含 `--worker-extension-cls` 对 `NPUWorker` 的混入），但 `VLLM_PLUGINS` 白名单与平台插件共用同一开关，导致 NPU 后端必须写成 `VLLM_PLUGINS=ascend,llm_d_server_introspection_*`——这一点在任何 OOT（out-of-tree）平台上都成立，不止 Ascend。建议 upstream 文档在 enablement 段落补一句"OOT 平台需把自身 platform plugin 名字一并列入 `VLLM_PLUGINS`"，或让 `PLATFORM_PLUGINS_GROUP` 不受 `VLLM_PLUGINS` 约束。
