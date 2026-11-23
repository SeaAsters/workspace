# SGLang NPU Profiling 采集与解析指南

> 适用环境：Ascend A3 / A5(950DT) + CANN 9.1.0 + torch_npu 2.10.0.post6 + sglang（glmx fork 或 nightly）
> 前提：容器内、已 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
> 更新：2026-11-23 —— **已并入 A3 实测踩坑修正**（鉴权、profiler_level、activities 降级、判据分层）

---

## 一、采集

### 1.1 服务端启动参数（关键项）

| 参数 / 环境变量 | 建议值 | 说明 |
|---|---|---|
| `--admin-api-key <key>` | **必须设置**（如 `profkey`） | ⚠️ 新提交起 `/start_profile` 等管理接口需鉴权：请求头带 `Authorization: Bearer <key>`，否则 **403** |
| `--max-running-requests <N>` | 按目标并发（如 8 / 32） | **决定 decode 窗口的 batch_size**；trace 中可用 `step[DECODE bs=<N>]` 核对 |
| `--cuda-graph-bs-decode` | `"1 8 64"`（A3 实测值） | 与窗口 bs 相关，按平台/版本取值 |
| `--tp-size` | 按机器核数 | TP=N → 每 rank 一套产物（核对产物目录数） |
| `--mem-fraction-static` | 依显存余量 | profiler 自身占 device 内存，边缘配置采集中可能 OOM |
| `--chunked-prefill-size` / `--model-path` / `--port` 等 | 常规 | 与本指南无冲突 |

```bash
export SGLANG_PROFILE_WITH_STACK=true                    # 默认就是 true！不改则每次必须显式 with_stack:false
export SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=4           # 按阶段开窗时，decode 等 bs≥N 才采（避开并发爬升期）
```

### 1.2 启动 → 探活 → 发流量

```bash
bash /mnt/share/l00982084/glm5.3_11_11/scripts/launch_server.sh > logs/server.log 2>&1 &

# 就绪判据
curl -s -m 5 http://127.0.0.1:15000/get_model_info -o /dev/null -w '%{http_code}\n'   # 200

# 探活
curl -s -m 20 -X POST http://127.0.0.1:15000/generate -H 'Content-Type: application/json' \
  -d '{"text":"hello","sampling_params":{"max_new_tokens":4,"temperature":0}}' | head -c 120

# 发流量（必须有负载；空跑采不到东西）
python /mnt/share/l00982084/glm5.3_11_11/scripts/bench_1k8k.py > logs/bench.log 2>&1 &
```

**触发时机**（profiler 不区分 prefill/decode，窗口内跑什么采什么）：

| 目标 | 判据（server 日志） |
|---|---|
| decode | `Decode batch, #running-req: <目标并发>` 连续出现（并发已爬满） |
| prefill | 流量刚进入、`Prefill batch` 活跃（chunked prefill 中） |

### 1.3 触发采集（完整请求体）

```bash
curl -s -X POST http://127.0.0.1:15000/start_profile \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer profkey' \
  -d '{
    "output_dir": "/mnt/share/l00982084/glm5.3_11_11/profiles/decode-mid",
    "activities": ["CPU", "NPU"],
    "with_stack": false,
    "record_shapes": true,
    "profile_by_stage": true,
    "profile_stages": ["decode"],
    "num_steps": 10,
    "start_step": null,
    "merge_profiles": true,
    "profile_id": "my_run",
    "profile_prefix": null,
    "detailed_annotations": false
  }'
# 返回 "Start profiling." HTTP=200
```

| 参数 | 建议值 | 说明 |
|---|---|---|
| `activities` | 显式 `["CPU","NPU"]` | ⚠️ **部分版本分支会静默丢弃 "NPU"**，只剩 CPU（纯 host 级数据）——必须用 2.① 前置检查校验 |
| `with_stack` | **false（显式）** | 不传默认 true（`SGLANG_PROFILE_WITH_STACK` 默认开），栈记录 +~30% host 开销 |
| `record_shapes` | true | GEMM FLOPs/MFU 分析必需 |
| `profile_by_stage` | true | prefill/decode 分两份独立 trace；decode 开窗自动排除 prefill 混入 |
| `profile_stages` | `["decode"]` | 配合 `profile_by_stage:true` **只开 decode 窗** |
| `num_steps` | 10 | 采 N 步自动停并自动解析；之后的 `stop_profile` 报 "not in progress" 属良性 |
| `merge_profiles` | true | 多 rank 合并 trace |
| `profile_id` | 建议填（如 `my_run`） | 便于识别批次 |
| `start_step` / `profile_prefix` / `detailed_annotations` | 可选 | — |

**`profiler_level` 是隐藏必需项**：device/kernel 级数据要求 **Level2**（sglang 内建默认 `Level1`，只出 API/算子级数据，采出的 kernel_details 无有效耗时）。当前版本需**通过补丁**把 `_ExperimentalConfig` 的 `profiler_level` 提到 `Level2`（各版本实现方式不同：补丁 / 平台参数 / 环境变量开关，需按版本确认）。

### 1.4 训练侧（对照）

slime/Megatron 训练用 hook（非 HTTP），挂 `--custom-megatron-before-train-step-hook-path`，环境变量控制：

```bash
GLM53_PROFILE_TRAIN=1
GLM53_PROFILE_STEP_START=3
GLM53_PROFILE_STEP_END=3
GLM53_PROFILE_DIR=/mnt/share/xxx/profiles/
GLM53_PROFILE_RANKS=0
GLM53_PROFILE_WITH_STACK=0        # 训练侧默认 0（无栈）
```

Hook 位置：`slime-ascend/scripts/lib/glm53/runtime/glm53_train_profiler.py`，每步调 `prof.step()`（有 ProfilerStep#N 标记）。

---

## 二、采集成功判据（按重要性排序，全满足才算成功）

### ① 前置检查（触发后立即做，不满足趁早停掉重采）

```bash
cat <任一 rank>/ASCEND_PROFILER_OUTPUT/profiler_info_*.json
```

- `activities` 数组**含设备活动**（NPU / ASCEND）；
- `_profiler_level` 为 **`Level2`**（或对应设备级档位）。

> 不满足 = 采出来的是 host 级空数据（kernel_details 全 0、op_statistic 只有 host 算子）。**别等解析完才发现**——重启改配置/打补丁重采。

### ② 解析完成

```bash
grep -c 'parsing profiling\|Profiling done' <server日志>   # ≥1
ls <output>/.../ASCEND_PROFILER_OUTPUT/analyse.done         # 存在
```

### ③ kernel 数据（主判据）

```bash
# kernel_details.csv 存在，且 Duration(us) 列有非 0 值
awk -F',' 'NR>1 && $10+0>0 {c++} END{print c}' kernel_details.csv
```

- ⚠️ **op_statistic.csv 存在不能算成功**——host-only 采集也会生成它，且 Device 列全 0。

### ④ 窗口内容

- decode 目标：trace 中存在 `step[DECODE ...]` 标记，数量 ≈ num_steps × rank 数；
- prefill 目标：对应阶段标记同理。

### ⑤ 反向检查

- 各 rank 产物目录数 = TP 数（`profile_by_stage:true` 会拆 prefill/decode 两波）。

> ⚠️ **`step_trace_time.csv` 不作判据**——非通用文件（A5/nightly 有，A3 某些提交无）。

---

## 三、解析产物

### 通用产出（各环境基本都有）

```
<output_dir>/<host>_<pid>_<ts>_ascend_pt/ASCEND_PROFILER_OUTPUT/
├── trace_view.json                  # 时间线（jsonl 风格）→ ui.perfetto.dev
├── kernel_details.csv               # 逐 kernel：Start Time / Duration(us) / Wait Time /
│                                    #   Input Shapes / aicore·aiv 分解 / cube_utilization（分析主力）
├── operator_details.csv             # 逐算子，含 Call Stack 列（with_stack=false 时 0 填充）
├── op_statistic.csv                 # 算子聚合
├── ascend_pytorch_profiler_<N>.db   # sqlite（PYTORCH_API 表、RANK_DEVICE_MAP 定位 rank）
├── profiler_info_<N>.json           # 会话配置（先查它！见判据①）
├── analyse.done
├── FRAMEWORK/  logs/
```

### 非通用（随版本/等级才产出，**不要列入必查和判据**）

`step_trace_time.csv`、`task_time.csv`、`api_statistic.csv`、`communication.json`、`communication_matrix.json`、`PROF_xxx/` 原始目录、`MSBackup`

### 文件 → 分析用途

| 分析目标 | 用什么 |
|---|---|
| 时间线肉眼看 | `trace_view.json` → [ui.perfetto.dev](https://ui.perfetto.dev)；或 MindStudio Insight 开 `PROF_xxx/` / `ascend_pytorch_profiler_<N>.db` |
| 算子占比快速看 | `op_statistic.csv`（注意 Device 列是否为 0——为 0 说明是 host-only 采集） |
| gap/Free、步内算子序列 | `kernel_details.csv`（区间合并法） |
| GEMM FLOPs/MFU | `kernel_details.csv` 的 Input Shapes（bf16 = 2 FLOPs/MAC，峰值 425 TFLOPS/卡） |
| host 提交延迟 | `kernel_details.csv` 的 Wait Time 列 |
| 通信分析（若有） | `communication.json`（Wait Time Ratio≈1.0 = 纯等待） |

### 解析结果使用注意

1. `kernel_details.csv` 的 `aicore_time(us)`、`cube_utilization(%)` 等**可能为 `N/A` 字符串**，脚本解析要容错（转换失败置 0）；
2. `trace_view.json` 可能是**单行大 JSON**（十万级事件）——按标记/切片分析时别只读文件头几 MB；
3. **聚合口径**：每 rank 一份（TP 切分后各自可见），跨 rank 求和 = 全 TP kernel 时间分布；
4. 解析由 sglang 内建自动完成（日志 `parsing profiling`，一般 **1~4 分钟**）；**无 msprof 工具链的环境（本栈）无法离线补解析**——只能以自动解析为准。

### 现成分析脚本（本地 D:\WorkSpace\glm53-mfu-profiling\）

| 脚本 | 功能 |
|---|---|
| `analyze_compare.py` | 两份 trace 算子类别对比（含 GEMM MFU） |
| `analyze_timeline.py` | gap 分布 + 步内算子序列模板 |
| `analyze_opt_points.py` | 优化点挖掘（gap 定位 / Wait Time / SwiGlu 链 / HCCL） |

改脚本顶部 `PATHS` 即复用。

---

## 四、已知坑（实战踩过）

1. **`/start_profile` 403**：新提交需 `--admin-api-key` + 请求头 `Authorization: Bearer <key>`。
2. **activities 的 "NPU" 被静默丢弃**：部分版本只剩 CPU → 纯 host 数据；靠判据①的 profiler_info 检查提前发现。
3. **profiler_level 默认 Level1**：只出 API/算子级，kernel 无有效耗时；必须补丁提到 Level2（判据①核对）。
4. **with_stack 默认 true**：不显式 false 就是带栈采（+~30% host 开销）。
5. **窗口内容由触发时机决定**：空 server 触发采不到东西；prefill 混入用 `profile_by_stage:true` + `profile_stages:["decode"]` 排除。
6. **op_statistic.csv 不能作为成功判据**：host-only 采集也生成。
7. **profiler 开销 ~30%**：步时长偏高，吞吐需修正；device 相对分布可信。纯净基线：`with_stack:false + record_shapes:false`。
8. **解析期间 HTTP 挂起**（1~4 分钟）正常；期间的第二个 start_profile 会静默丢失。
9. **显存余量**：300B + mem-fraction 0.9 边缘配置采集可能 OOM；300B 限 max-running 4（KDA 状态 180MB/条）。
10. **MindStudio Insight 无 Free 占比条**：sglang 调度循环无 `profiler.step()`（训练 hook 有）。解法：脚本合并区间法，或补丁 `process_batch_result` 加 `self.profiler_manager.step()`。
11. **权重加载加速**：`cat *.safetensors > /dev/null` 预热 page cache（2.2TB 内存可全缓存 620GB），837s → 75s。

## 五、最简命令卡片

```bash
# 触发（等 decode 稳定后；注意 admin key 和 with_stack）
curl -X POST http://127.0.0.1:15000/start_profile \
  -H 'Content-Type: application/json' -H 'Authorization: Bearer profkey' \
  -d '{"output_dir":"/mnt/share/xxx/profiles/x","activities":["CPU","NPU"],
       "with_stack":false,"record_shapes":true,"profile_by_stage":true,
       "profile_stages":["decode"],"num_steps":10,"merge_profiles":true,"profile_id":"run1"}'

# 前置检查（立即）
cat <out>/<rank>_ascend_pt/ASCEND_PROFILER_OUTPUT/profiler_info_*.json | grep -E 'activities|profiler_level'

# 解析完成确认（1~4 分钟后）
ls <out>/.../analyse.done && awk -F',' 'NR>1 && $10+0>0{c++} END{print "kernels_with_time:",c}' kernel_details.csv
```

## 六、相关路径

| 内容 | 路径 |
|---|---|
| server 启动脚本（300B Flash-BF16） | `/mnt/share/l00982084/glm5.3_11_11/scripts/launch_server.sh` |
| 压测脚本 | `/mnt/share/l00982084/glm5.3_11_11/scripts/bench_1k8k.py` |
| 已编译 kernel wheel | `/mnt/share/l00982084/sgl-kernel-npu/output/`（sgl_kernel_npu + deep_ep） |
| 本地分析脚本 | `D:\WorkSpace\glm53-mfu-profiling\{analyze_compare,analyze_timeline,analyze_opt_points}.py` |
| 本地 trace 数据 | `D:\WorkSpace\profiling\{glm5.3flash-rl, glm5.3falsh-infer}` + 4 个 tar |
