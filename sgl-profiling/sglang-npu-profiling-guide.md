# SGLang NPU Profiling 采集与解析指南

> 适用环境：Ascend 950DT + CANN 9.1.0 + torch_npu 2.10.0.post6 + sglang nightly（容器 `lzy_sglang_nightly`）
> 前提：容器内、已 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
> 更新：2026-11-20（基于 11-09 ~ 11-16 实战验证）

---

## 一、采集（5 步）

### 第 1 步：启动 server

```bash
bash /mnt/share/l00982084/glm5.3_11_11/scripts/launch_server.sh \
  > /mnt/share/l00982084/glm5.3_11_11/logs/server.log 2>&1 &
```

- 300B 权重首载 ~14 分钟；**page cache 预热后 ~75 秒**（预热命令：`cat /mnt/weight/GLM-5.3-Flash-BF16/*.safetensors > /dev/null &`，机器 2.2TB 内存足够缓存 620GB 权重）。
- 就绪判据：日志出现 `fired up`，或：

```bash
curl -s -m 5 http://127.0.0.1:15000/get_model_info -o /dev/null -w '%{http_code}\n'   # 200 即就绪
```

### 第 2 步：探活

```bash
curl -s -m 20 -X POST http://127.0.0.1:15000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello", "sampling_params": {"max_new_tokens": 4, "temperature": 0}}' | head -c 120
```

返回 `"text":...` 即生成链路正常。

### 第 3 步：发流量（必须有负载，空跑采不到东西）

```bash
python /mnt/share/l00982084/glm5.3_11_11/scripts/bench_1k8k.py \
  > /mnt/share/l00982084/glm5.3_11_11/logs/bench.log 2>&1 &
```

**等待目标阶段稳定再触发**——profiler 不区分 prefill/decode，窗口内跑什么采什么：

| 目标阶段 | 触发判据（看 server 日志） |
|---|---|
| decode | `Decode batch, #running-req: <目标并发>` 连续出现（并发已爬满） |
| prefill | 流量刚进入、`Prefill batch` 活跃（chunked prefill 中） |

### 第 4 步：触发采集

```bash
curl -s -X POST http://127.0.0.1:15000/start_profile -H 'Content-Type: application/json' -d '{
  "output_dir": "/mnt/share/l00982084/glm5.3_11_11/profiles/decode-mid",
  "num_steps": 10,
  "with_stack": false,
  "record_shapes": true
}'
# 返回 "Start profiling." HTTP=200 即触发成功
```

#### 参数说明（ProfileReq 全量）

| 参数 | 建议值 | 说明 |
|---|---|---|
| `output_dir` | 共享盘路径 | 产物位置（NFS 路径多机可取） |
| `num_steps` | 10 | 采 N 个调度步后**自动停止并落盘**，无需 stop_profile |
| `with_stack` | **false（显式传）** | ⚠️ 不传默认 true（`SGLANG_PROFILE_WITH_STACK` 默认开），栈记录加 ~30% host 开销 |
| `record_shapes` | true | 记录算子输入形状——做 GEMM FLOPs/MFU 分析必需；纯看时间线可 false |
| `profile_by_stage` | true（可选） | **prefill/decode 分成两份独立 trace**；decode 开窗自动排除 prefill 混入 |
| `start_step` | - | 从第 N 步开始采 |

#### 按阶段采集的环境变量

```bash
export SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=4   # decode 采等 batch_size≥4 才开窗（避开并发爬升期）
```

### 第 5 步：确认落盘

```bash
# 等 1~2 分钟（解析期间 HTTP 无响应是正常的）
ls /mnt/share/l00982084/glm5.3_11_11/profiles/decode-mid/*/ASCEND_PROFILER_OUTPUT/
```

看到 `trace_view.json`、`kernel_details.csv` 等即完成。TP8 会有 8 份（每 rank 一套）。

### 训练侧（对照）

slime/Megatron 训练用 hook 方式（非 HTTP）：模块挂 `--custom-megatron-before-train-step-hook-path`，环境变量控制：

```bash
GLM53_PROFILE_TRAIN=1            # 总开关
GLM53_PROFILE_STEP_START=3       # 采集起始 global step
GLM53_PROFILE_STEP_END=3         # 结束 step
GLM53_PROFILE_DIR=/mnt/share/xxx/profiles/   # 输出目录
GLM53_PROFILE_RANKS=0            # 只采 rank0
GLM53_PROFILE_WITH_STACK=0       # 默认 0（训练侧默认无栈）
```

Hook 位置：`slime-ascend/scripts/lib/glm53/runtime/glm53_train_profiler.py`，hook 内每步调 `prof.step()`（有 ProfilerStep#N 标记）。

---

## 二、解析

### 情况 1：`/start_profile` 采集 → 解析自动完成

profiler stop 后 sglang 内部自动调 msprof 解析（日志出现 `parsing profiling`，1-2 分钟），`ASCEND_PROFILER_OUTPUT/` 即全部解析结果，**无需手动解析**。

#### 产物文件一览

```
<output_dir>/<host>_<pid>_<ts>_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/              ← 解析结果（全部数据）
│   ├── trace_view.json                  # 时间线 → ui.perfetto.dev
│   ├── kernel_details.csv               # 逐 kernel：Start/Duration/Wait/Input Shapes（脚本分析主力）
│   ├── operator_details.csv             # 逐算子（含 Call Stack 列）
│   ├── op_statistic.csv                 # 算子聚合统计（几 KB，快速看占比）
│   ├── step_trace_time.csv              # 每步 Computing/Communication/Free 分解
│   ├── task_time.csv / api_statistic.csv
│   ├── communication.json               # HCCL 各算子带宽/等待占比
│   ├── communication_matrix.json        # HCCL 拓扑矩阵
│   ├── analysis.db / ascend_pytorch_profiler_0.db   # sqlite（Insight 可开）
│   └── ...
└── PROF_xxx/                            # 原始二进制（MindStudio Insight 打开）
```

#### 文件 → 分析用途映射

| 分析目标 | 用哪个文件 / 工具 |
|---|---|
| 时间线肉眼看 | `trace_view.json` → [ui.perfetto.dev](https://ui.perfetto.dev)；或 MindStudio Insight 开 PROF_xxx |
| 算子占比快速看 | `op_statistic.csv`（按 Device Duration 排序） |
| gap/Free 分析、步内算子序列 | `kernel_details.csv`（区间合并法；现成脚本见下） |
| GEMM FLOPs/MFU | `kernel_details.csv` 的 Input Shapes 列（bf16 = 2 FLOPs/MAC，峰值 425 TFLOPS/卡） |
| 通信是否瓶颈 | `communication.json`（Wait Time Ratio≈1.0 = 纯等待） |
| 每 token 成本归一化 | 窗口总时长 ÷ (num_steps × 并发数) |
| host 提交延迟 | `kernel_details.csv` 的 Wait Time 列 |

### 情况 2：只有原始 PROF_xxx、缺解析输出（手动解析兜底）

```bash
msprof --output=<PROF目录>      # 有 CANN msprof 的环境
# 或 MindStudio Insight 打开 PROF 目录，GUI 内重新解析
```

### 现成分析脚本（本地 D:\WorkSpace\glm53-mfu-profiling\）

| 脚本 | 功能 |
|---|---|
| `analyze_compare.py` | 两份 trace 的算子类别对比（含 GEMM MFU） |
| `analyze_timeline.py` | gap 分布 + 步内算子序列模板提取 |
| `analyze_opt_points.py` | 优化点挖掘（gap 定位到步内位置 / Wait Time / SwiGlu 展开链 / HCCL） |

改脚本顶部 `PATHS` 字典指向新 trace 即可复用。

---

## 三、验证清单（采集成功判据）

```bash
grep -c 'parsing profiling\|Profiling done' <server日志>       # ≥1 解析完成
ls <output_dir>/*/ASCEND_PROFILER_OUTPUT/op_statistic.csv      # 文件存在
head -2 <output_dir>/*/ASCEND_PROFILER_OUTPUT/step_trace_time.csv
```

## 四、已知坑（实战踩过）

1. **窗口内容由触发时机决定**：profiler 不区分 prefill/decode；空 server 触发采不到东西；prefill 混入用 `profile_by_stage: true` 排除。
2. **with_stack 默认值看版本**：nightly 的 `SGLANG_PROFILE_WITH_STACK` 默认 true——不显式传 false 就是带栈采。torch_npu 的 Call Stack 列（operator_details.csv）开关是有效的：关了就是 0 填充。
3. **profiler 开销 ~30%**：采出的步时长比真实高 ~30%，吞吐数字需修正；device kernel 相对分布仍可信。要纯净基线：`with_stack:false + record_shapes:false`。
4. **解析期间 HTTP 挂起**：正常；解析完成前发的第二个 start_profile 会静默丢失（HTTP=000/空），等落盘再发。
5. **显存余量**：profiler 本身占 device 内存，边缘配置（300B + mem-fraction 0.9）采集可能 OOM；300B 限 max-running 4（KDA 状态缓存 180MB/条，32 并发差 5.6GB 直接崩）。
6. **MindStudio Insight 看不到 Free 占比条**：sglang 调度循环没有 `profiler.step()` 标记（训练 hook 有）。解法：脚本合并区间法分析，或打补丁——`scheduler.py` 的 `process_batch_result` 加 `self.profiler_manager.step()`。
7. **权重加载加速**：先 `cat *.safetensors > /dev/null` 预热 page cache（2.2TB 内存可全缓存 620GB），重载 837s → 75s。

## 五、最简命令卡片

```bash
# 触发（等 decode 稳定后）
curl -X POST http://127.0.0.1:15000/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/mnt/share/l00982084/glm5.3_11_11/profiles/x","num_steps":10,"with_stack":false}'

# 确认落盘（1-2 分钟后）
ls <output_dir>/*/ASCEND_PROFILER_OUTPUT/ | head

# 时间线查看
# trace_view.json → https://ui.perfetto.dev
# PROF_xxx/       → MindStudio Insight
```

## 六、本仓库相关路径

| 内容 | 路径 |
|---|---|
| server 启动脚本（300B Flash-BF16） | `/mnt/share/l00982084/glm5.3_11_11/scripts/launch_server.sh` |
| 压测脚本 | `/mnt/share/l00982084/glm5.3_11_11/scripts/bench_1k8k.py` |
| 已编译 kernel wheel | `/mnt/share/l00982084/sgl-kernel-npu/output/`（sgl_kernel_npu 2026.11.0 tag + deep_ep 1.0.0+cann.9.1.0） |
| 本地分析脚本 | `D:\WorkSpace\glm53-mfu-profiling\{analyze_compare,analyze_timeline,analyze_opt_points}.py` |
| 本地 trace 数据 | `D:\WorkSpace\profiling\{glm5.3flash-rl, glm5.3falsh-infer}` + 4 个 tar |
