# SGLang extract / return / aux hidden states 增量跟踪：2026-10-30

UTC 窗口：`[2026-10-23T02:05:41Z, 2026-10-30T02:34:03Z)`。采集截止：`2026-10-30T02:34:03Z`（第三期，接续 10/23 第二期）。

代码基准：[官方 main `0f3f619850f2`](https://github.com/sgl-project/sglang/commit/0f3f619850f222b4f238ff3d336926e99a2a5c56)（该提交时间 `2026-10-30T02:08:22Z`，本地浅克隆观察于 `02:34:03Z`）。按用户要求，本轮 extract-hidden-status 按仓分文件夹成文，vLLM 与 vllm-ascend 侧见 [../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-10-30.md](<../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-10-30.md>)；既有 RL-feature 归档仅作增量基线。

## 本期结论

纳入实质变化或技术讨论 **13 条**，分类为 旧PR变化 2 条、本期新建 7 条、前期补漏续跟 4 条。这些不是同口径的新增功能数量：13 条中 9 条在本期合入 main，2 条为 open 提案，1 条为 open RFC，1 条为关闭未合入。

本期最重要的事实是两条 P0 都已合入，上一期「未集成、只移植 CLI」的判断已经过期：

- [#43449](https://github.com/sgl-project/sglang/pull/43449) 于 `2026-10-26T23:18:44Z` 合入，交付 `--enable-return-aux-hidden-states`：不加载 draft 也能在 target 上按 EAGLE3/DFLASH/DSPARK 约定捕获 aux hidden states，末层 hidden 另存 `last_hidden_states`。**该合入同时给 NPU graph runner 增加了 `last_hidden_states` 切片**，上一期记录的 NPU 缺字段问题已随代码变化。
- [#43450](https://github.com/sgl-project/sglang/pull/43450) 于 `2026-10-27T20:37:13Z` 合入，交付 `ModelRunner.forward_observer` 与 `Scheduler.defer_response` / `ResponseCompletion`，把「forward 后读取输出」和「异步发布响应」变成公共接口；文档同时写明 PD disaggregation 不支持。

仍然没有落地的边界：pp>1 的 aux relay 只有 [#46320](https://github.com/sgl-project/sglang/pull/46320) 提案（open）；PD 组合在 #43450 文档里被明确排除；激活级监视只有 [#46678](https://github.com/sgl-project/sglang/issues/46678) RFC。**结论不能写成「SGLang 在 NPU 上已支持 aux 捕获」**：合入只证明了接口与图路径字段存在，Ascend 侧没有任何本期设备验证。

## 增量索引

- [#43449](https://github.com/sgl-project/sglang/pull/43449) — 旧PR变化；期末 `merged`，采集时 `merged`（`2026-10-26T23:18:44Z`）；[Feat] spec: Capture aux hidden states on the target without a speculative draft。
- [#43450](https://github.com/sgl-project/sglang/pull/43450) — 旧PR变化；期末 `merged`（`2026-10-27T20:37:13Z`）；[Feat] scheduler: Add forward observer and deferred responses for plugins。
- [#46320](https://github.com/sgl-project/sglang/pull/46320) — 本期新建；期末 `open`；[Feat] spec: Support EAGLE3 under pipeline parallelism by relaying aux hidden states。
- [#46678](https://github.com/sgl-project/sglang/issues/46678) — 本期新建（RFC）；期末 `open`；[RFC] Activation monitors in SGLang。
- [#46290](https://github.com/sgl-project/sglang/pull/46290) — 本期新建；期末 `merged`（`2026-10-25T04:32:34Z`）；[Refactor] spec: Resolve draft parallel placement from model declarations。
- [#47072](https://github.com/sgl-project/sglang/pull/47072) — 本期新建；期末 `merged`（`2026-10-29T18:01:34Z`）；[Feat] scheduler: Add typed per-token and per-request customized_info values。
- [#46176](https://github.com/sgl-project/sglang/pull/46176) — 本期新建；期末 `merged`（`2026-10-24T00:24:07Z`）；[Perf] dsv41: Trim per-step DSpark decode overhead。
- [#47047](https://github.com/sgl-project/sglang/pull/47047) — 本期新建；期末 `open`；[Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE all-to-all backend。
- [#45144](https://github.com/sgl-project/sglang/pull/45144) — 前期补漏续跟；期末 `merged`（`2026-10-24T04:24:59Z`）；[Feat] spec: Let out-of-tree platforms provide their own speculative worker。
- [#45145](https://github.com/sgl-project/sglang/pull/45145) — 前期补漏续跟；期末 `merged`（`2026-10-24T04:28:56Z`）；[Feat] spec: Let out-of-tree platforms provide EAGLE draft graph runners。
- [#47216](https://github.com/sgl-project/sglang/pull/47216) — 本期新建；期末 `merged`（`2026-10-30T02:08:02Z`）；[Fix] [NPU] dspark: Avoid double-counting verify KV lengths in graph replay。
- [#42078](https://github.com/sgl-project/sglang/pull/42078) — 前期补漏续跟；期末 `closed`（superseded，`2026-10-30T01:50:34Z`）；[DeepSeek-V4.1] Allow bounded replay with breakable prefill。
- [#45925](https://github.com/sgl-project/sglang/pull/45925) — 前期补漏续跟；期末 `merged`（`2026-10-26T08:42:30Z`）；[Feat] minimax: Support MiniMax-M3.1-Flash。

## 重点 PR：机制、证据与 NPU 跟进

以下机制说明以采集时 head 与当前 main 源码为参照；本期进展单独列出，避免把历史实现全部归入当期。

### #43449 · P0 · 旧PR变化（本期合入）

[[Feat] spec: Capture aux hidden states on the target without a speculative draft](https://github.com/sgl-project/sglang/pull/43449)，目标分支 `main`，head `25303b32b5`，+942/-10，23 文件，合入者 Qiaolin-Yu。

机制：新增 `--enable-return-aux-hidden-states` 与 `--return-aux-hidden-state-layer-ids`。[`exec_.py` 的字段注释](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/arg_groups/fields/exec_.py#L133)写明：开启后 `--speculative-algorithm` **只用于选择捕获约定**（EAGLE3/DFLASH/DSPARK），`handle_aux_hidden_state_capture` 随后把它复位为 `None`，因此不加载 draft、不为 draft 预留资源、speculative 栈保持关闭；未显式设置时自动把 `--return-hidden-states-mode` 定为 `full`。校验规则集中在 [`validation_hook.py`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/arg_groups/validation_hook.py#L713)：约定名不在三者内、给了 draft 路径、`pp_size != 1`、DFLASH/DSPARK 未给层号、层号为负或重复，都会在启动期报错。输出侧，`hidden_states` 改为拼接后的 aux 捕获，`last_hidden_states` 单独保存供进程内消费者，不进响应路径。

本期证据（window 内 12 个非噪声事件、12 个 patch 提交）：

- `2026-10-23T05:49Z`–`20:31Z`：多轮 review_requested 与 label 变化，属合入前评审。
- `2026-10-26T20:22:54Z`、`20:37:01Z`：两次 head_ref_force_pushed（合入前压平历史）。
- `2026-10-26T22:53:24Z`：[作者评论 CI 通过](https://github.com/sgl-project/sglang/pull/43449)（引用 run 40283769296）。
- `2026-10-26T23:18:44Z`：merged；`23:18:45Z` closed。
- 末三个提交：`e71b7db5db`（keep auxiliary and final hidden-state rows aligned）、`aca27f3e19`（preserve Inkling capture states before logit）、`25303b32b5`（docs: clarify auxiliary capture layer-id conventions）。

NPU 判断：**部分覆盖，需实测**。合入给 NPU graph runner 补了 `last_hidden_states` 切片（见下文源码核对），说明 Ascend 图路径已能传播该字段；但 `pp_size == 1` 的硬限制、各模型 capture hook 在 Ascend 上是否被同样调用、EAGLE3/DFLASH/DSPARK 三种约定在 NPU 的实际层序与数值，都没有本期证据。不能由「CUDA 上合入」推断「Ascend 已支持」。

验证及限制：作者侧新增 `test_aux_hidden_state_capture.py`（394 行）、aux 与最终 hidden 行对齐测试（104 行）、Inkling 专项测试；本期未运行 GPU/NPU。NPU 验收建议覆盖：eager 与图两条路径的 aux 层序与维度、`last_hidden_states` 经 lm_head 得到 logits、padding 裁剪、禁用路径零开销、`pp>1` 走拒绝分支。

关键代码（当前 diff）：`python/sglang/srt/arg_groups/fields/exec_.py`、`python/sglang/srt/arg_groups/validation_hook.py`、`python/sglang/srt/arg_groups/pipeline.py`、`python/sglang/srt/layers/logits_processor.py`、`python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py`、`python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`、`docs/docs/advanced_features/server_arguments.mdx`。

[事件、提交和当前文件 diff](sglang-extract-hidden-states-delta-2026-10-30.json)。

### #43450 · P0 · 旧PR变化（本期合入）

[[Feat] scheduler: Add forward observer and deferred responses for plugins](https://github.com/sgl-project/sglang/pull/43450)，目标分支 `main`，head `9939f78d6f`，+650/-7，13 文件，合入者 Qiaolin-Yu。

机制：`ModelRunner.forward_observer` 是协议对象，回调签名为 `after_forward(batch, forward_batch, logits_output, *, can_run_graph)`，在 target forward 之后、采样清理张量之前执行，可返回 `DeviceAuxiliaryOutput` 交 scheduler 侧拷贝；`Scheduler.defer_response(req, completion)` 配合 `ResponseCompletion.done/result/cancel` 支持异步发布。模块文档明确了三条边界：持有响应不影响 `is_fully_idle`（只有 output rank 持有）、completion 在 KV 释放后拥有 payload、**PD disaggregation 不支持**；并提示图重放可能覆盖保留的 logits 输出视图，消费者必须自行复制。

本期证据（window 内 7 个非噪声事件、8 个 patch 提交）：

- `2026-10-26T23:26:12Z`：renamed（标题从 out-of-tree consumers 改为 plugins）。
- `2026-10-26T23:30:22Z`：head_ref_force_pushed。
- `2026-10-27T20:37:13Z`：merged；`20:37:13Z` closed。
- 提交序列含 `368a9c5816`（deferred abort 路由到嵌入式 Rust 前端）、`701e9adcc1`（scheduler 测试初始化）、`87fb013f6c`（让 observer/deferred 测试可单独执行）、`9939f78d6f`（收敛文档与测试）。

NPU 判断：**缺失/需实测**。这是设备无关的 scheduler/模型层协议，理论可复用；但 NPU 的 stream/event、owned buffer 与图重放语义需独立验证。尤其不能把 graph buffer view 交给延后消费者——文档已警告，但 NPU 上是否同样存在该风险没有证据。PD 组合被文档排除，若 Ascend 侧依赖 PD + 该接口，需要在 admission 阶段显式拒绝。

验证及限制：作者侧 `test_forward_observer.py`、`test_deferred_output.py`；本期未复测。NPU 验收覆盖连续 replay、延迟发布、abort、共享 Future 取消、以及 PD 组合被拒绝。

关键代码（当前 diff）：`python/sglang/srt/model_executor/forward_observer.py`、`python/sglang/srt/managers/deferred_output.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/managers/scheduler_components/output_streamer.py`、`python/sglang/srt/managers/tp_worker.py`。

[事件、提交和当前文件 diff](sglang-extract-hidden-states-delta-2026-10-30.json)。

### #46320 · P0 · 本期新建

[[Feat] spec: Support EAGLE3 under pipeline parallelism by relaying aux hidden states](https://github.com/sgl-project/sglang/pull/46320)，目标分支 `main`，head `8aeabdbd92`，+281/-73，18 文件，`open`。

机制：EAGLE3 draft 不只用 target 末层 hidden，而是读三个 target 层的 aux hidden states，因此 PP 下必须在阶段间 relay 这些张量；PR 同时改 `validation_hook.py` 的 PP 门禁、`runner/base_runner.py`、`runner_utils/buffers.py`、`kimi_k3.py` 等。作者自述这是 #39634（P2「EAGLE3/DFLASH ship aux across stages」）中的 EAGLE3 部分。它和已合入的 #43449 互补：后者明确要求 `pp_size == 1`。

本期证据：

- `2026-10-25T03:13:11Z`：ready_for_review。
- `2026-10-27T07:46:05Z`：[作者评论](https://github.com/sgl-project/sglang/pull/46320)请求评审，说明已在 Llama-3.1-8B、Qwen3-8B 上 PP2/PP4 验证。
- 3 个 patch 提交：`f90406c347`、`6675accd30`、`8aeabdbd92`（relay EAGLE3 aux hidden states across PP）。

NPU 判断：**缺失/需实测**。relay 依赖阶段切分与 send/recv 缓冲，Ascend 需独立验证 relay 顺序、PP2/PP4 下列表长度与数值、以及是否额外占用 draft KV；不能用 CUDA 上的 Llama/Qwen3 结果外推。

验证及限制：作者侧为自测；本期未复测，也未见 NPU 数据。验收应逐阶段比对 aux 层序与数值、覆盖 prefill/decode 两条路径，并确认与 #43449 的捕获约定不冲突。

关键代码（当前 diff）：`python/sglang/srt/arg_groups/validation_hook.py`、`python/sglang/srt/model_executor/runner/base_runner.py`、`python/sglang/srt/model_executor/runner_utils/buffers.py`、`python/sglang/srt/models/kimi_k3.py`。

[事件、提交和当前文件 diff](sglang-extract-hidden-states-delta-2026-10-30.json)。

### #46678 · P1 · 本期新建（RFC）

[[RFC] Activation monitors in SGLang](https://github.com/sgl-project/sglang/issues/46678)，`open`，创建于 `2026-10-26T23:21:43Z`。

机制/边界：提出在引擎内提供激活监视钩子，供外部在 forward 期间读取或拦截激活。与已合入的 forward observer 相比，差别在所有权与生命周期：RFC 若落地必须回答张量何时拷贝、图重放缓冲区如何避免被覆盖、是否沿用 `deferred response` 的终结语义。本期时间线没有非噪声事件（创建本身不计入），属纯设计讨论。

NPU 判断：**缺失**。RFC 文本不构成任何 Ascend 支持；若采纳，NPU 的 stream 语义与 buffer 生命周期需要单独定义。

验证及限制：无代码与设备验证。

### #46290 · P1 · 本期新建（合入）

[[Refactor] spec: Resolve draft parallel placement from model declarations](https://github.com/sgl-project/sglang/pull/46290)，head `08a8d91dd6`，+597/-158，29 文件，`2026-10-25T04:32:34Z` 合入。

机制：把 draft 的并行放置从隐式推断改为模型声明解析（新增 `configs/draft_parallel.py`、`mem_cache/draft_kv_budget.py`），并重写 `model_runner_components/spec_aux_hidden_state.py` 的解析入口。对捕获特性的意义是解耦：draft 放置变化不再改写 target 的捕获层号，但 EAGLE3/DFLASH/DSPARK 各自的层号语义仍分别解释，不能据此宣称三种约定统一。

本期证据：`2026-10-24T21:06:37Z`、`2026-10-25T04:30:06Z` 两次 force-push；`04:32:34Z` merged；2 个 patch 提交（`5a993bef5c`、`08a8d91dd6`）。

NPU 判断：**需实测**。解析层设备无关，但 Ascend 若另有一套 draft 放置/KV 预算实现，必须与该声明式解析一致，否则 aux 层号与 draft 放置错配。

验证及限制：作者以测试替换推断分支；NPU 需在 draft 放置变化时验证捕获层号稳定与 aux 维度一致。

关键代码：`python/sglang/srt/configs/draft_parallel.py`、`python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py`、`python/sglang/srt/mem_cache/draft_kv_budget.py`。

### #47072 · P1 · 本期新建（合入）

[[Feat] scheduler: Add typed per-token and per-request customized_info values](https://github.com/sgl-project/sglang/pull/47072)，head `8254d685ce`，+104/-4，`2026-10-29T18:01:34Z` 合入。

机制：`LogitsProcessorOutput.customized_info` 原先是每请求单值 dict + 逐提交 clone，生产者无法安全发布 device 张量，也没有声明值含义的机制。改为带类型的 per-token / per-request 收集后，捕获值不再依赖 clone 规避图 buffer 覆盖，而由类型与拷贝规则保证。这是 observer/aux 消费者契约层面的修复。

本期证据：`2026-10-28T15:59:57Z`、`17:02:36Z` 两次 force-push；`2026-10-28T21:12:59Z` 作者报告 base-a/base-b 全绿；`2026-10-29T18:01:34Z` merged。

NPU 判断：**需实测**。NPU 图重放下 buffer 生命周期与 stream 可见性需与 CUDA 等价验证，尤其 per-token device 值在 replay 前后不能串值。

验证及限制：作者侧更新 `test_batch_result_processor_hidden_states.py` 等；NPU 验收持续重放、abort 与取消路径。

### #46176 · P2 · 本期新建（合入，相邻性能）

[[Perf] dsv41: Trim per-step DSpark decode overhead](https://github.com/sgl-project/sglang/pull/46176)，head `05f9602021`，+201/-37，`2026-10-24T00:24:07Z` 合入。

机制：新增 `mhc_stream_mean` Triton 内核按 torch 顺序求四路 hc 流均值，供 DSpark 捕获与 replay tail 使用；aux 捕获直接写入 `AuxHiddenStatePacker` 的 slot，去掉 logits processor 里的 `torch.cat`。属捕获链路的相邻开销优化，不改变对外接口。

NPU 判断：**需实测**。Ascend 是否有等价内核；回退到 `x.mean(dim=1)` 时需确认 `-0` 行与 dtype 行为与 GPU 一致，否则捕获均值位级不一致。

验证及限制：作者附 `test_mhc_stream_mean.py` 与 dspark markov sample 测试；无 Ascend 证据。

### #47047 · P2 · 本期新建（open，仅交叉引用）

[[Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE all-to-all backend](https://github.com/sgl-project/sglang/pull/47047)，head `00fcb858a3`，+2/-8，单文件。

机制：把 GLM-5.3-Flash 的 DFlash aux 捕获 gather 条件从枚举 deepep/mooncake/deepep_v2 改为 `is_moe_input_scattered_across_dp_ranks()`，覆盖 FlashInfer FP4 all-gather 与 DWDP 等所有 scattered MoE 输入路径。本期只有 1 次 cross-reference（10/28），没有新提交或评审。

NPU 判断：**需实测**。若 Ascend 的 MoE 切分/通信后端不在该谓词覆盖内，aux 行数会与 attention 层边界行不一致。

验证及限制：作者未附 NPU 证据；建议用 GLM-5.3-Flash 在两种 MoE 后端下做 aux 行对齐对照。

### #45144 / #45145 · P2 · 前期补漏续跟（合入，插件基础设施）

[#45144](https://github.com/sgl-project/sglang/pull/45144)（`2026-10-24T04:24:59Z` 合入，head `78cbea036a`，+206）允许 out-of-tree 平台提供自己的 speculative worker；[#45145](https://github.com/sgl-project/sglang/pull/45145)（`04:28:56Z` 合入，head `f8e4e2b2e7`，+162/-21）允许 out-of-tree 平台提供 EAGLE draft graph runner，两者同一波合入并同步 `docs/docs/hardware-platforms/plugin.mdx`。

与 #43450 的 out-of-tree 消费者方向配套，构成 speculative 侧可替换点。NPU 判断：**需实测**——Ascend 若通过该接口替换 speculative worker 或 draft 图 runner，必须保证 aux 捕获约定与 target 层号仍由公共解析决定，且 `last_hidden_states` 传播不被插件绕过。

### #47216 · P2 · 本期新建（NPU 修复合入）

[[Fix] [NPU] dspark: Avoid double-counting verify KV lengths in graph replay](https://github.com/sgl-project/sglang/pull/47216)，head `45c09e78ff`，+5/-4，仅 `npu_graph_runner.py`，`2026-10-30T02:08:02Z` 由 sglang-npu-bot 合入。

机制：DSpark 图重放中 verify KV 长度被重复累计。与捕获特性的交集在于 NPU graph runner 现在是 aux/last hidden 输出裁剪与 DSpark 元数据的共同落点，长度记账错误会直接表现为捕获行错位或越界。NPU 判断：这是 NPU 自身修复，建议在 DSpark 图重放 + aux 捕获组合下验证行数与 KV 长度一致。

### #42078 · P2 · 前期补漏续跟（关闭，superseded）

[[DeepSeek-V4.1] Allow bounded replay with breakable prefill](https://github.com/sgl-project/sglang/pull/42078) 于 `2026-10-30T01:50:34Z` 以 superseded 关闭（未合入）。作者说明 main 已由 #43736 提供 bounded replay 支持，并由 #44850、#45301、#45407 补齐 tail buffer 修复。属跟踪清理，**不能把关闭本身当修复**；Ascend 若支持该模型与 bounded replay，仍需按 prompt logprob / hidden-state 读取路径单独验证。

### #45925 · P2 · 前期补漏续跟（模型基线合入）

[[Feat] minimax: Support MiniMax-M3.1-Flash](https://github.com/sgl-project/sglang/pull/45925)，head `65dc6195f2`，+6257/-324，63 文件，`2026-10-26T08:42:30Z` 合入；创建于 `2026-10-23T01:03:49Z`（本期窗口开始前 1 小时，上期未纳入）。其中包含 `model_runner_components` 的 aux 捕获测试更新与 speculative 改动，不是捕获接口变更，但改变了该模型在 aux 捕获路径上的可运行基线。NPU 判断：需实测，该模型在 Ascend 的可运行性与 aux 一致性不能由主仓合入推断。

## 当前 main 的接口与 NPU 现状（源码核对）

以下为 `0f3f619850f2` 上的直接核对，用于区分「接口已存在」和「硬件已验证」：

- 捕获开关：[`exec_.py#L133`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/arg_groups/fields/exec_.py#L133) `enable_return_aux_hidden_states`，以及 [`validation_hook.py#L713`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/arg_groups/validation_hook.py#L713) 的 `handle_aux_hidden_state_capture`（含 `pp_size != 1` 拒绝）。
- 响应字段：`return_hidden_states: Union[bool, Literal["last"]]`，见 `python/sglang/srt/entrypoints/openai/protocol.py`（completions/chat 各一处）；`--return-hidden-states-mode` 只有 `last`/`full`。
- NPU 图路径：[`npu_graph_runner.py#L451-L453`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py#L451) 已按 `raw_num_token` 裁剪 `output.last_hidden_states`。**上一期「NPU graph runner 无该字段」的结论已失效**，但这是字段传播，不是数值验证。
- 观测/异步：[`forward_observer.py`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/model_executor/forward_observer.py)、[`deferred_output.py`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/managers/deferred_output.py) 均为设备无关协议；`forward_observer` 引用点覆盖 `model_runner.py` 与 `tp_worker.py`。
- 无 NPU 专项门禁：`--enable-return-aux-hidden-states` 的校验里没有 `is_npu()` 分支，因此 Ascend 走的是同一套校验与 hook；是否真正可用取决于模型 capture hook 与图路径实测。

## 无实质进展与补漏边界

- [#43502](https://github.com/sgl-project/sglang/pull/43502)：本期无活动，事件类型 `{}`；采集时 `merged`。不计入本期实质条目。
- [#45407](https://github.com/sgl-project/sglang/pull/45407)：`2026-10-23` 合入，属 bounded replay 尾部修复，不改变 hidden-state 输出契约，仅作邻近背景。
- [#46481](https://github.com/sgl-project/sglang/pull/46481)：唯一在本期触及 `test_hidden_states.py` 等文件的 main 提交，仅调整 CI `est_time`，无功能变化。
- 普通 `return_hidden_states` 与新 aux 捕获接口应分开判断：前者是响应字段（`bool | "last"`），后者是 `--enable-return-aux-hidden-states` 的捕获约定，二者的层语义与验收方式不同。

## 检索与复核

基线为 10/10→10/16→10/23 三期归档：131 条 tracked PR，与历期 featured 记录合并去重后共 143 条做当前状态复核，另加本期新增发现。关键词、翻页、143 条归档条目状态复核、82 条候选判定与本期证据均见共享 JSON。

- [本期结构化数据](sglang-extract-hidden-states-delta-2026-10-30.json) · [本期 CSV](sglang-extract-hidden-states-delta-2026-10-30.csv)。
- vLLM + vllm-ascend 同期报告：[../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-10-30.md](<../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-10-30.md>)。
- 当前 PR body、current diff、CI 摘要均为采集快照。force-push 前的旧提交与当期 CI 状态无法完整恢复。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。

## 扫描覆盖与源码时间说明

- 关键词检索：sglang 17 个短语（`"hidden states"`、`hidden_states`、`aux_hidden_states`、`"aux hidden"`、`"forward observer"`、`"sampling mask"` 等）× `updated:>=2026-10-23`，另加 RFC 标签与按日切分的 `created:` 全量扫描；去重后 sglang 命中 1563 条 PR/Issue（1460 PR + 103 Issue）。
- 合入侧：直接扫描本地 main 浅克隆 `2026-10-23T02:05:41Z` 以来 552 个提交，按主题关键词与文件路径双重筛选，避免只依赖标题。
- 归档侧：143 条历史 tracked 条目做当前状态复核，其中 9 条在本期有更新。
- Issue/RFC 侧：sglang 本期 103 条 Issue 命中中 7 条进入 hidden 候选，最终仅 RFC #46678 纳入核心分析（其余为通用张量提及或已修复 bug 报告）。
- 候选侧：82 条关键词候选逐条判定，纳入 13 条为核心分析，其余记录在 `candidate_decisions`。

源码快照为远端 main `0f3f619850f222b4f238ff3d336926e99a2a5c56`（提交时间 `2026-10-30T02:08:22Z`），观察于 `2026-10-30T02:34:03Z`；PR/Issue 事件截止同一时刻。两者时间接近，但源码只用于当前覆盖核验，不代表窗口内任意中间状态的源码。
