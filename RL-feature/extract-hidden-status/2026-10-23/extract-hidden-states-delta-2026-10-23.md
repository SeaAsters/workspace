# SGLang extract / return / aux hidden states 增量跟踪：2026-10-23

UTC 窗口：`[2026-10-17T00:00:00Z, 2026-10-23T02:05:41Z)`。采集截止：`2026-10-23T02:05:41Z`；第二期不代表 10 月 23 日全天。

代码基准：[官方 main `2da6c0c23732`](https://github.com/sgl-project/sglang/commit/2da6c0c237324eefea6420e015beb1e6858cb80c)。按用户最新要求，未使用本地 SGLang 分支判断覆盖；既有 RL-feature 归档仅作增量基线。

## 本期结论

纳入实质变化或技术讨论 **7 条**，分类为 旧PR变化 2 条、前期补漏续跟 1 条、本期新建 3 条、前期新增续跟 1 条。这些不是同口径的新增功能数量。

第二期重点是 #43450 scheduler completion 方案、#44850 替代 #41448 合入，以及 #45096 的无用 CP gather 消除。#43449/#43450 仍在评审，不能将 GPU 集成验证等同官方 main 或 Ascend 已具备新接口。

## 增量索引

- [#41448](https://github.com/sgl-project/sglang/pull/41448) — 旧PR变化；期末 `closed`，采集时 `closed`；[AMD][DeepSeek-V4.1] Fix server crash on prompt-logprob and hidden-state requests under decoder SWA bounded replay。
- [#43449](https://github.com/sgl-project/sglang/pull/43449) — 旧PR变化；期末 `open`，采集时 `open`；[Feat] spec: Capture aux hidden states on the target without a speculative draft。
- [#43450](https://github.com/sgl-project/sglang/pull/43450) — 前期补漏续跟；期末 `open`，采集时 `open`；[Feature] Forward observer and deferred outputs for out-of-tree consumers。
- [#44850](https://github.com/sgl-project/sglang/pull/44850) — 本期新建；期末 `merged`，采集时 `merged`；[Fix] deepseek-v4.1: Serve prompt-logprob requests under decoder SWA bounded replay。
- [#43744](https://github.com/sgl-project/sglang/pull/43744) — 前期新增续跟；期末 `open`，采集时 `open`；[Feat] spec: Add aux hidden-state capture for Granite and GraniteMoe targets。
- [#45096](https://github.com/sgl-project/sglang/pull/45096) — 本期新建；期末 `open`，采集时 `open`；[Perf] DSv4/DSv4.1: Skip the pre_hc_head CP gather when DSpark captures aux hidden states。
- [#45142](https://github.com/sgl-project/sglang/pull/45142) — 本期新建；期末 `open`，采集时 `open`；[Fix] qwen4_exp: Capture DFlash aux hidden states on Qwen3.8-Flash-Next。

## 重点 PR：机制、证据与 NPU 跟进

以下机制说明以采集时 PR head 为参照；本期进展单独列出，避免把当前实现全部归入当期。

### #41448 · P0 · 旧PR变化

[[AMD][DeepSeek-V4.1] Fix server crash on prompt-logprob and hidden-state requests under decoder SWA bounded replay](https://github.com/sgl-project/sglang/pull/41448)，目标分支 `main`。期末 `closed`；采集时 `closed`。

第一期讨论是 bounded replay 遇 prompt-logprob/full hidden-state 请求时应拒绝还是 whole-extend fallback。第二期于 10/17 08:33:53Z 关闭未合入，作者明确注明由已合入的 #44850 替代；修复依据应引用 #44850 的代码，不能把关闭状态本身当修复。

本期证据：
- `2026-10-17T08:33:53Z`：closed。
- `2026-10-17T08:33:52Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41448#issuecomment-6231793764)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：需实测：相同组合若在 Ascend runner 不支持，应 admission 阶段明确拒绝或切换 eager，避免共享 GPU 局部修复未经核实照搬。

验证及限制：覆盖 full/last/关闭、prompt logprob、SWA bounded replay 开关；检查单个请求失败不拖垮服务。关闭原因与替代 PR 只按事件证据引用。

关键代码（当前 diff）：`python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py`、`python/sglang/srt/layers/attention/deepseek_v4_backend.py`、`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`、`python/sglang/srt/models/deepseek_v4.py`、`test/registered/unit/layers/test_dsv4_late_layer_tail.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

### #43449 · P0 · 旧PR变化

[[Feat] spec: Capture aux hidden states on the target without a speculative draft](https://github.com/sgl-project/sglang/pull/43449)，目标分支 `main`。期末 `open`；采集时 `open`。

将 aux hidden-state 捕获从加载 draft 解耦。当前方案以 enable-return-aux-hidden-states 和层选择参数配置 EAGLE3/DFLASH/DSPARK capture convention；捕获 aux 拼接之外另保留最后层 last_hidden_states 给进程内消费者。当前设计限制 PP>1，不能与 PP relay PR 合起来声称已支持。

本期证据：
- `2026-10-19T06:59:48Z` [`504577a5f5`](https://github.com/sgl-project/sglang/commit/504577a5f5ceaa5b97e3fa120b2a38603b24027b)：Merge branch 'main' into feat/aux-hidden-state-capture-without-draft。
- `2026-10-19T07:02:57Z` [`46f0562481`](https://github.com/sgl-project/sglang/commit/46f05624817c3b1c245087131e813f7c30d1eafc)：style: sort argument validation imports。
- `2026-10-19T19:58:19Z` [`56c3434c6c`](https://github.com/sgl-project/sglang/commit/56c3434c6c1fe58fdf4f8c1c259916890dfc0e61)：Merge branch 'main' into feat/aux-hidden-state-capture-without-draft。
- `2026-10-19T22:53:37Z` [`fd50fbc689`](https://github.com/sgl-project/sglang/commit/fd50fbc689a75e83fd37342d41c9d53298799132)：test: fix auxiliary hidden-state capture CI execution。
- `2026-10-19T23:43:04Z` [`35e5a73308`](https://github.com/sgl-project/sglang/commit/35e5a733082780d1e0b5fa19ed9b9a241b0f3eab)：test: pin aux_hidden_state_capture in the resolvable whitelist。
- `2026-10-20T19:09:52Z` [`8f2477d33f`](https://github.com/sgl-project/sglang/commit/8f2477d33f9e731064f14b0d5e8e174b34d2f980)：Merge branch 'main' into feat/aux-hidden-state-capture-without-draft。
- `2026-10-22T07:00:19Z` [`30cf354469`](https://github.com/sgl-project/sglang/commit/30cf3544692e9e59fb1c10e3f42d4001b07e3767)：Merge main into auxiliary hidden-state capture branch。
- `2026-10-22T07:23:41Z` [`c61626a2f3`](https://github.com/sgl-project/sglang/commit/c61626a2f31238e7993fd680baf6369ca7c1917e)：fix: preserve positional slots for auxiliary capture arguments。
- `2026-10-23T00:04:25Z` [`69bb13a600`](https://github.com/sgl-project/sglang/commit/69bb13a6009680193c3a83de8791af43112dc30d)：Merge main to pick up CI precision and AMD fixes。

NPU 判断：缺失：官方 main 有普通 hidden_states 图输出裁剪，但未集成此 capture-only 新接口和 last_hidden_states 字段传播。应补 resolver/参数校验、模型 capture hook、NPU graph runner 输出裁剪与 buffer 生命周期；不能只移植 CLI。

验证及限制：作者报告 GB300/HF bf16 对齐及训练 finite loss，非 NPU 证明。NPU 验收 eager/graph 的 aux 层序和维度、last hidden 经 lm_head 得到 logits、padding 裁剪、无 draft KV 预留、禁用路径开销。

关键代码（当前 diff）：`docs/docs/advanced_features/server_arguments.mdx`、`python/sglang/srt/arg_groups/field_order.py`、`python/sglang/srt/arg_groups/fields/exec_.py`、`python/sglang/srt/arg_groups/pipeline.py`、`python/sglang/srt/arg_groups/resolution_hooks.py`、`python/sglang/srt/arg_groups/serving_hook.py`、`python/sglang/srt/arg_groups/validation_hook.py`、`python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

### #43450 · P0 · 前期补漏续跟

[[Feature] Forward observer and deferred outputs for out-of-tree consumers](https://github.com/sgl-project/sglang/pull/43450)，目标分支 `main`。期末 `open`；采集时 `open`。

在 target forward 后、采样清理张量前提供 forward observer，并为异步发布保留响应。当前方案采用 completion.done/result/cancel，由 scheduler 完成元数据与终结响应；共享 Future、abort、失败和权重版本属于正确性边界。与 #43449 是不同 PR，训练消费链路组合验证不能替代各自合入。

本期证据：
- `2026-10-19T08:54:26Z` [`c78a9fe198`](https://github.com/sgl-project/sglang/commit/c78a9fe198419311c9f298647854ac6ccb6b06fa)：[Feature] Forward observer and deferred outputs for out-of-tree consumers。
- `2026-10-19T08:54:26Z` [`6fec313952`](https://github.com/sgl-project/sglang/commit/6fec3139520668affe34fc8f06da8083d2949c7f)：Keep idle state rank-consistent while deferred outputs are held。
- `2026-10-19T08:54:26Z` [`ac64094787`](https://github.com/sgl-project/sglang/commit/ac64094787df28bf70a684716a292d62eae418f6)：test: trim forward observer and deferred output fixtures。
- `2026-10-19T08:54:26Z` [`364656fc3a`](https://github.com/sgl-project/sglang/commit/364656fc3a2eedda9b63b2fcff0072cf5ffe27d9)：fix: own deferred responses in the scheduler。
- `2026-10-19T08:56:02Z`：head_ref_force_pushed。

NPU 判断：缺失/需实测：复用公共 completion 生命周期，重新核验 NPU stream/event 与 owned buffer；避免把 graph buffer view 交给延后消费者导致下一 replay 覆盖。Mooncake TCP/GPU 测试不能证明 Ascend RDMA 或跨节点可用。

验证及限制：作者报告 H200 TP1/TP2、真实 Mooncake TCP 与训练集成。早期 -0.21%/-0.04% 性能数字发生在 completion 重构前，不能用作当前版本性能结论。验收连续 replay、延迟发布、异常、abort 和共享 future 取消。

关键代码（当前 diff）：`python/sglang/srt/managers/deferred_output.py`、`python/sglang/srt/managers/schedule_batch.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/managers/scheduler_components/output_streamer.py`、`python/sglang/srt/managers/tp_worker.py`、`python/sglang/srt/managers/utils.py`、`python/sglang/srt/model_executor/forward_observer.py`、`python/sglang/srt/model_executor/model_runner.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

### #44850 · P0 · 本期新建

[[Fix] deepseek-v4.1: Serve prompt-logprob requests under decoder SWA bounded replay](https://github.com/sgl-project/sglang/pull/44850)，目标分支 `main`。期末 `merged`；采集时 `merged`。

取代 #41448 与 #42719，于 10/17 08:25:34Z 合入。对读取 prompt logprob 或完整 hidden states 的请求，late-layer tail 覆盖 whole extend，每行保留完整 sliding window；tail 超出 prefill graph block 时 eager 回退，并经 DP vote 保持各 rank 一致，避免 forward 报错拖垮 scheduler。全局 return-hidden-states 会关闭 bounded replay。

本期证据：
- 创建于 `2026-10-17T04:28:29Z`。
- `2026-10-17T04:27:58Z` [`d71ea6b4c4`](https://github.com/sgl-project/sglang/commit/d71ea6b4c4c116279d28e117aa33814d8e416b75)：serve prompt readers under decoder swa bounded replay; dp prefill-graph tail vote; tests。
- `2026-10-17T07:26:19Z` [`ba00ca41f3`](https://github.com/sgl-project/sglang/commit/ba00ca41f3f1c43279f0f5cd1cadf81a805a417e)：prompt readers run every row with plain windows, short prompts included; doc: rl keeps replay off。
- `2026-10-17T07:54:43Z` [`8ba4d7a86b`](https://github.com/sgl-project/sglang/commit/8ba4d7a86bdfaec50df4063ec197236c6dcf18fd)：move the dp tail vote into the backend; share tail lens; return-hidden-states only resolves replay off; merge overlapping tests。
- `2026-10-17T07:58:44Z` [`a6ee724fde`](https://github.com/sgl-project/sglang/commit/a6ee724fde1edee9f5868db466a9e87d69416325)：add co-authors。
- `2026-10-17T08:06:43Z` [`cf4fc09004`](https://github.com/sgl-project/sglang/commit/cf4fc090046a247c64766d7a7e8e93a4b692498d)：Merge remote-tracking branch 'origin/main' into lsyin/dsv41-replay-prompt-readers。
- `2026-10-17T08:25:34Z`：merged。
- `2026-10-17T08:25:34Z`：closed。

NPU 判断：需实测：CUDA/HIP 已共享修复逻辑，不能据此推断 NPU 同样覆盖。Ascend 若支持对应模型和 bounded replay，须核验 whole-extend layout、图容量回退及所有 rank 的 vote；不支持时应在 admission 明确拒绝组合。

验证及限制：上游附 dsv41_late_layer_tail_readers 单测。NPU 验收 full/last、prompt logprob、DSpark FULL 捕获特例、tail 超图容量、多 DP rank 的 eager/graph 一致性；RL trainer 对齐需要按上游说明关闭 bounded replay。

关键代码（当前 diff）：`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`、`python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py`、`python/sglang/srt/arg_groups/deepseek_v4_hook.py`、`python/sglang/srt/arg_groups/fields/exec_.py`、`python/sglang/srt/layers/attention/deepseek_v4_backend.py`、`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`、`python/sglang/srt/managers/scheduler_components/dp_attn.py`、`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

### #43744 · P1 · 前期新增续跟

[[Feat] spec: Add aux hidden-state capture for Granite and GraniteMoe targets](https://github.com/sgl-project/sglang/pull/43744)，目标分支 `main`。期末 `open`；采集时 `open`。

为 Granite/GraniteMoe target 增加 aux capture 模型覆盖。重点检查所捕获的是层输入/输出、是否包含残差及最终 norm，而非只确认张量形状。

本期证据：
- `2026-10-19T19:21:10Z` [`3aa3f2d32d`](https://github.com/sgl-project/sglang/commit/3aa3f2d32d3ef778d614a21294aaf9bc4321b904)：[Spec] Add aux hidden-state capture for Granite and GraniteMoe targets。
- `2026-10-19T19:25:38Z`：head_ref_force_pushed。
- `2026-10-20T14:29:33Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43744#issuecomment-6293253173)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。

NPU 判断：需实测：可复用设备无关 hook，但模型可运行性、Ascend fused residual 和 graph 路径需要独立验证，不把模型支持推导为硬件支持。

验证及限制：对照独立 HF forward 检查逐层数值和层序，EAGLE3/DFlash 配置、空层/越界层及图重放。

关键代码（当前 diff）：`python/sglang/srt/models/granite.py`、`python/sglang/srt/models/granitemoe.py`、`test/registered/unit/models/test_granite_aux_hidden_capture.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

### #45096 · P1 · 本期新建

[[Perf] DSv4/DSv4.1: Skip the pre_hc_head CP gather when DSpark captures aux hidden states](https://github.com/sgl-project/sglang/pull/45096)，目标分支 `main`。期末 `open`；采集时 `open`。

DSpark aux capture 场景在 CP gather 前丢弃稍后本就不使用的 pre_hc_head，消除大张量无用通信；保留真正 hidden-state gather。属于直接相关的特征传递成本优化。

本期证据：
- 创建于 `2026-10-19T05:25:08Z`。
- `2026-10-19T05:25:00Z` [`515f923ad0`](https://github.com/sgl-project/sglang/commit/515f923ad08b8a1b7882a76821b4975630945f37)：[Perf] dsv4: Skip the pre_hc_head CP gather when DSpark captures aux hidden states。
- `2026-10-19T05:26:09Z`：ready_for_review。

NPU 判断：需实测：仅当 Ascend 同样产生并丢弃 pre_hc_head 时才可采用。先证明无消费者，再缩减 HCCL 通信；图输出结构和其它 capture convention 保持一致。

验证及限制：作者 MI355X profile 给出末尾 gather 5.4ms、pre_hc_head 占 80% 字节，DSpark 开启后的加速结果仍待补，不得写成已验证提速。

关键代码（当前 diff）：`python/sglang/srt/model_executor/runner/eager_runner.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

### #45142 · P1 · 本期新建

[[Fix] qwen4_exp: Capture DFlash aux hidden states on Qwen3.8-Flash-Next](https://github.com/sgl-project/sglang/pull/45142)，目标分支 `main`。期末 `open`；采集时 `open`。

补 Qwen4-Exp/Qwen3.8-Flash-Next 的 DFlash aux hidden capture。模型配置名称和模型实际层边界必须匹配，不能仅沿用 EAGLE3 层选择。

本期证据：
- 创建于 `2026-10-19T08:58:17Z`。
- `2026-10-19T08:50:08Z` [`c98b5d20d0`](https://github.com/sgl-project/sglang/commit/c98b5d20d006eaa226c4ccd488daa3964b76a9cd)：[Fix] qwen4_exp: Capture DFlash aux hidden states on Qwen3.8-Flash-Next。
- `2026-10-19T09:03:05Z`：ready_for_review。
- `2026-10-22T13:49:19Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/45142#issuecomment-6342393965)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。

NPU 判断：需实测：先验证官方 Ascend 模型实现是否经过相同 capture hook；若另有 fused 路径，应确认 residual/norm 后特征与 GPU 参考一致。

验证及限制：10/22 评论报告单张 RTX PRO 6000 Blackwell SM120/NVFP4 上 DFlash2 能加载、捕获 CUDA graphs 并服务，旧版本在首次 warmup 解包失败；该用户复测不是 Ascend 测试。仍需独立 forward 特征对齐、指定层数/维度、prefill/decode 回归。

关键代码（当前 diff）：`python/sglang/srt/models/qwen4_exp.py`、`test/registered/unit/models/test_qwen4_exp_dflash_capture.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-23.json)。

## 无实质进展与补漏边界

- [#39290](https://github.com/sgl-project/sglang/pull/39290)：无本期活动，事件类型 `{}`；采集时 `closed`。不计入本期实质条目。
- [#43502](https://github.com/sgl-project/sglang/pull/43502)：无本期活动，事件类型 `{}`；采集时 `merged`。不计入本期实质条目。
- [#43854](https://github.com/sgl-project/sglang/pull/43854)：仅元数据/引用，事件类型 `{"referenced": 1}`；采集时 `merged`。不计入本期实质条目。
- [#44568](https://github.com/sgl-project/sglang/pull/44568)：无本期活动，事件类型 `{}`；采集时 `merged`。不计入本期实质条目。

普通 hidden_states 输出与新 aux capture 接口应分开判断：[远端 NPU graph runner](https://github.com/sgl-project/sglang/blob/2da6c0c237324eefea6420e015beb1e6858cb80c/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py#L428) 不能据已有字段推断 #43449 的 last_hidden_states 已接通。

## 检索与复核

基线为 10/10 归档，第二期接续本轮第一期结果；旧遗漏单列历史补漏。关键词、完整分页、131 条归档 PR 的状态/事件检查、远端文件历史和候选排除均见共享证据。

- 完整查询索引、131条旧PR审计、39条候选筛选及本期证据均嵌入 [extract-hidden-states-delta-2026-10-23.json](extract-hidden-states-delta-2026-10-23.json)。
- [本期结构化数据](extract-hidden-states-delta-2026-10-23.json) · [本期 CSV](extract-hidden-states-delta-2026-10-23.csv)。
- 深读与优先级由主代理完成；DeepSeek sidecar 承担归档整理、远端扫描、事件/diff 采集。
- 当前 PR body、current diff、CI 摘要均为采集快照。无法完整恢复 force-push 前所有源码和当期 CI 状态，未将现在的绿灯/红灯倒推到第一期。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。

## 扫描覆盖与源码时间说明

完成76个issue查询/76页、213个唯一检索PR；检查336个归档与发现PR的当前元数据及全部timeline页。直接扫描892个main提交/10页，56个路径窗口查询得到74个唯一提交。39条机器候选由主代理继续筛选，另补入 #35697、#43854 和替代链路 #44850 等关键证据。完整关键词查询、候选排除和旧PR审计在本期JSON中。

源码快照为远端 main `2da6c0c237324eefea6420e015beb1e6858cb80c`，观察于 `2026-10-23T02:44:27Z`；PR事件截止 `2026-10-23T02:05:41Z`。两者时间不同，源码用于当前覆盖核验，不能代表当期期末源码。
