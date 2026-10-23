# SGLang extract / return / aux hidden states 增量跟踪：2026-10-16

UTC 窗口：`[2026-10-11T00:00:00Z, 2026-10-17T00:00:00Z)`。采集截止：`2026-10-23T02:05:41Z`；第二期不代表 10 月 23 日全天。

代码基准：[官方 main `2da6c0c23732`](https://github.com/sgl-project/sglang/commit/2da6c0c237324eefea6420e015beb1e6858cb80c)。按用户最新要求，未使用本地 SGLang 分支判断覆盖；既有 RL-feature 归档仅作增量基线。

## 本期结论

纳入实质变化或技术讨论 **7 条**，分类为 旧PR变化 3 条、本期新建 3 条、历史补漏 1 条。这些不是同口径的新增功能数量。

第一期重点是 #43449 参数/边界收敛，#44568 回读生命周期修复与 #43854 capture no-grad 修复。#39290 因长期无更新关闭未合入，不能认定 PP aux relay 已落地。

## 增量索引

- [#41448](https://github.com/sgl-project/sglang/pull/41448) — 旧PR变化；期末 `open`，采集时 `closed`；[AMD][DeepSeek-V4.1] Fix server crash on prompt-logprob and hidden-state requests under decoder SWA bounded replay。
- [#43449](https://github.com/sgl-project/sglang/pull/43449) — 旧PR变化；期末 `open`，采集时 `open`；[Feat] spec: Capture aux hidden states on the target without a speculative draft。
- [#44568](https://github.com/sgl-project/sglang/pull/44568) — 本期新建；期末 `merged`，采集时 `merged`；[Perf] [NVIDIA] scheduler: Pageable host staging and inline hidden-state/DSpark readback under Confidential Computing。
- [#39290](https://github.com/sgl-project/sglang/pull/39290) — 旧PR变化；期末 `closed`，采集时 `closed`；[DeepSeek-V4] Forward auxiliary hidden states across PP stages。
- [#43502](https://github.com/sgl-project/sglang/pull/43502) — 历史补漏；期末 `merged`，采集时 `merged`；[Fix] DSv4.1: Size the encoder-replay SWA pool by the DSpark draft's stages, and keep more prefix tails。
- [#43744](https://github.com/sgl-project/sglang/pull/43744) — 本期新建；期末 `open`，采集时 `open`；[Feat] spec: Add aux hidden-state capture for Granite and GraniteMoe targets。
- [#43854](https://github.com/sgl-project/sglang/pull/43854) — 本期新建；期末 `merged`，采集时 `merged`；Fix autograd retention in CUDA graph capture tail hooks。

## 重点 PR：机制、证据与 NPU 跟进

以下机制说明以采集时 PR head 为参照；本期进展单独列出，避免把当前实现全部归入当期。

### #41448 · P0 · 旧PR变化

[[AMD][DeepSeek-V4.1] Fix server crash on prompt-logprob and hidden-state requests under decoder SWA bounded replay](https://github.com/sgl-project/sglang/pull/41448)，目标分支 `main`。期末 `open`；采集时 `closed`。

第一期讨论是 bounded replay 遇 prompt-logprob/full hidden-state 请求时应拒绝还是 whole-extend fallback。第二期于 10/17 08:33:53Z 关闭未合入，作者明确注明由已合入的 #44850 替代；修复依据应引用 #44850 的代码，不能把关闭状态本身当修复。

本期证据：
- `2026-10-13T18:51:12Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41448#issuecomment-6152792794)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：需实测：相同组合若在 Ascend runner 不支持，应 admission 阶段明确拒绝或切换 eager，避免共享 GPU 局部修复未经核实照搬。

验证及限制：覆盖 full/last/关闭、prompt logprob、SWA bounded replay 开关；检查单个请求失败不拖垮服务。关闭原因与替代 PR 只按事件证据引用。

关键代码（当前 diff）：`python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py`、`python/sglang/srt/layers/attention/deepseek_v4_backend.py`、`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`、`python/sglang/srt/models/deepseek_v4.py`、`test/registered/unit/layers/test_dsv4_late_layer_tail.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

### #43449 · P0 · 旧PR变化

[[Feat] spec: Capture aux hidden states on the target without a speculative draft](https://github.com/sgl-project/sglang/pull/43449)，目标分支 `main`。期末 `open`；采集时 `open`。

将 aux hidden-state 捕获从加载 draft 解耦。当前方案以 enable-return-aux-hidden-states 和层选择参数配置 EAGLE3/DFLASH/DSPARK capture convention；捕获 aux 拼接之外另保留最后层 last_hidden_states 给进程内消费者。当前设计限制 PP>1，不能与 PP relay PR 合起来声称已支持。

本期证据：
- `2026-10-13T23:40:02Z` [`50aabbc87f`](https://github.com/sgl-project/sglang/commit/50aabbc87fe72b8723ba143b8bb201a34d0c9abf)：Address review: reject pp_size > 1, scope last_hidden_states to in-process consumers。
- `2026-10-15T06:49:08Z` [`6f530d94cb`](https://github.com/sgl-project/sglang/commit/6f530d94cb1618c2e5b3cf3c171b1b091a29f9d6)：Address review: validate layer ids, rename to --return-aux-hidden-state-layer-ids。
- `2026-10-15T07:03:45Z` [`c0b5388753`](https://github.com/sgl-project/sglang/commit/c0b53887537f68700eabe59fc31d07b989194590)：Address review follow-ups: fail fast on uncapturable layers, default full capture。
- `2026-10-15T07:14:06Z` [`008c6d2106`](https://github.com/sgl-project/sglang/commit/008c6d21065c5600d82104aa88fa3b830a4d4df0)：Merge remote-tracking branch 'origin/main' into feat/aux-hidden-state-capture-without-draft。
- `2026-10-15T07:18:34Z` [`affe3e7ee0`](https://github.com/sgl-project/sglang/commit/affe3e7ee040a12c51547e5009ed57ca15d9024e)：Merge remote-tracking branch 'origin/main' into feat/aux-hidden-state-capture-without-draft。
- `2026-10-16T04:26:11Z` [`9f2856a9b1`](https://github.com/sgl-project/sglang/commit/9f2856a9b1a27561ccc1d06fe42f4ec73164c725)：Switch the CLI to --enable-return-aux-hidden-states with --speculative-algorithm naming the convention。
- `2026-10-12T20:58:24Z`：reviewed。
- `2026-10-14T23:11:36Z`：reviewed。

NPU 判断：缺失：官方 main 有普通 hidden_states 图输出裁剪，但未集成此 capture-only 新接口和 last_hidden_states 字段传播。应补 resolver/参数校验、模型 capture hook、NPU graph runner 输出裁剪与 buffer 生命周期；不能只移植 CLI。

验证及限制：作者报告 GB300/HF bf16 对齐及训练 finite loss，非 NPU 证明。NPU 验收 eager/graph 的 aux 层序和维度、last hidden 经 lm_head 得到 logits、padding 裁剪、无 draft KV 预留、禁用路径开销。

关键代码（当前 diff）：`docs/docs/advanced_features/server_arguments.mdx`、`python/sglang/srt/arg_groups/field_order.py`、`python/sglang/srt/arg_groups/fields/exec_.py`、`python/sglang/srt/arg_groups/pipeline.py`、`python/sglang/srt/arg_groups/resolution_hooks.py`、`python/sglang/srt/arg_groups/serving_hook.py`、`python/sglang/srt/arg_groups/validation_hook.py`、`python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

### #44568 · P0 · 本期新建

[[Perf] [NVIDIA] scheduler: Pageable host staging and inline hidden-state/DSpark readback under Confidential Computing](https://github.com/sgl-project/sglang/pull/44568)，目标分支 `main`。期末 `merged`；采集时 `merged`。

NVIDIA CC 下 worker 延后发起的 D2H 可能读到被下一 replay 覆盖的 hidden-state/DSpark buffer view；这两类批次保持 forward 后立即在 copy_stream 排队。pageable staging 仅适用于 NVIDIA CC 条件。

本期证据：
- 创建于 `2026-10-15T19:44:32Z`。
- `2026-10-15T19:34:22Z` [`1d44b8bd36`](https://github.com/sgl-project/sglang/commit/1d44b8bd366257c8b24022d158c244d3f05b12d1)：perf: pageable host staging and inline hidden-state/DSpark readback under Confidential Computing。
- `2026-10-16T00:48:50Z`：merged。
- `2026-10-16T00:48:51Z`：closed。
- `2026-10-16T00:23:46Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/44568#issuecomment-6205140618)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。

NPU 判断：需实测：可迁移 buffer ownership 与拷贝入队顺序约束；不可把 NVIDIA CC pageable staging 默认策略照搬 Ascend。公共 helper 存在不代表 NPU stream 同步已验证。

验证及限制：作者普通 GPU 上 8+20 项测试通过，无真实 CC 主机。NPU 用延迟 forward、连续 replay、不同 shape 请求及异步 consumer 验证每批数据属于正确请求。

关键代码（当前 diff）：`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/utils/common.py`、`test/registered/unit/disaggregation/test_prefill_result_polling.py`、`test/registered/unit/managers/test_async_d2h_copy_worker.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

### #39290 · P1 · 旧PR变化

[[DeepSeek-V4] Forward auxiliary hidden states across PP stages](https://github.com/sgl-project/sglang/pull/39290)，目标分支 `main`。期末 `closed`；采集时 `closed`。

跨 PP stage 携带按全局 layer-id 命名的 aux tensor，最后 stage 按配置顺序收集，处理无本地 capture 的 stage 与 mHC 延后计算。它是捕获 plumbing，不等同完整 DSpark speculative PP serving。当前 closed 未合入。 第一期开于 10/13 的关闭说明指出仍是 draft、29 天无更新；属于主动清理，未提供已完成替代实现的证据。

本期证据：
- `2026-10-13T01:38:29Z`：closed。
- `2026-10-13T01:38:28Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/39290#issuecomment-6136193758)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：缺失/需实测：如跟进 PP，需逐 stage 保持 tensor ownership、动态 token/padding 与固定图 buffer；暂不能绕过 #43449 的 PP 限制。

验证及限制：作者只有 CPU stub 与图 buffer 单测，缺真实权重 multi-GPU graph replay；NPU 必须验证三 stage、非单调层顺序、无本地捕获 stage 和前一 stage 缺失张量。

关键代码（当前 diff）：`python/sglang/srt/model_executor/runner/base_runner.py`、`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`、`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`、`python/sglang/srt/model_executor/runner_utils/buffers.py`、`python/sglang/srt/models/deepseek_v4.py`、`test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py`、`test/registered/unit/models/test_deepseek_v4_pipeline_aux_capture.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

### #43502 · P1 · 历史补漏

[[Fix] DSv4.1: Size the encoder-replay SWA pool by the DSpark draft's stages, and keep more prefix tails](https://github.com/sgl-project/sglang/pull/43502)，目标分支 `main`。期末 `merged`；采集时 `merged`。

按 DSpark draft 实际 stages 计算 encoder-replay SWA pool，避免沿用 target 的 num_hidden_layers 导致容量预算错误，并保留更多 prefix tails。属于隐藏特征消费端相邻的容量正确性修复，而非新 API。

本期证据：
- `2026-10-13T03:37:17Z` [`d402e6fa51`](https://github.com/sgl-project/sglang/commit/d402e6fa5169cc36426bb0bcfe93359b24e1c062)：[DSv4.1] Record DSpark stages only for the DeepSeek-V4 draft。
- `2026-10-13T03:37:17Z` [`fc187dc30a`](https://github.com/sgl-project/sglang/commit/fc187dc30a15f371f53824f42f20511f65f0dbb5)：[DSv4.1] Drop the duplicate DSpark replay pricing test。
- `2026-10-14T05:47:08Z` [`b81f06dc66`](https://github.com/sgl-project/sglang/commit/b81f06dc6637ce51d584d3e55b656f150a798fd2)：Merge branch 'main' into dsv41-dspark-swa-pricing。
- `2026-10-14T09:52:31Z` [`c763dae981`](https://github.com/sgl-project/sglang/commit/c763dae9815b6b892392b60e94fb86d9ed209212)：Merge branch 'main' into dsv41-dspark-swa-pricing。
- `2026-10-15T04:39:24Z` [`0de0506514`](https://github.com/sgl-project/sglang/commit/0de0506514547f6669532e94cffa82704e41c26f)：Merge branch 'main' into dsv41-dspark-swa-pricing。
- `2026-10-12T17:23:15Z`：ready_for_review。
- `2026-10-13T03:08:28Z`：reviewed。
- `2026-10-15T05:04:59Z`：merged。
- `2026-10-15T05:04:59Z`：closed。
- `2026-10-13T04:00:06Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43502#issuecomment-6137927134)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-14T05:48:06Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43502#issuecomment-6161837623)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-14T09:29:56Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43502#issuecomment-6165258108)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-15T04:37:48Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43502#issuecomment-6184530662)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。

NPU 判断：部分覆盖/需实测：官方公共 spec_aux_hidden_state resolver 有 dspark_num_stages，NPU 需核验是否使用该容量路径及图预算；不能以公共字段存在证明对应模型可运行。

验证及限制：检查 target层数与draft stages不等、prefix tails、不同decode/prefill图形状；本次仅静态核验。

关键代码（当前 diff）：`python/sglang/srt/arg_groups/fields/schedule.py`、`python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py`、`python/sglang/srt/model_executor/pool_configurator.py`、`test/registered/unit/model_executor/test_pool_configurator.py`、`test/registered/unit/model_executor/test_spec_aux_hidden_state_resolver.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

### #43744 · P1 · 本期新建

[[Feat] spec: Add aux hidden-state capture for Granite and GraniteMoe targets](https://github.com/sgl-project/sglang/pull/43744)，目标分支 `main`。期末 `open`；采集时 `open`。

为 Granite/GraniteMoe target 增加 aux capture 模型覆盖。重点检查所捕获的是层输入/输出、是否包含残差及最终 norm，而非只确认张量形状。

本期证据：
- 创建于 `2026-10-12T02:18:27Z`。
- `2026-10-13T02:27:51Z`：reviewed。
- `2026-10-14T15:51:45Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43744#issuecomment-6172409050)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：需实测：可复用设备无关 hook，但模型可运行性、Ascend fused residual 和 graph 路径需要独立验证，不把模型支持推导为硬件支持。

验证及限制：对照独立 HF forward 检查逐层数值和层序，EAGLE3/DFlash 配置、空层/越界层及图重放。

关键代码（当前 diff）：`python/sglang/srt/models/granite.py`、`python/sglang/srt/models/granitemoe.py`、`test/registered/unit/models/test_granite_aux_hidden_capture.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

### #43854 · P1 · 本期新建

[Fix autograd retention in CUDA graph capture tail hooks](https://github.com/sgl-project/sglang/pull/43854)，目标分支 `main`。期末 `merged`；采集时 `merged`。

startup graph capture tail hook 落在 model forward 的 no_grad 范围外，persistent 输出 buffer 保留 CopySlices autograd 历史。修复在 Scheduler.init_all_cuda_graphs 统一加 no_grad；它是采集链路相邻的图生命周期修复，不是新输出接口。

本期证据：
- 创建于 `2026-10-12T12:13:04Z`。
- `2026-10-12T11:56:36Z` [`2d6a5ec9f9`](https://github.com/sgl-project/sglang/commit/2d6a5ec9f9267ec8b82e77470931baf7f1761fed)：Fix autograd retention in CUDA graph capture tail hooks。
- `2026-10-12T19:05:40Z` [`63dccce565`](https://github.com/sgl-project/sglang/commit/63dccce565f40047acd5fa11a51e75abc05d0800)：move no-grad guard to scheduler startup capture; replace test。
- `2026-10-12T19:12:02Z` [`e17d0a04a4`](https://github.com/sgl-project/sglang/commit/e17d0a04a4d817135e306e2680b839d294d0212a)：drop scheduler grad-mode test。
- `2026-10-12T19:13:30Z` [`8f59ef1e66`](https://github.com/sgl-project/sglang/commit/8f59ef1e66c28071439da1392ffd3ea0906588ec)：trim comment。
- `2026-10-12T19:31:19Z`：merged。
- `2026-10-12T19:31:19Z`：closed。

NPU 判断：部分覆盖/需实测：公共初始化入口已有 guard；需确认 NPU 初始化实际经过该入口，并检查 grad_fn 与 warmup 后内存。不要随意换成 inference_mode，否则 disagg 原地写 inference tensor 可能失败。

验证及限制：作者 H20/H200 报告进程显存下降 806 MiB，非本次测量，也非 NPU 收益保证。检查启动/recapture、tail hook 参数 requires_grad 及 buffer 历史。

关键代码（当前 diff）：`python/sglang/srt/managers/scheduler.py`。

[事件、提交和当前文件 diff](extract-hidden-states-delta-2026-10-16.json)。

## 无实质进展与补漏边界

- [#43450](https://github.com/sgl-project/sglang/pull/43450)：仅元数据/引用，事件类型 `{"assigned": 1}`；采集时 `open`。不计入本期实质条目。

普通 hidden_states 输出与新 aux capture 接口应分开判断：[远端 NPU graph runner](https://github.com/sgl-project/sglang/blob/2da6c0c237324eefea6420e015beb1e6858cb80c/python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py#L428) 不能据已有字段推断 #43449 的 last_hidden_states 已接通。

## 检索与复核

基线为 10/10 归档，第二期接续本轮第一期结果；旧遗漏单列历史补漏。关键词、完整分页、131 条归档 PR 的状态/事件检查、远端文件历史和候选排除均见共享证据。

- 完整查询索引、131条旧PR审计、39条候选筛选及本期证据均嵌入 [extract-hidden-states-delta-2026-10-16.json](extract-hidden-states-delta-2026-10-16.json)。
- [本期结构化数据](extract-hidden-states-delta-2026-10-16.json) · [本期 CSV](extract-hidden-states-delta-2026-10-16.csv)。
- 深读与优先级由主代理完成；DeepSeek sidecar 承担归档整理、远端扫描、事件/diff 采集。
- 当前 PR body、current diff、CI 摘要均为采集快照。无法完整恢复 force-push 前所有源码和当期 CI 状态，未将现在的绿灯/红灯倒推到第一期。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。

## 扫描覆盖与源码时间说明

完成76个issue查询/76页、213个唯一检索PR；检查336个归档与发现PR的当前元数据及全部timeline页。直接扫描892个main提交/10页，56个路径窗口查询得到74个唯一提交。39条机器候选由主代理继续筛选，另补入 #35697、#43854 和替代链路 #44850 等关键证据。完整关键词查询、候选排除和旧PR审计在本期JSON中。

源码快照为远端 main `2da6c0c237324eefea6420e015beb1e6858cb80c`，观察于 `2026-10-23T02:44:27Z`；PR事件截止 `2026-10-23T02:05:41Z`。两者时间不同，源码用于当前覆盖核验，不能代表当期期末源码。
