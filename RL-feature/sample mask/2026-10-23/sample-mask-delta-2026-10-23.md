# SGLang sampling mask 增量跟踪：2026-10-23

UTC 窗口：`[2026-10-17T00:00:00Z, 2026-10-23T02:05:41Z)`。采集截止：`2026-10-23T02:05:41Z`；第二期不代表 10 月 23 日全天。

代码基准：[官方 main `2da6c0c23732`](https://github.com/sgl-project/sglang/commit/2da6c0c237324eefea6420e015beb1e6858cb80c)。按用户最新要求，未使用本地 SGLang 分支判断覆盖；既有 RL-feature 归档仅作增量基线。

## 本期结论

纳入实质变化或技术讨论 **6 条**，分类为 旧PR变化 3 条、前期补漏续跟 3 条。这些不是同口径的新增功能数量。

第二期重点是 #41916 的 CUDA unseeded rescaling 融合和 #43517 的 Ascend capture/测试迭代。#43047 只有 base branch force-push，不构成这条 PR 的功能实现进展。官方 main 仍拒绝 Ascend mask 请求。

## 增量索引

- [#38565](https://github.com/sgl-project/sglang/pull/38565) — 旧PR变化；期末 `open`，采集时 `open`；[Fix] sampling: deterministic top-p / top-k renorm when attn_tp_size > 1 (TP rank divergence in DFlash/DSpark)。
- [#41916](https://github.com/sgl-project/sglang/pull/41916) — 旧PR变化；期末 `open`，采集时 `open`；[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths。
- [#43517](https://github.com/sgl-project/sglang/pull/43517) — 前期补漏续跟；期末 `open`，采集时 `open`；[Feat] [NPU] sampling: Add sampling-mask support for Ascend。
- [#40953](https://github.com/sgl-project/sglang/pull/40953) — 旧PR变化；期末 `closed`，采集时 `closed`；[PD] Transfer EAGLE draft probabilities for rejection sampling。
- [#42352](https://github.com/sgl-project/sglang/pull/42352) — 前期补漏续跟；期末 `closed`，采集时 `closed`；[RL] Return replay outputs through a Mooncake output store。
- [#42647](https://github.com/sgl-project/sglang/pull/42647) — 前期补漏续跟；期末 `open`，采集时 `open`；[Feat] RL: Add radix transfer plugins。

## 重点 PR：机制、证据与 NPU 跟进

以下机制说明以采集时 PR head 为参照；本期进展单独列出，避免把当前实现全部归入当期。

### #38565 · P0 · 旧PR变化

[[Fix] sampling: deterministic top-p / top-k renorm when attn_tp_size > 1 (TP rank divergence in DFlash/DSpark)](https://github.com/sgl-project/sglang/pull/38565)，目标分支 `main`。期末 `open`；采集时 `open`。

FlashInfer 浮点 atomicAdd 归一化的不确定性可使 TP ranks 对相同 logits 得到略不同 target_probs，随后接受/bonus 决策分叉并导致 KV/radix 状态漂移甚至 collective hang。当前方案在 attn_tp_size>1 或确定性推理时使用 deterministic renorm，保留显式环境覆盖；这已不同于旧归档的纯 opt-in 描述。第一期存在 force-push，第二期继续收敛 TP 策略。

本期证据：
- `2026-10-17T02:28:56Z` [`53c104d152`](https://github.com/sgl-project/sglang/commit/53c104d1522dfbd285104a6ca83c89b2f733eded)：[Sampling] Deterministic renorm on every attn_tp_size > 1 path。

NPU 判断：需实测：不要移植 FlashInfer CUDA kernel 到 NPU；应在 Ascend 实际 renorm 算子上检查跨 rank bitwise 一致性、共享随机数和 token 决策。与 #41916 的过滤语义一起验收。

验证及限制：作者报告 H100/B200 无 broadcast 的 fast 版本六次均挂起、deterministic 五次均完成30分钟；不是本次复测，也不证明 Ascend 会发生同样问题。需限制 top_p<1，跟踪 target_logits→target_probs→accept_len/bonus→KV 提交的首个分歧。

关键代码（当前 diff）：`python/sglang/srt/environ.py`、`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/layers/sampling_renorm.py`、`python/sglang/srt/speculative/dflash_utils.py`、`python/sglang/srt/speculative/eagle_utils.py`、`python/sglang/srt/speculative/spec_tp_sync.py`、`test/registered/kernels/ops/sampling/test_renorm.py`、`test/registered/unit/sampling/test_renorm_policy.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-23.json)。

### #41916 · P0 · 旧PR变化

[[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths](https://github.com/sgl-project/sglang/pull/41916)，目标分支 `main`。期末 `open`；采集时 `open`。

修复方向是保留原分布的 top-p nucleus，再与 top-k support 相交（joint），不是采用 #41955 的 top_k_first 默认切换。第一期重构 FlashInfer min-p、EAGLE、dense/sparse target 和 UNO 五条路径；第二期 b94a95d5 为 CUDA unseeded 路径融合 top-p rescaling，共享 retained-maximum reduction，保留 seeded/non-CUDA 原计算，并补 is_cuda 绑定修复。FP32 cutoff 边界可能有差异，RL support 与归一化概率必须同时校验。

本期证据：
- `2026-10-20T01:09:06Z` [`7c5a3174b2`](https://github.com/sgl-project/sglang/commit/7c5a3174b28f87cefccc4f21ddaf61b4f5de5676)：Merge branch 'main' into fix/top-k-top-p-filtering。
- `2026-10-20T02:40:41Z` [`3d63a0ade8`](https://github.com/sgl-project/sglang/commit/3d63a0ade8e167c118971c154ef314a0254f0b1b)：Clarify joint sampling filter comments。
- `2026-10-20T17:58:37Z` [`c71e432288`](https://github.com/sgl-project/sglang/commit/c71e432288582122c682b9fd0a8fd0c2fb4bf202)：Fix contig post-quant import in masked quantization test。
- `2026-10-20T21:23:52Z` [`f0c0a64ad1`](https://github.com/sgl-project/sglang/commit/f0c0a64ad1b70d9a72f9d741b9eaa2b51d088c03)：Merge branch 'main' into fix/top-k-top-p-filtering。
- `2026-10-21T23:17:55Z` [`b94a95d5e8`](https://github.com/sgl-project/sglang/commit/b94a95d5e805e0ee214c6d355a3a6cc79dd5e073)：[Perf] sampling: Fuse top-p rescaling for unseeded top-k paths。
- `2026-10-21T23:48:14Z` [`c55b455da8`](https://github.com/sgl-project/sglang/commit/c55b455da88dc9ff85533d7850a2a3878c6823e0)：[Fix] sampling: Bind CUDA predicate in speculative verification。
- `2026-10-22T23:49:32Z` [`1f6d554f4c`](https://github.com/sgl-project/sglang/commit/1f6d554f4c3a749b08fe51afce9809f990c808c5)：Merge branch 'main' into fix/top-k-top-p-filtering。
- `2026-10-20T01:14:02Z`：reviewed。
- `2026-10-21T19:41:05Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6324918711)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-21T19:41:10Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6324920382)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-21T23:21:39Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6328764169)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-21T23:48:29Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6329129671)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。

NPU 判断：需实测：Ascend fused helper 与 torch fallback 的过滤顺序必须逐分布比对。先定义与 GPU 一致的语义再迁移 kernel；不能据 PR open 状态宣称默认策略已改变。

验证及限制：10/14 作者报告 H200 上五路径 18 passed + 6 subtests，修复前 support/probability 断言失败。10/21 作者报告 11 项原测试、12 项 seeded bitwise 比较和 seeded CUDA Graph 验证通过；H200 helper DFlash 1198→899 us、UNO 368→271 us，未宣称端到端精度或 trainer 对齐。本次未复现；NPU 需覆盖 joint/top_k_first 反例、min_p、cutoff ties、混合 batch 和 emitted token 对齐。

关键代码（当前 diff）：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/speculative/dflash_utils.py`、`python/sglang/srt/speculative/eagle_utils.py`、`python/sglang/srt/speculative/uno_utils.py`、`test/registered/kernels/ops/speculative/test_speculative_sampling.py`、`test/registered/sampling/test_sampling_mask.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-23.json)。

### #43517 · P0 · 前期补漏续跟

[[Feat] [NPU] sampling: Add sampling-mask support for Ascend](https://github.com/sgl-project/sglang/pull/43517)，目标分支 `main`。期末 `open`；采集时 `open`。

直接导出 Ascend 实际采样的 post-filter 权重，接入公共 _SamplingMaskCapture、打包与 host 物化；fused 与 torch fallback 均覆盖，避免重放过滤器造成 support 偏差。当前仍 open，不能当作官方 main 已支持。

本期证据：
- `2026-10-19T05:19:46Z` [`b08f0e2fc2`](https://github.com/sgl-project/sglang/commit/b08f0e2fc2807f2f35c5ab33fe1d8bc07668551b)：[NPU] Handle min-p and cutoff ties in sampling masks。
- `2026-10-19T08:12:29Z` [`13e72b46aa`](https://github.com/sgl-project/sglang/commit/13e72b46aab1a2502e71179ba25e272a8c981f4f)：Refine Ascend sampling mask capture and tests。
- `2026-10-19T09:32:38Z` [`e5aa76d458`](https://github.com/sgl-project/sglang/commit/e5aa76d4580b5aa2cfebc1d8378b6a4a257056a0)：Expand Ascend sampling mask regression coverage。
- `2026-10-19T10:41:36Z` [`f2f46f6d87`](https://github.com/sgl-project/sglang/commit/f2f46f6d87cd3084ab84df2adcc40986df0dc448)：Merge upstream main to satisfy CI minimum-base requirement。
- `2026-10-22T11:23:22Z` [`2acb16f1d2`](https://github.com/sgl-project/sglang/commit/2acb16f1d2b6647dca1f11ded4bc410f6cc91cd3)：Refine Ascend sampling mask buffer reuse and reference tests。
- `2026-10-22T12:10:51Z` [`1cd3123faa`](https://github.com/sgl-project/sglang/commit/1cd3123faad690c1ce2bc3ac2c8ca9ef4580af88)：Move Ascend sampling mask reference outside registered tests。
- `2026-10-20T12:18:54Z`：reviewed。

NPU 判断：缺失：官方 main scheduler 仍拒绝 Ascend return_sampling_mask，helper 只返回 token IDs。优先推动该 PR 的接口和正确性审阅；有限 top_k、cap/overflow、ties 需沿用公共契约。

验证及限制：PR 描述含 NPU/CPU mock 验证，均为作者报告。验收 top_k=1025 fallback、min_p、greedy、support/selected、混合 opted-in rows、seeded mask-on/off 及连续图重放。

关键代码（当前 diff）：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/test/ascend/sampling_mask_reference.py`、`test/registered/npu/rl/test_npu_sampling_mask.py`、`test/registered/npu/rl/test_npu_sampling_mask_bf16_models.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-23.json)。

### #40953 · P1 · 旧PR变化

[[PD] Transfer EAGLE draft probabilities for rejection sampling](https://github.com/sgl-project/sglang/pull/40953)，目标分支 `main`。期末 `closed`；采集时 `closed`。

PD 分离下传输 EAGLE draft 完整概率分布以支持 rejection sampling；与 support mask 不同，只有候选集合不足以恢复提议分布。第二期因 draft 28天无更新关闭未合入，属于跟踪风险清理。

本期证据：
- `2026-10-22T01:23:47Z`：closed。
- `2026-10-22T01:23:46Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/40953#issuecomment-6330395605)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：缺失/需实测：如果 Ascend 跟进 PD speculative，先定义概率 tensor/精度/布局与 token 对齐，不能只传 sampling mask 冒充 proposal distribution。

验证及限制：未见可据此认定 main 已支持的合入证据；关闭评论明确要求相关时再 reopen。

关键代码（当前 diff）：`docs/docs/advanced_features/pd_disaggregation.mdx`、`python/sglang/srt/disaggregation/common/conn.py`、`python/sglang/srt/disaggregation/decode.py`、`python/sglang/srt/disaggregation/prefill.py`、`python/sglang/srt/disaggregation/utils.py`、`python/sglang/srt/managers/schedule_batch.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/speculative/eagle_disaggregation.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-23.json)。

### #42352 · P1 · 前期补漏续跟

[[RL] Return replay outputs through a Mooncake output store](https://github.com/sgl-project/sglang/pull/42352)，目标分支 `jiajun/chat-session-return-indexer-topk`。期末 `closed`；采集时 `closed`。

拟将 sampling masks、routed experts、indexer top-k 从 inline meta_info 改为 Mooncake 引用返回；stack 依赖 #42349。第二期关闭未合入，不能将接口作为已可用能力；与后续 #42647 的共享包方案比较，不能无事件证据宣称自动继承。

本期证据：
- `2026-10-21T20:25:12Z`：closed。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：缺失/需实测：先有 Ascend mask 捕获，再评估 host/IPC 和 store transport，不能以传输方案替代采样正确性。

验证及限制：作者主要 fake Mooncake/CPU 测试，真实 Mooncake 未重跑；跟进 dtype/shape、逐 token 长度、写入失败、取消、disconnect、删除所有权。

关键代码（当前 diff）：`python/sglang/srt/arg_groups/fields/exec_.py`、`python/sglang/srt/entrypoints/openai/protocol.py`、`python/sglang/srt/entrypoints/openai/serving_chat.py`、`python/sglang/srt/managers/detokenizer_manager.py`、`python/sglang/srt/managers/io_struct.py`、`python/sglang/srt/managers/multi_tokenizer_mixin.py`、`python/sglang/srt/managers/output_store.py`、`python/sglang/srt/managers/tokenizer_manager.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-23.json)。

### #42647 · P1 · 前期补漏续跟

[[Feat] RL: Add radix transfer plugins](https://github.com/sgl-project/sglang/pull/42647)，目标分支 `sglang-miles`。期末 `open`；采集时 `open`。

共享 radix_transfer_plugins 包保存 rollout 输出，SGLang 负责 TokenOutputCollector 与 OutputStoreWriter。只保存 support-sized mask rows，selected logprobs 仍在 meta_info；要求单 prompt/n=1/无 beam，writer 关闭和取消清理是重点。第二期重写提交，当前 open。

本期证据：
- `2026-10-21T21:35:15Z` [`0c1978910c`](https://github.com/sgl-project/sglang/commit/0c1978910c3f9edf45ad171c818c6ff0d9858b7b)：[RL] Return replay outputs through the radix_transfer_plugins output store。
- `2026-10-21T21:35:15Z` [`f378094c9c`](https://github.com/sgl-project/sglang/commit/f378094c9c9938aeef831d88772347720176e2d8)：[RL] Keep sampled-token logprobs in meta_info with the output store。
- `2026-10-21T21:35:15Z` [`34470a3e11`](https://github.com/sgl-project/sglang/commit/34470a3e11811ac180fc3666380c4af52f3d474b)：[RL] Add OutputStore.read_many: chunked reads into caller-owned arrays。
- `2026-10-21T21:35:15Z` [`f396edb143`](https://github.com/sgl-project/sglang/commit/f396edb143e23affcaac8797fc6637d51820a236)：[RL] Output store: optional package import, no mask row past finished_len, retry a full buffer alone。
- `2026-10-21T22:48:30Z` [`f21c0cd45d`](https://github.com/sgl-project/sglang/commit/f21c0cd45d3e62d9883e7057b5623d871a19ac84)：[RL] Output store: one output sequence per request, close the writer on Engine shutdown。
- `2026-10-22T05:47:42Z` [`c577937f17`](https://github.com/sgl-project/sglang/commit/c577937f1730e5c51344f1a15569480ab8027860)：[RL] Output store: copy read fields with numpy, off the GIL。
- `2026-10-17T20:25:41Z`：head_ref_force_pushed。
- `2026-10-21T21:23:16Z`：head_ref_force_pushed。
- `2026-10-21T21:35:18Z`：head_ref_force_pushed。
- `2026-10-21T22:11:38Z`：reviewed。

NPU 判断：缺失/需实测：框架/host 逻辑可复用；Ascend 输出需先接通 #43517，再核对 store ref 对齐、buffer 复制所有权和真实 NPU→host→Mooncake 路径。

验证及限制：作者 macOS fake Mooncake 32项、相关CPU343 passed/2 CUDA skipped；未运行 GPU 或真实 Mooncake。不能外推生产吞吐或 Ascend 支持。

关键代码（当前 diff）：`.github/workflows/_pr-test-stage-cpu.yml`、`docs/docs/advanced_features/server_arguments.mdx`、`python/sglang/srt/arg_groups/fields/exec_.py`、`python/sglang/srt/entrypoints/engine.py`、`python/sglang/srt/entrypoints/openai/protocol.py`、`python/sglang/srt/entrypoints/openai/serving_chat.py`、`python/sglang/srt/managers/detokenizer_manager.py`、`python/sglang/srt/managers/io_struct.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-23.json)。

## 无实质进展与补漏边界

- [#35697](https://github.com/sgl-project/sglang/pull/35697)：无本期活动，事件类型 `{}`；采集时 `open`。不计入本期实质条目。
- [#43047](https://github.com/sgl-project/sglang/pull/43047)：仅元数据/引用，事件类型 `{"base_ref_force_pushed": 1}`；采集时 `open`。不计入本期实质条目。
- [#43346](https://github.com/sgl-project/sglang/pull/43346)：仅元数据/引用，事件类型 `{"referenced": 1}`；采集时 `merged`。不计入本期实质条目。
- #41955 / #41957 / #42802 / #39934 ：采集时 updated_at 均早于 10/11，作为旧背景，不重复计入。#41957 目标分支是 sampling-top-k-first，不能当 main 独立落地。

官方 Ascend 缺口的直接依据：[scheduler 拒绝分支](https://github.com/sgl-project/sglang/blob/2da6c0c237324eefea6420e015beb1e6858cb80c/python/sglang/srt/managers/scheduler.py#L3135)、[Ascend helper 返回契约](https://github.com/sgl-project/sglang/blob/2da6c0c237324eefea6420e015beb1e6858cb80c/python/sglang/srt/layers/sampler.py#L861)。

## 检索与复核

基线为 10/10 归档，第二期接续本轮第一期结果；旧遗漏单列历史补漏。关键词、完整分页、131 条归档 PR 的状态/事件检查、远端文件历史和候选排除均见共享证据。

- 完整查询索引、131条旧PR审计、39条候选筛选及本期证据均嵌入 [sample-mask-delta-2026-10-23.json](sample-mask-delta-2026-10-23.json)。
- [本期结构化数据](sample-mask-delta-2026-10-23.json) · [本期 CSV](sample-mask-delta-2026-10-23.csv)。
- 深读与优先级由主代理完成；DeepSeek sidecar 承担归档整理、远端扫描、事件/diff 采集。
- 当前 PR body、current diff、CI 摘要均为采集快照。无法完整恢复 force-push 前所有源码和当期 CI 状态，未将现在的绿灯/红灯倒推到第一期。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。

## 扫描覆盖与源码时间说明

完成76个issue查询/76页、213个唯一检索PR；检查336个归档与发现PR的当前元数据及全部timeline页。直接扫描892个main提交/10页，56个路径窗口查询得到74个唯一提交。39条机器候选由主代理继续筛选，另补入 #35697、#43854 和替代链路 #44850 等关键证据。完整关键词查询、候选排除和旧PR审计在本期JSON中。

源码快照为远端 main `2da6c0c237324eefea6420e015beb1e6858cb80c`，观察于 `2026-10-23T02:44:27Z`；PR事件截止 `2026-10-23T02:05:41Z`。两者时间不同，源码用于当前覆盖核验，不能代表当期期末源码。

本期没有新建且纳入核心分析的 sampling-mask PR；新增候选与旧遗漏明确分开统计。#41205 的本期事件仅作引用/背景审计，不表示代码更新。
