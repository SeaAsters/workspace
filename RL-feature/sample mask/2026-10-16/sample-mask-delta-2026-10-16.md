# SGLang sampling mask 增量跟踪：2026-10-16

UTC 窗口：`[2026-10-11T00:00:00Z, 2026-10-17T00:00:00Z)`。采集截止：`2026-10-23T02:05:41Z`；第二期不代表 10 月 23 日全天。

代码基准：[官方 main `2da6c0c23732`](https://github.com/sgl-project/sglang/commit/2da6c0c237324eefea6420e015beb1e6858cb80c)。按用户最新要求，未使用本地 SGLang 分支判断覆盖；既有 RL-feature 归档仅作增量基线。

## 本期结论

纳入实质变化或技术讨论 **4 条**，分类为 历史补漏 1 条、旧PR变化 3 条。这些不是同口径的新增功能数量。

第一期重点是 #43346 合入与 #41916 的过滤语义修正及 GPU 验证；#35697 是旧遗漏 PR 的混合 batch 确定性问题复现。#43517 于 10/12 转为可评审，但创建于 10/10，不能算本期新 PR。

## 增量索引

- [#35697](https://github.com/sgl-project/sglang/pull/35697) — 历史补漏；期末 `open`，采集时 `open`；[Bug] Seeded sampling changes token when a batchmate enables top-k/top-p。
- [#38565](https://github.com/sgl-project/sglang/pull/38565) — 旧PR变化；期末 `open`，采集时 `open`；[Fix] sampling: deterministic top-p / top-k renorm when attn_tp_size > 1 (TP rank divergence in DFlash/DSpark)。
- [#41916](https://github.com/sgl-project/sglang/pull/41916) — 旧PR变化；期末 `open`，采集时 `open`；[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths。
- [#43346](https://github.com/sgl-project/sglang/pull/43346) — 旧PR变化；期末 `merged`，采集时 `merged`；[Sampler] Sort only the max(top_k) leading columns in the torch sampler。

## 重点 PR：机制、证据与 NPU 跟进

以下机制说明以采集时 PR head 为参照；本期进展单独列出，避免把当前实现全部归入当期。

### #35697 · P0 · 历史补漏

[[Bug] Seeded sampling changes token when a batchmate enables top-k/top-p](https://github.com/sgl-project/sglang/pull/35697)，目标分支 `main`。期末 `open`；采集时 `open`。

seeded Gumbel 噪声按词表 token id 还是排序 rank 哈希会导致混合 batch 改变相同请求结果。方案在过滤后 scatter 回词表顺序再采样，同时覆盖 Ascend fallback；它是旧遗漏候选，不能按本期新创建统计。

本期证据：
- `2026-10-13T10:24:54Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/35697#issuecomment-6143483007)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：需实测：官方 Ascend fallback 仍在排序域 seeded sampling。优先校验单请求与带过滤 batchmate 的一致性；修复可能改变历史 seeded rollout 的 bit-match。

验证及限制：固定 seed/position/distribution，solo 对比 batchmate top_k=2/greedy；top_k=vocab 应等于未过滤；support 外 token 不可达。

关键代码（当前 diff）：`python/sglang/srt/layers/sampler.py`、`test/registered/sampling/test_deterministic_vocab_id_noise.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-16.json)。

### #38565 · P0 · 旧PR变化

[[Fix] sampling: deterministic top-p / top-k renorm when attn_tp_size > 1 (TP rank divergence in DFlash/DSpark)](https://github.com/sgl-project/sglang/pull/38565)，目标分支 `main`。期末 `open`；采集时 `open`。

FlashInfer 浮点 atomicAdd 归一化的不确定性可使 TP ranks 对相同 logits 得到略不同 target_probs，随后接受/bonus 决策分叉并导致 KV/radix 状态漂移甚至 collective hang。当前方案在 attn_tp_size>1 或确定性推理时使用 deterministic renorm，保留显式环境覆盖；这已不同于旧归档的纯 opt-in 描述。第一期存在 force-push，第二期继续收敛 TP 策略。

本期证据：
- `2026-10-16T18:35:45Z` [`719874d725`](https://github.com/sgl-project/sglang/commit/719874d725c795d256a1b1913fe28db250f90c7b)：[Sampling] Deterministic top-p / top-k renorm policy (fixes TP rank divergence in DFlash/DSpark)。
- `2026-10-14T20:24:56Z`：head_ref_force_pushed。
- `2026-10-15T02:52:15Z`：reviewed。
- `2026-10-15T02:53:12Z`：reviewed。
- `2026-10-16T21:25:38Z`：head_ref_force_pushed。

NPU 判断：需实测：不要移植 FlashInfer CUDA kernel 到 NPU；应在 Ascend 实际 renorm 算子上检查跨 rank bitwise 一致性、共享随机数和 token 决策。与 #41916 的过滤语义一起验收。

验证及限制：作者报告 H100/B200 无 broadcast 的 fast 版本六次均挂起、deterministic 五次均完成30分钟；不是本次复测，也不证明 Ascend 会发生同样问题。需限制 top_p<1，跟踪 target_logits→target_probs→accept_len/bonus→KV 提交的首个分歧。

关键代码（当前 diff）：`python/sglang/srt/environ.py`、`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/layers/sampling_renorm.py`、`python/sglang/srt/speculative/dflash_utils.py`、`python/sglang/srt/speculative/eagle_utils.py`、`python/sglang/srt/speculative/spec_tp_sync.py`、`test/registered/kernels/ops/sampling/test_renorm.py`、`test/registered/unit/sampling/test_renorm_policy.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-16.json)。

### #41916 · P0 · 旧PR变化

[[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths](https://github.com/sgl-project/sglang/pull/41916)，目标分支 `main`。期末 `open`；采集时 `open`。

修复方向是保留原分布的 top-p nucleus，再与 top-k support 相交（joint），不是采用 #41955 的 top_k_first 默认切换。第一期重构 FlashInfer min-p、EAGLE、dense/sparse target 和 UNO 五条路径；第二期 b94a95d5 为 CUDA unseeded 路径融合 top-p rescaling，共享 retained-maximum reduction，保留 seeded/non-CUDA 原计算，并补 is_cuda 绑定修复。FP32 cutoff 边界可能有差异，RL support 与归一化概率必须同时校验。

本期证据：
- `2026-10-14T13:32:19Z` [`98d8264ea5`](https://github.com/sgl-project/sglang/commit/98d8264ea54d456692d8f7d28896439b428af284)：Fix FlashInfer min-p filter ordering。
- `2026-10-14T13:32:20Z` [`19b987d3bf`](https://github.com/sgl-project/sglang/commit/19b987d3bf68b2d382705af82e1245b90f69a678)：Fix speculative sampling filter ordering。
- `2026-10-14T13:32:20Z` [`a6a9034a55`](https://github.com/sgl-project/sglang/commit/a6a9034a551826956dde064c8d816d01e4a1da85)：Fix DFlash joint sampling filters。
- `2026-10-14T13:32:21Z` [`8f31f64624`](https://github.com/sgl-project/sglang/commit/8f31f64624d786bf0bd2eb19eae38e334fca79b0)：Reject unsupported speculative min-p requests。
- `2026-10-14T13:32:21Z` [`e9a056ce5b`](https://github.com/sgl-project/sglang/commit/e9a056ce5b8da39fe73ec55990cb3753a2720649)：Adapt sampling filter tests to current CI layout。
- `2026-10-14T13:32:21Z` [`599a7bd14f`](https://github.com/sgl-project/sglang/commit/599a7bd14f6670b5fd491cc1fd9e4dab5b1a1dcf)：Fix UNO joint sampling filter semantics。
- `2026-10-14T13:32:21Z` [`50e6ca6b73`](https://github.com/sgl-project/sglang/commit/50e6ca6b73cc49214b5e9e28068b187ac3d7de2f)：Consolidate sampling regressions and use production kernels。
- `2026-10-15T20:05:28Z` [`94c3514810`](https://github.com/sgl-project/sglang/commit/94c35148104f26a8c6740582571e528f116eeb94)：Merge branch 'main' into fix/top-k-top-p-filtering。
- `2026-10-14T13:36:28Z`：head_ref_force_pushed。
- `2026-10-14T11:15:19Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6167080316)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-14T16:49:55Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6173553964)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- `2026-10-15T20:04:50Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/41916#issuecomment-6201187503)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。

NPU 判断：需实测：Ascend fused helper 与 torch fallback 的过滤顺序必须逐分布比对。先定义与 GPU 一致的语义再迁移 kernel；不能据 PR open 状态宣称默认策略已改变。

验证及限制：10/14 作者报告 H200 上五路径 18 passed + 6 subtests，修复前 support/probability 断言失败。10/21 作者报告 11 项原测试、12 项 seeded bitwise 比较和 seeded CUDA Graph 验证通过；H200 helper DFlash 1198→899 us、UNO 368→271 us，未宣称端到端精度或 trainer 对齐。本次未复现；NPU 需覆盖 joint/top_k_first 反例、min_p、cutoff ties、混合 batch 和 emitted token 对齐。

关键代码（当前 diff）：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/managers/scheduler.py`、`python/sglang/srt/speculative/dflash_utils.py`、`python/sglang/srt/speculative/eagle_utils.py`、`python/sglang/srt/speculative/uno_utils.py`、`test/registered/kernels/ops/speculative/test_speculative_sampling.py`、`test/registered/sampling/test_sampling_mask.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-16.json)。

### #43346 · P1 · 旧PR变化

[[Sampler] Sort only the max(top_k) leading columns in the torch sampler](https://github.com/sgl-project/sglang/pull/43346)，目标分支 `main`。期末 `merged`；采集时 `merged`。

将 torch probs sampler 的全词表排序缩为 max(top_k) 个候选，host 端上界避免每步 D2H 同步。seeded 与 return_filtered_probs 路径明确保留 full sort，因此不能把 AMD 常规采样吞吐收益直接算到 mask 输出。

本期证据：
- `2026-10-11T07:54:58Z`：reviewed。
- `2026-10-11T08:01:00Z`：merged。
- `2026-10-11T08:01:00Z`：closed。
- `2026-10-11T00:45:44Z`：[技术讨论原文](https://github.com/sgl-project/sglang/pull/43346#issuecomment-6103886956)，完整上下文保存在时间线 JSON；技术判断及验证范围见本节说明。
- 当前可达提交列表未发现本期提交；评论、评审或状态变化不等同代码新增。

NPU 判断：部分覆盖：公共 sampler 已有该优化；官方 Ascend logits fallback 是独立实现，仍全量排序。可移植上界传播，但必须保留 token-id 映射及 seed/tie 契约。

验证及限制：作者报告 MI355X 常规 serving 吞吐提升 9.4%–35.2%，不是 mask-on 的基准。本次未复现。验收 mixed top_k、全词表行、merge/filter、seeded 与 mask-on 的回退。

关键代码（当前 diff）：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/sampling/sampling_batch_info.py`、`test/registered/sampling/test_sampling_mask.py`、`test/registered/unit/sampling/test_sampling_batch_info.py`、`test/registered/unit/sampling/test_torch_sampling_top_k_bound.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-16.json)。

## 无实质进展与补漏边界

- [#40953](https://github.com/sgl-project/sglang/pull/40953)：无本期活动，事件类型 `{}`；采集时 `closed`。不计入本期实质条目。
- [#42352](https://github.com/sgl-project/sglang/pull/42352)：仅元数据/引用，事件类型 `{"referenced": 1}`；采集时 `closed`。不计入本期实质条目。
- [#42647](https://github.com/sgl-project/sglang/pull/42647)：仅元数据/引用，事件类型 `{"cross-referenced": 4, "commented": 1}`；采集时 `open`。不计入本期实质条目。
- [#43047](https://github.com/sgl-project/sglang/pull/43047)：无本期活动，事件类型 `{}`；采集时 `open`。不计入本期实质条目。
- [#43517](https://github.com/sgl-project/sglang/pull/43517)：仅元数据/引用，事件类型 `{"ready_for_review": 1, "review_requested": 11}`；采集时 `open`。不计入本期实质条目。
- #41955 / #41957 / #42802 / #39934 ：采集时 updated_at 均早于 10/11，作为旧背景，不重复计入。#41957 目标分支是 sampling-top-k-first，不能当 main 独立落地。

官方 Ascend 缺口的直接依据：[scheduler 拒绝分支](https://github.com/sgl-project/sglang/blob/2da6c0c237324eefea6420e015beb1e6858cb80c/python/sglang/srt/managers/scheduler.py#L3135)、[Ascend helper 返回契约](https://github.com/sgl-project/sglang/blob/2da6c0c237324eefea6420e015beb1e6858cb80c/python/sglang/srt/layers/sampler.py#L861)。

## 检索与复核

基线为 10/10 归档，第二期接续本轮第一期结果；旧遗漏单列历史补漏。关键词、完整分页、131 条归档 PR 的状态/事件检查、远端文件历史和候选排除均见共享证据。

- 完整查询索引、131条旧PR审计、39条候选筛选及本期证据均嵌入 [sample-mask-delta-2026-10-16.json](sample-mask-delta-2026-10-16.json)。
- [本期结构化数据](sample-mask-delta-2026-10-16.json) · [本期 CSV](sample-mask-delta-2026-10-16.csv)。
- 深读与优先级由主代理完成；DeepSeek sidecar 承担归档整理、远端扫描、事件/diff 采集。
- 当前 PR body、current diff、CI 摘要均为采集快照。无法完整恢复 force-push 前所有源码和当期 CI 状态，未将现在的绿灯/红灯倒推到第一期。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。

## 扫描覆盖与源码时间说明

完成76个issue查询/76页、213个唯一检索PR；检查336个归档与发现PR的当前元数据及全部timeline页。直接扫描892个main提交/10页，56个路径窗口查询得到74个唯一提交。39条机器候选由主代理继续筛选，另补入 #35697、#43854 和替代链路 #44850 等关键证据。完整关键词查询、候选排除和旧PR审计在本期JSON中。

源码快照为远端 main `2da6c0c237324eefea6420e015beb1e6858cb80c`，观察于 `2026-10-23T02:44:27Z`；PR事件截止 `2026-10-23T02:05:41Z`。两者时间不同，源码用于当前覆盖核验，不能代表当期期末源码。

本期没有新建且纳入核心分析的 sampling-mask PR；新增候选与旧遗漏明确分开统计。#41205 的本期事件仅作引用/背景审计，不表示代码更新。
