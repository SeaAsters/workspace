# SGLang sampling mask 增量跟踪：2026-10-30

UTC 窗口：`[2026-10-23T02:05:41Z, 2026-10-30T02:34:03Z)`。采集截止：`2026-10-30T02:34:03Z`（第三期，接续 10/23 第二期）。

代码基准：[官方 main `0f3f619850f2`](https://github.com/sgl-project/sglang/commit/0f3f619850f222b4f238ff3d336926e99a2a5c56)（提交时间 `2026-10-30T02:08:22Z`，观察于 `02:34:03Z`）。sample mask 照旧只跟踪 SGLang；vLLM 侧 mask 相关 PR 仅作背景列出，不纳入判断。

## 本期结论

纳入实质变化或技术讨论 **9 条**，分类为 旧PR变化 5 条、前期补漏续跟 2 条、本期新建 2 条。其中本期合入 1 条（#42647，合入目标是 `sglang-miles` 而非 main），open 7 条，关闭未合入 1 条。

**最重要的事实是 main 在本窗口内没有任何 sampling-mask 功能合入**：唯一触及 `test/registered/sampling/test_sampling_mask.py` 的 main 提交是 [#46481](https://github.com/sgl-project/sglang/pull/46481)（只调整 CI `est_time`）。因此「mask 能力有没有进展」在本期只能从 PR 侧回答，不能用 main 的形态推断。

三条 open 的 P0 都没有收敛：#43517（Ascend 支持）仍被 main 的 scheduler 拒绝，#38565（确定性 renorm）本期在修冲突与补测试，#41916（过滤一致性）把 main 合进分支并补了两个后续修复提交。官方 main 仍然以「ascend 采样后端直接从 logits 采样、不构造 top-k/top-p support」为由拒绝 `return_sampling_mask`，[`scheduler.py#L3287-L3294`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/managers/scheduler.py#L3287) 的分支没有变化。

## 增量索引

- [#43517](https://github.com/sgl-project/sglang/pull/43517) — 前期补漏续跟；期末 `open`，采集时 `open`；[Feat] [NPU] sampling: Add sampling-mask support for Ascend。
- [#38565](https://github.com/sgl-project/sglang/pull/38565) — 旧PR变化；期末 `open`；[Fix] sampling: deterministic top-p / top-k renorm when tp_size > 1 (TP rank divergence in DFlash/DSpark)。
- [#41916](https://github.com/sgl-project/sglang/pull/41916) — 旧PR变化；期末 `open`；[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths。
- [#46080](https://github.com/sgl-project/sglang/pull/46080) — 本期新建；期末 `open`；[Perf] DFLASH: Verify top-k sampled batches on the all-gathered top-k support (opt-in)。
- [#46186](https://github.com/sgl-project/sglang/pull/46186) — 本期新建；期末 `open`；[Perf] spec: radix top-k for speculative verify target probabilities。
- [#39934](https://github.com/sgl-project/sglang/pull/39934) — 前期补漏续跟；期末 `open`；[Test] [NPU] parameter: Add sampling mask max tokens e2e test。
- [#41205](https://github.com/sgl-project/sglang/pull/41205) — 旧PR变化；期末 `closed`（stale draft，`2026-10-24T01:24:49Z`）；[Sampling] Write /generate sampling masks from numpy rows。
- [#42647](https://github.com/sgl-project/sglang/pull/42647) — 旧PR变化；期末 `merged`（`2026-10-23T06:17:53Z`，目标分支 `sglang-miles`）；[Feat] RL: Add radix transfer plugins。
- [#43047](https://github.com/sgl-project/sglang/pull/43047) — 旧PR变化；期末 `open`（draft）；[sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding。

## 重点 PR：机制、证据与 NPU 跟进

### #43517 · P0 · 前期补漏续跟

[[Feat] [NPU] sampling: Add sampling-mask support for Ascend](https://github.com/sgl-project/sglang/pull/43517)，目标分支 `main`，head `22ca97cc63`，+567/-37，`open`。

机制：让 Ascend 采样后端在 `return_sampling_mask=true` 时导出实际采样的 post-filter 权重（support / selected 两种模式），接入公共 `_SamplingMaskCapture`、打包与 host 物化。仍 open，不能被当作 main 已支持。

本期证据：

- `2026-10-24T06:50:13Z`、`07:22:00Z`：两次 head_ref_force_pushed（分支重写）。
- `2026-10-24T14:20:33Z`（+0800）提交 `c1cc2213db`：`[PATCH 11/12] Reorganize Ascend sampling-mask tests and retain …`。
- `2026-10-26T01:32:12Z`：[floatlibai 评论 LGTM](https://github.com/sgl-project/sglang/pull/43517)。
- `2026-10-29T13:07:33Z`（+0800）提交 `22ca97cc63`：`[PATCH 12/12] [NPU] Unify Ascend sampler returns and refine …`。
- 测试落点已重组为 `test/registered/npu/basic_function/backends/` 与 `test/registered/npu/rl/sampling_mask/` 两处，并新增 `test/registered/unit/sampling/test_ascend_sampler.py`。

NPU 判断：**这是缺口本身**。main 的 scheduler 通过 `get_exec().kernel.sampling_backend == "ascend"` 直接拒绝 `return_sampling_mask`（理由：Ascend 后端从 logits 直接采样，不构造 support），所以 Ascend 用户目前拿不到 mask。该 PR 未合入前，任何「Ascend 已支持采样 mask」的说法都不成立。

验证及限制：作者侧为 NPU/CPU mock 验证与参考实现对照。验收应覆盖 `top_k=1025` 回退、`min_p`、greedy（`top_k=1`）、support/selected 两模式、混合 opted-in 行、seeded mask on/off 与连续图重放。

关键代码（当前 diff）：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/managers/scheduler.py`、`test/registered/npu/rl/sampling_mask_reference.py`、`test/registered/unit/sampling/test_ascend_sampler.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-10-30.json)。

### #38565 · P0 · 旧PR变化

[[Fix] sampling: deterministic top-p / top-k renorm when tp_size > 1 (TP rank divergence in DFlash/DSpark)](https://github.com/sgl-project/sglang/pull/38565)，目标分支 `main`，head `b8d4668008`，+237/-15，`open`。

机制：FlashInfer 浮点 `atomicAdd` 归一化的不确定性会让 TP ranks 对相同 logits 得到略不同的 `target_probs`，随后接受/bonus 决策分叉，导致 KV/radix 状态漂移甚至 collective hang。方案是在 `tp_size > 1` 或确定性推理时使用 deterministic renorm，并保留显式环境覆盖。本期标题由 `attn_tp_size` 改为 `tp_size`，覆盖面变大。

本期证据：

- `2026-10-29T06:52:09Z`：[Fridge003 评论 `@claude review`](https://github.com/sgl-project/sglang/pull/38565)。
- `2026-10-29T08:12:27Z`：[Fridge003 评论 `Please fix conflicts`](https://github.com/sgl-project/sglang/pull/38565)。
- `2026-10-30T00:15:19Z`：head_ref_force_pushed；`00:16:12Z` renamed。
- 新提交：`2026-10-30T00:02:48Z` `c33aeb0b18`、`01:09:51Z` `b8d4668008`（`[Test] sampling: Publish a runtime config`）。
- 当前 diff 新增 `python/sglang/kernels/ops/sampling/renorm.py`，并触及 `test/registered/sampling/test_sampling_mask.py`。

NPU 判断：**需实测**。不要移植 FlashInfer CUDA kernel；应在 Ascend 实际 renorm 算子上检查跨 rank 位级一致性、共享随机数与 token 决策，并与 #41916 的过滤语义一起验收。

验证及限制：作者早前报告 H100/B200 上 fast 版本六次均挂起、deterministic 五次完成 30 分钟；本期为新提交与冲突修复，未提供新的设备数据。需限制 `top_p<1`，跟踪 `target_logits → target_probs → accept_len/bonus → KV 提交` 的首个分歧。

关键代码：`python/sglang/srt/layers/sampling_renorm.py`、`python/sglang/kernels/ops/sampling/renorm.py`、`python/sglang/srt/speculative/spec_tp_sync.py`、`test/registered/sampling/test_sampling_mask.py`。

### #41916 · P0 · 旧PR变化

[[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths](https://github.com/sgl-project/sglang/pull/41916)，目标分支 `main`，head `e84984ce7c`，+226/-44，`open`。

机制：保留原分布的 top-p nucleus，再与 top-k support 相交（joint），而不是 #41955 的 `top_k_first` 默认切换；覆盖 FlashInfer min-p、EAGLE、dense/sparse target 与 UNO 五条路径。`FP32` cutoff 边界可能有差异，RL support 与归一化概率必须同时校验。

本期证据：

- `2026-10-23T15:12:01Z`、`2026-10-24T00:08:38Z`、`00:14:47Z`：3 次 cross-reference。
- `2026-10-28T00:05:01Z`：[guapisolo 评论](https://github.com/sgl-project/sglang/pull/41916)说明将以 `c49d5a0950` 合并 main（`655059819c`），保留双方 joint 语义。
- `2026-10-28T07:53:46Z`：请求 Qiaolin-Yu 评审。
- `2026-10-30T00:20:02Z`：`/rerun-test test/registered/e2e/models/test_transformers_backend_eval.py`，随后 bot 报告通过。
- 新提交：`2026-10-29T16:20:04Z`（-0700）`3d1e9fddf3`（mem_cache: build an empty KVLayerRouter）、`cb72322eab`（pass request pools to dummy tree caches）。

NPU 判断：**需实测**。Ascend fused helper 与 torch fallback 的过滤顺序必须逐分布比对；先定义与 GPU 一致的语义再迁移 kernel，不能据 PR open 状态宣称默认策略已改变。

验证及限制：作者早前报告五路径 18 passed + 6 subtests、12 项 seeded 位级比较与 CUDA Graph 验证；本期为合并 main 与重跑，没有新的 Ascend 数据。

关键代码：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/speculative/{eagle,dflash,uno}_utils.py`、`test/registered/kernels/ops/speculative/test_speculative_sampling.py`。

### #46080 · P1 · 本期新建

[[Perf] DFLASH: Verify top-k sampled batches on the all-gathered top-k support (opt-in)](https://github.com/sgl-project/sglang/pull/46080)，目标分支 `main`，head `a43baa6f83`，+930/-8，`open`。

机制：TP>1 + softcap lm_head 时，DFLASH verify 原本要 all-gather 全词表 fp32 logits、softcap、全词表 radix top-k、再跑整词表扫描的 chain sampler。当请求使用 top-k 采样（多数 generation config 的默认）时，只需在 all-gather 的 top-k support 上重算概率即可，因此新增 `kernels/ops/speculative/dflash_sparse_verify.py` 与 env 开关（opt-in）。

本期证据：3 个提交均在 `2026-10-23`（`22f36afe91`、`5581b95cc9`、`a43baa6f83`）推送；PR 时间线本期无评审事件，属「已建未评审」。

NPU 判断：**需实测**。稀疏 verify 只在 support 上重算概率，Ascend 需确认 support 与实际采样 support 完全一致，否则会改变 support 语义（这正是 mask 正确性的核心）。

验证及限制：作者附 `test_dflash_sparse_verify.py`、`test_dflash_logits.py` 与 `test_graph_pool_borrow.py`；NPU 需覆盖 top_k 边界与 support 一致性。

关键代码：`python/sglang/kernels/ops/speculative/dflash_sparse_verify.py`、`python/sglang/srt/speculative/dflash_worker_v2.py`、`python/sglang/srt/environ.py`。

### #46186 · P1 · 本期新建

[[Perf] spec: radix top-k for speculative verify target probabilities](https://github.com/sgl-project/sglang/pull/46186)，目标分支 `main`，head `837aa2162d`，+277/-39，`open`。

机制：把 spec verify 的目标概率构造改为 radix top-k（`kernels/ops/sampling/topk.py`、`speculative/dflash_utils.py`），而 `build_speculative_verify_target_probs` 同时是 sampling-mask 捕获的入口。属相邻性能改动，但一旦语义偏移会直接污染 mask 行。

本期证据：1 个提交 `837aa2162d`（`2026-10-23`）；`2026-10-24` 两次 cross-reference（jianc99）。作者注明这是 DFlash 自适应验证系列的第 3 个组件，基准与 GPU 测试在原始分支上跑过、合到上游后未重跑。

NPU 判断：**需实测**。radix top-k 的分块与归约在 Ascend 上需与 torch 版做容差比对，否则 support/mask 行会偏移。

验证及限制：作者附 `test_verify_target_probs.py`；本期未复测，且作者自述更新后未重跑基准。

### #39934 · P2 · 前期补漏续跟

[[Test] [NPU] parameter: Add sampling mask max tokens e2e test](https://github.com/sgl-project/sglang/pull/39934)，目标分支 `main`，head `34b628dd28`，+266/-0，`open`。

机制：为 `--sampling-mask-max-tokens` 补 NPU 端到端回归（单文件 `test/registered/npu/basic_function/parameter/test_npu_sampling_mask_max_tokens.py`），覆盖「support 超过上限即 abort」的行为。这是 #43517 合入前的验收护栏之一。

本期证据：`2026-10-27T06:07:30Z` bot 报告 `c0b9a7c` 必需检查失败；`08:16:20Z` renamed；新提交 `2026-10-27T14:03:26Z`（+0800）`c0b9a7ce52`、`16:26:33Z` `34b628dd28`，提交信息均为 `fix`。

NPU 判断：直接是 NPU 测试补齐，不改变实现；但当前 CI 未过，不能当作已验证。

验证及限制：作者侧测试仍在修，本期未复测。

### #41205 · P2 · 旧PR变化（关闭）

[[Sampling] Write /generate sampling masks from numpy rows](https://github.com/sgl-project/sglang/pull/41205)，draft，`closed`。

本期被 bot 以「仍是 draft 且 28 天未更新」关闭（`2026-10-24T01:24:49Z`），未合入。其思路仍是 mask 返回路径的性能参考：`/generate` 直接从 numpy 行序列化 mask，避免逐元素 Python list 构造（作者测 batch 256 + top_k=8192 时 64.9s vs 11.2s）。NPU 判断：与 #43517 的 host 物化路径相关，但关闭状态不代表能力可用。

### #42647 · P1 · 旧PR变化（合入 sglang-miles）

[[Feat] RL: Add radix transfer plugins](https://github.com/sgl-project/sglang/pull/42647)，目标分支 `sglang-miles`，head `c577937f17`，+2395/-46，`merged`。

机制：把 sampling masks、routed experts、indexer top-k 从 inline `meta_info`（长序列下 6.3 MB 的 base64/列表）改为共享 `radix_transfer_plugins` 包的 `OutputStore` 引用返回，selected logprobs 仍留在 `meta_info`；要求单 prompt/n=1/无 beam，并补齐 writer 关闭、取消清理、`read_many` 分块读取与 numpy 拷贝。

本期证据：`2026-10-23T06:17:53Z` merged（及 closed）；之后 `06:31:10Z`、`08:37:31Z` 等多次 cross-reference。6 个 patch 提交，最后一个为 `2026-10-21T22:47:42Z`（-0700）`c577937f17`。

NPU 判断：**缺失/需实测**。框架与 host 侧逻辑可复用，但 Ascend 输出需先接通 #43517，再核对 store 引用对齐、buffer 拷贝所有权与真实 NPU→host→Mooncake 路径。

验证及限制：作者侧为 macOS fake Mooncake 与 CPU 测试（未跑真实 Mooncake）；本期未复测。

### #43047 · P2 · 旧PR变化（无本期事件）

[[sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding](https://github.com/sgl-project/sglang/pull/43047)，draft，目标分支 `sglang-miles`，head `0363eeae5b`。分支相对 main 有 218 文件（含整条 release 分支的累计 diff，+15060/-2083），本期时间线无实质事件，仅作 branch 背景。NPU 判断：不适用——目标分支不是 main，不能据此判断 main 或 Ascend 能力。

## 当前 main 的 mask 现状（源码核对）

- 拒绝分支仍在：[`scheduler.py#L3287-L3294`](https://github.com/sgl-project/sglang/blob/0f3f619850f222b4f238ff3d336926e99a2a5c56/python/sglang/srt/managers/scheduler.py#L3287) 对 `return_sampling_mask and get_exec().kernel.sampling_backend == "ascend"` 返回 `return_sampling_mask is not supported with the ascend sampling backend.`。
- 公共约束依旧：同一函数里先检查 PD 分离需 `SGLANG_ENABLE_DISAGG_SAMPLING_MASK=1`，再检查 `top_k == 1`（greedy）或 `1 < top_k <= sampling_mask_max_tokens`，然后才进入 `validate_spec_sampling_mask_request`。
- 因此 Ascend 侧只能走 #43517；`--sampling-mask-max-tokens` 的行为护栏由 #39934 补测。
- 本期 main 没有任何 mask 功能提交：`test/registered/sampling/test_sampling_mask.py` 只在 #46481（`est_time`）里被改动。

## 无实质进展与补漏边界

- [#41955](https://github.com/sgl-project/sglang/pull/41955)、[#41957](https://github.com/sgl-project/sglang/pull/41957)：目标分支是 `sampling-top-k-first` 分支线，本期无实质事件；不能当 main 独立落地。
- [#43346](https://github.com/sgl-project/sglang/pull/43346)：本期无实质事件（`merged`），仅作背景。
- [#46481](https://github.com/sgl-project/sglang/pull/46481)：唯一触及 mask 测试的 main 提交，只调整 CI `est_time`。
- 本期采样主题没有 Issue/RFC 命中：sglang 103 条 Issue 发现中，属采样主题的候选为 0，因此本轮 mask 侧全部为 PR 条目。
- 背景（不纳入判断）：vLLM 侧本期有两条 mask 相关 PR——[#64231](https://github.com/vllm-project/vllm/pull/64231)（[XPU] SamplingMaskTensors 走 sycl kernel）与 [#64130](https://github.com/vllm-project/vllm/pull/64130)（[Perf][ROCm] 小 batch 下拆分 top-k mask），均 open，与 SGLang 无关。

## 检索与复核

- [本期结构化数据](sample-mask-delta-2026-10-30.json) · [本期 CSV](sample-mask-delta-2026-10-30.csv)。
- 基线为 9/29→10/10→10/16→10/23 四期归档；本轮 9 条核心条目全部取到当前 state/merged_at，重点条目取到 timeline 与 patch 提交序列。
- 当前 PR body、current diff、CI 摘要均为采集快照；force-push 前的旧提交无法恢复（#43517、#38565 本期都有 force-push）。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。

## 扫描覆盖与源码时间说明

- 关键词检索：sglang 8 个采样短语（`"sampling mask"`、`sampling_mask`、`"sample mask"`、`return_sampling_mask`、`"top-p mask"`、`"top-k mask"`、`"mask capture"`、`sample_mask`）× `updated:>=2026-10-23`，另加 RFC 标签与按日切分的 `created:` 全量扫描。
- 合入侧：本地 main 浅克隆 `2026-10-23T02:05:41Z` 以来 552 个提交，按 `sampling mask` / `sampling_mask` / `sample mask` / `top-k support` / `mask capture` 关键词与文件路径双重筛选；命中的功能提交为 0，仅 1 条 CI 时序提交触及测试文件。
- 归档侧：143 条历史 tracked 条目状态复核（其中 35 条属 sampling_mask 主题）；本期 9 条有更新（含 hidden 主题条目）。
- 候选侧：18 条采样主题关键词候选逐条判定，纳入 9 条为核心分析，其余在 `candidate_decisions`。

源码快照为远端 main `0f3f619850f222b4f238ff3d336926e99a2a5c56`（提交时间 `2026-10-30T02:08:22Z`），观察于 `2026-10-30T02:34:03Z`；PR/Issue 事件截止同一时刻。
