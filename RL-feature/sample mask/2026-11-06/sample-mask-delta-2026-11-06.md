# SGLang sampling mask 增量跟踪：2026-11-06

UTC 窗口：`[2026-10-30T02:34:03Z, 2026-11-06T06:35:52Z)`。采集截止：`2026-11-06T06:35:52Z`（第四期，接续 10/30 第三期）。

代码基准：[官方 main `10be18dcfc46`](https://github.com/sgl-project/sglang/commit/10be18dcfc466a2171b4892de214909a328ff737)。sample mask 照旧只跟踪 SGLang；vLLM 侧 mask 相关条目仅作背景。

## 本期结论

纳入 **6 条**，分类为 前期补漏续跟 4 条、本期新建 2 条。本期最大的变化是**上期跟踪的两条 P0 全部合入 main**：

- [#41916](https://github.com/sgl-project/sglang/pull/41916)（`2026-11-03T12:06-08:00`）：top-k/top-p 过滤统一为 joint 语义（先保留 top-p nucleus，再与 top-k support 相交），并给 scheduler 加了 15 行校验；`test_sampling_mask.py` +28。
- [#38565](https://github.com/sgl-project/sglang/pull/38565)（`2026-10-30T17:07-07:00`）：TP>1 下的确定性 renorm 合入，新增 `srt/layers/sampling_renorm.py`（98 行），`test_sampling_mask.py` +7。

也就是说：**语义与确定性这两件事在 main 上定案了**，但这是 CUDA 侧的实现与测试。Ascend 仍然拿不到 mask：

- [#43517](https://github.com/sgl-project/sglang/pull/43517)（Ascend 采样 mask）仍 open，更新到 11-06；
- main 的 scheduler 仍以「ascend 采样后端直接从 logits 采样、不构造 support」拒绝 `return_sampling_mask`；
- #38565 的确定性策略是否在 Ascend 算子上成立**没有证据**，而同一问题在 XPU 上已经被单独提出（[#47967](https://github.com/sgl-project/sglang/pull/47967)）。

## 增量索引

- [#41916](https://github.com/sgl-project/sglang/pull/41916) — 前期补漏续跟；期末 `merged`（`2026-11-03T12:06:33-08:00`）；[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths。
- [#38565](https://github.com/sgl-project/sglang/pull/38565) — 前期补漏续跟；期末 `merged`（`2026-10-30T17:07:31-07:00`）；[Fix] sampling: deterministic top-p / top-k renorm when tp_size > 1。
- [#43517](https://github.com/sgl-project/sglang/pull/43517) — 前期补漏续跟；期末 `open`；[Feat] [NPU] sampling: Add sampling-mask support for Ascend。
- [#47967](https://github.com/sgl-project/sglang/pull/47967) — 本期新建；期末 `open`；[Fix] [XPU] sampling: Make deterministic top-k renorm reproducible on XPU。
- [#48487](https://github.com/sgl-project/sglang/pull/48487) — 本期新建；期末 `open`；[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank。
- [#43047](https://github.com/sgl-project/sglang/pull/43047) — 前期补漏续跟；期末 `open`（draft，base `sglang-miles`）；[sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding。

## 重点条目

### #41916 · P0 · 前期补漏续跟（合入）

[[Fix] sampling: Make top-k/top-p filtering consistent across sampling paths](https://github.com/sgl-project/sglang/pull/41916)，`2026-11-03T12:06:33-08:00` 合入，7 文件，+228/-44。

机制：五条路径（FlashInfer min-p、EAGLE、DFLASH、UNO、dense/sparse target）统一为 joint 语义——保留原分布的 top-p nucleus，再与 top-k support 相交；`scheduler.py` 增加 15 行校验，`test_speculative_sampling.py` +89、`test_sampling_mask.py` +28 固定语义。这与 #41955 的 `top_k_first` 默认切换是相反方向，`sampling-top-k-first` 分支线不能当作 main 行为。

NPU 判断：需实测。合入只代表**语义定案**，Ascend fused helper 与 torch fallback 的过滤顺序、cutoff 并列项处理仍需逐分布比对。mask 的 support 与 selected 两种模式都要对齐。

验证及限制：作者侧五路径 18 passed + 6 subtests、seeded 位级比较；本期未复测。

### #38565 · P0 · 前期补漏续跟（合入）

[[Fix] sampling: deterministic top-p / top-k renorm when tp_size > 1](https://github.com/sgl-project/sglang/pull/38565)，`2026-10-30T17:07:31-07:00` 合入，9 文件，+237/-15。

机制：新增 `python/sglang/srt/layers/sampling_renorm.py`（98 行）作为确定性归一化实现，`sampler.py` 按 TP 组大小选择路径；测试覆盖 `test_renorm.py`（+68）、`test_renorm_policy.py`（+49）与 `test_sampling_mask.py`（+7）。合入口径是 `tp_size > 1`，比原 `attn_tp_size` 覆盖面更大——现在只要 TP 组多于一个 rank 就默认确定性。

NPU 判断：需实测，且**这是本期最需要 Ascend 自证的一条**。确定性 renorm 的价值在于跨 rank 位级一致；Ascend 的累加顺序、bf16 归约与 CUDA 不同，必须用 NPU 自己的算子验证，不能因为 CUDA 合入就认为 Ascend 一致。

验证及限制：作者侧 H100/B200 数据（fast 版本挂起、deterministic 版本完成）；本期未复测。

### #43517 · P0 · 前期补漏续跟（open）

[[Feat] [NPU] sampling: Add sampling-mask support for Ascend](https://github.com/sgl-project/sglang/pull/43517)，`open`，更新到 11-06（本期 13 个提交）。

现状：main 仍通过 `get_exec().kernel.sampling_backend == "ascend"` 拒绝 `return_sampling_mask`（理由：ascend 后端直接从 logits 采样、不构造 top-k/top-p support）。因此 Ascend 用户拿不到 mask，该 PR 仍是唯一通路。

NPU 判断：缺失。合入前不能宣称 Ascend 已支持；合入后需按公共契约验收 `top_k=1025` 回退、`min_p`、greedy、support/selected、混合 opted-in 行、seeded mask on/off 与连续图重放。

验证及限制：作者侧 NPU/mock 对照；本期未复测。

### #47967 · P2 · 本期新建（XPU 跟进）

[[Fix] [XPU] sampling: Make deterministic top-k renorm reproducible on XPU](https://github.com/sgl-project/sglang/pull/47967)，`open`。

机制：#38565 的确定性策略在 XPU 上会走到 sgl-kernel 之外的路径，需要单独保证可复现。对 Ascend 是直接的方法论参考：**同一语义在不同后端要各自证明确定性**。

NPU 判断：需实测。Ascend 上应照此补一个「确定性策略在 NPU 算子上是否真的确定」的验证，而不是假设。

### #48487 · P2 · 本期新建（采样同步优化）

[[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank](https://github.com/sgl-project/sglang/pull/48487)，`open`。

机制：`Sampler._sync_token_ids_across_tp` 对所有 grammar-constrained batch（以及开启 `SYNC_TOKEN_IDS_ACROSS_TP` 时的所有 batch）做 token-id all-reduce；当 sync group 只有一个 rank 时跳过。与 mask 的关系是相邻的：同步被跳过时，mask 行的选中集必须自洽，否则「省一次通信」会变成「行不一致」。

NPU 判断：需实测。Ascend 上需确认分组判断与 HCCL all-reduce 等价，且跳过同步后 mask 选中行保持确定。

### #43047 · P2 · 前期补漏续跟（无实质）

[[sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding](https://github.com/sgl-project/sglang/pull/43047)，draft，base 为 `sglang-miles`，分支相对 main 有 222 文件（整分支 diff）。本期仅状态更新，目标分支不是 main，不能据此判断 main 或 Ascend 能力。

## 无实质进展与补漏边界

- [#46080](https://github.com/sgl-project/sglang/pull/46080)（DFLASH top-k support verify）与 [#46186](https://github.com/sgl-project/sglang/pull/46186)（radix top-k）本期无活动，不计入。
- [#39934](https://github.com/sgl-project/sglang/pull/39934)（NPU mask max tokens e2e）本期无更新。
- 背景：vLLM 侧 [#65649](https://github.com/vllm-project/vllm/pull/65649) 用 FlashInfer top-k selection 做 top-k/top-p masking，与 SGLang 无关；vllm-ascend 的 [#17907](https://github.com/vllm-project/vllm-ascend/pull/17907) 是 mask 打包签名适配，已在 hidden-state 报告里提及。
- main 本期没有「新增 mask 能力」的提交；两条合入都是既有 PR 的语义定案与确定性实现。

## 检索与复核

- [本期结构化数据](sample-mask-delta-2026-11-06.json) · [本期 CSV](sample-mask-delta-2026-11-06.csv)。
- 未运行 GPU/NPU 或模型测试；作者性能与接受率数字均为作者报告。

## 扫描覆盖与源码时间说明

- 关键词检索：8 个采样短语 × `updated:>=2026-10-30` + `created:` 按日切分，去重后 sglang 1598 条（1463 PR + 135 Issue）；属采样主题的候选 16 条，纳入 6 条。
- 合入侧：本地 main 浅克隆扫描窗口内 537 个提交，按 `sampling mask` / `sampling_mask` / `top-k support` / `mask capture` 与文件路径双重筛选；命中的功能提交即本期两条 P0 的合入提交（`a2cafa5676`、`71ef675ff5`）。
- 基线：156 条历史 tracked 条目状态复核；源码快照 `10be18dcfc466a2171b4892de214909a328ff737`。
