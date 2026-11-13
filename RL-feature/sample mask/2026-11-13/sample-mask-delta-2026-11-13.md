# SGLang sampling mask 增量跟踪：2026-11-13

UTC 窗口：`[2026-11-06T06:35:52Z, 2026-11-13T04:40:41Z)`。采集截止：`2026-11-13T04:40:41Z`（第五期，接续 11/06 第四期）。

代码基准：[官方 main `4f8b6867534c`](https://github.com/sgl-project/sglang/commit/4f8b6867534c8a6632845353997555fca857dd63)（`2026-11-13T10:52+08:00`）。sample mask 照旧只跟踪 SGLang；vLLM 侧 mask 条目仅作背景。

## 本期结论

纳入 **4 条**（2 条合入、2 条新建）。本期是这条特性的**转折点**：

1. **Ascend 采样 mask 终于合入**：[#43517](https://github.com/sgl-project/sglang/pull/43517) 于 `2026-11-12T09:52+08:00` 合入（`f211f410`，4 文件，+448/-37）。这是跟踪四期以来的第一次实质变化：
   - `managers/scheduler.py` **删掉了 10 行**，正是那条「ascend 采样后端不构造 support、拒绝 `return_sampling_mask`」的分支；
   - `layers/sampler.py` 改为 `_forward_ascend_backend` 返回三元组 `(batch_next_token_ids, logprobs, sampling_mask_capture)`，新增 `_build_ascend_sampling_mask_capture`，并把 `return_filtered_probs=return_sampling_mask` 传进 `top_k_top_p_min_p_sampling_from_logits_ascend`；
   - 新增 `test/registered/npu/rl/sampling_mask/test_npu_sampling_mask.py`（211 行）与 `test/registered/unit/sampling/test_ascend_sampler.py`（+167）。

   也就是说：**main 现在允许 Ascend 返回 mask，并且有 NPU 侧用例**。我核对了 `4f8b6867534c` 的 `scheduler.py`，确实已无该拒绝分支。

2. 但「合入」不等于「已验证」：本期没有 Ascend 设备侧复测数据，`min_p`、cutoff ties、`top_k=1025` 回退、seeded 图重放这些边界只能按测试清单核对，不能宣称全绿。

3. 返回路径出现两条新的性能/API 方向：[#50126](https://github.com/sgl-project/sglang/pull/50126) 把 mask 改成扁平数组返回，[#50155](https://github.com/sgl-project/sglang/pull/50155) 允许只返回采样 logprob 而不带整张 mask。两者都直接关系 RL 侧拿到 mask 的成本。

## 增量索引

- [#43517](https://github.com/sgl-project/sglang/pull/43517) — 前期补漏续跟；期末 `merged`（`2026-11-12T09:52:37+08:00`）；[Feat] [NPU] sampling: sampling-mask support for Ascend。
- [#50126](https://github.com/sgl-project/sglang/pull/50126) — 本期新建；期末 `open`；[Perf] tokenizer: Return output sampling masks as flat arrays on request。
- [#50155](https://github.com/sgl-project/sglang/pull/50155) — 本期新建；期末 `open`；[Feat] sampling: Return sampling logprobs without the sampling mask。
- [#48487](https://github.com/sgl-project/sglang/pull/48487) — 前期补漏续跟；期末 `merged`（`2026-11-06T13:58:57-08:00`）；[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank。

## 重点条目

### #43517 · P0 · 前期补漏续跟（本期合入）

[[Feat] [NPU] sampling: Add sampling-mask support for Ascend](https://github.com/sgl-project/sglang/pull/43517)，`merged`。

机制（按合入 diff）：Ascend 采样后端原本只返回 token ids 与 logprobs；本条让它同时导回**实际采样的 post-filter 权重**（support 模式）或选中项（selected 模式），
经 `_build_ascend_sampling_mask_capture` 填成 `_SamplingMaskCapture`，再由公共打包路径产出 mask。同时删除 scheduler 的拒绝分支，使请求不再被挡在入口。

NPU 判断：**这是本期唯一的能力级变化**，但仍需按公共契约逐项验收：
`top_k=1025`（上限回退）、`min_p`（含 cutoff 并列）、greedy（`top_k=1`）、support/selected 两模式、混合 opted-in 行（部分请求要 mask、部分不要）、
seeded 下 mask on/off 一致，以及连续图重放。作者侧用例在下述两个文件里，但本期没有设备复测数据。

验证及限制：`test_npu_sampling_mask.py`（211 行，新增）+ `test_ascend_sampler.py`（+167 行）。合入提交自带测试，但「测试存在」≠「在 NPU 上全绿」。

关键代码：`python/sglang/srt/layers/sampler.py`、`python/sglang/srt/managers/scheduler.py`、`test/registered/npu/rl/sampling_mask/test_npu_sampling_mask.py`、`test/registered/unit/sampling/test_ascend_sampler.py`。

[事件、提交和当前文件 diff](sample-mask-delta-2026-11-13.json)。

### #50126 · P1 · 本期新建（返回序列化）

[[Perf] tokenizer: Return output sampling masks as flat arrays on request](https://github.com/sgl-project/sglang/pull/50126)，`open`。

机制：`return_sampling_mask` 现在把每个输出 token 的 support 返回成嵌套 JSON（`output_token_sampling_mask` 每 token 一个 list，对应 logprob 再一层），
本条改为按请求返回扁平数组。对 RL 侧是纯收益：大 top-k 下嵌套列表的序列化/反序列化占了相当比例的时间（上期 #41205 的 64.9s vs 11.2s 就是这类成本）。

NPU 判断：需实测。Ascend 也走 tokenizer manager 的返回路径，扁平化后必须保证 mask 行与 token 的对齐关系不变，且 `support` 长度与 `top_k` 的边界一致。

验证及限制：作者侧基准；本期未复测。

### #50155 · P1 · 本期新建（API 拆分）

[[Feat] sampling: Return sampling logprobs without the sampling mask](https://github.com/sgl-project/sglang/pull/50155)，`open`。

机制：RL trainer 需要的是「采样后概率」（温度与 top-k/top-p/min-p 之后再归一化），而 `output_token_logprobs` 是温度后的 logprob；本条允许只取 logprob 而不返回整张 mask。
对 Ascend 的现实意义：**在 mask 打包仍有风险时，这是一条不依赖打包的近似替代路径**——前提是该 logprob 与 mask 模式的 selected 概率一致。

NPU 判断：需实测。要求两者位级一致（同 seed、同 support），否则 trainer 的 off-policy 校正会漂移。

验证及限制：作者侧单测；本期未复测。

### #48487 · P2 · 前期补漏续跟（合入）

[[Fix] sampler: skip the token-id all-reduce when the TP sync group has one rank](https://github.com/sgl-project/sglang/pull/48487)，`2026-11-06T13:58:57-08:00` 合入。

机制：`Sampler._sync_token_ids_across_tp` 在 grammar-constrained batch（及开启 `SYNC_TOKEN_IDS_ACROSS_TP` 时）做 token-id all-reduce；当 sync group 只有一个 rank 时跳过。
与 mask 的交集是：跳过同步后，mask 的选中行仍需自洽（单 rank 时不存在跨 rank 分歧，但复制/广播语义要一致）。

NPU 判断：需实测。Ascend 上要确认分组判断与 HCCL all-reduce 等价，且跳过同步不影响 mask 行。

## 无实质进展与补漏边界

- [#47967](https://github.com/sgl-project/sglang/pull/47967)（XPU 确定性 renorm）、[#43047](https://github.com/sgl-project/sglang/pull/43047)（sglang-miles draft）：本期无更新。
- [#46080](https://github.com/sgl-project/sglang/pull/46080) / [#46186](https://github.com/sgl-project/sglang/pull/46186)：本期无活动。
- 背景（不纳入判断）：vLLM 侧本期有 [#66825](https://github.com/vllm-project/vllm/pull/66825)（RL：返回与 sampling mask 对齐的 logprobs）、
  [#66521](https://github.com/vllm-project/vllm/pull/66521)（CUDA top-k/top-p masking kernel）、[#66797](https://github.com/vllm-project/vllm/pull/66797)（CPU mask 打包加速），均 open。
- vllm-ascend [#17907](https://github.com/vllm-project/vllm-ascend/pull/17907) 上期已提，本期无更新。

## 检索与复核

- [本期结构化数据](sample-mask-delta-2026-11-13.json) · [本期 CSV](sample-mask-delta-2026-11-13.csv)。
- 本期改用 `gh` CLI（已认证）拉取，每条记录都有 PR 详情 + 提交 + 文件清单；合入条目以本地浅克隆的合并提交为证据。
- 未运行 GPU/NPU 或模型测试；「main 已允许 Ascend 返回 mask」是对源码的核对结论，不代表 Ascend 上数值与边界已验证。

## 扫描覆盖与源码时间说明

- 关键词检索：8 个采样短语 × `updated:>=2026-11-06` + `created:` 按日切分；去重后 sglang 1604 条（1442 PR + 162 Issue），属采样主题的候选 17 条，纳入 4 条。
- 合入侧：本地 main 浅克隆扫描窗口内 571 个提交，命中 mask 功能提交仅 `f211f410`（#43517）与 `48487` 的合入提交。
- 源码快照 `4f8b6867534c8a6632845353997555fca857dd63`（`2026-11-13T10:52+08:00`）；已在其中确认 `scheduler.py` 无 ascend 拒绝分支。
