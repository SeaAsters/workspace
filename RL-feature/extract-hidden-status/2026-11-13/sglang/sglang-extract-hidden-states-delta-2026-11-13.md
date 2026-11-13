# SGLang extract / return / aux hidden states 增量跟踪：2026-11-13

UTC 窗口：`[2026-11-06T06:35:52Z, 2026-11-13T04:40:41Z)`。采集截止：`2026-11-13T04:40:41Z`（第五期，接续 11/06 第四期）。

代码基准：[官方 main `4f8b6867534c`](https://github.com/sgl-project/sglang/commit/4f8b6867534c8a6632845353997555fca857dd63)（`2026-11-13T10:52+08:00`）。vLLM / vllm-ascend 侧见 [../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-13.md](<../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-13.md>)；采样 mask 见 [../../../sample mask/2026-11-13/sample-mask-delta-2026-11-13.md](<../../../sample mask/2026-11-13/sample-mask-delta-2026-11-13.md>)。

## 本期结论

纳入 **8 条**（本期新建 5、前期补漏续跟 3），其中 4 条合入 main（#49664、#47493、#47750、#48412）。本期没有新的「接口」，变化集中在三件事：

1. **draft 侧行/槽位映射在动**：[#49664](https://github.com/sgl-project/sglang/pull/49664) 把滑窗 MLA draft 的 KV 改成每请求环形缓冲（20 文件），[#47493](https://github.com/sgl-project/sglang/pull/47493) 修好「图补齐行 vs draft 消费行」错位。两条都会改变 draft 读取 target hidden 的行映射。
2. **PP relay 有了可对照的实现**：[#48412](https://github.com/sgl-project/sglang/pull/48412) 把 routed experts 跨 PP 阶段 relay 并合并——与 aux hidden states 的 relay 是同一类「每段只抓自己层」问题，Ascend 侧可直接对照其缓冲/校验约定。
3. **静默失败案例被记录**：[#38589](https://github.com/sgl-project/sglang/issues/38589) 关闭，但结论是「Qwen4Exp 根本产不出 DFlash aux hidden states，而 `set_dflash_layers_to_capture` 调用看起来成功」。这类失败不会报错，只会让 drafter 起不来或接受率异常，是验收时必须用「aux 非空 + 层数正确」两条断言覆盖的场景。

另外 [#50039](https://github.com/sgl-project/sglang/pull/50039) 把「bounded replay 与 aux 输出生命周期」的交叉问题摆上台面：compact 输出可能被后续步骤覆盖。

## 增量索引

- [#49664](https://github.com/sgl-project/sglang/pull/49664) — 本期新建；期末 `merged`（`2026-11-11T19:09:14-08:00`）；[Perf] dflash: sliding-window MLA draft KV in a per-request ring。
- [#47493](https://github.com/sgl-project/sglang/pull/47493) — 本期新建；期末 `merged`（`2026-11-09T15:38:10+08:00`）；[Fix] dp-attention: trim MLP-sync pad rows before draft extend。
- [#47750](https://github.com/sgl-project/sglang/pull/47750) — 本期新建；期末 `merged`（`2026-11-09T12:12:58-08:00`）；[Fix] score: validate MIS hidden-state row counts。
- [#48412](https://github.com/sgl-project/sglang/pull/48412) — 本期新建；期末 `merged`（`2026-11-12T19:48:45+08:00`）；[Feat] pp: relay routed experts through PP stages。
- [#38589](https://github.com/sgl-project/sglang/issues/38589) — 前期补漏续跟；期末 `closed`（未修）；Qwen4Exp 无法产出 DFlash aux hidden states。
- [#48188](https://github.com/sgl-project/sglang/pull/48188) — 前期补漏续跟；期末 `open`；[Fix] spec: chain MTP prefill hidden states under DP attention。
- [#47047](https://github.com/sgl-project/sglang/pull/47047) — 前期补漏续跟；期末 `open`；[Fix] glm5_next: gather DFlash aux hidden states for every scattered MoE backend。
- [#50039](https://github.com/sgl-project/sglang/pull/50039) — 本期新建；期末 `open`；[Fix] deepseek-v4.1: preserve compact DSpark outputs with CP bounded replay。

## 重点条目

### #49664 · P1 · 本期新建（合入）

[[Perf] dflash: Keep a sliding-window MLA draft's KV in a per-request ring](https://github.com/sgl-project/sglang/pull/49664)，`2026-11-11T19:09-08:00` 合入，20 文件。

机制：滑窗 MLA draft 的 KV 从共享池改为每请求环形缓冲，减少 draft KV 的碎片与占用。对 aux 捕获的关联：draft KV 布局变化会改变 draft 侧读取 target hidden 的行映射与槽位索引。

NPU 判断：需实测。Ascend 的 draft KV 分配若仍走共享池，环形缓冲语义需重新对照；draft 读取 aux 行的映射必须同步改，否则会读错行。

### #47493 · P1 · 本期新建（合入）

[[Fix] dp-attention: Trim MLP-sync pad rows from extend outputs before the draft extend](https://github.com/sgl-project/sglang/pull/47493)，`2026-11-09T15:38+08:00` 合入，2 文件。

机制：DP attention + EAGLE、target 用 flashinfer A2A、draft 用 DeepEP 时，prefill CUDA graph 按最大 rank 的 DP bucket 补齐，draft extend 却按未补齐的 token 数消费 → draft 侧行错位。修复在进入 draft extend 前裁掉补齐行。

NPU 判断：需实测，且这条最容易在 Ascend 复现：只要图路径有 bucket 补齐，就要验证 aux/draft 消费的是裁剪后的行。

### #47750 · P2 · 本期新建（合入）

[[Fix] score: Validate logical item and MIS hidden-state row counts](https://github.com/sgl-project/sglang/pull/47750)，`2026-11-09T12:12-08:00` 合入。

机制：为评分路径加显式断言：逻辑 item 数与 MIS hidden-state 行数不一致时早失败，而不是让错位数据进 logprob。

NPU 判断：需实测。Ascend 上 CP/SP shard 后的行数要与断言一致，否则会从「静默错」变成「误报」。

### #48412 · P1 · 本期新建（合入）

[[Feat] pp: Relay routed experts through pipeline-parallel stages](https://github.com/sgl-project/sglang/pull/48412)，`2026-11-12T19:48+08:00` 合入，14 文件。

机制：`--enable-return-routed-experts` 在 PP>1 时各段只抓自己层，本条把 routed experts 跨阶段 relay 并按请求合并。与 aux hidden states 的 PP relay 同构，是本期最有参考价值的一条。

NPU 判断：需实测。Ascend 的 `pp_transport_aux_hidden_states_` 与这条 relay 的缓冲/校验方式是否兼容，决定 NPU 侧能否直接复用。

### #38589 · P1 · 前期补漏续跟（关闭，未修）

Qwen4Exp（Qwen3.8-Flash-Next）**无法产出 DFlash aux hidden states**：`set_dflash_layers_to_capture` 继承自基类、调用「看起来成功」，但捕获从未发生，因此该 target 上任何 DFlash/DSpark drafter 都无法服务。本期关闭，对应 PR #45142 仍停在「待 GPU 验证」。

NPU 判断：缺失。这是「捕获开关成功但不出数据」的静默失败形态；Ascend 上同类模型必须用两条断言验证：`aux_hidden_states` 非空、捕获层数等于配置层数。

### #48188 · P1 · 前期补漏续跟（open）

[[Fix] spec: Keep chain MTP prefill hidden states under DP attention](https://github.com/sgl-project/sglang/pull/48188)：DP attention 下多层 EAGLE + chain MTP draft 在第一步后用**全零 hidden** 构造 prompt 的 draft KV。本期更新到 11-12，仍 open。NPU 需验证 Ascend 的 DP attention + chain MTP 是否命中同一分支；外部表征是接受率异常。

### #47047 · P2 · 前期补漏续跟（open，无新提交）

[[Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE all-to-all backend](https://github.com/sgl-project/sglang/pull/47047)：gather 条件从枚举 deepep/mooncake/deepep_v2 改成 `is_moe_input_scattered_across_dp_ranks()`。本期仅状态更新。NPU 需确认 Ascend 的 MoE 切分在该谓词覆盖内，否则 aux 行数与 attention 边界行不一致。

### #50039 · P1 · 本期新建（open）

[[Fix] deepseek-v4.1: Preserve compact DSpark outputs with CP bounded replay](https://github.com/sgl-project/sglang/pull/50039)：接续 #47406，在 Prefill CP + DSpark + decoder SWA bounded replay 组合下保证 compact 输出（含 DSpark 需要的 hidden 片段）不被后续步骤覆盖/截断。属「bounded replay × aux 生命周期」交叉问题；NPU 若支持该组合需同样的保留语义。

## 无实质进展与补漏边界

- [#48569](https://github.com/sgl-project/sglang/pull/48569)（NPU return-aux e2e）与 [#48659](https://github.com/sgl-project/sglang/pull/48659)（TBO aux）本期无更新，仍 open。
- [#45142](https://github.com/sgl-project/sglang/pull/45142)（qwen4_exp aux）无更新，仍待 GPU 验证；其 issue #38589 已关闭未修。
- [#46320](https://github.com/sgl-project/sglang/pull/46320)（EAGLE3 PP relay）本期无活动。
- 采样 mask 侧的 Ascend 合入（#43517）见 [sample mask 2026-11-13](<../../../sample mask/2026-11-13/sample-mask-delta-2026-11-13.md>)。

## 检索与复核

- [本期结构化数据](sglang-extract-hidden-states-delta-2026-11-13.json) · [本期 CSV](sglang-extract-hidden-states-delta-2026-11-13.csv)。
- 本期改用已认证的 `gh` CLI 拉取（PR 详情 + 时间线 + 提交 + 文件），合入条目以本地浅克隆的合并提交为证据。
- 未运行 GPU/NPU 或模型测试；作者数据均为作者报告。

## 扫描覆盖与源码时间说明

- 关键词检索：17 个短语 × `updated:>=2026-11-06` + RFC 标签 + `created:` 按日切分；去重后 sglang 1604 条（1442 PR + 162 Issue）；属 hidden 主题的候选 66 条，纳入 8 条。
- 合入侧：本地 main 浅克隆扫描窗口内 571 个提交，按关键词与文件路径双重筛选。
- 源码快照 `4f8b6867534c8a6632845353997555fca857dd63`（`2026-11-13T10:52+08:00`）。
