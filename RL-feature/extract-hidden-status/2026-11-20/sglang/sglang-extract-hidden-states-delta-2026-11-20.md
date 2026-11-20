# SGLang extract / return / aux hidden states 增量跟踪：2026-11-20

UTC 窗口：`[2026-11-13T04:40:41Z, 2026-11-20T06:54:42Z)`。采集截止：`2026-11-20T06:54:42Z`（第六期，接续 11/13 第五期）。

代码基准：[官方 main `9866f0a0c798`](https://github.com/sgl-project/sglang/commit/9866f0a0c798ffb85da1122c09f8f135f66085d7)（`2026-11-19T22:30-08:00`）。vLLM / vllm-ascend 侧见 [../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-20.md](<../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-20.md>)；采样 mask 见 [../../../sample mask/2026-11-20/sample-mask-delta-2026-11-20.md](<../../../sample mask/2026-11-20/sample-mask-delta-2026-11-20.md>)。

## 本期结论

纳入 **7 条**（本期新建 4、前期补漏续跟 3），其中 2 条合入（#50555、#50039）。本期的主线是**公共接口的回归与契约收紧**，而不是新能力：

1. **`return_hidden_states` 在 `/generate` 上有一个版本级回归**：[#50879](https://github.com/sgl-project/sglang/issues/50879) 报告自 **0.5.22** 起 `"return_hidden_states": "last"` 直接 400（0.5.21 可用）。根因是 api.v1 schema 把该字段声明成 `BoolOrList`，契约检查因此拒绝字符串。[#51475](https://github.com/sgl-project/sglang/pull/51475) 一次修三个问题：`"last"` 被拒、响应形状、单 prompt 时该返回 list。**在修复合入前，按文档取 last hidden 的调用方会直接失败。**
2. **契约收紧**：[#50555](https://github.com/sgl-project/sglang/pull/50555)（`2026-11-16T09:43-08:00` 合入，10 文件）让 rust-server 在请求模式超过服务端上限时直接拒绝，而不是按较低模式静默返回。这与 tokenizer 侧已有的模式检查一致（`get_server_return_hidden_states_mode`）。
3. **图内 aux buffer 所有权**：[#51566](https://github.com/sgl-project/sglang/pull/51566) 把 prefill 图捕获的 aux hidden states 收进单个 runner 持有的缓冲，避免每尺寸累积常驻——与上游 vLLM 的 #63427/#63406 是同一类问题。
4. **bounded replay × aux 生命周期**：[#50039](https://github.com/sgl-project/sglang/pull/50039)（`2026-11-13T23:28+08:00` 合入）修好 compact DSpark 输出被后续步骤覆盖的问题。

## 增量索引

- [#51475](https://github.com/sgl-project/sglang/pull/51475) — 本期新建；期末 `open`；[Fix] api: fix `return_hidden_states` on `/generate`（"last"、响应形状、单 prompt list）。
- [#50879](https://github.com/sgl-project/sglang/issues/50879) — 本期新建；期末 `open`；[Bug] `/generate` 自 0.5.22 起拒绝 `"last"`。
- [#51566](https://github.com/sgl-project/sglang/pull/51566) — 本期新建；期末 `open`；[Fix] cuda-graph: pack prefill aux hidden states into one runner-owned buffer。
- [#50555](https://github.com/sgl-project/sglang/pull/50555) — 本期新建；期末 `merged`（`2026-11-16T09:43:33-08:00`）；[Fix] rust-server: refuse return_hidden_states above the server's mode。
- [#50039](https://github.com/sgl-project/sglang/pull/50039) — 前期补漏续跟；期末 `merged`（`2026-11-13T23:28:02+08:00`）；[Fix] deepseek-v4.1: preserve compact DSpark outputs。
- [#47047](https://github.com/sgl-project/sglang/pull/47047) — 前期补漏续跟；期末 `open`（11-18 更新）；[Fix] glm5_next: gather DFlash aux hidden states。
- [#48188](https://github.com/sgl-project/sglang/pull/48188) — 前期补漏续跟；期末 `open`（11-14 更新）；[Fix] spec: chain MTP prefill hidden states under DP attention。

## 重点条目

### #50879 / #51475 · P0 · 本期新建（公共 API 回归）

[#50879](https://github.com/sgl-project/sglang/issues/50879)：`/generate` 自 **0.5.22** 起对 `"return_hidden_states": "last"` 返回 400，0.5.21 正常。[#51475](https://github.com/sgl-project/sglang/pull/51475)（13 个提交、32 文件）一次修三件事：

1. api.v1 schema 把字段声明为 `BoolOrList`，契约检查拒绝 `"last"`；
2. 响应形状与文档不一致；
3. 单个 prompt 时应返回 list 而非裸张量。

NPU 判断：**需实测**。Ascend 走同一 `/generate` 与 tokenizer 契约检查，修复后要复核 `"last"` 在 NPU 上返回的形状与 dtype；修复合入前，Ascend 用户按文档调用同样 400。

验证及限制：作者侧测试；本期未复测。这是**接口层**回归，与数值无关，优先看形状/类型断言。

### #50555 · P1 · 本期新建（合入，契约收紧）

[[Fix] rust-server: Refuse return_hidden_states above the server's mode](https://github.com/sgl-project/sglang/pull/50555)，`2026-11-16T09:43-08:00` 合入，10 文件。

机制：请求的 hidden 模式超过服务端上限时直接拒绝。属「宁可报错也不要悄悄降级」——静默降级会让 RL 侧以为拿到 full，实际是 last。

NPU 判断：需实测。Ascend 若走 rust-server 或同一模式检查，要确认拒绝语义一致（400，而不是返回空 hidden）。

### #51566 · P1 · 本期新建（图内 aux buffer）

[[Fix] cuda-graph: Pack prefill aux hidden states into one runner-owned buffer](https://github.com/sgl-project/sglang/pull/51566)，`open`。

机制：prefill 图捕获的 aux hidden states 收进单个 runner 持有的缓冲，而不是每次捕获新申请/保留。直接关系两件事：图捕获尺寸增加时的 aux 常驻占用、以及 aux 视图的生命周期（谁持有、何时被下一次 replay 覆盖）。

NPU 判断：需实测。Ascend 图路径同样需要「runner 持有 + 按 `raw_num_token` 裁剪」；否则每次重捕获都会累积常驻，而 NPU 显存更紧。

### #50039 · P1 · 前期补漏续跟（合入）

[[Fix] deepseek-v4.1: Preserve compact DSpark outputs with CP bounded replay](https://github.com/sgl-project/sglang/pull/50039)，`2026-11-13T23:28+08:00` 合入，2 文件。

机制：Prefill CP + DSpark + decoder SWA bounded replay 组合下，compact 输出（含 DSpark 需要的 hidden 片段）会被后续步骤覆盖或截断；修复保证保留。属「bounded replay 与 aux 输出生命周期」交叉问题，上期新增、本期合入。

NPU 判断：需实测。Ascend 若支持该组合需要同样的保留语义，否则 DSpark 读到被覆盖的 buffer——不报错，只是数据错。

### #47047 / #48188 · P2/P1 · 持续跟进

- [#47047](https://github.com/sgl-project/sglang/pull/47047)：GLM5_next 的 DFlash aux gather 条件改为 `is_moe_input_scattered_across_dp_ranks()`；本期新增 2 个提交（11-18），仍 open。NPU 需确认 Ascend MoE 切分在该谓词覆盖内。
- [#48188](https://github.com/sgl-project/sglang/pull/48188)：DP attention 下 chain MTP 用全零 hidden 构造 draft KV；本期更新到 11-14，仍 open。外部表征是接受率异常。

## 无实质进展与补漏边界

- [#51074](https://github.com/sgl-project/sglang/pull/51074)（11-18 合入，删除废弃 flag/环境变量）经核对**未触及** hidden/aux 相关字段，不计入。
- [#48569](https://github.com/sgl-project/sglang/pull/48569)（NPU return-aux e2e）、[#48659](https://github.com/sgl-project/sglang/pull/48659)（TBO aux）、[#45142](https://github.com/sgl-project/sglang/pull/45142)、[#46320](https://github.com/sgl-project/sglang/pull/46320)：本期无更新。
- 采样 mask 侧本期有 abort 路径与扁平数组两条，见 [sample mask 2026-11-20](<../../../sample mask/2026-11-20/sample-mask-delta-2026-11-20.md>)。

## 检索与复核

- [本期结构化数据](sglang-extract-hidden-states-delta-2026-11-20.json) · [本期 CSV](sglang-extract-hidden-states-delta-2026-11-20.csv)。
- 使用已认证 `gh` CLI 拉取 PR 详情/时间线/提交/文件；合入条目以本地浅克隆的合并提交为证据。
- 未运行 GPU/NPU 或模型测试。

## 扫描覆盖与源码时间说明

- 关键词检索：17 个短语 × `updated:>=2026-11-13` + RFC 标签 + `created:` 按日切分；去重后 sglang 1679 条（1505 PR + 174 Issue），hidden 候选 66 条，纳入 7 条。
- 合入侧：本地 main 浅克隆扫描窗口内 572 个提交。
- 源码快照 `9866f0a0c798ffb85da1122c09f8f135f66085d7`（`2026-11-19T22:30-08:00`）。
