# SGLang extract / return / aux hidden states 增量跟踪：2026-11-06

UTC 窗口：`[2026-10-30T02:34:03Z, 2026-11-06T06:35:52Z)`。采集截止：`2026-11-06T06:35:52Z`（第四期，接续 10/30 第三期）。

代码基准：[官方 main `10be18dcfc46`](https://github.com/sgl-project/sglang/commit/10be18dcfc466a2171b4892de214909a328ff737)（提交时间 `2026-11-05T21:55:45-08:00`）。vLLM 与 vllm-ascend 侧见 [../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-06.md](<../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-06.md>)；采样 mask 见 [../../sample mask/2026-11-06/sample-mask-delta-2026-11-06.md](<../../../sample mask/2026-11-06/sample-mask-delta-2026-11-06.md>)。

## 本期结论

纳入实质变化或技术讨论 **12 条**，分类为 本期新建 8 条、前期补漏续跟 4 条；其中 3 条在本期合入（#47445、#42248、#43902），9 条 open 或已关闭未合入。

本期的主线不是「新能力上线」，而是**正确性与契约收紧**：

1. **NPU 首次出现 aux 捕获的专门 e2e 测试**：[#48569](https://github.com/sgl-project/sglang/pull/48569) 为 `--enable-return-aux-hidden-states` 补 NPU 端到端测试。上一期只能说「NPU 图路径有 `last_hidden_states` 字段」，本期开始有测试意图，但**尚未合入**，仍不能宣称 Ascend 已验证。
2. **overlap 与 DP attention 下的 aux 状态错误集中暴露**：[#48660](https://github.com/sgl-project/sglang/issues/48660) 报告 TBO 会改变确定性 prefill 输出并**丢掉投机 aux hidden states**，#48659 修复；[#48188](https://github.com/sgl-project/sglang/pull/48188) 修 DP attention 下 chain MTP 用全零 hidden 构造 draft KV；[#47493](https://github.com/sgl-project/sglang/pull/47493) 修图补齐行与 draft 消费行不一致。这三条都是「不崩溃但数据错」的类型。
3. **返回路径的 dtype/张量契约被修**：[#47445](https://github.com/sgl-project/sglang/pull/47445) 让 hidden states 的 dtype 在 msgpack IPC 下正确往返；[#42248](https://github.com/sgl-project/sglang/pull/42248) 把 pooled hidden states 改成原生 tensor 直传。
4. **#43450 的后续重构落地**：[#43902](https://github.com/sgl-project/sglang/pull/43902) 把 deferred responses 收进对象（`scheduler_components/deferred_responses.py`），接口不变、状态机集中。

## 增量索引

- [#48569](https://github.com/sgl-project/sglang/pull/48569) — 本期新建；期末 `open`；[Test] [NPU] return-aux-hidden-states e2e。
- [#48659](https://github.com/sgl-project/sglang/pull/48659) — 本期新建；期末 `open`；[Fix] tbo: Preserve child offsets and auxiliary hidden states。
- [#48660](https://github.com/sgl-project/sglang/issues/48660) — 本期新建；期末 `open`；[Bug] TBO can change deterministic prefill outputs and drop speculative auxiliary hidden states。
- [#48188](https://github.com/sgl-project/sglang/pull/48188) — 本期新建；期末 `open`；[Fix] spec: Keep chain MTP prefill hidden states under DP attention。
- [#47493](https://github.com/sgl-project/sglang/pull/47493) — 本期新建；期末 `open`；[Fix] dp-attention: Trim MLP-sync pad rows from extend outputs before the draft extend。
- [#47445](https://github.com/sgl-project/sglang/pull/47445) — 本期新建；期末 `merged`（`2026-10-30T13:46:00-07:00`）；[Fix] ipc: hidden states 的 dtype 在 msgpack IPC 下往返。
- [#42248](https://github.com/sgl-project/sglang/pull/42248) — 前期补漏续跟；期末 `merged`（`2026-10-30T15:50:50-07:00`）；[Perf] srt: Native tensor output and prefill scoring。
- [#43902](https://github.com/sgl-project/sglang/pull/43902) — 前期补漏续跟；期末 `merged`（`2026-11-01T21:42:33-08:00`）；[Refactor] scheduler: Move deferred responses into an object。
- [#36829](https://github.com/sgl-project/sglang/issues/36829) — 前期补漏续跟；期末 `closed`；GLM-5.3-Flash EAGLE3/MTP `fused_eh_norm` 宽度与 mHC 展平 aux hidden 不匹配。
- [#47750](https://github.com/sgl-project/sglang/pull/47750) — 本期新建；期末 `open`；[Fix] score: Validate logical item and MIS hidden-state row counts。
- [#45142](https://github.com/sgl-project/sglang/pull/45142) — 前期补漏续跟；期末 `open`；[Fix] qwen4_exp: Capture DFlash aux hidden states（待 GPU 验证）。
- [#48412](https://github.com/sgl-project/sglang/pull/48412) — 本期新建；期末 `open`；[Feat] pp: Relay routed experts through pipeline-parallel stages。

## 重点条目

### #48569 · P0 · 本期新建（NPU 测试）

[[Test] [NPU] return-aux-hidden-states: Add NPU e2e test](https://github.com/sgl-project/sglang/pull/48569)，`open`。

机制：为 `--enable-return-aux-hidden-states`（上期 #43449 合入）补 NPU 端到端测试。PR 描述复述该开关的语义：**复用投机解码的 aux 捕获**，在 target 上抓取中间层 hidden states 打包返回，但不加载 draft。

NPU 判断：这是本期最有价值的一条 Ascend 证据，但仍未合入。上一期的结论是「NPU 图 runner 有字段、无验证」，本期推进到「有人写 e2e」，仍**不能**说 Ascend 已验证 aux 捕获。验收要看它是否覆盖 eager/图两条路径、是否断言层号与 `last_hidden_states` 经 lm_head 的 logits。

验证及限制：作者侧 NPU e2e；本期未复测，也未看到 CI 结果。

### #48659 / #48660 · P1 · 本期新建（TBO 下 aux 丢失）

[#48660](https://github.com/sgl-project/sglang/issues/48660) 报告 TBO（`--enable-two-batch-overlap`）的两个正确性问题：microbatch 的 `extend_start_loc` 仍是父级相对（第二个 microbatch 读错 KV 槽），以及 **TBO 路径丢掉投机 auxiliary hidden states**。[#48659](https://github.com/sgl-project/sglang/pull/48659) 修复：保留子偏移，并把 aux 捕获带过 TBO 边界。

NPU 判断：需实测。Ascend 若启用 TBO + aux 捕获，需要单独验证；这类错误表现为接受率下降或 logprob 漂移，不会崩溃。

验证及限制：作者侧以 issue 复现为准；本期未复测。

### #48188 · P1 · 本期新建（DP attention 下 chain MTP）

[[Fix] spec: Keep chain MTP prefill hidden states under DP attention](https://github.com/sgl-project/sglang/pull/48188)，`open`。

机制：DP attention 下，多层 EAGLE + chain MTP draft（Step3p5MTP、Inkling MTP、GigaChat35NextN）在第一步之后用**全零 hidden** 构造 prompt 的 draft KV。属「hidden-state 传了但传错」的静默错误。

NPU 判断：需实测。Ascend 的 DP attention 与 chain MTP 是否命中同一分支需要单独确认，接受率异常是唯一外部表征。

### #47493 · P1 · 本期新建（图补齐行 vs 消费行）

[[Fix] dp-attention: Trim MLP-sync pad rows before the draft extend](https://github.com/sgl-project/sglang/pull/47493)，`open`。

机制：DP attention + EAGLE、target 用 flashinfer A2A、draft 用 DeepEP 时，prefill CUDA graph 按最大 rank 的 DP bucket 补齐，draft extend 却按未补齐的 token 数消费，导致 draft 侧行错位。修复在进入 draft extend 前裁掉补齐行。本期 11 条非噪声事件、1 个提交，仍在迭代。

NPU 判断：需实测。Ascend 图路径同样有 bucket 补齐，必须确认 aux/draft 消费的是裁剪后的行。

### #47445 · P1 · 本期新建（合入）

[[Fix] ipc: Make multimodal payloads, hidden states and weight-update dtypes round-trip under msgpack IPC](https://github.com/sgl-project/sglang/pull/47445)，`2026-10-30T13:46:00-07:00` 合入，10 文件。

机制：`msgpack_utils.py`（+98）、`io_struct.py`（+47）、`schedule_batch.py`（+39）等一起修 dtype 往返。对 hidden states 的意义：序列化时 dtype 丢失会把 bf16/fp32 静默换型，训练侧数据与参考不一致且难定位。

NPU 判断：需实测。Ascend 侧同一 IPC 路径需验证 bf16 与 fp32 mixing 的往返。

### #42248 · P1 · 前期补漏续跟（合入）

[[Perf] srt: Native tensor output and prefill scoring optimizations](https://github.com/sgl-project/sglang/pull/42248)，`2026-10-30T15:50:50-07:00` 合入，21 文件。

机制：`pooled_hidden_states` 等输出改为原生 tensor 直传（`batch_result_processor`、`output_streamer`、`io_struct`、`engine` 同步改），并优化 prefill 评分。返回路径少了 list/tensor 转换，形状与 dtype 契约随之收紧。

NPU 判断：需实测。Ascend 输出同样走 `batch_result_processor`，需确认 tensor 直传后 D2H 与裁剪一致。

### #43902 · P2 · 前期补漏续跟（合入）

[[Refactor] scheduler: Move deferred responses into an object](https://github.com/sgl-project/sglang/pull/43902)，`2026-11-01T21:42:33-08:00` 合入，30 文件。

机制：新增 `scheduler_components/deferred_responses.py`，把 #43450 引入的 deferred responses 收进对象，并把 PD prefill inflight 队列改名为 transfer queue。接口不变、所有权与状态机集中，便于后续在 PP/PD 下复用。

NPU 判断：需实测。Ascend 复用同一 scheduler 组件，需验证对象化后 resume/abort 与图重放 buffer 生命周期。

### #36829 · P1 · 前期补漏续跟（关闭）

GLM-5.3-Flash（glm5_next、mHC）EAGLE3/MTP：`fused_eh_norm` 宽度与 mHC 展平后的 aux hidden 不匹配，自行绕过后接受率约 1.0。这是「aux 捕获层/宽度与 draft 期望不一致」的典型现象——**不是 shape 崩溃，而是接受率异常**。本期关闭说明结论已收敛，但关闭本身不等于代码已修。NPU 需核对 mHC 展平顺序与 `fused_eh_norm` 宽度。

### #47750 / #45142 / #48412 · P2 · 其他

- [#47750](https://github.com/sgl-project/sglang/pull/47750)：为评分路径加显式行数校验（逻辑 item 数 vs MIS hidden-state 行数），把静默错位变成显式报错。
- [#45142](https://github.com/sgl-project/sglang/pull/45142)：qwen4_exp 的 DFlash aux 捕获，作者明示 **CPU 单测通过、GPU 未验证**，等 GB300 验收数字。
- [#48412](https://github.com/sgl-project/sglang/pull/48412)：把 routed experts 跨 PP relay；与 aux hidden states 的 PP relay 是同一类「每段只抓自己层、需合并」问题。

## 无实质进展与补漏边界

- [#46320](https://github.com/sgl-project/sglang/pull/46320)（EAGLE3 PP relay aux hidden states）与 [#47047](https://github.com/sgl-project/sglang/pull/47047)（GLM5_next DFlash aux gather）本期无活动，不计入。
- [#47200](https://github.com/sgl-project/sglang/pull/47200)（speculative CUDA fast paths，11-05 合入）触及 `logits_processor.py` 与 Inkling aux 测试，但主体是 CUDA runtime 快路径，仅作相邻背景。
- 采样 mask 侧本期有两项 P0 合入，见 [sample mask 2026-11-06](<../../../sample mask/2026-11-06/sample-mask-delta-2026-11-06.md>)。

## 检索与复核

- [本期结构化数据](sglang-extract-hidden-states-delta-2026-11-06.json) · [本期 CSV](sglang-extract-hidden-states-delta-2026-11-06.csv)。
- 本期未运行 GPU/NPU 或模型测试；作者性能与接受率数字均为作者报告。

## 扫描覆盖与源码时间说明

- 关键词检索：17 个短语 × `updated:>=2026-10-30` + RFC 标签 + 按日 `created:` 切分，去重后 sglang 命中 1598 条（1463 PR + 135 Issue）。
- 合入侧：本地 main 浅克隆扫描窗口内 537 个提交，按关键词与文件路径双重筛选；合入条目直接用提交本身作为证据（省掉 PR 详情请求）。
- 基线：156 条历史 tracked 条目做当前状态复核，其中 9 条本期有更新；103 条关键词候选逐条判定，纳入 12 条。
- 源码快照 `10be18dcfc466a2171b4892de214909a328ff737`（`2026-11-05T21:55:45-08:00`），PR 事件截止 `2026-11-06T06:35:52Z`。
