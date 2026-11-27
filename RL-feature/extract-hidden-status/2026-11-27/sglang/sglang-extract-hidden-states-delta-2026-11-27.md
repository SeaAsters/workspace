# SGLang extract / return / aux hidden states 增量跟踪：2026-11-27

UTC 窗口：`[2026-11-20T06:54:42Z, 2026-11-27T01:49:24Z)`。采集截止：`2026-11-27T01:49:24Z`（第七期）。

代码基准：[官方 main `93939d5534df`](https://github.com/sgl-project/sglang/commit/93939d5534df01946af942b3fdbeed96d0061969)（`2026-11-27T09:37+08:00`）。vLLM / vllm-ascend 见 [../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-27.md](<../vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-11-27.md>)；采样 mask 见 [../../../sample mask/2026-11-27/sample-mask-delta-2026-11-27.md](<../../../sample mask/2026-11-27/sample-mask-delta-2026-11-27.md>)。

本期按用户要求改变写法：**只对 1 条重点 PR 做代码级展开，其余条目一句话带过。**

## 深读一：#52507 —— 可断图 prefill 重放把 aux hidden states 截断成 SP 行数

[[Fix] cuda-graph: Keep full aux hidden states in breakable prefill replays with SP](https://github.com/sgl-project/sglang/pull/52507)（`open`，改动 1 个实现文件 + 1 个新单测）

### 问题

`python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` 里，`_output_rows()` 负责回答「这次 replay 的 body 在 dim 0 上真正产了多少行」，供 `_slice_output()` 与 `_copy_output_to_buffer()` 裁剪输出。改前它有两条分支：

```python
if isinstance(output, (list, tuple)) and output:
    return min(self._output_rows(o, cap) for o in output if o is not None)
return cap
```

对 list/tuple 输出，它把所有输出压成**一个标量**——所有输出里最小的行数。SEQUENCE PARALLEL（SP）下 `hidden_states` 被分片、行数本来就少；而 aux hidden states 按契约是 **full-token** 的、行数多。于是 `min()` 取到 SP 的行数，`_slice_output()` 用同一个 `num_tokens` 去切 `aux_hidden_states`——**aux 被静默截断**。不报错、不崩，只是交给 drafter 的行数变少。

### 修法

- `_output_rows()` 改为返回**逐输出**的行数：list/tuple 分支返回 `[None if o is None else self._output_rows(o, cap) for o in output]`，保留结构。
- 新增 `_rows_per_item(num_tokens, count)`：`num_tokens` 是 list 就按项返回，否则广播成 `[num_tokens] * count`。
- `_slice_output()` 与 `_copy_output_to_buffer()` 都用它按项取行数（tuple 仍返回 tuple）。

### 证据与验收含义

新单测 `test_breakable_cuda_graph_backend.py` 构造 `hidden=(2,4)`、`aux=(8,6)` 两个形状不同的输出，`capture_one` 后 `replay`，断言两者**各自保持原形状**：

```python
torch.testing.assert_close(replayed[0], hidden)
torch.testing.assert_close(replayed[1], aux)
```

**NPU 判断：这是本期最值得在 Ascend 上直接复刻的断言。** Ascend 的图重放同样会在 SP/CP 下裁剪输出；只要 aux 与 hidden_states 行数不同，就会被最小值截断。验收不该是「replay 不报错」，而是「aux 行数 == full-token 行数，且逐行数值与 eager 一致」。

## 其余条目（一句话）

- [#53074](https://github.com/sgl-project/sglang/pull/53074)（新）：TBO 重设子 extend offset + 在 overlap 层补 aux 捕获，收口上期 #48660 的两个问题；open。
- [#47047](https://github.com/sgl-project/sglang/pull/47047)（**已合入**，11-21）：GLM5_next 的 DFlash aux gather 改用 `is_moe_input_scattered_across_dp_ranks()`，并补 deferred-allreduce 单测——跟踪三期收口。
- [#51566](https://github.com/sgl-project/sglang/pull/51566)（open）：prefill aux 收进单个 runner 缓冲，与 #52507 同一片区域（缓冲所有权/裁剪）。
- [#51475](https://github.com/sgl-project/sglang/pull/51475)（open）：`/generate` 的 `return_hidden_states: "last"` 回归**仍未修完**（0.5.22+ 依旧 400）。
- [#51926](https://github.com/sgl-project/sglang/pull/51926)（已合入 11-26）：Inkling-2-Mini 新模型，顺带改了 `spec_aux_hidden_state.py` 的 forward 形态，属新模型接入 aux 捕获面。

## 无实质进展与边界

- [#46320](https://github.com/sgl-project/sglang/pull/46320)（EAGLE3 PP relay）、[#48569](https://github.com/sgl-project/sglang/pull/48569)（NPU aux e2e）、[#48659](https://github.com/sgl-project/sglang/pull/48659)、[#45142](https://github.com/sgl-project/sglang/pull/45142)：本期无活动。
- [#51074](https://github.com/sgl-project/sglang/pull/51074) 类「删废弃 flag」的提交已核对不涉及 hidden/aux 字段。

## 检索与复核

- [本期结构化数据](sglang-extract-hidden-states-delta-2026-11-27.json) · [本期 CSV](sglang-extract-hidden-states-delta-2026-11-27.csv)。
- 深读依据：`gh pr diff 52507` 全文 + 新单测源码；其余条目用 PR 状态/提交/时间线。未运行 GPU/NPU。

## 扫描覆盖

- 关键词 17 短语 × `updated:>=2026-11-20` + RFC 标签 + `created:` 按日切分，去重 1178 条（1080 PR + 98 Issue），hidden 候选 73 条 → 纳入 6 条（3 新/3 补漏）。
- 合入侧扫本地 main 浅克隆窗口内 464 个提交；源码快照 `93939d5534df01946af942b3fdbeed96d0061969`。
