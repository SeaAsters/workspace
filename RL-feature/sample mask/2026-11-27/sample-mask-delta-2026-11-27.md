# SGLang sampling mask 增量跟踪：2026-11-27

UTC 窗口：`[2026-11-20T06:54:42Z, 2026-11-27T01:49:24Z)`。采集截止：`2026-11-27T01:49:24Z`（第七期）。

代码基准：[官方 main `93939d5534df`](https://github.com/sgl-project/sglang/commit/93939d5534df01946af942b3fdbeed96d0061969)。sample mask 照旧只跟踪 SGLang。

本期按用户要求：**只深读 1 条重点 PR，其余一句话带过。**

## 深读：#53418 —— PD 分离下「mask 行」与「实际发出的 token」对齐

[[Fix] Match prefill sampling masks to emitted tokens](https://github.com/sgl-project/sglang/pull/53418)（`open`；改 `disaggregation/utils.py`、`scheduler.py`，测试加在 PD 与 mask 相关 4 个文件）

### 问题：三类「有人发 token，但没人产出 mask 行」的路径

PD（prefill/decode 分离）里 mask 行的归属不像单机那么显然。这个 PR 一次修三处，都指向同一件事——**mask 行必须跟着真正过 sampler 的那个 token 走**：

1. **standalone prefill 白占 wire buffer**。`disaggregation/utils.py` 的 `set_buf()` 原先对任何 `return_sampling_mask` 请求都申请可选的 mask wire buffer。但 standalone prefill 采样后直接把这一行返回给客户端，根本不做 KV handoff，两槽 wire buffer 反而让「没有真正 transfer」的请求也走跨机通道。修法：

   ```python
   if req.return_sampling_mask and self.enable_sampling_mask:
   ```

   加了 `enable_sampling_mask` 这个前提——只有真的要 handoff 才申请。

2. **fake PD decode 的合成首 token**。scheduler 现在对「fake PD decode」显式拒绝：

   ```python
   self._reject_bad_request(req, "Fake PD decode synthesizes its first token without a sampler. "
                                 "return_sampling_mask requires a real prefill result.")
   ```

   fake decode 的首 token 是合成的、没过 sampler，因此**不存在**合法的 mask 行；此前会返回错位或空行。

3. **无输出 token 的捕获**。对 tokenizer 会返回空行的场景，直接把 `req.return_sampling_mask = False`，避免出现「有 token 无 mask」或「mask 行数比 token 少」。

### 与此前修复的关系

上期合入的 [#50859](https://github.com/sgl-project/sglang/pull/50859) 解决的是「waiting queue 里被 abort 的请求仍要返回已生成 token 的 mask」；本期 #53418 解决的是另一头——PD 里**根本不该有 mask** 的路径不要伪造 mask。两条合起来才把「mask 行 ↔ token」定为硬约束。同一周还合入了 [#53153](https://github.com/sgl-project/sglang/pull/53153)（prefill 直接回答单 token 请求、不再经 decode），它进一步把「token 由 prefill 产生」变成常态，mask 的产出点也随之必须留在 prefill 侧。

### 验收含义（含 Ascend）

断言应该是「**`len(mask rows) == len(emitted tokens)`，且逐 token 对应**」，而不是「请求成功且返回了 mask 字段」。Ascend 的 PD 路径同样有 standalone prefill 与 fake decode 两种形态；NPU 上还要额外验证 abort/retract 后 buffer 回收不会把 mask 行丢掉。

## 其余条目（一句话）

- [#52270](https://github.com/sgl-project/sglang/pull/52270)（**已合入** 11-22，12 文件）：seed 与 sampling-mask 行改为**逐 step 构建**，删掉 `SamplingBatchInfo` 的批级 filter/merge——mask 行归属的结构性简化。
- [#50859](https://github.com/sgl-project/sglang/pull/50859)（**已合入** 11-20）：abort 的请求返回已生成 token 的 mask（上期新建、本期收口；发布分支 cherry-pick 已关闭）。
- [#53153](https://github.com/sgl-project/sglang/pull/53153)（**已合入** 11-26）：PD 单 token 请求由 prefill 直接回答，mask 产出点随之前移。

## 无实质进展与边界

- 上期合入的 Ascend 支持 [#43517](https://github.com/sgl-project/sglang/pull/43517) 本期无新增；**仍无 Ascend 设备侧复测**（`min_p`、cutoff ties、`top_k=1025` 回退、seeded 图重放）。
- [#47967](https://github.com/sgl-project/sglang/pull/47967)、[#43047](https://github.com/sgl-project/sglang/pull/43047)、[#46080](https://github.com/sgl-project/sglang/pull/46080)、[#46186](https://github.com/sgl-project/sglang/pull/46186)：本期无更新。
- 背景：vLLM [#69921](https://github.com/vllm-project/vllm/pull/69921)（rust 前端 mask 校验）、#68871/#68624（XPU mask dtype）未纳入判断。

## 检索与复核

- [本期结构化数据](sample-mask-delta-2026-11-27.json) · [本期 CSV](sample-mask-delta-2026-11-27.csv)。
- 深读依据：`gh pr diff 53418`（`disaggregation/utils.py` 与 `scheduler.py` 的 hunk + 新增测试）；未运行 GPU/NPU。

## 扫描覆盖

- 8 个采样短语 × `updated:>=2026-11-20` + `created:` 切分；去重 1178 条（1080 PR + 98 Issue），采样主题候选 20 条 → 纳入 4 条。
- 合入侧：本地 main 窗口内 464 个提交，命中 mask 功能提交为 #52270、#50859、#53153 的合入提交。
