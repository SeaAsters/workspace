# SGLang sampling mask 增量跟踪：2026-11-27

UTC 窗口：`[2026-11-20T06:54:42Z, 2026-11-27T01:49:24Z)`。采集截止：`2026-11-27T01:49:24Z`（第七期）。

代码基准：[官方 main `93939d5534df`](https://github.com/sgl-project/sglang/commit/93939d5534df01946af942b3fdbeed96d0061969)。sample mask 照旧只跟踪 SGLang。

本期按要求：**主深读 #52270（每步重建 seed 与 mask 行）**，#53418 作为次要项，其余一句话带过。

## 深读：#52270 —— 采样状态从「建批时构建」改为「每步构建」

[[Refactor] sampling: Build seeds and sampling-mask rows per step; drop SamplingBatchInfo filter/merge](https://github.com/sgl-project/sglang/pull/52270)，`2026-11-22T03:43-08:00` 合入（12 文件，含 6 个测试文件）

### 改前：mask 行索引靠「建批 + filter/merge 维护」

`SamplingBatchInfo.from_schedule_batch(batch, vocab_size)` 在建批时一次性算出整套 per-row 采样状态：`sampling_seed` 张量、`rows_with_seed` 布尔列表、`return_sampling_masks` 布尔列表，以及派生出的 `sampling_mask_batch_indices`（哪些 batch 行要 mask）与 support 行索引。此后只要 batch 被裁剪或合并，就必须同步维护这些数组：

```python
self.sampling_info.filter_batch(keep_indices, keep_indices_device)   # filter_batch 里
self.sampling_info.merge_batch(other.sampling_info, self_len=..., other_len=...)  # merge_batch 里
```

问题在于这是一份**与 reqs 并行、需要手工保持同步的副本**：retraction、PD、chunked prefill、chunked merge 任意一处漏维护，mask 行就会指向错的行——不报错，只是 support/logprob 落到了别的请求上。被删掉的 4 个单测（`TestSamplingMaskBatchIndices` 整类）正好覆盖这类偏移：「filter 重建行索引」「merge 给 rhs 行加偏移」「filter 去掉最后一个 opted-in 行后 merge 恢复捕获」。

### 改后：每步从当前 reqs 重算

- `SamplingBatchInfo` 在建批时只建空壳：`SamplingBatchInfo(vocab_size=..., device=...)`（`prepare_for_extend`、`prepare_for_built`、`_build_hisparse_decode_batch`、`prepare_for_idle` 全部改掉）。
- 每次 launch 时调用 `set_seeds_and_mask_rows(reqs)`：`scheduler.run_batch` 里紧跟 `set_plugin_state`；`prepare_for_idle` 传 `[]`。代码搜索确认**没有 NPU 专用调用点**，Ascend 走的就是这条公共路径。
- 删掉 `rows_with_seed`、`return_sampling_masks`、`_make_sampling_mask_batch_indices`，以及 `filter_batch`/`merge_batch` 对采样数组的维护——**没有副本，就没有不同步**。
- 未播种请求的 filler seed 从「每建批重抽」改为记在请求上：`req.filler_sampling_seed`，同请求跨步复用。

### 新增断言（比旧测试强）

```python
def test_mask_rows(self):
    """Mask rows index the batch; support rows index the mask rows."""
    info.set_seeds_and_mask_rows(reqs)          # 4 个请求，3 个要 mask
    self.assertEqual(info.sampling_mask_batch_indices.tolist(), [1, 2, 3])
    self.assertEqual(info.sampling_support_logprobs_capture_indices.tolist(), [1, 2])
    self.assertEqual(info.sampling_mask_top_ks, [1, 7, 50, 3])
    ...
    info.set_seeds_and_mask_rows(reqs[:2])      # 收缩后索引随之收缩
    self.assertEqual(info.sampling_mask_batch_indices.tolist(), [1])
    self.assertIsNone(info.sampling_support_logprobs_capture_indices)
```

```python
def test_unseeded_row_keeps_its_filler_seed_across_steps(self):
    """An unseeded request samples with one random seed in every seeded step,
    wherever its row is, and unseeded once no seeded row is left."""
```

第二条是本期最容易被忽略、但对 RL 很关键的一条：当批里存在「有 seed 的行」时，未播种行会被塞一个 filler seed 一起走 seeded sampler；如果这个 seed **每步重抽**，那么同一请求的多步采样在不同 step 上用的是不同随机流，mask/logprob 重放就对不上。改成按请求记忆后，行位置变化（`[seeded, a]` → `[b, a, seeded]`）不影响 `a` 的 seed。

### NPU 判断

Ascend 的 `_build_ascend_sampling_mask_capture` 读的正是 `sampling_info.sampling_mask_batch_indices`，而它们现在**每步在 `run_batch` 里重建**。两点需要实测：

1. **读取时机**：Ascend 的 mask capture 必须在同一 step、`set_seeds_and_mask_rows` 之后读取；代码搜索显示没有 NPU 专用调用点，理论上继承公共顺序，但一旦引入 NPU 侧独立的 batch/launch 路径，就会读到空或上一步的索引。
2. **不要缓存**：任何缓存 per-row 采样参数（种子、mask 行索引、`sampling_mask_top_ks`）的图/缓冲路径，都必须每步刷新，否则跨步复用旧索引。

代价是每步一次张量分配 + 索引构建（不再只在建批时做一次）；PR 未给基准数据，属于「换正确性」的取舍。

## 次要：#53418 —— PD 下 mask 行必须跟着「真正过 sampler 的 token」

[[Fix] Match prefill sampling masks to emitted tokens](https://github.com/sgl-project/sglang/pull/53418)（`open`）修三处「有人发 token、没人产 mask」：`disaggregation/utils.py` 只在 `self.enable_sampling_mask` 时才申请可选 wire buffer（standalone prefill 直接回客户端，不需要）；scheduler 显式拒绝 fake PD decode（`return_sampling_mask requires a real prefill result.`，其首 token 是合成、不过 sampler）；无输出 token 的捕获直接把 `req.return_sampling_mask = False`。

与 #52270 互补：#52270 保证「索引每步与 reqs 对齐」，#53418 保证「不该有 mask 的路径不要造 mask」。验收断言仍应是 **mask 行数 == 发出 token 数，且逐 token 对应**。

## 其余条目（一句话）

- [#50859](https://github.com/sgl-project/sglang/pull/50859)（**已合入** 11-20）：waiting queue 里被 abort 的请求也返回已生成 token 的 mask；发布分支 cherry-pick 本期关闭。
- [#53153](https://github.com/sgl-project/sglang/pull/53153)（**已合入** 11-26）：PD 单 token 请求由 prefill 直接回答、不再经 decode——mask 产出点随之前移到 prefill 侧。

## 无实质进展与边界

- Ascend 支持 [#43517](https://github.com/sgl-project/sglang/pull/43517)（上期合入）本期无新增，**仍无 Ascend 设备侧复测**：`min_p`、cutoff ties、`top_k=1025` 回退、seeded 图重放四项待验。
- [#47967](https://github.com/sgl-project/sglang/pull/47967)、[#43047](https://github.com/sgl-project/sglang/pull/43047)、[#46080](https://github.com/sgl-project/sglang/pull/46080)、[#46186](https://github.com/sgl-project/sglang/pull/46186)：本期无更新。
- 背景：vLLM [#69921](https://github.com/vllm-project/vllm/pull/69921)（rust 前端 mask 校验）、#68871/#68624（XPU mask dtype）未纳入判断。

## 检索与复核

- [本期结构化数据](sample-mask-delta-2026-11-27.json) · [本期 CSV](sample-mask-delta-2026-11-27.csv)。
- #52270 深读依据：`gh pr diff 52270`（`sampling_batch_info.py` 的 `from_schedule_batch`→`set_seeds_and_mask_rows` 改写、`filter_batch`/`merge_batch` 删除、新增两个单测）+ 代码搜索确认调用点；#53418 依据 `gh pr diff 53418` 的 scheduler/`disaggregation/utils.py` hunk。未运行 GPU/NPU。

## 扫描覆盖

- 8 个采样短语 × `updated:>=2026-11-20` + `created:` 切分；去重 1178 条（1080 PR + 98 Issue），采样主题候选 20 条 → 纳入 4 条。
- 合入侧：本地 main 窗口内 464 个提交，mask 功能提交为 #52270、#50859、#53153 的合入提交。
