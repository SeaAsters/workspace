# SGLang sampling mask 增量跟踪：2026-11-20

UTC 窗口：`[2026-11-13T04:40:41Z, 2026-11-20T06:54:42Z)`。采集截止：`2026-11-20T06:54:42Z`（第六期，接续 11/13 第五期）。

代码基准：[官方 main `9866f0a0c798`](https://github.com/sgl-project/sglang/commit/9866f0a0c798ffb85da1122c09f8f135f66085d7)。sample mask 照旧只跟踪 SGLang；vLLM 侧 mask 条目仅作背景。

## 本期结论

纳入 **4 条**（本期新建 2、前期补漏续跟 2），其中 1 条合入 main（#50126）。上期 Ascend 支持已合入（#43517），本期回到**返回路径的正确性与成本**：

1. **abort 路径的 mask 补齐**：[#50859](https://github.com/sgl-project/sglang/pull/50859) 让「等待队列中被 abort（retract/取消）」的请求仍返回它**已生成 token** 的 mask（按请求选择的嵌套或扁平格式）。这是一个真实的一致性缺口：此前 abort 响应没有 mask，RL 侧会看到「有 token、无 mask」。维护者把它也 cherry-pick 到发布分支（[#51378](https://github.com/sgl-project/sglang/pull/51378)），说明被视为需要回移的缺陷。
2. **扁平数组返回合入 main**：[#50126](https://github.com/sgl-project/sglang/pull/50126) 于 `2026-11-17T11:45-08:00` 合入（10 文件），并在 11-18 被 cherry-pick 到发布分支（#51376）。大 top-k 下嵌套 JSON 的序列化成本问题在 main 上已解。
3. **「只返回采样 logprob」没有进 main**：[#50155](https://github.com/sgl-project/sglang/pull/50155) 本期以 `closed`（未合入）结束，其 cherry-pick #51377 同样关闭。因此 main 上**没有**「不要 mask、只取采样概率」的路径；RL 侧仍要按 mask 方案取概率。

## 增量索引

- [#50859](https://github.com/sgl-project/sglang/pull/50859) — 本期新建；期末 `open`；[Fix] tokenizer: return masks on requests aborted from the waiting queue。
- [#50126](https://github.com/sgl-project/sglang/pull/50126) — 前期补漏续跟；期末 `merged`（`2026-11-17T11:45:00-08:00`）；[Perf] tokenizer: return masks as flat arrays。
- [#50155](https://github.com/sgl-project/sglang/pull/50155) — 前期补漏续跟；期末 `closed`（未合入）；[Feat] return sampling logprobs without the mask。
- [#51378](https://github.com/sgl-project/sglang/pull/51378) — 本期新建；期末 `open`；[cherrypick from #50859] return masks on aborted requests。

## 重点条目

### #50859 · P0 · 本期新建（abort 路径）

[[Fix] tokenizer: Return sampling masks on requests aborted from the waiting queue](https://github.com/sgl-project/sglang/pull/50859)，`open`，6 个提交、38 条时间线事件。

机制：请求要求 `return_sampling_mask`，但从等待队列被 abort（retract 后重新排队、或被取消）时，响应现在仍带上**已生成 token** 的 mask，格式遵循请求选择的嵌套/扁平。

NPU 判断：需实测。Ascend 的 retract/abort 路径（含图重放下的 buffer 回收）必须同样验证；否则 NPU 上会出现「有 token 无 mask」，而这类不一致在训练侧表现为 off-policy 校正丢失。

验证及限制：作者侧测试；本期未复测。

### #50126 · P1 · 前期补漏续跟（合入）

[[Perf] tokenizer: Return output sampling masks as flat arrays on request](https://github.com/sgl-project/sglang/pull/50126)，`2026-11-17T11:45-08:00` 合入，10 文件。

机制：mask 不再按 token 嵌套 JSON，而是按请求返回扁平数组，显著降低大 top-k 下的序列化/反序列化成本。上期新建、本期合入；11-18 又被 cherry-pick 到发布分支（#51376）。

NPU 判断：需实测。Ascend 走同一 tokenizer 返回路径，扁平化后 mask 行与 token 的对齐关系必须不变；注意 `support` 长度与 `top_k` 的上限边界。

### #50155 · P2 · 前期补漏续跟（关闭，未合入）

[[Feat] sampling: Return sampling logprobs without the sampling mask](https://github.com/sgl-project/sglang/pull/50155)，`closed`（未合入 main），其 cherry-pick #51377 同样关闭。

结论：**main 没有「只取采样 logprob」的路径**。上期曾把它视为「不依赖 mask 打包的替代路径」，本期必须撤回这一判断。RL 侧要拿到与采样分布一致的概率，仍需 mask 方案（或等新提案）。

### #51378 · P2 · 本期新建（发布分支 cherry-pick）

[[cherrypick from #50859]](https://github.com/sgl-project/sglang/pull/51378)：把 abort 路径 mask 修复搬到发布分支（open）。同期 #51376（#50126 的 cherry-pick）已于 11-18 合入发布分支。这说明 mask 的两条修复被视为需要回移稳定版的能力；判断版本可用性时要区分 main 与发布分支。

## 无实质进展与补漏边界

- [#43517](https://github.com/sgl-project/sglang/pull/43517)（Ascend 支持）上期已合入，本期无新增；**仍未看到 Ascend 设备侧复测数据**，`min_p`/cutoff ties/`top_k=1025` 回退/seeded 图重放这几项仍属「已合入、待验证」。
- [#47967](https://github.com/sgl-project/sglang/pull/47967)（XPU 确定性 renorm）、[#43047](https://github.com/sgl-project/sglang/pull/43047)（sglang-miles draft）、[#46080](https://github.com/sgl-project/sglang/pull/46080)、[#46186](https://github.com/sgl-project/sglang/pull/46186)：本期无更新。
- 背景（不纳入判断）：vLLM 侧 [#68634](https://github.com/vllm-project/vllm/pull/68634)（RL 独立采样 mask opt-out 原型）、#68871/#68624（XPU mask dtype 修复，已关闭）、#67542（AMD sampling-mask D2H race，已关闭）均在 hidden-state 报告中作相邻提及。

## 检索与复核

- [本期结构化数据](sample-mask-delta-2026-11-20.json) · [本期 CSV](sample-mask-delta-2026-11-20.csv)。
- 使用已认证 `gh` CLI 拉取；合入条目以本地浅克隆的合并提交为证据。
- 未运行 GPU/NPU 或模型测试。

## 扫描覆盖与源码时间说明

- 关键词检索：8 个采样短语 × `updated:>=2026-11-13` + `created:` 按日切分；去重后 sglang 1679 条（1505 PR + 174 Issue），采样主题候选 26 条，纳入 4 条。
- 合入侧：本地 main 浅克隆扫描窗口内 572 个提交，命中 mask 功能提交为 `50126` 的合入提交。
- 源码快照 `9866f0a0c798ffb85da1122c09f8f135f66085d7`。
