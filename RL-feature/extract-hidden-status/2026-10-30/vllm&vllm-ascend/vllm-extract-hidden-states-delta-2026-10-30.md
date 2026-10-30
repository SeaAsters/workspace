# vLLM / vllm-ascend hidden state extract / return / aux relay 增量跟踪：2026-10-30

UTC 窗口：`[2026-10-23T02:05:41Z, 2026-10-30T02:34:03Z)`。采集截止：`2026-10-30T02:34:03Z`。本报告同时覆盖 **vLLM 主仓**与 **vllm-ascend**，两者分开列条目但合并判断；SGLang 侧见 [../sglang/sglang-extract-hidden-states-delta-2026-10-30.md](<../sglang/sglang-extract-hidden-states-delta-2026-10-30.md>)。

代码基准：

- vLLM：[官方 main `f81a414faed6`](https://github.com/vllm-project/vllm/commit/f81a414faed63b576768487236f98ea5e655fc57)（提交时间 `2026-10-30T01:34:21Z`）。
- vllm-ascend：[官方 main `c9fc4538526b`](https://github.com/vllm-project/vllm-ascend/commit/c9fc4538526ba9b04d7f85c4f2766620bc1c4049)（提交时间 `2026-10-30T02:36:06Z`）。
- 两者均以本地浅克隆观察，vllm-ascend HEAD 略晚于 PR 事件截止（见文末说明）。

基线说明：两仓都没有历史归档，本期为首版基线。`前期补漏续跟` 的含义是「本期才被纳入跟踪的旧条目」，不是「上期已跟踪并继续」。**更正上一版判断**：本次调研推翻了「vllm-ascend 只有内部张量提及、没有 hidden-state 工作」的说法——Ascend 侧既有模型内 PP 传输实现，也有本期在做的 SP 分片与 DSpark/DFlash 消费方改造（见「vllm-ascend：当前能力」）。

## 本期结论

纳入实质变化或技术讨论 **27 条**（vLLM 主仓 19 + vllm-ascend 8），分类为 本期新建 10 条、前期补漏续跟 17 条；其中 8 条是 Issue/RFC，6 条在本期合入（vLLM 5 + vllm-ascend 1），其余 21 条 open。

四条主线：

1. **vLLM 主仓的 PP × aux hidden states 仍是逐模型 opt-in**。#60477 修好 DSA 的 partial hidden_states 必须在 PP send 前 all-reduce；#64458 把「模型是否支持 aux over PP」判定收敛到设置阶段；#64320、#61180、#57917 三条 open PR 分别补 Qwen3-MoE、Qwen3.5、GLM-5.2/DSv3.2。#63157 是这条缺口最典型的用户可见形态（worker 初始化直接拒绝）。
2. **vLLM 对外返回 hidden state 仍停在 RFC**。#59542（按请求返回采样位置的最后 hidden state）本期讨论升温，#48743 有 follow-up 实现分支；但 main 的 OpenAI 协议里没有 `return_hidden_states`，`extract_hidden_states` 仍只是 spec-decode 的一种 draft 方法。
3. **vllm-ascend 的能力是「模型内实现 + 公共校验」的组合**：`supports_aux_hidden_states_over_pp = True` 出现在 DeepSeek-V4、Kimi-K3、MiniMax-M3，并通过补丁打开 DeepseekV2；`DeepseekV4Model` 用 `AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"` 从 `intermediate_tensors` 读写 aux 张量。也就是说 **Ascend 的 aux relay 由模型自己搬运**，上游只做校验与转发。
4. **vllm-ascend 本期在补的正是这套机制的边界**：#20075（draft）在保持 SP 分片的同时让 aux/Top-K 保持 full-token；#19284 把 HC 激活、FP32 mixing、DSpark aux 一起塞进 PP intermediate tensors；#19537 已合入，为 DSpark 供 aux hidden；#17557 让 DFlash2 drafter 按 `hc_contract(hidden_states + residual)` 消费逐层 hidden。另有 xlite graph 的硬约束：**带中间 hidden states 的投机方法（eagle3/extract_hidden_states/dflash/dspark）不能在 xlite graph 下启用，且 xlite 要求 PP=1**。

## 增量索引

### vLLM 主仓（19 条）

- [#63436](https://github.com/vllm-project/vllm/pull/63436) — 本期新建；期末 `merged`（`2026-10-26T20:58:42Z`）；[Bugfix][Spec Decode] DFlash: run the aux hidden state FC in the draft dtype。
- [#63427](https://github.com/vllm-project/vllm/pull/63427) — 本期新建；期末 `merged`（`2026-10-27T03:04:50Z`）；[Bugfix][CUDA Graph] Weak-ref nested outputs of captured graphs。
- [#64320](https://github.com/vllm-project/vllm/pull/64320) — 本期新建；期末 `open`；[Bugfix][PP][Spec Decode] Relay Qwen3-MoE EAGLE3 aux hidden states over PP。
- [#64458](https://github.com/vllm-project/vllm/pull/64458) — 本期新建；期末 `merged`（`2026-10-29T16:41:39Z`）；[MRV2][PP] Fix spec-decode draft/sampled-token broadcast issues。
- [#60477](https://github.com/vllm-project/vllm/pull/60477) — 前期补漏续跟；期末 `merged`（`2026-10-29T05:14:24Z`）；[Bugfix] DSA: all-reduce partial hidden_states before PP send。
- [#61699](https://github.com/vllm-project/vllm/pull/61699) — 前期补漏续跟；期末 `merged`（`2026-10-28T22:13:50Z`）；[Bugfix][Spec Decode] Honor standalone draft load config。
- [#57736](https://github.com/vllm-project/vllm/pull/57736) — 前期补漏续跟；期末 `open`；feat: add windowed hidden-state capture for vLLM rollout。
- [#61180](https://github.com/vllm-project/vllm/pull/61180) — 前期补漏续跟；期末 `open`；[Bugfix][Spec Decode] Relay Qwen3.5 auxiliary states over PP。
- [#57917](https://github.com/vllm-project/vllm/pull/57917) — 前期补漏续跟；期末 `open`；[Spec Decode] Enable GLM-5.2 DSpark under pipeline parallelism。
- [#56822](https://github.com/vllm-project/vllm/pull/56822) — 前期补漏续跟；期末 `open`；[Bugfix] Bill hidden-state extraction at its own KV cache group width。
- [#50187](https://github.com/vllm-project/vllm/pull/50187) — 前期补漏续跟；期末 `open`（stale）；[Bugfix][Spec Decode] Fix misspelled EAGLE3 aux-hidden-state layer method overrides。
- [#63406](https://github.com/vllm-project/vllm/pull/63406) — 本期新建；期末 `open`；[Core][CUDA Graph] Separate prefill (token) and decode (request) CUDA graph capture size。
- [#59542](https://github.com/vllm-project/vllm/issues/59542) — 前期补漏续跟（RFC）；期末 `open`；[RFC]: Opt-in return of the last hidden state at sampled positions, per request (Model Runner V2)。
- [#48743](https://github.com/vllm-project/vllm/issues/48743) — 前期补漏续跟（RFC）；期末 `open`；[RFC]: Return extracted hidden states in the generation response。
- [#63157](https://github.com/vllm-project/vllm/issues/63157) — 前期补漏续跟；期末 `open`；[Spec Decode][DeepSeek-V4.1] Support DSpark with pipeline parallelism。
- [#61539](https://github.com/vllm-project/vllm/issues/61539) — 前期补漏续跟（RFC）；期末 `open`；[RFC]: Explicit row layouts for routed-expert capture。
- [#63818](https://github.com/vllm-project/vllm/issues/63818) — 本期新建（RFC）；期末 `open`；[RFC]: Pluggable pipeline-parallel transport…。
- [#36998](https://github.com/vllm-project/vllm/issues/36998) — 前期补漏续跟（RFC）；期末 `open`（stale）；[RFC]: Observation Plugin for Intercepting & Routing on Activations。
- [#37002](https://github.com/vllm-project/vllm/pull/37002) — 前期补漏续跟；期末 `open`（stale）；[Core][Feature] Observation Plugin for Intercepting & Routing on Activations。

### vllm-ascend（8 条）

- [#20075](https://github.com/vllm-project/vllm-ascend/pull/20075) — 本期新建（draft）；期末 `open`；[Feature][MRV2] Preserve SP token shards across pipeline stages。
- [#19284](https://github.com/vllm-project/vllm-ascend/pull/19284) — 前期补漏续跟；期末 `open`；[Feature][Model] Enable DeepSeek V4.1 MRV2 PP with arbitrary layer partitions。
- [#19537](https://github.com/vllm-project/vllm-ascend/pull/19537) — 前期补漏续跟；期末 `merged`（`2026-10-29T07:36:54Z`）；[Feature] Add MiniMax-M3 DSpark draft FlashAttn support on A5。
- [#17557](https://github.com/vllm-project/vllm-ascend/pull/17557) — 前期补漏续跟；期末 `open`；[Feature][Model] Support GLM-5.3-Flash DFlash2 speculative decoding on model runner V2。
- [#19801](https://github.com/vllm-project/vllm-ascend/pull/19801) — 本期新建；期末 `open`；[Refactor][MRV2][PCP][SpecDecode] Align PCP speculative decoding with upstream draft routes。
- [#19150](https://github.com/vllm-project/vllm-ascend/pull/19150) — 前期补漏续跟；期末 `open`；[Feature] Expose EAGLE3 support for Qwen3-ASR。
- [#20231](https://github.com/vllm-project/vllm-ascend/issues/20231) — 本期新建（RFC）；期末 `open`；[RFC]: Support xPress Causal Refiner for DFlash on Ascend NPU。
- [#19845](https://github.com/vllm-project/vllm-ascend/issues/19845) — 本期新建（RFC）；期末 `open`；[RFC]: Extend adaptive verification support across ascend attention backends。

## vLLM 主仓：机制、证据与 NPU 跟进

### #63436 · P0 · 本期新建（合入）

[[Bugfix][Spec Decode] DFlash: run the aux hidden state FC in the draft dtype](https://github.com/vllm-project/vllm/pull/63436)，`main`，head `d234a0b828`，+44/-4，合入者 mgoin。

机制：DFlash/DFlash2 drafter 在 target 给出 FP32 aux hidden states 时（Muse Spark 残差流保持 FP32）会让 bf16 的 `fc` 收到 FP32 输入并在启动阶段失败；修复把 `fc` 输入转成 draft dtype。通用约束：**aux hidden states 的 dtype 由 target 决定，不能假设与 draft 一致**。

本期证据：`2026-10-24T02:24Z` 建 PR；`2026-10-26T15:03:13Z` force-push；`19:19:02Z` ready_for_review、`19:19:07Z` `/ci run`（Buildkite #97391，提示落后 main 113 个提交）；`19:51:03Z` auto_squash；`20:58:42Z` merged。

NPU 判断：**需实测**。Ascend 上若 target 残差流为 FP32 而 draft 为 bf16，需要同等的 dtype 门禁；NPU fused fc 的 dtype 提升行为需与 CUDA 对照。

验证及限制：作者附 `tests/v1/spec_decode/test_dflash2.py` 的 FC dtype 用例；同题 CI 失败记录见 #63972（closed）。本期未复测。

关键代码（当前 diff）：`vllm/model_executor/models/qwen3_dflash.py`、`tests/v1/spec_decode/test_dflash2.py`。

[事件、提交和当前文件 diff](vllm-extract-hidden-states-delta-2026-10-30.json)。

### #63427 · P1 · 本期新建（合入）

[[Bugfix][CUDA Graph] Weak-ref nested outputs of captured graphs](https://github.com/vllm-project/vllm/pull/63427)，`main`，head `6ee3f40d61`，+33/-4，合入者 ZJY0516。

机制：捕获图输出的嵌套结构改为弱引用保存。按 PR 描述，此前每个 PIECEWISE 捕获尺寸都会保留 `num_aux × num_tokens × hidden_size × dtype` 字节的 aux hidden states（EAGLE3/DSpark/DFlash/`extract_hidden_states`），属捕获特性的内存泄漏；修复同时把 `LayerName` 透传给 `weak_ref_tensors`。

本期证据：`2026-10-23T23:40:59Z` ready_for_review；`23:41:09Z` codex 评审摘要；`2026-10-26T21:22:59Z` `/ci run`（Buildkite #97427）；`2026-10-26T23:55:40Z` `--allow-stale` 重跑；`2026-10-27T03:04:50Z` merged。

NPU 判断：**需实测**。Ascend 图捕获若同样持有 aux 输出引用，需要等价弱引用或显式释放；vllm-ascend 的 `_310p/worker/v2/aclgraph.py` 已经能产出 aux hidden outputs，建议按捕获尺寸数量测显存曲线。

关键代码：`vllm/utils/torch_utils.py`、`tests/utils_/test_torch_utils.py`。

### #60477 · P1 · 前期补漏续跟（合入，正确性）

[[Bugfix] DSA: all-reduce partial hidden_states before PP send](https://github.com/vllm-project/vllm/pull/60477)，`main`，head `ea165f4168`，+59/-17，合入者 aoshen02。

机制：PP 的 send/recv 需要 TP-replicated 张量，而中间 stage 的 partial `hidden_states` 尚未归约。修复在中段先做 `tensor_model_parallel_all_reduce`，并把 `IntermediateTensors` 的键从 `["hidden_states","residual"]` 收敛为 `["hidden_states"]`；同时修正 `aux_hidden_states` 的 `hidden_states + residual` 拼接。对 aux 捕获是**跨阶段还原正确性**：捕获值错了，下游 draft 训练数据就错了，且不会以崩溃形式暴露。

本期证据：`2026-10-28T22:47:32Z` 交接评论、`22:55:46Z` pre-commit 失败；`2026-10-29T03:36:34Z` `/ci run`（Buildkite #98186）；`04:59:01Z` auto_squash；`05:14:24Z` merged。

NPU 判断：**需实测且与 Ascend 强相关**。vllm-ascend 的 `DeepseekV4Model` 在 SP 模式下会先 `sp_shard` 再跨 PP 传；进入下一 stage 前要求 full-sequence，因此 Ascend 的归约/还原顺序必须与这里一致，否则 aux 行数与残差流会错位。

关键代码：`vllm/models/deepseek_v32/amd/model.py`、`vllm/models/deepseek_v32/nvidia/model.py`、`vllm/models/dots3_note/nvidia/model.py`。

### #64458 · P1 · 本期新建（合入，MRV2 PP）

[[MRV2][PP] Fix spec-decode draft/sampled-token broadcast issues](https://github.com/vllm-project/vllm/pull/64458)，`main`，head `f5ea95f1c2`，+75/-65，合入者 WoosukKwon。

机制：规范 MRV2 下 PP 的 draft / sampled-token 广播，并把 EAGLE3 aux-hidden-state 的 PP 支持检查移入 `set_eagle3_aux_hidden_state_layers`——能力判定收敛到设置阶段，不支持的模型在配置期失败，而不是运行期静默错位。

本期证据：`2026-10-29T07:03:09Z` `/ci run`（#98229）、`15:26:25Z` `/ci retry`（#98358）；`16:41:39Z` merged。2 个 patch 提交（`9ea39b7506`、`65d21d7a92`）。

NPU 判断：**需实测**。vllm-ascend 的 MRV2 spec decode（`worker/v2/spec_decode/*`）直接调用上游判定，需验证 Ascend 上该判定位置与广播顺序一致。

关键代码：`vllm/v1/worker/gpu/pp_utils.py`、`vllm/v1/worker/gpu/model_runner.py`、`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`。

### #57736 · P0 · 前期补漏续跟（open，本期实质推进）

[feat: add windowed hidden-state capture for vLLM rollout](https://github.com/vllm-project/vllm/pull/57736)，`main`，head `0ad6f72436`，+1547/-22。

机制：调用方给出响应区间 `[start, end)`，只返回该区间的 hidden-state 行，并可附对齐的 raw top-k logprobs；窗口在生成过程中经 **AuxOutput Connector** 收集，以减少 GPU→CPU 传输，同时支持最终 hidden states 与配置的投机辅助输出（上游需求 #56916）。本期新增 6 个提交，落地 `vllm/distributed/aux_output_connector/`（connector / hidden_states / worker）、`vllm/config/aux_output.py`、`docs/features/speculative_decoding/windowed_hidden_states.md` 与 benchmark，并处理 preemption 已交付行、直接/流式准入与请求校验。

本期证据：`2026-10-29T08:29:55Z` force-push；`08:30:41Z`、`12:06:48Z` 两次 mergify 冲突提示。提交序列含 `7386d5d4a1`（capture hidden-state windows through AuxOutput）、`9d54e01314`（exclude delivered hidden rows during preemption）、`40c384ccd2`（admission for direct and streaming）、`1165f9a930`（validate requests and isolate logprob）。

NPU 判断：**缺失/需实测**。这是目前最接近「RL 训练要窗口化 hidden state」的实现，但未合入；Ascend 侧要先有接口，再评估 AuxOutput Connector 的 D2H 路径（vllm-ascend 目前没有对应实现）。

关键代码：`vllm/distributed/aux_output_connector/hidden_states.py`、`vllm/config/aux_output.py`、`vllm/v1/worker/gpu/model_runner.py`（准入）。

### #64320 / #61180 / #57917 · P0/P1 · PP aux relay 的三条模型侧 PR（open）

- [#64320](https://github.com/vllm-project/vllm/pull/64320)：Qwen3-MoE 的 EAGLE3 aux relay（`qwen3_moe.py`、`qwen3_omni_moe_thinker.py`、`qwen3_vl_moe.py`，+175/-13）。本期 1 个提交（`ebe90f2108`，10/29）、1 次 force-push，mergify 报冲突待 rebase，无评审结论。
- [#61180](https://github.com/vllm-project/vllm/pull/61180)：Qwen3.5 的有序 EAGLE3/DFlash aux relay（`qwen3_5.py`、`qwen3_next.py`），修复 #60765。本期无新提交，10/29 作者请求评审并说明带 CPU 单测，mergify 报冲突待 rebase。
- [#57917](https://github.com/vllm-project/vllm/pull/57917)：GLM-5.2 DSpark 在 PP 下运行（`vllm/models/deepseek_v32/nvidia/model.py`，+16/-5），依赖 #50514 的 `supports_aux_hidden_states_over_pp`。本期唯一事件是 10/29 冲突提示。

共同机制：vLLM 的 aux-over-PP 不是全局开关，而是逐模型类声明 `supports_aux_hidden_states_over_pp`（[`interfaces.py#L1611`](https://github.com/vllm-project/vllm/blob/f81a414faed63b576768487236f98ea5e655fc57/vllm/model_executor/models/interfaces.py#L1611)），由 [`verify_supports_aux_hidden_states_over_pp`](https://github.com/vllm-project/vllm/blob/f81a414faed63b576768487236f98ea5e655fc57/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py#L57) 在初始化期校验。**新增一个模型不等于新增硬件支持**：Ascend 侧要按模型逐个验证 relay 顺序、阶段层号与 dtype。

### #59542 · P0 · 前期补漏续跟（RFC，讨论升温）

[[RFC]: Opt-in return of the last hidden state at sampled positions, per request (Model Runner V2)](https://github.com/vllm-project/vllm/issues/59542)，`open`。

机制/诉求：带小头的部署（线性探针、任务分类器、校准层）希望一次 forward 同时拿到 token 分数与其对应 hidden state，现状要么再起一个 pooling 引擎、要么二次前向。本期 3 条评论：10/26 有用户给出 RAG 证据门控 + serving 插件的实际用法；10/29 有维护者按 popularity 口径汇总「三个独立团队遇到同一问题」。

NPU 判断：**缺失**。无实现即无 Ascend 路径；若采纳，需要同时定义 NPU 的 D2H 布局、采样位置对齐与图重放下的 buffer 语义。

### #48743 · P1 · 前期补漏续跟（RFC，有实现分支）

[[RFC]: Return extracted hidden states in the generation response](https://github.com/vllm-project/vllm/issues/48743)，`open`，标签 `RFC`。

机制：主张把 `extract_hidden_states` 的产物通过生成响应返回，而不是仅供 draft 使用；与 #59542 的差别在响应体契约与 opt-in 范围。本期 `2026-10-27T16:33:15Z` 有一条评论给出幻灯片与 follow-up 实现分支（`Davidnet/vllm:feat/inline-hidde…`）。

NPU 判断：缺失。需先有实现；Ascend 还要确认响应序列化与张量裁剪。

### #56822 / #50187 / #61699 / #63406 · P2 · 持续跟进项

- [#56822](https://github.com/vllm-project/vllm/pull/56822)：hybrid Mamba/MLA 模型上 `extract_hidden_states` 的 KV-cache 准入按最宽 peer group 计费（1 层 hidden-state group 被按 24 层 MLA 宽度收费，约 5.4 GiB 多算），改为按自身 group 宽度计费。本期仅 10/29 冲突提示。
- [#50187](https://github.com/vllm-project/vllm/pull/50187)：修正 EAGLE3 aux-hidden-state 层方法 override 的拼写错误——拼错会让 override 静默失效、捕获层回落默认值。本期 10/27 被 stale bot 标记。
- [#61699](https://github.com/vllm-project/vllm/pull/61699)（`merged`，`2026-10-28T22:13:50Z`）：standalone draft 加载配置未被遵守，涉及 `vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py` 与 `standalone_ar/speculator.py`。
- [#63406](https://github.com/vllm-project/vllm/pull/63406)（`open`）：把 prefill（token）与 decode（request）的 CUDA Graph 捕获尺寸分开，与 #63427 的内存分解互相引用。

NPU 判断：均需实测。vllm-ascend 的 KV 池分组/计费、模型 override 命名与图捕获尺寸策略各自独立，需按 Ascend 实际配置复算。

### #63157 / #61539 / #63818 · P1/P2 · Issue 与 RFC

- [#63157](https://github.com/vllm-project/vllm/issues/63157)：DeepSeek-V4.1 的 DSpark + PP 在 worker 初始化被 `verify_supports_aux_hidden_states_over_pp` 拒绝；10/23 维护者回复认为 PP+DSpark 值得支持（尤其 PD 分离下用 PP+DSpark 做 prefill），本期无修复合入。
- [#61539](https://github.com/vllm-project/vllm/issues/61539)：routed-expert 捕获目前按张量行数与 `dp_metadata` 反推行归属，对 out-of-tree MoE 后端不成立；提议显式声明行布局。本期 2 次引用（含 vllm-ascend 侧维护者的 cross-reference）。
- [#63818](https://github.com/vllm-project/vllm/issues/63818)：可插拔 PP 传输与 out-of-process PP 阶段；作者 10/27 给出 4×A40（PCIe）+ 100Gb IB 的实测与故障切换结果，配套草案 PR #63897/#63898/#63899。

NPU 判断：三条都是**缺失**（对 Ascend 而言）。HCCL 不在 #63818 的默认假设内；#61539 的显式布局若落地，Ascend MoE 后端可直接受益。

### #36998 / #37002 · P2 · 前期补漏续跟（Observation Plugin，stale）

[#36998](https://github.com/vllm-project/vllm/issues/36998)（RFC）与 [#37002](https://github.com/vllm-project/vllm/pull/37002)（Phase 1 实现，17 文件 +1257）本期都只被 stale bot 在 10/29 标记，最后一次实质提交停在 4 月。与 SGLang 已合入的 forward observer 属同类需求的两条不同工程路径：**不能把 RFC 存在当作能力可用**。

## vllm-ascend：当前能力（源码核对）

以下为 `c9fc4538526b` 上的直接核对，用于把「Ascend 侧到底有什么」说清楚：

- **逐模型的 aux-over-PP 声明**：`supports_aux_hidden_states_over_pp = True` 出现在 [`models/deepseek_v4/model.py#L844`](https://github.com/vllm-project/vllm-ascend/blob/c9fc4538526ba9b04d7f85c4f2766620bc1c4049/vllm_ascend/models/deepseek_v4/model.py#L844)、`models/kimi_k3.py#L760`、`models/minimax_m3/minimax_m3.py#L1110`；并通过 [`patch/worker/patch_deepseek_v2.py`](https://github.com/vllm-project/vllm-ascend/blob/c9fc4538526ba9b04d7f85c4f2766620bc1c4049/vllm_ascend/patch/worker/patch_deepseek_v2.py) 给 `DeepseekV2Model` 打上 `EagleModelMixin` 的 `AUX_HIDDEN_STATE_KEY`、`pack_local_aux_hidden_states`、`collect_remote_aux_hidden_states`、`_cache_aux_pp_layout` 等成员后置为 `True`。
- **模型自己搬 aux 张量**：`DeepseekV4Model` 定义 `AUX_HIDDEN_STATE_KEY = "pp_transport_aux_hidden_states_"`，用 `get_pp_transport_tensors(intermediate_tensors, PPTransportDataType.AUX_HIDDEN_STATES)` 取上一阶段写入的 aux；本阶段按 `layer_idx + 1 in aux_hidden_state_layers` 捕获 `hidden_states.mean(dim=1)`（对 hc 流取均），SP 模式下再 `sp_all_gather` 还原 full-token。注释明确「vLLM #50514 validates and relays the model's existing PP aux payload」——**上游只做校验与转发，实现由模型提供**。
- **投机方法白名单**：`ascend_config.py` 的 xlite graph 配置在 `_validate_preconditions` 里拒绝 `eagle3`/`extract_hidden_states`/`dflash`/`dspark`（理由：带中间 hidden states），并要求 `pipeline_parallel_size == 1`。这是 Ascend 侧一条明确的组合约束，配置错会在启动期报错而不是静默降级。
- **310P 图路径**：`_310p/worker/v2/aclgraph.py` 的 `capture()` 接受 `use_aux_hidden_state_outputs`，说明 310P 的 aclgraph 路径能产出 aux hidden 输出。
- **没有公共返回接口**：与主仓一致，`return_hidden_states` 在 vllm-ascend 中不存在；`extract_hidden_states` 只作为 spec 方法名出现在配置校验与方法分派里。
- **采样 mask 侧（相邻特性）**：vllm-ascend 已有 Triton 打包内核 [`ops/triton/v2/sample/sampling_mask.py`](https://github.com/vllm-project/vllm-ascend/blob/c9fc4538526ba9b04d7f85c4f2766620bc1c4049/vllm_ascend/ops/triton/v2/sample/sampling_mask.py) 与设计文档 `ops/triton/docs/v2/sample/sampling_mask.md`（`SamplingMaskTensors.from_logits` → `_pack_sampling_mask_kernel_ascend`），本期还有两条在修 PR（#19866 请求级 mask、#20129 mask 打包槽位对齐）与一条把 mask 算子测试加入夜间的 CI 提交（#18431）。这部分属 sample mask 主题，本报告只作相邻提及。

## vllm-ascend：本期条目

### #20075 · P0 · 本期新建（draft）

[[Feature][MRV2] Preserve SP token shards across pipeline stages](https://github.com/vllm-project/vllm-ascend/pull/20075)，`main`，head `8850a6f867`，+1742/-61，25 文件，**draft**。

机制：MRV2 跨 PP 传输时保持 SP token 分片，同时把 **auxiliary hidden states 与 Top-K payload 保持为 full-token**，只在既有 send 边界恢复额外 TP padding，并在图重捕获时复用常驻 PP 缓冲。PR 说明 #16511 是 Kimi 专用 V1/V2+DSpark 实现，本 PR 提供通用 MRV2 buffer/graph 支持。

本期证据：3 个提交（`0ec5b8a498` 10/28「preserve SP token shards across MRV2 pipeline」、`a42f8f6878`「preserve GLM token shards」、`8850a6f867`「reuse persistent PP buffers when recapturing」）；`2026-10-28T07:37:00Z` force-push；`2026-10-29T08:06:01Z` bot 报冲突。

NPU 判断：这条本身是 Ascend 的 aux 跨 PP 实现，风险在于 full-token 语义与 TP padding 恢复是否在所有阶段一致，以及常驻 buffer 在图重捕获后是否被覆盖（与 #43450/#63427 的 buffer 所有权问题同源）。

验证及限制：作者附 `tests/ut/worker/v2/test_pp_transport.py`、`test_pp_token_shard_graph.py` 与 A3 多卡 SP+PP e2e；draft 且冲突未解，未合入。

关键代码（当前 diff）：`vllm_ascend/worker/v2/`（PP transport / graph）、`docs/source/user_guide/feature_guide/sequence_parallelism.md`、`tests/ut/worker/v2/test_pp_transport.py`。

[事件、提交和当前文件 diff](vllm-extract-hidden-states-delta-2026-10-30.json)。

### #19284 · P0 · 前期补漏续跟（open，长期迭代）

[[Feature][Model] Enable DeepSeek V4.1 MRV2 PP with arbitrary layer partitions](https://github.com/vllm-project/vllm-ascend/pull/19284)，`main`，head `0eec97fb58`，+1903/-76，21 文件。

机制：把 HC 激活、FP32 mixing state 与 **DSpark auxiliary states** 一并放进 PP intermediate tensors 传输，并保留 payload 校验、快照与不透明运行时导入/导出；支持任意层切分。评论把「增量 source-cache 传输 + stage-local mirror」视为后续方向，并另开 #19892 跟踪 A2 双节点 RoCE 的验证范围。

本期证据：`2026-10-26T08:07:27Z` 冲突提示；`09:01:03Z`/`10:03:03Z` 两次 renamed；`2026-10-26T09:59:24Z`、10/27 两条实质评论；6 次 force-push（10/26–10/29）；`2026-10-29T12:35:42Z` 再次冲突。仅 1 个 patch 提交（`0eec97fb58`）。

NPU 判断：直接决定 Ascend 上 PP>1 时 aux 状态是否可用；需验证任意层切分下 DSpark aux 槽位、HC 激活维度与 FP32 mixing 精度一致。

验证及限制：作者附 `tests/ut/models/test_deepseek_v41_{pp_cache,dspark,hyper_connection,layer_plan,cache}.py` 与 PP cache transfer e2e；未合入，本期未复测。

### #19537 · P1 · 前期补漏续跟（合入）

[[Feature] Add MiniMax-M3 DSpark draft FlashAttn support on A5](https://github.com/vllm-project/vllm-ascend/pull/19537)，`main`，head `4d58a50493`，+442/-38，13 文件，`2026-10-29T07:36:54Z` 由 ZT-AIA 合入。

机制：为 DSpark 提供 target 的 **auxiliary hidden states**，并修正混合 FP8 target / BF16 draft 的 KV page 尺寸与 indexer kernel page 视图，保留既有 64/128 token 分块行为。

本期证据：10/23–10/28 共 12 次 force-push、两次 renamed、`ready_for_review`；`2026-10-28T14:47:20Z`、`16:21:53Z` 两次 `/rerun`；`2026-10-29T07:36:54Z` merged。

NPU 判断：这是 Ascend 上 DSpark 消费 aux hidden 的落地证据；仍需逐层比对 aux 层序与数值，并验证 FP8 target / BF16 draft 混用时捕获 dtype 与 cast 位置（与上游 #63436 同类问题）。

验证及限制：作者附 `tests/ut/spec_decode/test_dspark_graph_contract.py`、`minimax_m3` 单测与 flash-attn 精度 e2e。

关键代码：`vllm_ascend/models/minimax_m3/minimax_m3.py`、`vllm_ascend/attention/fa_v1.py`、`vllm_ascend/platform.py`。

### #17557 · P1 · 前期补漏续跟（open）

[[Feature][Model] Support GLM-5.3-Flash DFlash2 speculative decoding on model runner V2](https://github.com/vllm-project/vllm-ascend/pull/17557)，`main`，head `c7ca3b9a1f`，+759/-35，15 文件。

机制：DFlash drafter 消费 target 的逐层 hidden states，并按 `hc_contract(hidden_states + residual)` 收缩；同时补 rope/NoPE 缓存跳过、`rope is_neox_style` 透传与 QKNorm-Rope 融合的跳过条件。测试含 `test_glm5next_dflash_capture.py` 与 `test_glm5next_pp.py`。

本期证据：`2026-10-23T02:11:26Z`、`2026-10-29T01:43:02Z` 两次冲突提示；`2026-10-25T07:14:12Z` force-push。

NPU 判断：需验证 `hc_contract` 在 Ascend 上与目标模型残差流语义一致（含 `hc_mult` 维），以及 PP 下 aux 层号与捕获顺序。

验证及限制：作者侧单测与 e2e；本期未合入、未复测。

### #19801 · P2 · 本期新建（open）

[[Refactor][MRV2][PCP][SpecDecode] Align PCP speculative decoding with upstream draft routes](https://github.com/vllm-project/vllm-ascend/pull/19801)，`main`，head `0d8b9fc163`，+615/-1524，18 文件。

机制：把 PCP 下的投机解码路由与上游 draft route 对齐，10/29 的提交进一步澄清 PCP 采样的 auxiliary-state 路由；净删除代码为主，触及 `worker/v2/model_runner.py`、aclgraph 与 dspark/mtp speculator 测试。

本期证据：10/26–10/28 共 10 次 force-push、多次冲突提示；`2026-10-29T09:33:00Z` renamed，`09:33:23Z` `/rerun`。

NPU 判断：需验证 PCP 分片下 aux 状态的路由顺序与 full-token 还原，避免与 #20075 的 SP 分片语义冲突。

### #19150 · P2 · 前期补漏续跟（open）

[[Feature] Expose EAGLE3 support for Qwen3-ASR](https://github.com/vllm-project/vllm-ascend/pull/19150)，`main`，head `8403b12063`，+97，3 文件。

机制：为 Qwen3-ASR 注册 Ascend 子类并继承 `SupportsEagle3`，否则能力检查会拒绝 EAGLE3 draft（PR 说明当前验证的 vLLM ref 缺少该声明）。属能力注册而非捕获实现，但决定该模型能否走 aux hidden 路径。本期仅 `2026-10-30T02:34:40Z` 冲突提示。

### #20231 / #19845 · P2 · 本期新建（RFC）

- [#20231](https://github.com/vllm-project/vllm-ascend/issues/20231)（10/29 创建）：xPress Causal Refiner for DFlash —— 复用 DFlash proposer 的输入准备（hidden states、token IDs、上下文元数据），把精炼头作为后置步骤挂载，不替代 DFlash。本期无评论。
- [#19845](https://github.com/vllm-project/vllm-ascend/issues/19845)（10/26 创建）：把自适应验证（DFlash 系列）扩展到更多 Ascend attention 后端。自适应验证依赖 verify 阶段的 target 概率与 support，与采样 mask 捕获同源；本期仅 `parent_issue_added` 事件。

NPU 判断：两条均为设计阶段，无代码；若采纳需定义读取的 hidden state 层、dtype 与图捕获下的 buffer 所有权。

## vLLM 主仓接口现状（源码核对）

- **没有公共 `return_hidden_states`**：`vllm/entrypoints/openai/protocol.py` 与 `vllm/outputs.py` 中都没有该字段（`outputs.py` 只有 pooling/embedding 的 hidden dim 描述）。请求级返回 final hidden state 仍只有 RFC（#59542 / #48743）。
- **`extract_hidden_states` 是 spec-decode 的 draft 方法**：`vllm/config/speculative.py` 把它与 eagle/eagle3/DFlash/MTP 并列；`vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py` 要求 `draft_sample_method='greedy'`、`num_speculative_tokens == 1`、且 draft config 必须给出 `eagle_aux_hidden_state_layer_ids`。
- **aux over PP 是逐模型 opt-in**：`supports_aux_hidden_states_over_pp` 在 `interfaces.py` 默认 `False`，当前声明 `True` 的包括 llama、qwen2、sarvam、deepseek_v4(nvidia)、kimi_k3、minimax_m3(amd/nvidia) 等；校验入口是 `verify_supports_aux_hidden_states_over_pp`。
- **sampling mask 与 hidden state 是两条独立链路**：`SamplingMaskTensors` 存在于 `vllm/v1/worker/gpu/sample/*` 与 rejection sampler，但与本报告的 hidden-state 返回无关。

## 无实质进展与补漏边界

- vLLM [#63972](https://github.com/vllm-project/vllm/issues/63972)：`test_dflash_fc_runs_fp32_target_hidden_states_in_draft_dtype` 的 CPU CI 失败记录，随 #63436 修复关闭，仅作证据链。
- vLLM [#63406](https://github.com/vllm-project/vllm/pull/63406)：与 #63427 只是互相引用，不构成 hidden-state 接口变化。
- vLLM [#50187](https://github.com/vllm-project/vllm/pull/50187)、[#37002](https://github.com/vllm-project/vllm/pull/37002)、[#36998](https://github.com/vllm-project/vllm/issues/36998)：本期均只有 stale 标记，属「90 天无活动」的风险提示，不是技术进展。
- vllm-ascend [#19866](https://github.com/vllm-project/vllm-ascend/pull/19866)、[#20129](https://github.com/vllm-project/vllm-ascend/pull/20129)：属采样 mask 主题（请求级 mask 与打包槽位对齐），本报告只作相邻提及，不计入 hidden-state 条目。
- vllm-ascend [#20288](https://github.com/vllm-project/vllm-ascend/pull/20288)（MRV2 eager Draft sliding window，含 DFlash2，draft、+12419/-159）：体量大且仍为 draft，本期仅建立索引，待其收敛后再判断对 aux 捕获的影响。

## 检索与复核

- [本期结构化数据](vllm-extract-hidden-states-delta-2026-10-30.json) · [本期 CSV](vllm-extract-hidden-states-delta-2026-10-30.csv)。
- SGLang 侧同期报告：[../sglang/sglang-extract-hidden-states-delta-2026-10-30.md](<../sglang/sglang-extract-hidden-states-delta-2026-10-30.md>)。
- 当前 PR body、current diff、CI 摘要均为采集快照；force-push 前的旧提交与当期 CI 状态无法完整恢复（本期 vllm-ascend 多条条目有 10–12 次 force-push）。
- 未运行模型、GPU/NPU 或远程设备测试；作者性能数字均不代表本次复测。
- **更正记录**：本报告上一版（同日早些时候）曾写「vllm-ascend 同期没有 hidden-state 返回/relay 相关条目」，该结论只基于标题匹配，已被本轮的代码核对与 #20075/#19284/#19537 等条目推翻。

## 扫描覆盖与源码时间说明

- 关键词检索：vLLM 9 个短语 × `updated:>=2026-10-23` + `label:RFC` + 按日 `created:` 全量扫描，去重后 1635 条（1362 PR + 273 Issue）。
- vllm-ascend：11 个短语 × `updated:>=2026-10-01`（含 `"pipeline parallel" hidden`、`"spec decode" "pipeline parallel"`、`"sampling mask"`）+ `label:RFC` + 按日 `created:` 全量扫描，去重后 957 条（768 PR + 189 Issue）。
- 合入侧：两个浅克隆分别扫描窗口内 527（vLLM）与 178（vllm-ascend）个提交，按主题关键词与文件路径双重筛选；vllm-ascend 另做 pickaxe（`-S`）定位 `supports_aux_hidden_states_over_pp`、`pp_transport_aux_hidden_states_`、`extract_hidden_states` 的引入提交。
- 基线：两仓均无历史归档，分别用「关键词命中且创建早于窗口」的 63（vLLM）与 191（vllm-ascend）条建立首版基线；本期 138 条关键词候选逐条判定（vLLM 87 + ascend 51），纳入 27 条为核心分析。
- 代码核对：在 `f81a414faed6` 与 `c9fc4538526b` 上直接 grep 关键符号，用于区分「接口存在」与「硬件已验证」。

源码快照：vLLM main `f81a414faed63b576768487236f98ea5e655fc57`（提交时间 `2026-10-30T01:34:21Z`）；vllm-ascend main `c9fc4538526ba9b04d7f85c4f2766620bc1c4049`（提交时间 `2026-10-30T02:36:06Z`，略晚于 PR 事件截止 `02:34:03Z`，因此 Ascend 侧源码用于能力核对、不代表事件期末快照）。
