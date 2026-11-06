# vLLM / vllm-ascend hidden state extract / return / aux relay 增量跟踪：2026-11-06

UTC 窗口：`[2026-10-30T02:34:03Z, 2026-11-06T06:35:52Z)`。采集截止：`2026-11-06T06:35:52Z`。本报告同时覆盖 **vLLM 主仓**与 **vllm-ascend**；SGLang 侧见 [../sglang/sglang-extract-hidden-states-delta-2026-11-06.md](<../sglang/sglang-extract-hidden-states-delta-2026-11-06.md>)。

代码基准：vLLM [main `de05fd1db93f`](https://github.com/vllm-project/vllm/commit/de05fd1db93f712288e654bc28a8480a67a7a8e2)（`2026-11-06T06:11:41Z`）；vllm-ascend [main `1c2efa8ac24b`](https://github.com/vllm-project/vllm-ascend/commit/1c2efa8ac24ba14fd69f12609b850d4e6daf0865)（`2026-11-06T04:08:21Z`）。

## 本期结论

纳入 **19 条**（vLLM 10 + vllm-ascend 9），分类为 本期新建 7 条、前期补漏续跟 12 条；其中 3 条合入（vLLM #63205、#63406，vllm-ascend #20617），其余 open。

四条主线：

1. **`extract_hidden_states` 的成本被量化，并出现了关掉它的开关**。[#65250](https://github.com/vllm-project/vllm/issues/65250) 测出该方法每步为每请求排一个 draft token、接受率约 0.03%，在 L4 上损失 37%–39% 饱和吞吐；[#65991](https://github.com/vllm-project/vllm/pull/65991) 随之加 `disable_extract_hidden_states_decode_drafts`（把动态 SD 调度设为 K=0）。这是本期最实用的一条：**用 spec 方法做纯提取要付 spec 的代价**。
2. **窗口化 hidden-state 捕获扩到 teacher scoring**。[#57736](https://github.com/vllm-project/vllm/pull/57736) 标题与范围改为「capture selected hidden-state rows for rollout windows and teacher scoring」，同一原语服务两类负载。但主仓仍**没有** `return_hidden_states` 响应字段，`aux_output_connector` 只有 `routed_experts` payload，没有 `hidden_states.py`（源码核对见下）。
3. **PP × aux 仍是逐模型 opt-in**。Qwen3.5 的缺口 issue [#60765](https://github.com/vllm-project/vllm/issues/60765) 与实现 [#61180](https://github.com/vllm-project/vllm/pull/61180)、Qwen3-MoE 的 [#64320](https://github.com/vllm-project/vllm/pull/64320) 本期都还在 open；#63205 把 speculator 配置对象化（新增 `extract_hidden_states_config.py`），是扩模型的接口基础。
4. **vllm-ascend 本期的主战场是 aux 层号正确性**。[#21022](https://github.com/vllm-project/vllm-ascend/issues/21022) 报 DeepSeek V4.1 在 MRV2 上 **DSpark aux 早抓一层**（零基 attention-input 边界换算错），[#21023](https://github.com/vllm-project/vllm-ascend/pull/21023) 修复；同时 [#16993](https://github.com/vllm-project/vllm-ascend/pull/16993) 把 DSpark aux 的 ReduceMean 融进 HCPost、[#16621](https://github.com/vllm-project/vllm-ascend/pull/16621) 跳过未被消费的 aux buffer 操作、[#19157](https://github.com/vllm-project/vllm-ascend/pull/19157) 打通 V4.1 Flash 的 PP。另一侧 [#8513](https://github.com/vllm-project/vllm-ascend/issues/8513) 仍是 `Unknown speculative decoding method: extract_hidden_states`。

## 增量索引

### vLLM（10 条）

- [#57736](https://github.com/vllm-project/vllm/pull/57736) — 前期补漏续跟；期末 `open`；窗口化 hidden-state 捕获（扩到 teacher scoring）。
- [#66050](https://github.com/vllm-project/vllm/pull/66050) — 本期新建；期末 `open`；[Model] Ming ASR 音频条件 hidden-state 提取。
- [#65991](https://github.com/vllm-project/vllm/pull/65991) — 本期新建；期末 `open`；[Spec Decode] 关掉 extract_hidden_states decode draft 的开关。
- [#65250](https://github.com/vllm-project/vllm/issues/65250) — 本期新建；期末 `open`；extract_hidden_states 损失 ~38% 吞吐。
- [#60765](https://github.com/vllm-project/vllm/issues/60765) — 前期补漏续跟；期末 `open`；Qwen3.5 在 PP 下无法跑 dflash（缺 aux opt-in）。
- [#64320](https://github.com/vllm-project/vllm/pull/64320) — 前期补漏续跟；期末 `open`；Qwen3-MoE EAGLE3 aux 跨 PP relay。
- [#61180](https://github.com/vllm-project/vllm/pull/61180) — 前期补漏续跟；期末 `open`；Qwen3.5 aux 跨 PP relay（修 #60765）。
- [#63205](https://github.com/vllm-project/vllm/pull/63205) — 前期补漏续跟；期末 `merged`（`2026-10-30T10:52:23-07:00`）；Speculator 配置类与自定义 speculator。
- [#63406](https://github.com/vllm-project/vllm/pull/63406) — 前期补漏续跟；期末 `merged`（`2026-11-03T22:50:27-08:00`）；prefill/decode 图捕获尺寸拆分。
- [#61539](https://github.com/vllm-project/vllm/issues/61539) — 前期补漏续跟；期末 `open`；routed-expert 捕获的显式行布局 RFC。

### vllm-ascend（9 条）

- [#21023](https://github.com/vllm-project/vllm-ascend/pull/21023) — 本期新建；期末 `open`；DeepSeek V4.1 DSpark aux 捕获边界修复。
- [#21022](https://github.com/vllm-project/vllm-ascend/issues/21022) — 本期新建；期末 `open`；DSpark aux 早抓一层的 bug 报告。
- [#16621](https://github.com/vllm-project/vllm-ascend/pull/16621) — 前期补漏续跟；期末 `open`；跳过未被消费的 MTP hidden-state buffer 操作。
- [#16993](https://github.com/vllm-project/vllm-ascend/pull/16993) — 前期补漏续跟；期末 `open`；DSpark aux ReduceMean 融进 HCPost。
- [#19157](https://github.com/vllm-project/vllm-ascend/pull/19157) — 前期补漏续跟；期末 `open`；DeepSeek V4.1 Flash PP（含 aux 传输）。
- [#8513](https://github.com/vllm-project/vllm-ascend/issues/8513) — 前期补漏续跟；期末 `open`；`Unknown speculative decoding method: extract_hidden_states`。
- [#17907](https://github.com/vllm-project/vllm-ascend/pull/17907) — 前期补漏续跟；期末 `open`；适配 `SamplingMaskTensors.from_logits` 新签名（mask 相邻）。
- [#21056](https://github.com/vllm-project/vllm-ascend/pull/21056) — 本期新建；期末 `open`；draft KV 滑窗。
- [#20617](https://github.com/vllm-project/vllm-ascend/pull/20617) — 本期新建；期末 `merged`（`2026-11-03T17:43:08+08:00`）；用模型 tag 识别 V2 EAGLE MLA draft。

## vLLM：机制、证据与 NPU 跟进

### #65250 / #65991 · P1 · 本期新建（成本量化 + 开关）

[#65250](https://github.com/vllm-project/vllm/issues/65250)：`extract_hidden_states` 每个 decode step 给每个请求排一个 draft token，接受率约 0.03%；L4 上饱和吞吐损失 37%–39%。[#65991](https://github.com/vllm-project/vllm/pull/65991) 加 `disable_extract_hidden_states_decode_drafts=True`：把动态 SD 调度设为 K=0（`[(1,1,0)]`），约 45 行非测试改动，计划额外报一个 warning 提示「关掉后不再有 decode draft」。

NPU 判断：需实测。Ascend 上 draft 前向更贵，同一方法的开销占比可能更高；需要按接受率与吞吐自测，不能引用 L4 数字。

### #57736 · P0 · 前期补漏续跟（范围改写）

[feat: capture selected hidden-state rows for rollout windows and teacher scoring](https://github.com/vllm-project/vllm/pull/57736)：同一原语服务两类负载——按绝对位置区间抓 rollout 窗口行（drafter 训练数据），以及教师评分需要的行；仍经 AuxOutput Connector 返回，支持最终 hidden states 与配置的投机辅助输出。本期 12 个提交、33 文件、4 条非噪声事件，仍 open。

NPU 判断：缺失。主仓尚无公共返回字段；Ascend 需先有接口再评估 connector 的 D2H 与图重放 buffer。

### #60765 / #61180 / #64320 · P1 · PP × aux 的逐模型 opt-in

- [#60765](https://github.com/vllm-project/vllm/issues/60765)：Qwen3.5（hybrid GDN）在 PP 下跑 dflash 时，因缺 `supports_aux_hidden_states_over_pp` opt-in，引擎初始化被拒（用户环境 PP2、RTX 3090 + A4000 异构）。
- [#61180](https://github.com/vllm-project/vllm/pull/61180)：为 Qwen3.5 提供 relay 实现（修 #60765），含 NVIDIA 侧；本期更新到 11-06。
- [#64320](https://github.com/vllm-project/vllm/pull/64320)：Qwen3-MoE 的 EAGLE3 aux relay，沿用 Qwen2（#50514）与 MiniMax-M3（#57197）模式；本期更新到 11-04。

NPU 判断：缺失/需实测。Ascend 的 opt-in 名单只覆盖 DeepSeek-V4、Kimi-K3、MiniMax-M3 与 DeepseekV2 补丁，Qwen3.5 / Qwen3-MoE 不在其中（源码核对见下），PP 下同样会被拒。

### #63205 / #63406 · P1/P2 · 本期合入

- [#63205](https://github.com/vllm-project/vllm/pull/63205)（`2026-10-30T10:52:23-07:00`，67 文件）：MRV2 的 speculator 配置类与自定义 speculator，新增 `vllm/v1/worker/gpu/spec_decode/extract_hidden_states_config.py`，把该方法从「名字 + 散落参数」收敛成配置对象。
- [#63406](https://github.com/vllm-project/vllm/pull/63406)（`2026-11-03T22:50:27-08:00`，17 文件）：prefill（token）与 decode（request）分别使用 CUDA Graph 捕获尺寸，减少每个尺寸保留 aux hidden states 的常驻开销（与 #63427 的弱引用修复同一目标）。

NPU 判断：需实测。Ascend 的 speculator 需对齐配置对象；图池按什么维度分配尺寸要对照，否则 aux 常驻占用可能不降反升。

### #66050 / #61539 · P1/P2 · 扩展与 RFC

- [#66050](https://github.com/vllm-project/vllm/pull/66050)：为 Ming-3.0-ASR 打开音频条件下的 decoder hidden-state 提取（走离线 `LLM` API），blocked on #54174。hidden-state 提取开始扩到多模态。
- [#61539](https://github.com/vllm-project/vllm/issues/61539)：routed-expert 捕获的显式行布局 RFC，与 hidden-state 捕获同属「行布局必须显式」问题族；本期仅状态更新。

## vllm-ascend：机制、证据与 NPU 跟进

### #21022 / #21023 · P0 · 本期新建（aux 早抓一层）

[#21022](https://github.com/vllm-project/vllm-ascend/issues/21022)（A3 910_9392、CANN 9.2.0、1P1D、DP8/TP2、DSpark 5 tokens）：DeepSeek V4.1 checkpoint 的 `dspark_target_layer_ids=[37,38,39]` 是**零基 attention-input 边界**，MRV2 上 `set_eagle3_aux_hidden_state_layers()` 原样透传后，Ascend 侧又多减一层，导致 **aux 比预期早抓一层**。[#21023](https://github.com/vllm-project/vllm-ascend/pull/21023) 只改边界换算，不动捕获机制。

NPU 判断：这是 Ascend 侧最典型的 aux 错误形态——不崩溃、不报错，只是**早一层**，会让 DSpark 训练数据与推理不一致。同类零基/一基约定应逐模型核对（DFlash/DFlash2 也在列）。

### #16621 / #16993 · P1 · aux 传输与计算的开销治理

- [#16621](https://github.com/vllm-project/vllm-ascend/pull/16621)：Ascend 的 DeepSeek V4 已按层收集 aux 并经 `PPTransportDataType.AUX_HIDDEN_STATES` 跨 PP 传输，但 draft 侧仍对**不会被消费**的 buffer 做拷贝/清零；本条跳过这些操作。风险是删除后必须确认 PP 中段的 aux 槽位仍被正确初始化（否则脏数据而非崩溃）。
- [#16993](https://github.com/vllm-project/vllm-ascend/pull/16993)：reapply #16544，并把 DSpark aux 的 ReduceMean 融进 HCPost（`npu_hc_post_with_mean`）；只有配置的 aux 目标层请求 mean，非目标层仍走 `npu_hc_post`，且需保持 SP 顺序。这正是「hc 流取均值」的 Ascend 融合版（对应上游 `hidden_states.mean(dim=1)`）。

NPU 判断：融合后必须与 torch 顺序位级一致（含 `-0` 行与 dtype 提升），否则 aux 数值偏移会静默传播到训练数据。

### #19157 · P1 · 前期补漏续跟（PP 端到端）

[#19157](https://github.com/vllm-project/vllm-ascend/pull/19157)：在 MRV2 上支持 DeepSeek V4.1 Flash 的 PP——stage-local Engram、mHC/auxiliary-state transport、投影后的 KV-cache 布局与 DSpark+PP 一起打通，并为旧 vLLM 版本通过 model-state hook 恢复原始 token ids。属「PP 下 aux 状态可搬」的端到端模型支持。

### #8513 · P1 · 前期补漏续跟（方法分派缺口）

`ValueError: Unknown speculative decoding method: extract_hidden_states`：用户按上游文档跑 KV connector 的 `extract_hidden_states` 集成测试时，Ascend 侧方法分派不认识该方法。说明「Ascend 认识这个方法名」与「该方法在 Ascend 可用」仍有差距——`ascend_config.py` 只在校验层出现该名字。

### #17907 / #21056 / #20617 · P2 · 相邻

- [#17907](https://github.com/vllm-project/vllm-ascend/pull/17907)：适配 vLLM #59359 改过的 `SamplingMaskTensors.from_logits` 新签名（新增 `cu_num_logits`、`rows_per_request`），让 request-major mask 打包继续工作（采样 mask 主题）。
- [#21056](https://github.com/vllm-project/vllm-ascend/pull/21056)：`additional_config["draft_kv_window"]` 把 EAGLE/EAGLE3/MTP 的 draft KV 限制在最近 W 个 token。
- [#20617](https://github.com/vllm-project/vllm-ascend/pull/20617)（`2026-11-03T17:43:08+08:00` 合入）：用上游 model tag 识别 V2 EAGLE MLA draft，识别错会影响 aux 读取路径。

## 源码核对（`de05fd1db93f` / `1c2efa8ac24b`）

- **vLLM 仍无公共 `return_hidden_states`**：`vllm/entrypoints/openai/protocol.py` 中无该字段。
- **`aux_output_connector` 只有 routed-experts payload**：目录内为 `__init__.py`、`connector.py`、`routed_experts.py`、`store.py`、`worker.py`，**没有** `hidden_states.py`；scheduler 已接 `aux_output_connector_metadata`（`v1/core/sched/output.py`）。也就是说「窗口化 hidden state」仍未进 main。
- **vLLM 的 aux-over-PP 名单**：`supports_aux_hidden_states_over_pp = True` 在 llama、qwen2、ricursive、sarvam、deepseek_v4(nvidia)、kimi_k3、minimax_m3(amd/nvidia)。
- **vllm-ascend 的名单**：`models/deepseek_v4/model.py:837`、`models/kimi_k3.py:762`、`models/minimax_m3/minimax_m3.py:1110`，加 `patch/worker/patch_deepseek_v2.py:339` 的 `DeepseekV2Model` 补丁；Qwen3.5 / Qwen3-MoE 不在其中。
- **`extract_hidden_states` 仍是 spec 方法**：需 `draft_sample_method='greedy'`、`num_speculative_tokens == 1`、draft config 给出 `eagle_aux_hidden_state_layer_ids`；#63205 之后另有 `extract_hidden_states_config.py` 承载配置对象。

## 无实质进展与补漏边界

- [#50187](https://github.com/vllm-project/vllm/pull/50187)、[#36998](https://github.com/vllm-project/vllm/issues/36998)、[#37002](https://github.com/vllm-project/vllm/pull/37002)：无本期更新。
- vllm-ascend [#19866](https://github.com/vllm-project/vllm-ascend/pull/19866) / [#20129](https://github.com/vllm-project/vllm-ascend/pull/20129)：属采样 mask 主题，仅相邻提及。
- vLLM 主仓本期还合入 #62702（prompt token ids 改 int32），仅间接影响缓冲布局，未纳入。

## 检索与复核

- [本期结构化数据](vllm-extract-hidden-states-delta-2026-11-06.json) · [本期 CSV](vllm-extract-hidden-states-delta-2026-11-06.csv)。
- 本期未运行 GPU/NPU 或模型测试；作者性能数字均为作者报告。

## 扫描覆盖与源码时间说明

- 关键词检索：vLLM 9 个短语 × `updated:>=2026-10-30`、vllm-ascend 11 个短语 × `updated:>=2026-10-23`，均加 RFC 标签与按日 `created:` 切分；去重后 vLLM 1665 条（1367 PR + 298 Issue）、vllm-ascend 1008 条（832 PR + 176 Issue）。
- 合入侧：两个浅克隆分别扫描窗口内 598（vLLM）与 205（vllm-ascend）个提交；合入条目直接以提交为证据。
- 候选：vLLM 88 条、vllm-ascend 30 条逐条判定，纳入 19 条。
- 源码快照：vLLM `de05fd1db93f712288e654bc28a8480a67a7a8e2`（`2026-11-06T06:11:41Z`）、vllm-ascend `1c2efa8ac24ba14fd69f12609b850d4e6daf0865`（`2026-11-06T04:08:21Z`）。
