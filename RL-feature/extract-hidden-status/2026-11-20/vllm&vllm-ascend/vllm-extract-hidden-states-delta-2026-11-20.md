# vLLM / vllm-ascend hidden state extract / return / aux relay 增量跟踪：2026-11-20

UTC 窗口：`[2026-11-13T04:40:41Z, 2026-11-20T06:54:42Z)`。采集截止：`2026-11-20T06:54:42Z`。本报告同时覆盖 **vLLM 主仓**与 **vllm-ascend**；SGLang 侧见 [../sglang/sglang-extract-hidden-states-delta-2026-11-20.md](<../sglang/sglang-extract-hidden-states-delta-2026-11-20.md>)。

代码基准：vLLM [main `aefea5c8e00c`](https://github.com/vllm-project/vllm/commit/aefea5c8e00c47032cda6d0b75704f6cd0a7c66d)（`2026-11-20T14:42+08:00`）；vllm-ascend [main `1464e7683088`](https://github.com/vllm-project/vllm-ascend/commit/1464e76830886683866aa70f941c6ff41ff6b19e)（`2026-11-20T14:53+08:00`）。

## 本期结论

纳入 **15 条**（vLLM 9 + vllm-ascend 6），本期新建 8 条、前期补漏续跟 7 条；5 条合入（vLLM #67648、#66323，vllm-ascend #22122、#22074、#21469，其中 #21469 为 main2main）。

本期主题是**「少一层」与「错一层」两类静默错误被同时修掉**：

1. **vLLM 丢最后一层 aux**：[#68813](https://github.com/vllm-project/vllm/issues/68813) 报告 DeepSeek-V3.2 / GLM-5.x 的 `extract_hidden_states` 返回层数比请求**少一个**（GLM-5.2、78 层、TP=8、H200）。根因在 [#68824](https://github.com/vllm-project/vllm/pull/68824)：`DeepseekV32Model.forward` 只在 `idx in [start_layer, end_layer)` 内记录 aux，`end_layer == num_hidden_layers` 时最后一层落在循环外。修复把最终层显式纳入。
2. **vllm-ascend 早抓一层被真正修掉**：[#22122](https://github.com/vllm-project/vllm-ascend/pull/22122)（`2026-11-17T19:21+08:00` 合入，4 文件）说明 MRV2 resolver 保留**零基**层号、target 侧却用**一基**比较，于是 `[37,38,39]` 实际选到第 36/37/38 层输入。上期 #21022/#21023 的问题是同一个，现在由已合入的 #22122 收口。
3. **层号约定被写进文档/解析**：[#68769](https://github.com/vllm-project/vllm/pull/68769) 读取 EAGLE-3 checkpoint 的嵌套 `eagle_config['eagle_aux_hidden_state_layer_ids']` 并写明 aux 层号约定——此前只查 drafter 风格键与顶层属性，导致这类 checkpoint 的层号被静默忽略。
4. **PP 阶段间 hidden 语义**：[#67648](https://github.com/vllm-project/vllm/pull/67648)（`2026-11-20T00:38:48Z` 合入）修 Transformers 后端在后续 PP 阶段重复应用预处理（learned pos-emb、embedding scaling）的问题。

其余方向：CPU-pages 仍 open（vLLM #66785 / Ascend #21792）、模型侧 aux 继续逐个补（#66323 合入、#68289 新增）、成本开关 #65991 仍 open。

## 增量索引

### vLLM（9 条）

- [#68813](https://github.com/vllm-project/vllm/issues/68813) — 本期新建；`open`；[Bug] 最后一层 aux 丢失，返回层数少一。
- [#68824](https://github.com/vllm-project/vllm/pull/68824) — 本期新建；`open`；[Bugfix] capture final-layer aux hidden state in DeepSeek-V3.2。
- [#68769](https://github.com/vllm-project/vllm/pull/68769) — 本期新建；`open`；[Bugfix][SpecDecode] 读嵌套层号 + 写明 aux id 约定。
- [#67648](https://github.com/vllm-project/vllm/pull/67648) — 本期新建；`merged`（`2026-11-20T00:38:48Z`）；[Bugfix][Transformers] preserve received hidden states on later PP stages。
- [#66323](https://github.com/vllm-project/vllm/pull/66323) — 前期补漏续跟；`merged`（`2026-11-16T03:56:51Z`）；[Model] EAGLE-3/DSpark aux for Lfm3。
- [#68289](https://github.com/vllm-project/vllm/pull/68289) — 本期新建；`open`；[Model] EAGLE-3 aux for EXAONE 4 / DeciLM。
- [#66785](https://github.com/vllm-project/vllm/pull/66785) — 前期补漏续跟；`open`；extract_hidden_states → CPU pages。
- [#65991](https://github.com/vllm-project/vllm/pull/65991) — 前期补漏续跟；`open`；关掉 extract_hidden_states decode draft 的开关。
- [#68634](https://github.com/vllm-project/vllm/pull/68634) — 本期新建；`open`；[Feature][RL] 独立 sampling-mask opt-out（mask 相邻）。

### vllm-ascend（6 条）

- [#22122](https://github.com/vllm-project/vllm-ascend/pull/22122) — 本期新建；`merged`（`2026-11-17T19:21:41+08:00`）；[Bugfix] 修正 DeepSeek V4.1 DSpark aux 层号捕获。
- [#22074](https://github.com/vllm-project/vllm-ascend/pull/22074) — 本期新建；`merged`（`2026-11-18T20:09:58+08:00`）；[BugFix] 保留 DFlash2/DSpark 的 draft LM-head TP。
- [#21792](https://github.com/vllm-project/vllm-ascend/pull/21792) — 前期补漏续跟；`open`（11-17 更新）；`hs_store_device=cpu` for extract_hidden_states。
- [#21023](https://github.com/vllm-project/vllm-ascend/pull/21023) — 前期补漏续跟；`open`（已被 #22122 替代）；DSpark aux 捕获边界。
- [#21469](https://github.com/vllm-project/vllm-ascend/pull/21469) — 前期补漏续跟；`merged`（`2026-11-19T16:46:20+08:00`）；main2main 1113。
- [#17557](https://github.com/vllm-project/vllm-ascend/pull/17557) — 前期补漏续跟；`closed`（未合入）；GLM-5.3-Flash DFlash2 on MRV2（落点为 #21439）。

## 重点条目

### #68813 / #68824 · P0 · 本期新建（少一层）

[#68813](https://github.com/vllm-project/vllm/issues/68813)：DeepSeek-V3.2 / GLM-5.x 的 `extract_hidden_states` 返回层数比请求少一，阻塞 EAGLE-3/DFlash 训练数据生成。[#68824](https://github.com/vllm-project/vllm/pull/68824)：`DeepseekV32Model.forward`（nvidia 与 amd 两份）只在 decoder layer 循环内 `idx in [start_layer, end_layer)` 记录 aux；`end_layer == num_hidden_layers` 时最后一层在循环外，永不捕获。修复显式纳入最终层。

NPU 判断：**必须逐模型核对**。Ascend 的 V4/V4.1 与 GLM 实现若同一循环边界，同样少一层；验收断言应是「返回层数 == 请求层数」，而不是「拿到非空张量」。这类错误不会崩溃，只会让 draft 输入维度与 checkpoint 约定不符。

### #22122 · P0 · 本期新建（Ascend 合入，早抓一层）

[[Bugfix] Correct DeepSeek V4.1 DSpark auxiliary layer capture](https://github.com/vllm-project/vllm-ascend/pull/22122)，`2026-11-17T19:21+08:00` 合入，4 文件。

机制：MRV2 的 resolver 保留 checkpoint 的**零基**层号，target 侧此前用**一基**比较，因此 `[37,38,39]` 选到的是第 36/37/38 层的输入。修复统一为零基比较。

NPU 判断：这是「层号约定错一层」的标准修复样本。同类风险仍在其他模型（GLM、DFlash2 各自实现捕获点），应按模型逐个核对并固化为断言。

### #68769 · P1 · 本期新建（约定与解析）

[[Bugfix][SpecDecode] Read nested eagle_aux_hidden_state_layer_ids and document the aux id convention](https://github.com/vllm-project/vllm/pull/68769)，`open`。

机制：参考实现转换出的 EAGLE-3 checkpoint 把层号放在 `eagle_config['eagle_aux_hidden_state_layer_ids']`；`get_eagle3_aux_layers_from_config()` 只查 drafter 风格的 `layer_ids` 与顶层属性，导致层号被静默忽略（回落到默认层）。修复读嵌套键并写明约定。

NPU 判断：需实测。Ascend 侧解析同一配置，嵌套键与零基/一基约定必须一致——否则会出现「配置正确但捕获层不对」。

### #67648 · P1 · 本期新建（合入，PP hidden 语义）

[[Bugfix][Transformers] Preserve received hidden states on later pipeline stages](https://github.com/vllm-project/vllm/pull/67648)，`2026-11-20T00:38:48Z` 合入，4 文件。后续 PP 阶段不再重复应用 decoder 对输入的预处理（learned position embeddings、embedding scaling），修掉 GPTBigCode 的 PP 缺陷（#66290）。NPU 若走 Transformers 后端 PP 需同样语义。

### #66323 / #68289 · P1/P2 · 模型侧继续逐个补

- [#66323](https://github.com/vllm-project/vllm/pull/66323)（`2026-11-16T03:56:51Z` 合入）：Lfm3ForCausalLM 的 EAGLE-3/DSpark aux 支持，使其可作为 target。
- [#68289](https://github.com/vllm-project/vllm/pull/68289)（open）：EXAONE 4 与 DeciLM 的 EAGLE-3 aux。
- 仍说明：aux 捕获/PP relay 是**逐模型声明**，不是全局能力；新增模型 ≠ 新增硬件支持。

### #21792 / #21023 / #21469 / #17557 · Ascend 其余

- [#21792](https://github.com/vllm-project/vllm-ascend/pull/21792)（open，11-17 更新）：`hs_store_device=cpu` 把抽取的 hidden states 搬到主机每请求分页，缓解 NPU cache-only KV 池压力；仍无 D2H/回收/dtype 三项设备数据。
- [#21023](https://github.com/vllm-project/vllm-ascend/pull/21023)（open）：与已合入的 #22122 同题，应视为 superseded；跟踪能力时以 #22122 为准。
- [#21469](https://github.com/vllm-project/vllm-ascend/pull/21469)（`2026-11-19T16:46+08:00` 合入，123 文件）：main2main 1113，对齐上游。上游本周期发生了 aux relay 集中化（#66177）与 CPU-pages（#66785），需要确认 Ascend 自有 aux 传输与集中化后的 `PPHandler` 是否一致。
- [#17557](https://github.com/vllm-project/vllm-ascend/pull/17557)（closed）：GLM-5.3-Flash DFlash2 的 MRV2 路径关闭，落点是已合入的 #21439。

### #22074 · P1 · 本期新建（Ascend 合入）

[[BugFix] Preserve draft LM-head TP for DFlash2 and DSpark](https://github.com/vllm-project/vllm-ascend/pull/22074)，`2026-11-18T20:09:58+08:00` 合入，15 文件。DFlash2 在 `finegrained_tp_config.lmhead_tensor_parallel_size > 1` 时构造失败：candidate selector 绕过常规 draft-logits hook，DP rank 上的请求未对齐。修复保留 draft LM-head 的 TP 语义。与 aux 的关联：selector 直接消费 target 的 aux/logits，绕过 hook 就会让「aux 行 → 请求」映射错位。

### #66785 / #65991 / #68634 · 仍 open

- [#66785](https://github.com/vllm-project/vllm/pull/66785)：CPU pages + token-hash 前缀回填，11-16 更新。
- [#65991](https://github.com/vllm-project/vllm/pull/65991)：`disable_extract_hidden_states_decode_drafts`（K=0），11-19 更新。
- [#68634](https://github.com/vllm-project/vllm/pull/68634)：RL 独立 sampling-mask opt-out 原型（mask 相邻，非 hidden-state）。

## 无实质进展与补漏边界

- vLLM [#57736](https://github.com/vllm-project/vllm/pull/57736)（窗口化捕获）11-17 更新但无实质合入；[#64320](https://github.com/vllm-project/vllm/pull/64320)、[#56442](https://github.com/vllm-project/vllm/pull/56442) 本期无活动。
- vllm-ascend #21735（删 V1 AUX 分支）上期合入（11-13），本期无新增；#11066 无活动。
- vLLM #68871/#68624（XPU mask dtype）、#67542（AMD mask D2H race）属 mask 主题且多已关闭，未列条。

## 检索与复核

- [本期结构化数据](vllm-extract-hidden-states-delta-2026-11-20.json) · [本期 CSV](vllm-extract-hidden-states-delta-2026-11-20.csv)。
- 使用已认证 `gh` CLI 拉取；合入条目以本地浅克隆的合并提交为证据。
- 未运行 GPU/NPU 或模型测试。

## 扫描覆盖与源码时间说明

- 关键词检索：vLLM 9 个短语、vllm-ascend 10 个短语 × `updated:>=2026-11-13`，加 RFC 标签与 `created:` 按日切分；去重后 vLLM 1728 条、vllm-ascend 963 条（771 PR + 192 Issue）。
- 合入侧：两个浅克隆分别扫描窗口内 630（vLLM）与 239（vllm-ascend）个提交。
- 源码快照：vLLM `aefea5c8e00c47032cda6d0b75704f6cd0a7c66d`、vllm-ascend `1464e76830886683866aa70f941c6ff41ff6b19e`。
