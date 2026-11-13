# vLLM / vllm-ascend hidden state extract / return / aux relay 增量跟踪：2026-11-13

UTC 窗口：`[2026-11-06T06:35:52Z, 2026-11-13T04:40:41Z)`。采集截止：`2026-11-13T04:40:41Z`。本报告同时覆盖 **vLLM 主仓**与 **vllm-ascend**；SGLang 侧见 [../sglang/sglang-extract-hidden-states-delta-2026-11-13.md](<../sglang/sglang-extract-hidden-states-delta-2026-11-13.md>)。

代码基准：vLLM [main `c410543d9fff`](https://github.com/vllm-project/vllm/commit/c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15)（`2026-11-13T12:39+08:00`）；vllm-ascend [main `671292ee7ee2`](https://github.com/vllm-project/vllm-ascend/commit/671292ee7ee2c6a620f307ce7930d84d772f4d5a)（`2026-11-13T11:42+08:00`）。

## 本期结论

纳入 **18 条**（vLLM 11 + vllm-ascend 7），其中 10 条本期新建、8 条前期补漏续跟；4 条合入（vLLM #66177、#66063、#66546，vllm-ascend #21439）。

本期是这条特性的**结构性变化期**，四条主线：

1. **aux relay 在 vLLM 主仓被集中化**：[#66177](https://github.com/vllm-project/vllm/pull/66177)（`2026-11-09T14:27+08:00`，23 文件）把 EAGLE3/DFlash/DSpark 的 aux relay 与组装收进 MRV2 的 `PPHandler`，使 gpt-oss、Qwen3-VL-MoE、Gemma4 能跑 PP。源码核对：`vllm/v1/worker/gpu/pp_utils.py:100` 的 `PPHandler` 现在有 `configure_aux_hidden_state_relay` / `relay_aux_hidden_states` / `add_upstream_aux_hidden_states`，并读 `decoder.aux_hidden_state_slot_keys()`。**relay 从「每模型自己搬」变成「公共层搬 + 模型声明槽位」。**
2. **draft 成为独立 PP 阶段**：[#66063](https://github.com/vllm-project/vllm/pull/66063)（`2026-11-12T13:40-08:00`，57 文件）把 draft 模型作为单个 PP 阶段运行，target/draft 的阶段拓扑随之改变，aux 的产出与消费跨不同阶段集合。
3. **提取路径开始离开 KV 池**：[#66785](https://github.com/vllm-project/vllm/pull/66785)（+ RFC [#66794](https://github.com/vllm-project/vllm/issues/66794)）给 `extract_hidden_states` 加 `hs_store_device=cpu`：hidden states 经 KV connector 流式写主机侧每请求 CPU 分页，用 token-hash 前缀回填处理缓存命中。**同一天**，vllm-ascend 出现对应实现 [#21792](https://github.com/vllm-project/vllm-ascend/pull/21792)（`hs_store_device=cpu` 适配 MRV2）——这是本期最值得关注的跨仓同步。
4. **Ascend 侧在收敛模型侧 aux 接口**：GLM-5.3-Flash DFlash2 合入 [#21439](https://github.com/vllm-project/vllm-ascend/pull/21439)（实现 Eagle3 aux 接口 + mHC 延后残差补全 + SP 安全捕获），[#21735](https://github.com/vllm-project/vllm-ascend/pull/21735) 删除不可达的 V1 AUX 分支（消除双轨），[#21501](https://github.com/vllm-project/vllm-ascend/issues/21501) RFC 提出「target 侧 aux 接口应与 draft 方法无关」。层号问题仍在：[#21023](https://github.com/vllm-project/vllm-ascend/pull/21023)（DSpark aux 早抓一层）本期还没合入。

## 增量索引

### vLLM（11 条）

- [#66177](https://github.com/vllm-project/vllm/pull/66177) — 本期新建；`merged`（`2026-11-09T14:27:11+08:00`）；Centralize auxiliary hidden-state relay。
- [#66063](https://github.com/vllm-project/vllm/pull/66063) — 本期新建；`merged`（`2026-11-12T13:40:32-08:00`）；Run draft models as a single pipeline stage。
- [#66546](https://github.com/vllm-project/vllm/pull/66546) — 本期新建；`merged`（`2026-11-09T19:52:26Z`）；Transformers backend EAGLE3 aux + layer 0 under PP。
- [#66785](https://github.com/vllm-project/vllm/pull/66785) — 本期新建；`open`；extract_hidden_states → CPU pages with token-hash backfill。
- [#66794](https://github.com/vllm-project/vllm/issues/66794) — 本期新建；`open`；上条 RFC。
- [#66323](https://github.com/vllm-project/vllm/pull/66323) — 本期新建；`open`；Lfm3 的 EAGLE-3/DSpark aux 支持。
- [#56442](https://github.com/vllm-project/vllm/pull/56442) — 前期补漏续跟；`open`；多 token 时用最后一个有效 token。
- [#65991](https://github.com/vllm-project/vllm/pull/65991) — 前期补漏续跟；`open`；关闭 extract_hidden_states decode draft 的开关。
- [#57736](https://github.com/vllm-project/vllm/pull/57736) — 前期补漏续跟；`open`；窗口化捕获（rollout + teacher scoring）。
- [#64320](https://github.com/vllm-project/vllm/pull/64320) — 前期补漏续跟；`open`；Qwen3-MoE EAGLE3 aux 跨 PP。
- [#61539](https://github.com/vllm-project/vllm/issues/61539) — 前期补漏续跟；`closed`（未合入）；routed-expert 显式行布局 RFC。

### vllm-ascend（7 条）

- [#21439](https://github.com/vllm-project/vllm-ascend/pull/21439) — 本期新建；`merged`（`2026-11-12T16:21:24+08:00`）；GLM-5.3-Flash DFlash2。
- [#21792](https://github.com/vllm-project/vllm-ascend/pull/21792) — 本期新建；`open`；`hs_store_device=cpu` for extract_hidden_states on MRV2。
- [#17557](https://github.com/vllm-project/vllm-ascend/pull/17557) — 前期补漏续跟；`open`；GLM-5.3-Flash DFlash2 on MRV2（与 #21439 同模型不同路径）。
- [#11066](https://github.com/vllm-project/vllm-ascend/pull/11066) — 前期补漏续跟；`open`；GLM-5.3-Flash DSpark target states 与 draft caches。
- [#21735](https://github.com/vllm-project/vllm-ascend/pull/21735) — 本期新建；`open`；删除 V1 spec 兼容路径（消除 AUX 双轨）。
- [#21023](https://github.com/vllm-project/vllm-ascend/pull/21023) — 前期补漏续跟；`open`；DSpark aux 捕获边界修复。
- [#21501](https://github.com/vllm-project/vllm-ascend/issues/21501) — 本期新建；`open`；RFC：GLM-5.3 target 侧架构（draft-method-agnostic）。

## 重点条目

### #66177 · P0 · 本期新建（合入）

[[Spec Decode][PP] Centralize auxiliary hidden-state relay](https://github.com/vllm-project/vllm/pull/66177)，`2026-11-09T14:27+08:00` 合入，23 文件。

机制：把 EAGLE3 / DFlash / DSpark 的 aux hidden-state relay 与组装集中到 MRV2 `PPHandler`。模型只需声明槽位（`aux_hidden_state_slot_keys()`、`aux_hidden_state_layers`），公共层负责打包/转发/组装；PR 说明这解锁了 gpt-oss、Qwen3-VL-MoE、Gemma4 的 PP。

NPU 判断：**需实测，且是本期的关键决策点**。vllm-ascend 目前是模型自己搬 aux（`pp_transport_aux_hidden_states_`）；集中化后有两种可能：Ascend 改走公共 `PPHandler`，或继续双轨。双轨并存时，槽位顺序与 full-token 语义必须两边一致，否则 aux 静默错位。

### #66063 · P0 · 本期新建（合入）

[[Spec Decode][PP] Run draft models as a single pipeline stage](https://github.com/vllm-project/vllm/pull/66063)，`2026-11-12T13:40-08:00` 合入，57 文件。

机制：draft 模型作为单个 PP 阶段运行，target/draft 不再共享同一阶段切分。对 aux 的意义：aux 的产出阶段集合与消费阶段集合分离，relay 的拓扑假设要重算。

NPU 判断：需实测。Ascend 的 `supports_aux_hidden_states_over_pp` 判定与 PP 拓扑强相关，需要按新拓扑重新验证每个模型。

### #66785 / #66794 · P0/P1 · 本期新建（CPU pages）

[#66785](https://github.com/vllm-project/vllm/pull/66785)（RFC [#66794](https://github.com/vllm-project/vllm/issues/66794)）：`hs_store_device=cpu`，hidden states 不再占 cache-only KV 池，而是经 KV connector 流式写主机侧每请求 CPU 分页，并用 token-hash 前缀回填处理缓存命中。

NPU 判断：**跨仓同步已发生**——vllm-ascend [#21792](https://github.com/vllm-project/vllm-ascend/pull/21792) 同一天把该特性适配到 MRV2。但 Ascend 的 D2H 带宽与分页语义需独立验证：分页切换成本、请求结束回收、以及 aux 行在 host 上的 dtype 往返（参考 SGLang 侧 #47445 的 msgpack dtype 修复）。

### #21439 · P0 · 本期新建（Ascend 合入）

[[Feature][SpecDecode] Support GLM-5.3-Flash DFlash2](https://github.com/vllm-project/vllm-ascend/pull/21439)，`2026-11-12T16:21+08:00` 合入，12 文件。

机制：GLM-5.3-Flash 原本起不了 DFlash2，因为 target 没有暴露 Eagle3 aux 接口；本条在 text 与多模态模型上实现该接口，包含 **mHC 层的延后 `hc_post` 残差补全**与 **SP 安全捕获**。同题更早的 #21431 已关闭，以本条为准。

NPU 判断：Ascend 模型侧 aux 接口的标准形态——mHC 残差补全必须延后到 aux 捕获之后再合并，否则 hidden 语义与上游不一致；SP 下捕获还需 all-gather 还原 full-token。

### #21792 · P0 · 本期新建（Ascend CPU pages）

[[SpecDecode][Feature] Support hs_store_device=cpu for extract_hidden_states on MRV2](https://github.com/vllm-project/vllm-ascend/pull/21792)，`open`。

机制：把上游 #66785 的 CPU-pages 路径适配到 vllm-ascend MRV2：抽取的 hidden states 不再占 NPU 上 cache-only KV 池，由 KV connector 每步搬到主机侧每请求 CPU 分页，请求结束回收。

NPU 判断：这是本期 Ascend 侧最有价值的新方向（直接缓解 RL 抽特征的显存压力），但代价是 D2H 带宽与分页管理；需要吞吐/显存曲线，而不是只跑通。

### #11066 / #17557 / #21735 / #21501 · Ascend 模型侧 aux

- [#11066](https://github.com/vllm-project/vllm-ascend/pull/11066)：GLM-5.3-Flash 的 DSpark target states 与 draft caches。PR 明说 GLM5Next 用 hyper-connections + hybrid cache，中间张量**不能**直接当普通 decoder hidden 导出。
- [#17557](https://github.com/vllm-project/vllm-ascend/pull/17557)：同模型的 DFlash2 MRV2 路径，按 `hc_contract(hidden_states + residual)` 收缩。与已合入的 #21439 是两条路径，统计覆盖时不能重复计数。
- [#21735](https://github.com/vllm-project/vllm-ascend/pull/21735)：Ascend 已强制 MRV2，删除不可达的 V1 AUX payload 分支，统一走 `EagleModelMixin` AUX slot relay——**消除双轨**，与上游 #66177 的集中化方向一致。
- [#21501](https://github.com/vllm-project/vllm-ascend/issues/21501)：RFC 提出 target 侧 aux 接口应与 draft 方法无关（当前 dflash 走 Eagle3 共享接口、dspark 另有路径）。

### #56442 · P1 · 前期补漏续跟（多 token 对齐）

[[Bugfix][Spec Decode] Fix extract_hidden_states to use last valid token for multi-token outputs](https://github.com/vllm-project/vllm/pull/56442)：`ExtractHiddenStatesProposer.prepare_next_token_ids_padded()` 只读 `sampled_token_ids` 第一列，多 token 输出时读到错 token；修复改用最后一个有效 token。直接影响提取数据的 token 对齐，NPU 若沿用同一实现同样错位。

### #65991 / #57736 / #64320 / #61539 · 其余

- [#65991](https://github.com/vllm-project/vllm/pull/65991)：`disable_extract_hidden_states_decode_drafts=True`（K=0），回应上期 #65250 的 37%–39% 吞吐损失；本期更新到 11-12。
- [#57736](https://github.com/vllm-project/vllm/pull/57736)：窗口化捕获（rollout 窗口 + teacher scoring），19 提交、30 文件、118 条时间线事件，仍 open。
- [#64320](https://github.com/vllm-project/vllm/pull/64320)：Qwen3-MoE EAGLE3 aux 跨 PP；#66177 集中化后形态可能需调整。
- [#61539](https://github.com/vllm-project/vllm/issues/61539)：routed-expert 显式行布局 RFC **以关闭收尾（未合入）**，行布局仍靠反推。

## 源码核对（`c410543d9fff` / `671292ee7ee2`）

- aux relay 已集中：`vllm/v1/worker/gpu/pp_utils.py:100` `class PPHandler`，含 `configure_aux_hidden_state_relay`、`relay_aux_hidden_states`、`add_upstream_aux_hidden_states`。
- `vllm/distributed/aux_output_connector/` 仍只有 `connector.py`、`routed_experts.py`、`store.py`、`worker.py` —— **没有** `hidden_states.py`，即 CPU-pages/窗口化捕获尚未进 main。
- vllm-ascend 仍保留自己的 PP aux 传输（`pp_transport_aux_hidden_states_` 与 `EagleModelMixin` AUX slot），#21735 正在清理 V1 分支。

## 无实质进展与补漏边界

- [#66794](https://github.com/vllm-project/vllm/issues/66794)、[#21501](https://github.com/vllm-project/vllm-ascend/issues/21501)：均为 RFC，本期无代码。
- vllm-ascend #21209（LM head TP + probabilistic draft sampling）、#21475（MRV2 padding mask 刷新）、#20454（PCP overlap）本期合入，但属 draft 采样/通信优化，未单独列条；需注意 #21475 会改变 padding 行，与 aux 行对齐相关。
- vLLM #66050（Ming ASR）、#52074（chunked prefill 文档）本期无更新。

## 检索与复核

- [本期结构化数据](vllm-extract-hidden-states-delta-2026-11-13.json) · [本期 CSV](vllm-extract-hidden-states-delta-2026-11-13.csv)。
- 本期改用已认证 `gh` CLI 拉取（PR 详情 + 时间线 + 提交 + 文件），合入条目以本地浅克隆的合并提交为证据。
- 未运行 GPU/NPU 或模型测试；作者性能数字均为作者报告。

## 扫描覆盖与源码时间说明

- 关键词检索：vLLM 9 个短语 × `updated:>=2026-11-06`、vllm-ascend 10 个短语 × `updated:>=2026-11-06`，均加 RFC 标签与 `created:` 按日切分；去重后 vLLM 1587 条（1310 PR + 277 Issue）、vllm-ascend 753 条（608 PR + 145 Issue）。
- 合入侧：两个浅克隆分别扫描窗口内 520（vLLM）与 129（vllm-ascend）个提交。
- 源码快照：vLLM `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`、vllm-ascend `671292ee7ee2c6a620f307ce7930d84d772f4d5a`。
