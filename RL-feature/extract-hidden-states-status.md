# extract / return hidden states 特性提交更新情况（vLLM · SGLang）

> 归档日期：2026-10-10
> 数据来源：GitHub Search API 全量检索（官方仓库 vllm-project/vllm、sgl-project/sglang）＋ 本地仓库 git log 交叉核对
> 远端快照：vLLM `main` = `3c802af3cf62`（2026-10-10T01:33Z）｜ SGLang `main` = `77ba54f7df57`（2026-10-10T01:45Z）
> 本地检出：`D:\WorkSpace\vllm`（HEAD `0c1bf2c2`，`origin/main` 停在 `b205750` 2026-09-02，**落后上游约 38 天**）｜`D:\WorkSpace\sglang`（`upstream/main` 停在 `7b6b6ee` 2026-10-08，落后约 2 天）
> 归档产物：本文件（分析 ＋ 全量索引表）、[extract-hidden-states-prs.json](extract-hidden-states-prs.json)、[extract-hidden-states-prs.csv](extract-hidden-states-prs.csv)

## 1. 结论速览

- **特性域整体高位活跃**：命中高相关 PR 共 **183 条**（vLLM 87 / SGLang 96）。其中 **2026-09-18 ~ 2026-10-10 三周窗口内有 29 条 PR 有更新**（vLLM 17、SGLang 12）。
- **vLLM：已合入 35、进行中 27、已关闭未合入 25。**核心实现文件 `vllm/model_executor/models/extract_hidden_states.py` 在 main 上最后一次功能性变更是 2026-06-30（#46301），此后只被格式化/lint（09-16 #52136）与 KV-cache 重构（08-22 #51718）顺带触碰。也就是说：**近三个月的提交不在抽取内核，而在 MRV2 兼容、KV cache 计费与布局、PP/PD 传输、以及把 hidden state 变成引擎的一等输出。**
- **SGLang：已合入 47、进行中 9、已关闭未合入 40。**`--return-hidden-states-mode`（#30177，支持 `"last"`）与模型侧 aux-hidden 捕获链路已成型；**10-10 新开的 #43449 / #43450 是本周期最重要的方向性变化**——把 aux hidden-state 采集从「必须挂一个 draft 模型」解绑为独立 server 参数，并给 out-of-tree consumer（SpecForge / Mooncake）补上 forward observer 与可延迟的 `consume`。
- **需求驱动正在迁移**：从「投机解码 draft 训练数据（aux hidden states）」扩展到「RL rollout / 训练数据管道（按请求返回 hidden states、窗口化采集、跨进程跨节点传输）」。vLLM 侧对应 #57736 / #57663（windowed capture for rollout）、#59543（+ RFC #59542）、#57185；SGLang 侧对应 #43449 / #43450、#41448。
- **传输与落地层成为新战场，且两端都押注 Mooncake**：vLLM #47302 `MooncakeStoreECConnector`（多模态 hidden-state 传输，11 条评论、needs-rebase）；SGLang「DSpark PD 1/3–3/3」三段式（#32422 / #32423 / #32424）中 **2/3、3/3 已于 2026-10-09 关闭未合入**，1/3 更早关闭——这条路线在 SGLang 侧没有落地。vLLM 侧则以 #60545（把 hidden states 塞进 prompt 之后的 KV slot）另起一条 P/D 实现。
- **仍有长期挂起的正确性修复**：vLLM #43869（CUDA graph padding token 的 `-1` slot 会静默覆盖 KV cache 最后一个有效槽位，2026-05-28 开出，标签 stale/needs-rebase）、#56822（hidden-state 层的 KV cache 被按最宽的 peer group 计费）、#56442（投机解码下多 token 输出只读第 0 列）。

## 2. 术语：三条容易被混淆的技术路线

| 路线 | vLLM | SGLang | 用途 |
|---|---|---|---|
| **A. 抽取中间层 aux hidden states** | `extract_hidden_states`（KV-connector 承载）＋ 模型侧 `aux_hidden_states` 捕获 | `capture_aux_hidden_states` / `set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture` | 给 EAGLE3 / DFlash / DSpark / MTP drafter 提供目标模型中间层特征，用于 draft 训练或在线投机 |
| **B. 把 hidden states 作为请求输出** | `return_last_hidden_states`（#59543）、`hidden_state_window`（#57736 / #57663）、最终 prompt hidden state（#57185） | `--return-hidden-states-mode` / `return_hidden_states`（#30177 起支持 `"last"`） | RL rollout / 数据生产：把隐状态直接回传给训练侧 |
| **C. 传输层** | KV-connector、`MooncakeStoreECConnector`（#47302）、PP relay（#57197 / #60477） | DSpark PD hidden-state 传输（#32422–32424）、Mooncake streaming、forward observer（#43450） | 跨 PP stage / P-D 分离 / 跨进程把 A、B 的产物送到消费者手里 |

本报告把 A/B/C 视为同一特性域（两个仓库的实际改动也确实互相交织），但三者的评审人、CI 测试集与风险面并不相同：A 的正确性由模型精度/接受率兜底，B 是 API 契约问题，C 是部署形态问题。

## 3. vLLM（vllm-project/vllm）

### 3.1 近三周更新（2026-09-18 ~ 2026-10-10）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [60545](https://github.com/vllm-project/vllm/pull/60545) | [PoC] Transfer hidden states for P/D | 进行中 | 2026-10-08 | 2026-10-10 | — | @njhill | 5 | speculative-decoding needs-rebase … |
| [60477](https://github.com/vllm-project/vllm/pull/60477) | [Bugfix] DSA: all-reduce partial hidden_states before PP send | 进行中 | 2026-10-07 | 2026-10-09 | — | @SumanthRH | 7 | bug ready deepseek |
| [60855](https://github.com/vllm-project/vllm/pull/60855) | [Model] Support EAGLE3/DFlash aux hidden states for Granite | 进行中 | 2026-10-09 | 2026-10-09 | — | @devpramod | 2 | dflash |
| [60763](https://github.com/vllm-project/vllm/pull/60763) | [Bugfix][Spec Decode] Fix GLM-5.3-Flash DFlash2/EAGLE3 aux hidden-state capture with the upstream config | 已合入 | 2026-10-09 | 2026-10-09 | 2026-10-09 | @ischencheng | 3 | bug ready dflash glm |
| [60603](https://github.com/vllm-project/vllm/pull/60603) | [Hybrid KV Cache] [2/N] Support finalized decode checkpoints with hidden-state drafting | 进行中 | 2026-10-08 | 2026-10-08 | — | @qianlihuang | 1 | needs-rebase scheduler kv-cache-ma… |
| [57736](https://github.com/vllm-project/vllm/pull/57736) | feat: add windowed hidden-state capture for vLLM rollout | 进行中 | 2026-09-20 | 2026-10-08 | — | @vx120 | 2 | performance frontend ci/build mrv2… |
| [57185](https://github.com/vllm-project/vllm/pull/57185) | [Feature] Return the final prompt hidden state from generation | 进行中 | 2026-09-16 | 2026-10-05 | — | @SorenDreano | 3 | documentation frontend needs-rebas… |
| [59543](https://github.com/vllm-project/vllm/pull/59543) | [Core] Opt-in per-request return of the last hidden states in generation (MRV2) | 进行中 | 2026-10-01 | 2026-10-03 | — | @aminry | 3 | documentation mrv2 scheduler |
| [59810](https://github.com/vllm-project/vllm/pull/59810) | [Bugfix][DBO] Fix crash when merging auxiliary hidden states | 进行中 | 2026-10-02 | 2026-10-02 | — | @Venkat2811 | 1 | bug mrv2 |
| [59037](https://github.com/vllm-project/vllm/pull/59037) | [KV Cache] Support hidden-state extraction with packed BLHNC layouts | 进行中 | 2026-09-28 | 2026-10-01 | — | @tomasruizt | 2 | documentation rocm speculative-dec… |
| [47302](https://github.com/vllm-project/vllm/pull/47302) | [EC Transfer] Add MooncakeStoreECConnector for multimodal hidden-state transfer | 进行中 | 2026-07-01 | 2026-10-01 | — | @kanceler | 11 | needs-rebase v1 mooncake |
| [57197](https://github.com/vllm-project/vllm/pull/57197) | [Bugfix][PP][Spec Decode] MiniMax-M3 EAGLE3 aux-state relay at PP > 1 and per-stage FlashInfer autotune | 已合入 | 2026-09-16 | 2026-09-30 | 2026-09-30 | @venkywonka | 23 | bug ready nvidia minimax |
| [43869](https://github.com/vllm-project/vllm/pull/43869) | [BugFix] Fix extract_hidden_states CUDA graph padding polluting KV cache | 进行中 | 2026-05-28 | 2026-09-30 | — | @Ag-Cu | 4 | bug needs-rebase stale nvidia |
| [57663](https://github.com/vllm-project/vllm/pull/57663) | [V1] Add SamplingParams.hidden_state_window + position math for windowed hidden-state capture (#56916) | 进行中 | 2026-09-19 | 2026-09-28 | — | @HEMU-VERMA | 2 | needs-rebase |
| [55575](https://github.com/vllm-project/vllm/pull/55575) | [Performance][Model] DeepSeek-V4: reuse fused post residual for aux hidden states | 进行中 | 2026-09-06 | 2026-09-26 | — | @TeloySXH | 10 | needs-rebase deepseek mrv2 DSv4 df… |
| [55682](https://github.com/vllm-project/vllm/pull/55682) | [Spec Decode] GLM-5.3-Flash DFlash2 aux hidden-state capture | 进行中 | 2026-09-07 | 2026-09-21 | — | @Leoyzen | 4 | needs-rebase dflash glm |
| [57290](https://github.com/vllm-project/vllm/pull/57290) | [Bugfix][Spec Decode] Keep hidden-state extractor drafts at PP1 | 已关闭 | 2026-09-17 | 2026-09-18 | 2026-09-18 | @venkywonka | 8 | bug |

要点：

- **窗口内只合入 2 条**：#60763（10-09，当天开当天合）修的是 `Glm5NextModel._aux_hidden_state` 读 `mhc_num_residual_streams`，而上游 `Glm5NextTextConfig` 在 #57387 之后改名为 `hc_mult`，导致 DFlash2 / EAGLE3 drafter 启动即 `AttributeError`——典型的「两条并行开发流在重命名上撞车」；#57197（09-30，23 条评论）补 MiniMax-M3 在 PP>1 下的 EAGLE3 aux-state relay。窗口前一日 #49819（09-17）合入 Cohere2MoE 的 Eagle3 aux hidden states。
- **最值得关注的在途项是 #60545「[PoC] Transfer hidden states for P/D」**（@njhill，10-08 开出，9 commits / 26 files，**draft**，标签 needs-rebase / kv-connector / mrv2 / scheduler / kv-cache-manager）。思路是**把最终 hidden states（含 aux 层、以及 eagle3 / multi-module MTP 的额外位置）写入 prompt 末尾之后的 KV cache token slot**：通常复用 block padding，因此零额外显存与传输开销；同时避免 D 侧重算最后一个 prompt token、避免 D 侧加载多模态编码器权重（可把省下的显存给 KV cache），并保住 target 模型 uniform decode 的全 CUDA graph。作者列出的收益还包括绕开 N-1 截断在 prompt logprobs 上的各种特例。这是 P/D 分离下 hidden-state 传递的新路线，与 2026-02-17 关闭的 #34388（WIP：为 EAGLE warm-up 传输 hidden states）目标一致、实现不同。
- **RL 方向的三个入口全部在评审中**：#57736（request-local 窗口化采集，`HiddenStateCapturePlan` 穿通 scheduler / model runner / engine / output processor，需处理 chunked prefill、decode、投机解码三类 step）、#57663（只做 `SamplingParams.hidden_state_window` 与「窗口 → 变长 step」的位置数学；作者明确说明 GPUModelRunner 侧抽取尚未实现，先要设计反馈）、#59543（RFC #59542 的落地，返回「生成 token 的 logits 所依据的那个 hidden state」，CPU、模型 dtype，engine flag 关闭时零开销、输出与 stock 逐位一致）。
- **KV cache 计费/布局是积压重灾区**：#56822（1 层 hidden-state group 被按 24 层 MLA 的宽度计费，5.4 GiB 实收约 130 GiB，启动直接失败）、#53074（把 hidden-state cache 从 DeepSeek-V4 MLA 分组中隔离）、#59037（在 packed BLHNC 布局下支持抽取，服务 Speculators 场景的 GLM-5.3-Flash）、#60603（[Hybrid KV Cache][2/N] finalized decode checkpoint ＋ hidden-state drafting）、#49301（限制导出 staging 内存）。这四条都带 needs-rebase。
- **#60477 虽标题只提 `hidden_states`，实际修的是 DSA 模型在 PP 边界的激活值错误**（最后一层未归约的 MLP/MoE 输出需在 PP send 前 all-reduce）：GLM-5.3 BF16 TP8PP2 由此从掉 7–9 分、多生成 7.6 倍 token 恢复到与 TP16 一致，带 `ready` 标签，是本窗口性价比最高的一条。
- **稳定性长尾**：#43869 已挂 4 个多月并被标 stale；其根因值得单独记一笔——`basic_cache()` 直接用 `slot_mapping` 索引 KV cache 而未处理 `PADDING_SLOT_ID = -1`，Python 中 `-1 // block_size == -1`、`-1 % block_size == block_size - 1`，于是 padding token 的隐状态**静默覆盖最后一个有效 KV 槽位**。

### 3.2 全部已合入（35 条，按合入时间倒序）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [60763](https://github.com/vllm-project/vllm/pull/60763) | [Bugfix][Spec Decode] Fix GLM-5.3-Flash DFlash2/EAGLE3 aux hidden-state capture with the upstream config | 已合入 | 2026-10-09 | 2026-10-09 | 2026-10-09 | @ischencheng | 3 | bug ready dflash glm |
| [57197](https://github.com/vllm-project/vllm/pull/57197) | [Bugfix][PP][Spec Decode] MiniMax-M3 EAGLE3 aux-state relay at PP > 1 and per-stage FlashInfer autotune | 已合入 | 2026-09-16 | 2026-09-30 | 2026-09-30 | @venkywonka | 23 | bug ready nvidia minimax |
| [49819](https://github.com/vllm-project/vllm/pull/49819) | [Model] Add Cohere2MoE Eagle3 auxiliary hidden states | 已合入 | 2026-07-25 | 2026-09-17 | 2026-09-17 | @sdougbrown | 10 | speculative-decoding ready v1 dfla… |
| [50894](https://github.com/vllm-project/vllm/pull/50894) | [Bugfix] Scale KV page size for hidden states extraction with TP | 已合入 | 2026-08-03 | 2026-09-14 | 2026-09-14 | @orestis-z | 16 | bug ready kv-cache-manager |
| [53862](https://github.com/vllm-project/vllm/pull/53862) | [XPU][CI] increase timeout of extract_hidden_states tp2 | 已合入 | 2026-08-26 | 2026-09-16 | 2026-08-26 | @zhenwei-intel | 3 | intel-gpu kv-connector |
| [49811](https://github.com/vllm-project/vllm/pull/49811) | [Feature][Model Runner V2] Support extract_hidden_states speculation | 已合入 | 2026-07-25 | 2026-08-20 | 2026-08-20 | @zupengwang | 12 | speculative-decoding ready v1 mrv2 |
| [51108](https://github.com/vllm-project/vllm/pull/51108) | [BugFix][KV Cache] Fix hybrid prefix caching with hidden-state extraction | 已合入 | 2026-08-05 | 2026-09-14 | 2026-08-06 | @gcanlin | 11 | bug ready |
| [48161](https://github.com/vllm-project/vllm/pull/48161) | [CI] Increase extract hidden states TP2 timeout | 已合入 | 2026-07-09 | 2026-07-09 | 2026-07-09 | @LucasWilkinson | 0 | ready v1 kv-connector |
| [46301](https://github.com/vllm-project/vllm/pull/46301) | [Spec Decode] Fix hidden-state extraction block size for hybrid verifiers | 已合入 | 2026-06-21 | 2026-06-30 | 2026-06-30 | @imargulis | 4 | ready v1 kv-connector |
| [46973](https://github.com/vllm-project/vllm/pull/46973) | [Bugfix] Capture final-layer aux hidden state in deepseek_v2 backbone | 已合入 | 2026-06-28 | 2026-06-29 | 2026-06-29 | @mgoin | 0 | bug speculative-decoding ready dee… |
| [46968](https://github.com/vllm-project/vllm/pull/46968) | [Spec Decode] Avoid redundant hidden-states gather in draft prefill | 已合入 | 2026-06-28 | 2026-06-29 | 2026-06-29 | @WoosukKwon | 0 | ready v1 |
| [46786](https://github.com/vllm-project/vllm/pull/46786) | [Model Runner V2][Spec Decode] Handle tuple hidden states from MTP draft models | 已合入 | 2026-06-26 | 2026-06-28 | 2026-06-28 | @chaunceyjiang | 0 | v1 |
| [45849](https://github.com/vllm-project/vllm/pull/45849) | [BUG] fix hidden states nan for hybrid attention models | 已合入 | 2026-06-16 | 2026-06-17 | 2026-06-17 | @shanjiaz | 0 | bug ready v1 kv-connector |
| [45294](https://github.com/vllm-project/vllm/pull/45294) | Update hidden states extraction integration test triggers | 已合入 | 2026-06-11 | 2026-06-12 | 2026-06-12 | @fynnsu | 1 | ready ci/build |
| [43805](https://github.com/vllm-project/vllm/pull/43805) | Hidden states extraction improvements | 已合入 | 2026-05-27 | 2026-06-11 | 2026-06-11 | @fynnsu | 6 | documentation performance speculat… |
| [43482](https://github.com/vllm-project/vllm/pull/43482) | [Bugfix] Apply fc_norm in Eagle3DeepseekV2 combine_hidden_states | 已合入 | 2026-05-23 | 2026-05-26 | 2026-05-26 | @yubofredwang | 1 | bug speculative-decoding ready dee… |
| [37374](https://github.com/vllm-project/vllm/pull/37374) | [Perf] Optimize hidden state extraction logic | 已合入 | 2026-03-18 | 2026-05-22 | 2026-05-22 | @benchislett | 8 | documentation performance speculat… |
| [39949](https://github.com/vllm-project/vllm/pull/39949) | [Spec Decode] Support hybrid attention models in extract_hidden_states | 已合入 | 2026-04-15 | 2026-05-18 | 2026-05-13 | @mgoin | 12 | speculative-decoding ready ci/buil… |
| [38610](https://github.com/vllm-project/vllm/pull/38610) | [Spec Decode] fix returning size mismatch on extract hidden states proposer | 已合入 | 2026-03-31 | 2026-04-09 | 2026-04-09 | @zzaebok | 2 | speculative-decoding ready v1 |
| [39160](https://github.com/vllm-project/vllm/pull/39160) | [Bugfix] Fix extract_hidden_states crash with quantized KV cache dtype | 已合入 | 2026-04-07 | 2026-04-07 | 2026-04-07 | @yubofredwang | 2 | bug ready |
| [38987](https://github.com/vllm-project/vllm/pull/38987) | [Bugfix][Spec Decode] Fix extract_hidden_states for VLM models | 已合入 | 2026-04-04 | 2026-04-05 | 2026-04-05 | @abatilo | 1 | bug speculative-decoding ready v1 |
| [38723](https://github.com/vllm-project/vllm/pull/38723) | Fix shape comment in extract_hidden_states example | 已合入 | 2026-04-01 | 2026-04-01 | 2026-04-01 | @fynnsu | 1 | documentation ready |
| [37818](https://github.com/vllm-project/vllm/pull/37818) | [MRV2] Skip hidden states allocation for PW CUDA graphs | 已合入 | 2026-03-22 | 2026-03-22 | 2026-03-22 | @WoosukKwon | 0 | ready v1 nvidia |
| [37013](https://github.com/vllm-project/vllm/pull/37013) | [Spec Decode] Update extract_hidden_states to use deferred kv_connector clear | 已合入 | 2026-03-13 | 2026-03-16 | 2026-03-16 | @fynnsu | 0 | speculative-decoding ready v1 kv-c… |
| [33736](https://github.com/vllm-project/vllm/pull/33736) | [Spec Decode] Add hidden states extraction system | 已合入 | 2026-02-03 | 2026-04-09 | 2026-03-02 | @fynnsu | 9 | documentation new-model speculativ… |
| [26686](https://github.com/vllm-project/vllm/pull/26686) | [Model][7/N] Improve all pooling task / Deprecation as_reward_model. Extract hidden states prefer using new multi-vector retrieval API | 已合入 | 2025-10-13 | 2025-12-08 | 2025-12-08 | @noooop | 6 | documentation ready |
| [27688](https://github.com/vllm-project/vllm/pull/27688) | [Spec Decode] Add support for EAGLE3 heads that do not use_aux_hidden_states | 已合入 | 2025-10-28 | 2026-03-03 | 2025-11-25 | @hjjq | 1 | speculative-decoding ready v1 llam… |
| [28740](https://github.com/vllm-project/vllm/pull/28740) | [Bugfix] Fix incorrect use of hidden_states for shared_experts due to do_naive_dispatch_combine | 已合入 | 2025-11-14 | 2025-11-14 | 2025-11-14 | @alexm-redhat | 1 | bug ready deepseek |
| [24056](https://github.com/vllm-project/vllm/pull/24056) | [bugfix]fix MTP hidden states | 已合入 | 2025-09-01 | 2025-09-01 | 2025-09-01 | @luccafong | 0 | speculative-decoding ready v1 |
| [20152](https://github.com/vllm-project/vllm/pull/20152) | [Bugfix] Mark 'hidden_states' as mutable in moe_forward registration. | 已合入 | 2025-06-26 | 2025-09-20 | 2025-06-27 | @bnellnm | 2 | ready |
| [18543](https://github.com/vllm-project/vllm/pull/18543) | [Bugfix] Use random hidden states in dummy sampler run | 已合入 | 2025-05-22 | 2025-05-23 | 2025-05-22 | @abmfy | 4 | ready v1 |
| [14369](https://github.com/vllm-project/vllm/pull/14369) | [Bugfix][Disaggregated] Add a check in send_kv_caches_and_hidden_states and fix the reshape of the KVCache | 已合入 | 2025-03-06 | 2025-03-11 | 2025-03-08 | @hasB4K | 1 | ready |
| [7508](https://github.com/vllm-project/vllm/pull/7508) | [Speculative Decoding] Fixing hidden states handling in batch expansion | 已合入 | 2024-08-14 | 2024-08-20 | 2024-08-20 | @abhigoyal1997 | 9 | ready |
| [3539](https://github.com/vllm-project/vllm/pull/3539) | Fix 1D query issue from `_prune_hidden_states` | 已合入 | 2024-03-21 | 2024-03-21 | 2024-03-21 | @rkooo567 | 1 | — |
| [707](https://github.com/vllm-project/vllm/pull/707) | Improve _prune_hidden_states micro-benchmark | 已合入 | 2023-08-09 | 2023-08-31 | 2023-08-31 | @tmm1 | 7 | — |

读法：表内包含 2023–2025 的早期基础提交（#7508、#3539、#707 等）与 2025-12 的 pooling / 多向量检索条目（#26686，共用 `extract_hidden_states` 命名但语义不同，属「用隐状态做 embedding」而非本节主线）。主线里程碑是：

1. **#33736（2026-03-02）引入 hidden states extraction system** —— 特性起点；
2. #37013 / #37818 / #38723 / #38987 / #39160 / #38610（2026-03 ~ 04）打磨 kv_connector 清理时机、CUDA graph 显存、VLM、量化 KV dtype、proposer 返回尺寸；
3. #39949 / #45849 / #46301（2026-05 ~ 06）集中解决 **hybrid attention 模型**，同期 #46786 / #46968 / #46973 处理 MTP tuple hidden states 与 deepseek_v2 末层捕获；
4. **#49811（2026-08-20）把抽取接到 MRV2**；
5. #50894（09-14，KV page size）、#51108（09-14，hybrid prefix caching）、#49819（09-17，Cohere2MoE）、#57197（09-30，PP relay）——**2026-09 之后战线整体转向工程化落地**。

### 3.3 全部进行中（27 条）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [60545](https://github.com/vllm-project/vllm/pull/60545) | [PoC] Transfer hidden states for P/D | 进行中 | 2026-10-08 | 2026-10-10 | — | @njhill | 5 | speculative-decoding needs-rebase … |
| [60477](https://github.com/vllm-project/vllm/pull/60477) | [Bugfix] DSA: all-reduce partial hidden_states before PP send | 进行中 | 2026-10-07 | 2026-10-09 | — | @SumanthRH | 7 | bug ready deepseek |
| [60855](https://github.com/vllm-project/vllm/pull/60855) | [Model] Support EAGLE3/DFlash aux hidden states for Granite | 进行中 | 2026-10-09 | 2026-10-09 | — | @devpramod | 2 | dflash |
| [60603](https://github.com/vllm-project/vllm/pull/60603) | [Hybrid KV Cache] [2/N] Support finalized decode checkpoints with hidden-state drafting | 进行中 | 2026-10-08 | 2026-10-08 | — | @qianlihuang | 1 | needs-rebase scheduler kv-cache-ma… |
| [57736](https://github.com/vllm-project/vllm/pull/57736) | feat: add windowed hidden-state capture for vLLM rollout | 进行中 | 2026-09-20 | 2026-10-08 | — | @vx120 | 2 | performance frontend ci/build mrv2… |
| [57185](https://github.com/vllm-project/vllm/pull/57185) | [Feature] Return the final prompt hidden state from generation | 进行中 | 2026-09-16 | 2026-10-05 | — | @SorenDreano | 3 | documentation frontend needs-rebas… |
| [59543](https://github.com/vllm-project/vllm/pull/59543) | [Core] Opt-in per-request return of the last hidden states in generation (MRV2) | 进行中 | 2026-10-01 | 2026-10-03 | — | @aminry | 3 | documentation mrv2 scheduler |
| [59810](https://github.com/vllm-project/vllm/pull/59810) | [Bugfix][DBO] Fix crash when merging auxiliary hidden states | 进行中 | 2026-10-02 | 2026-10-02 | — | @Venkat2811 | 1 | bug mrv2 |
| [59037](https://github.com/vllm-project/vllm/pull/59037) | [KV Cache] Support hidden-state extraction with packed BLHNC layouts | 进行中 | 2026-09-28 | 2026-10-01 | — | @tomasruizt | 2 | documentation rocm speculative-dec… |
| [47302](https://github.com/vllm-project/vllm/pull/47302) | [EC Transfer] Add MooncakeStoreECConnector for multimodal hidden-state transfer | 进行中 | 2026-07-01 | 2026-10-01 | — | @kanceler | 11 | needs-rebase v1 mooncake |
| [43869](https://github.com/vllm-project/vllm/pull/43869) | [BugFix] Fix extract_hidden_states CUDA graph padding polluting KV cache | 进行中 | 2026-05-28 | 2026-09-30 | — | @Ag-Cu | 4 | bug needs-rebase stale nvidia |
| [57663](https://github.com/vllm-project/vllm/pull/57663) | [V1] Add SamplingParams.hidden_state_window + position math for windowed hidden-state capture (#56916) | 进行中 | 2026-09-19 | 2026-09-28 | — | @HEMU-VERMA | 2 | needs-rebase |
| [55575](https://github.com/vllm-project/vllm/pull/55575) | [Performance][Model] DeepSeek-V4: reuse fused post residual for aux hidden states | 进行中 | 2026-09-06 | 2026-09-26 | — | @TeloySXH | 10 | needs-rebase deepseek mrv2 DSv4 df… |
| [55682](https://github.com/vllm-project/vllm/pull/55682) | [Spec Decode] GLM-5.3-Flash DFlash2 aux hidden-state capture | 进行中 | 2026-09-07 | 2026-09-21 | — | @Leoyzen | 4 | needs-rebase dflash glm |
| [54822](https://github.com/vllm-project/vllm/pull/54822) | [Bugfix][Speculative Decoding] Fix Gemma4 MTP hidden-state buffer sizing | 进行中 | 2026-09-01 | 2026-09-17 | — | @Archmonger | 2 | bug speculative-decoding |
| [56822](https://github.com/vllm-project/vllm/pull/56822) | [Bugfix] Bill hidden-state extraction at its own KV cache group width | 进行中 | 2026-09-14 | 2026-09-16 | — | @gokay-ai | 1 | bug kv-cache-manager |
| [56442](https://github.com/vllm-project/vllm/pull/56442) | [Bugfix][Spec Decode] Fix extract_hidden_states to use last valid token for multi-token outputs | 进行中 | 2026-09-11 | 2026-09-11 | — | @anishesg | 1 | bug speculative-decoding |
| [51328](https://github.com/vllm-project/vllm/pull/51328) | [Bugfix] Use per-group slot_mapping in ExtractHiddenStatesProposer | 进行中 | 2026-08-07 | 2026-09-11 | — | @ys2025-AI | 3 | bug speculative-decoding mrv1-only |
| [49301](https://github.com/vllm-project/vllm/pull/49301) | [Bugfix] Bound hidden-state export staging memory | 进行中 | 2026-07-21 | 2026-09-05 | — | @yongfuFang | 3 | bug documentation needs-rebase v1 … |
| [52184](https://github.com/vllm-project/vllm/pull/52184) | feat(inkling): add Eagle3 hidden state extraction support | 进行中 | 2026-08-13 | 2026-09-03 | — | @orestis-z | 8 | documentation new-model ready ci/b… |
| [53074](https://github.com/vllm-project/vllm/pull/53074) | [Bugfix][Core] Isolate hidden-state cache from DeepSeek-V4 MLA groups | 进行中 | 2026-08-20 | 2026-08-29 | — | @my0901 | 2 | bug needs-rebase deepseek kv-conne… |
| [52073](https://github.com/vllm-project/vllm/pull/52073) | [Spec Decode] Test extract_hidden_states on NemotronH hybrid models | 进行中 | 2026-08-13 | 2026-08-26 | — | @aminsamir45 | 2 | needs-rebase kv-connector |
| [48598](https://github.com/vllm-project/vllm/pull/48598) | [DeepSeek-V4] Handle 3-D packed hidden states in attention forward | 进行中 | 2026-07-14 | 2026-08-20 | — | @jimmy-adams | 3 | needs-rebase deepseek DSv4 |
| [43671](https://github.com/vllm-project/vllm/pull/43671) | [Model][MiniMax-M2] Fix EAGLE-3 aux hidden-state layer off-by-one | 进行中 | 2026-05-26 | 2026-08-20 | — | @pbkowalski | 6 | speculative-decoding ready needs-r… |
| [52074](https://github.com/vllm-project/vllm/pull/52074) | [Docs] extract_hidden_states supports chunked prefill | 进行中 | 2026-08-13 | 2026-08-13 | — | @aminsamir45 | 2 | documentation |
| [50187](https://github.com/vllm-project/vllm/pull/50187) | [Bugfix][Spec Decode] Fix misspelled EAGLE3 aux-hidden-state layer method overrides | 进行中 | 2026-07-28 | 2026-07-28 | — | @Henry-Cheng | 1 | bug qwen deepseek mistral kimi |
| [46421](https://github.com/vllm-project/vllm/pull/46421) | [Test] Add MoE dtype asserts for hidden states and logits | 进行中 | 2026-06-22 | 2026-06-22 | — | @bnellnm | 0 | — |

分类看：**KV cache / 内存计费与布局**类最多（#56822、#53074、#59037、#60603、#48598、#49301、#43869，共 7 条），**模型侧 aux 捕获**次之（#60855 Granite、#50815 Kimi-K3、#43671 MiniMax-M2、#52184 inkling、#55575 DeepSeek-V4、#55682 GLM-5.3-Flash），**对外输出/接口** 3 条（#59543、#57185、#57736/#57663），**测试与文档** 3 条（#52073 NemotronH 测试、#52074 文档、#46421 MoE dtype 断言），其余为分散的模型/框架 bugfix（如 #54822 Gemma4 MTP buffer sizing）。`needs-rebase` / `stale` 出现在约 11 条上——**积压主要卡在 rebase 与评审带宽，而不是设计分歧**。

### 3.4 已关闭未合入（此处列 2026-06 之后关闭的 12 条）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [57290](https://github.com/vllm-project/vllm/pull/57290) | [Bugfix][Spec Decode] Keep hidden-state extractor drafts at PP1 | 已关闭 | 2026-09-17 | 2026-09-18 | 2026-09-18 | @venkywonka | 8 | bug |
| [51760](https://github.com/vllm-project/vllm/pull/51760) | [XPU][Test] Fix OOT model registration in extract_hidden_states test | 已关闭 | 2026-08-11 | 2026-09-16 | 2026-08-26 | @zhenwei-intel | 3 | intel-gpu needs-rebase ci/build kv… |
| [55620](https://github.com/vllm-project/vllm/pull/55620) | [Spec Decode][GLM-5.3] Expose auxiliary hidden states (EAGLE-3 / DFlash) from Glm5Next | 已关闭 | 2026-09-07 | 2026-09-07 | 2026-09-07 | @cameronzucker | 5 | dflash glm |
| [53015](https://github.com/vllm-project/vllm/pull/53015) | [Feature][Model Runner V2] Support extract_hidden_states with pipeline parallelism | 已关闭 | 2026-08-19 | 2026-08-19 | 2026-08-19 | @yubofredwang | 0 | speculative-decoding mrv2 kimi k3 … |
| [35425](https://github.com/vllm-project/vllm/pull/35425) | [Spec Decode] implement online hidden state extraction | 已关闭 | 2026-02-26 | 2026-07-22 | 2026-07-22 | @harryzorus | 13 | documentation performance new-mode… |
| [48163](https://github.com/vllm-project/vllm/pull/48163) | Update timeout for test_extract_hidden_states_tp2 | 已关闭 | 2026-07-09 | 2026-07-16 | 2026-07-09 | @mgoin | 2 | ready v1 ci-failure kv-connector |
| [48152](https://github.com/vllm-project/vllm/pull/48152) | [Bugfix] Fix deepseek_v2 missing last layer aux hidden state extraction | 已关闭 | 2026-07-09 | 2026-07-10 | 2026-07-10 | @factnn | 3 | bug deepseek |
| [30315](https://github.com/vllm-project/vllm/pull/30315) | Generalize pooling model support with multi-task, multi-layer, multi-label classification that can be pooled from both hidden states and LM head's logits. | 已关闭 | 2025-12-09 | 2026-07-10 | 2026-07-10 | @kflu | 9 | needs-rebase stale |
| [48151](https://github.com/vllm-project/vllm/pull/48151) | [Bugfix] Fix deepseek_v2 missing last layer hidden states extraction | 已关闭 | 2026-07-09 | 2026-07-09 | 2026-07-09 | @factnn | 0 | bug deepseek |
| [44328](https://github.com/vllm-project/vllm/pull/44328) | [Core] Support hidden-state extraction for hybrid models | 已关闭 | 2026-06-02 | 2026-06-21 | 2026-06-21 | @imargulis | 7 | speculative-decoding needs-rebase … |
| [36731](https://github.com/vllm-project/vllm/pull/36731) | [Bugfix] Fix Eagle3 aux_hidden_state_layers indexing with pipeline parallelism | 已关闭 | 2026-03-11 | 2026-06-07 | 2026-06-07 | @alvinttang | 4 | bug llama qwen |
| [34388](https://github.com/vllm-project/vllm/pull/34388) | [WIP][Spec Decode] P/D disaggregation: transfer hidden states for EAGLE warm-up | 已关闭 | 2026-02-11 | 2026-06-04 | 2026-02-17 | @ZhanqiuHu | 2 | documentation speculative-decoding… |

值得记一笔的三组：

- **#57290「Keep hidden-state extractor drafts at PP1」**（8 条评论，关闭次日）与 **#53015「MRV2 支持 extract_hidden_states + PP」**（0 评论关闭）表明 **「MRV2 × 流水线并行」下的抽取路线被改写或搁置**，最终能力由 #49811 ＋ #57197 的组合覆盖。
- **#55620（GLM-5.3 暴露 aux hidden states）当天关闭**，随后由 #56983 → #60763 系列接手；#55682 同主题仍在 open 且 needs-rebase。
- **#44328「hybrid 模型 hidden-state 抽取」被 #39949 取代**（两者标题几乎相同，属重复实现收敛）。

更早关闭的条目（如 2025 年的 #15434、#7892、#21621 等）见 JSON / CSV。

## 4. SGLang（sgl-project/sglang）

### 4.1 近三周更新（2026-09-18 ~ 2026-10-10）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [43449](https://github.com/sgl-project/sglang/pull/43449) | [Feature] Capture aux hidden states on the target without a speculative draft | 进行中 | 2026-10-10 | 2026-10-10 | — | @maocheng23 | 0 | documentation |
| [41448](https://github.com/sgl-project/sglang/pull/41448) | [AMD][DeepSeek-V4.1] Fix server crash on prompt-logprob and hidden-state requests under decoder SWA bounded replay | 进行中 | 2026-09-27 | 2026-10-09 | — | @amd-pedghazi | 0 | documentation quant amd dependenci… |
| [32424](https://github.com/sgl-project/sglang/pull/32424) | [DSpark PD 3/3] Implement streaming hidden-state transfer for Mooncake | 已关闭 | 2026-07-26 | 2026-10-09 | 2026-10-09 | @zhangxiaolei123456 | 4 | deepseek |
| [32423](https://github.com/sgl-project/sglang/pull/32423) | [DSpark PD 2/3] Integrate hidden-state bootstrap into PD scheduling | 已关闭 | 2026-07-26 | 2026-10-09 | 2026-10-09 | @zhangxiaolei123456 | 2 | deepseek |
| [38642](https://github.com/sgl-project/sglang/pull/38642) | [DFLASH] Support Qwen4-Exp hidden-state capture | 进行中 | 2026-09-09 | 2026-10-08 | — | @Leslie360 | 7 | — |
| [40888](https://github.com/sgl-project/sglang/pull/40888) | fix(mimo-vl): keep multimodal features when capturing aux hidden states, and correct vision preprocessing | 已合入 | 2026-09-23 | 2026-10-06 | 2026-10-06 | @jetd1 | 1 | run-ci highest-priority |
| [39644](https://github.com/sgl-project/sglang/pull/39644) | [Spec][PP] Reject EAGLE3/DFLASH aux hidden-state capture under pipeline parallelism | 已关闭 | 2026-09-15 | 2026-09-29 | 2026-09-29 | @YAMY1234 | 3 | — |
| [41080](https://github.com/sgl-project/sglang/pull/41080) | [Fix] Complete the deferred FFN all-reduce before deepstack addition and aux hidden-state capture | 已合入 | 2026-09-24 | 2026-09-24 | 2026-09-24 | @ch-wan | 0 | — |
| [40801](https://github.com/sgl-project/sglang/pull/40801) | [Fix] Capture complete Nemotron auxiliary hidden states | 已合入 | 2026-09-23 | 2026-09-23 | 2026-09-23 | @ch-wan | 0 | — |
| [32103](https://github.com/sgl-project/sglang/pull/32103) | fix(hidden_states): keep prefill offset aligned for non-requesting reqs to stop cross-request leak | 已关闭 | 2026-07-22 | 2026-09-23 | 2026-09-23 | @Anai-Guo | 2 | — |
| [40332](https://github.com/sgl-project/sglang/pull/40332) | fix: transfer native hidden states from prefill to decode | 已关闭 | 2026-09-19 | 2026-09-19 | 2026-09-19 | @ishandhanani | 0 | — |
| [39776](https://github.com/sgl-project/sglang/pull/39776) | [NPU][Bugfix][MiMo-V2.5-Pro] Capture post-final-norm hidden state for the final DFlash aux layer | 已关闭 | 2026-09-16 | 2026-09-18 | 2026-09-18 | @iridiumine | 0 | — |

要点：

- **10-10 双开 #43449 / #43450（@maocheng23）是本周期最关键的信号：**

  - **#43449** 新增两个默认关闭的 `exec.features` 参数：`--aux-hidden-state-capture {eagle3,dflash,dspark}` 与 `--aux-hidden-state-layer-ids`，让**纯 target server（不加载 draft）也能产出 EAGLE3 / DFlash / DSpark 的逐层拼接隐状态**，并复用各模型已有的 capture hook（`set_eagle3_layers_to_capture` / `set_dflash_layers_to_capture` / 原生的 `set_dspark_layers_to_capture`）。动机写得很直白：数据生产与 draft 训练管线此前只能 patch SGLang 或另起一个 target 进程；`ModelRunner` 中那句 `TODO: expose this to server args?` 正是这个洞。顺带说明：**同样的需求在 2026-08-11 曾以 #12183「Option to enable auxiliary hidden capture mode」提出并关闭**，本次是重启。
  - **#43450** 补上两块缺失能力：一是让钩子能看到「forward 的其他输出（hidden states）＋ 本批次请求」，而不只是 logits（现有 `SamplingObserver` 只看 logits）；二是让 `consume` 支持异步，避免请求在数据落盘（例如 Mooncake）之前就返回给客户端、产生悬空引用。作者提到 SpecForge 目前为此 patch 了约 14 个 SGLang 文件。

- **10-06 合入 #40888**（mimo-vl：aux hidden-state 捕获时丢弃了多模态特征，并修正 vision 预处理），是窗口内唯一带 `highest-priority` 的合入；其余两条合入是 09-23/09-24 的 #40801（Nemotron 完整 aux hidden states）与 #41080（deepstack 加法与 aux 捕获前必须完成 deferred FFN all-reduce）。
- **#38642（[DFLASH] Qwen4-Exp hidden-state 捕获，7 条评论）**：诊断出 `qwen4_exp.py` 捕获链三处断裂（`Qwen4ExpLayerExtensionMixin` 删除了 `layer_communicator`，而父类 `qwen3_5.py` 的 `prepare_attn_and_capture_last_layer_outputs` 正是捕获入口），按 GLM-5.3-Flash（#36708）先例修复，仍在评审。
- **#41448 揭示了一类危险失败模式**：开启 `--enable-decoder-swa-bounded-replay` 后，`return_hidden_states: true`（全量捕获）或需要 prompt 全量 logprob 的请求会在 **model forward 内**抛 `ValueError`，从而**拖垮整个 server 与所有在途请求**，而不是只让该请求失败。
- **DSpark PD 传输三段式收尾**：**#32423 / #32424 于 10-09 关闭未合入**，加上更早关闭的 #32422，「PD 分离下把 target hidden states 传给 draft 以初始化 draft 侧 KV」这条路线在 SGLang 侧到此为止（对比 vLLM #60545 正在另起炉灶）。
- 其余窗口内关闭项：#40332（prefill→decode 传递 native hidden states）、#32103（非请求 req 的 prefill offset 对齐，修跨请求泄漏）、#39644（PP 下直接拒绝 EAGLE3/DFLASH aux 捕获）、#39776（NPU MiMo-V2.5-Pro 最终 DFlash aux 层的 post-final-norm）、#28261（Qwen3-VL DFlash aux 捕获 setup）。

### 4.2 全部已合入（47 条，按合入时间倒序）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [40888](https://github.com/sgl-project/sglang/pull/40888) | fix(mimo-vl): keep multimodal features when capturing aux hidden states, and correct vision preprocessing | 已合入 | 2026-09-23 | 2026-10-06 | 2026-10-06 | @jetd1 | 1 | run-ci highest-priority |
| [41080](https://github.com/sgl-project/sglang/pull/41080) | [Fix] Complete the deferred FFN all-reduce before deepstack addition and aux hidden-state capture | 已合入 | 2026-09-24 | 2026-09-24 | 2026-09-24 | @ch-wan | 0 | — |
| [40801](https://github.com/sgl-project/sglang/pull/40801) | [Fix] Capture complete Nemotron auxiliary hidden states | 已合入 | 2026-09-23 | 2026-09-23 | 2026-09-23 | @ch-wan | 0 | — |
| [37989](https://github.com/sgl-project/sglang/pull/37989) | [Docs] Document --retraction-policy, --return-hidden-states-mode, --language-model-only | 已合入 | 2026-09-04 | 2026-09-07 | 2026-09-04 | @hickeyma | 3 | documentation run-ci |
| [36755](https://github.com/sgl-project/sglang/pull/36755) | Fix DFLASH aux hidden-state capture on mHC models | 已合入 | 2026-08-28 | 2026-08-28 | 2026-08-28 | @zijiexia | 0 | — |
| [36708](https://github.com/sgl-project/sglang/pull/36708) | [DFLASH] Support GLM-5.3-Flash hidden-state capture | 已合入 | 2026-08-27 | 2026-08-27 | 2026-08-27 | @jianc99 | 0 | — |
| [34771](https://github.com/sgl-project/sglang/pull/34771) | [Spec] Wire DFLASH aux-hidden capture into the Qwen3.5 text-only wrapper | 已合入 | 2026-08-13 | 2026-08-15 | 2026-08-15 | @gilfordting | 2 | — |
| [30177](https://github.com/sgl-project/sglang/pull/30177) | [Feature] Support return_hidden_states="last" | 已合入 | 2026-07-05 | 2026-08-02 | 2026-08-02 | @ltaodream | 11 | run-ci |
| [32334](https://github.com/sgl-project/sglang/pull/32334) | [Speculative Decoding] Fix GPT-OSS EAGLE3 hidden states | 已合入 | 2026-07-24 | 2026-07-31 | 2026-07-31 | @nvpohanh | 9 | bug run-ci |
| [28956](https://github.com/sgl-project/sglang/pull/28956) | Pack aux hidden states into a preallocated buffer | 已合入 | 2026-06-22 | 2026-07-28 | 2026-07-28 | @chromecast56 | 5 | deepseek run-ci |
| [31870](https://github.com/sgl-project/sglang/pull/31870) | [lora] Fix WAR race: never write MoE runner output into hidden_states in place | 已合入 | 2026-07-21 | 2026-07-24 | 2026-07-24 | @sshleifer | 3 | lora run-ci |
| [31157](https://github.com/sgl-project/sglang/pull/31157) | Extract spec aux-hidden-state resolution into a module | 已合入 | 2026-07-14 | 2026-07-14 | 2026-07-14 | @fzyzcjy | 0 | — |
| [28577](https://github.com/sgl-project/sglang/pull/28577) | [Test] Fold EAGLE `return_hidden_states` regression into spec triton suite | 已合入 | 2026-06-17 | 2026-06-17 | 2026-06-17 | @hnyls2002 | 3 | — |
| [28496](https://github.com/sgl-project/sglang/pull/28496) | [Spec] Fix return_hidden_states under spec V2 (issue #26163) | 已合入 | 2026-06-17 | 2026-06-17 | 2026-06-17 | @kpham-sgl | 4 | run-ci |
| [28032](https://github.com/sgl-project/sglang/pull/28032) | [Spec] Centralize dummy verify-input capture; add `carries_draft_hidden_states` | 已合入 | 2026-06-12 | 2026-06-12 | 2026-06-12 | @hnyls2002 | 2 | — |
| [26779](https://github.com/sgl-project/sglang/pull/26779) | [core] Compute dimensions/return_pooled_hidden_states in ForwardBatch.init_new | 已合入 | 2026-05-30 | 2026-05-31 | 2026-05-31 | @hnyls2002 | 0 | run-ci run-ci-extra |
| [25155](https://github.com/sgl-project/sglang/pull/25155) | [perf] avoid hidden states d2h when return_hidden_states=false | 已合入 | 2026-05-13 | 2026-05-14 | 2026-05-14 | @Qiaolin-Yu | 2 | high priority run-ci bypass-fastfa… |
| [25037](https://github.com/sgl-project/sglang/pull/25037) | spec: STANDALONE skips hidden_states end-to-end (Optional schema + None-safe consumers) | 已合入 | 2026-05-12 | 2026-05-12 | 2026-05-12 | @hnyls2002 | 1 | high priority speculative-decoding… |
| [24939](https://github.com/sgl-project/sglang/pull/24939) | spec: Optional spec_info.hidden_states; STANDALONE skips capture | 已合入 | 2026-05-11 | 2026-05-12 | 2026-05-12 | @hnyls2002 | 8 | speculative-decoding |
| [24926](https://github.com/sgl-project/sglang/pull/24926) | spec: centralize EagleDraft{,Extend}Input.hidden_states shape | 已合入 | 2026-05-11 | 2026-05-11 | 2026-05-11 | @hnyls2002 | 5 | run-ci |
| [23190](https://github.com/sgl-project/sglang/pull/23190) | [NPU] add split_qkv_tp_rmsnorm_rope ops for minimax2 & fix eagle3 hidden states capture in dp attn mode | 已合入 | 2026-04-20 | 2026-04-30 | 2026-04-30 | @heziiop | 1 | run-ci |
| [22427](https://github.com/sgl-project/sglang/pull/22427) | [Score API] Add return_pooled_hidden_states to Scoring API for SequenceClassification / RewardModel | 已合入 | 2026-04-09 | 2026-04-15 | 2026-04-15 | @sundar24295s | 2 | run-ci |
| [22836](https://github.com/sgl-project/sglang/pull/22836) | [Speculative] Fix Eagle3/DFLASH aux hidden state capture during CUDA graph init | 已合入 | 2026-04-15 | 2026-04-15 | 2026-04-15 | @merrymercy | 2 | high priority run-ci |
| [19643](https://github.com/sgl-project/sglang/pull/19643) | [BUG] Support tuple hidden_states from fused MXFP4/FP8 quantization | 已合入 | 2026-03-02 | 2026-03-03 | 2026-03-03 | @zyzshishui | 2 | run-ci |
| [18500](https://github.com/sgl-project/sglang/pull/18500) | [Flashinfer Autotune] Fix FlashInfer FP4 MoE autotuning crash by removing incorrect flatten on hidden_states_scale | 已合入 | 2026-02-09 | 2026-08-25 | 2026-02-12 | @YAMY1234 | 13 | run-ci |
| [16480](https://github.com/sgl-project/sglang/pull/16480) | Bugfix: EagleDraftWorker has not attribute "eagle_use_aux_hidden_state" | 已合入 | 2026-01-05 | 2026-01-12 | 2026-01-12 | @chenxu214 | 2 | run-ci |
| [15485](https://github.com/sgl-project/sglang/pull/15485) | Clean `hidden_states_before_norm` | 已合入 | 2025-12-19 | 2025-12-20 | 2025-12-20 | @hnyls2002 | 3 | run-ci |
| [13590](https://github.com/sgl-project/sglang/pull/13590) | [Bugfix] Fix hidden state size in EAGLE PD disaggregation buffers | 已合入 | 2025-11-19 | 2026-04-20 | 2025-11-21 | @michelemarzollo | 9 | run-ci |
| [12798](https://github.com/sgl-project/sglang/pull/12798) | Support capturing aux_hidden_states for minimax m2. | 已合入 | 2025-11-07 | 2025-11-14 | 2025-11-08 | @pyc96 | 0 | run-ci |
| [10801](https://github.com/sgl-project/sglang/pull/10801) | [PD] Improve eagle acceptance rate by transferring draft model hidden states | 已合入 | 2025-09-23 | 2025-10-20 | 2025-10-20 | @ZeldaHuang | 1 | run-ci |
| [9976](https://github.com/sgl-project/sglang/pull/9976) | [BugFix] Fix incorrect hidden_states_tensor in pd disaggregation + eagle | 已合入 | 2025-09-03 | 2025-09-17 | 2025-09-17 | @ZeldaHuang | 1 | high priority run-ci |
| [10228](https://github.com/sgl-project/sglang/pull/10228) | Add self.capture_aux_hidden_states For GLM-4.5V | 已合入 | 2025-09-09 | 2025-09-14 | 2025-09-14 | @zRzRzRzRzRzRzR | 1 | — |
| [9005](https://github.com/sgl-project/sglang/pull/9005) | Fuse two kernels of hidden states padding into quantization kernel | 已合入 | 2025-08-09 | 2025-08-12 | 2025-08-12 | @fzyzcjy | 0 | — |
| [8511](https://github.com/sgl-project/sglang/pull/8511) | Bug: apply final_hidden_states*=self.routed_scaling_factor at MoE lay… | 已合入 | 2025-07-29 | 2025-08-01 | 2025-08-01 | @byjiang1996 | 1 | — |
| [7705](https://github.com/sgl-project/sglang/pull/7705) | saving hidden_states.clone() | 已合入 | 2025-07-02 | 2025-07-05 | 2025-07-05 | @ch-wan | 0 | ready-to-merge |
| [7507](https://github.com/sgl-project/sglang/pull/7507) | feat: add return hidden_states at async generation | 已合入 | 2025-06-25 | 2025-06-25 | 2025-06-25 | @yyihuang | 0 | — |
| [7516](https://github.com/sgl-project/sglang/pull/7516) | [PD][Spec] Fix hidden state transfer for spec decode | 已合入 | 2025-06-25 | 2025-06-25 | 2025-06-25 | @ShangmingCai | 0 | — |
| [7329](https://github.com/sgl-project/sglang/pull/7329) | [OAI Server Refactor] [ChatCompletions & Completions] Support Return Hidden State | 已合入 | 2025-06-18 | 2025-06-21 | 2025-06-21 | @key4ng | 2 | — |
| [7242](https://github.com/sgl-project/sglang/pull/7242) | [PD] Transfer hidden states for mtp when disaggregation | 已合入 | 2025-06-16 | 2025-06-19 | 2025-06-19 | @Atream | 7 | high priority |
| [6716](https://github.com/sgl-project/sglang/pull/6716) | Open AI API hidden states | 已合入 | 2025-05-28 | 2025-09-14 | 2025-06-10 | @kyle-pena-kuzco | 18 | high priority |
| [6440](https://github.com/sgl-project/sglang/pull/6440) | Revert "Implement `return_hidden_states` for the OpenAI API (#6137)" | 已合入 | 2025-05-20 | 2025-05-20 | 2025-05-20 | @zhyncs | 7 | — |
| [6137](https://github.com/sgl-project/sglang/pull/6137) | Implement `return_hidden_states` for the OpenAI API | 已合入 | 2025-05-09 | 2025-05-19 | 2025-05-19 | @kyle-pena-kuzco | 9 | high priority |
| [4269](https://github.com/sgl-project/sglang/pull/4269) | Fix the output of hidden states after HTTP requests | 已合入 | 2025-03-10 | 2025-03-13 | 2025-03-13 | @Qiaolin-Yu | 6 | — |
| [4074](https://github.com/sgl-project/sglang/pull/4074) | Add examples for returning hidden states when using the server | 已合入 | 2025-03-05 | 2025-03-05 | 2025-03-05 | @Qiaolin-Yu | 1 | — |
| [3985](https://github.com/sgl-project/sglang/pull/3985) | Move return_hidden_states to the generate input | 已合入 | 2025-03-01 | 2025-03-02 | 2025-03-02 | @Qiaolin-Yu | 0 | — |
| [3897](https://github.com/sgl-project/sglang/pull/3897) | Add return hidden state in the native API  | 已合入 | 2025-02-26 | 2025-03-01 | 2025-02-27 | @Qiaolin-Yu | 0 | — |
| [3364](https://github.com/sgl-project/sglang/pull/3364) | [Feat] return hidden states | 已合入 | 2025-02-07 | 2025-02-24 | 2025-02-10 | @Jackmin801 | 6 | — |

主线脉络（2025-02 → 2026-10）：

1. 2025-02 #3364 引入 `return hidden states` → 2025-03 #3985 / #4074 迁到 generate input 并补示例 → 2025-06 #7329 / #7507 上 OpenAI API 与 async 生成（#6137 曾因 OOM 被 #6440 revert，#6716 重新落地）；
2. 2025-10 #10801、#9976、#7516、#7242 打通 **PD 分离下 draft/MTP hidden states 的传递**；
3. 2026-04 #22836（CUDA graph init 期的 aux 捕获）、2026-05 #24926 / #24939 / #25037 / #25155 把 `hidden_states` 改为 Optional、并在 STANDALONE 下端到端跳过捕获（性能 ＋ 接口简洁性）；
4. 2026-07 #28956（aux hidden states 打进预分配 buffer）、#31157（把 spec aux-hidden-state 的层解析抽成独立模块）；
5. **2026-08-02 #30177 支持 `return_hidden_states="last"`** —— 对外接口的最后一次功能性扩展；
6. 2026-08 / 09 密集修模型侧捕获：GLM-5.3-Flash（#36708）、mHC（#36755）、Nemotron（#40801）、mimo-vl（#40888）。

### 4.3 全部进行中（9 条）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [43449](https://github.com/sgl-project/sglang/pull/43449) | [Feature] Capture aux hidden states on the target without a speculative draft | 进行中 | 2026-10-10 | 2026-10-10 | — | @maocheng23 | 0 | documentation |
| [41448](https://github.com/sgl-project/sglang/pull/41448) | [AMD][DeepSeek-V4.1] Fix server crash on prompt-logprob and hidden-state requests under decoder SWA bounded replay | 进行中 | 2026-09-27 | 2026-10-09 | — | @amd-pedghazi | 0 | documentation quant amd dependenci… |
| [38642](https://github.com/sgl-project/sglang/pull/38642) | [DFLASH] Support Qwen4-Exp hidden-state capture | 进行中 | 2026-09-09 | 2026-10-08 | — | @Leslie360 | 7 | — |
| [39290](https://github.com/sgl-project/sglang/pull/39290) | [DeepSeek-V4] Forward auxiliary hidden states across PP stages | 进行中 | 2026-09-13 | 2026-09-13 | — | @gerayking | 0 | — |
| [39286](https://github.com/sgl-project/sglang/pull/39286) | [Bugfix] Fix Gemma auxiliary hidden-state captures | 进行中 | 2026-09-13 | 2026-09-13 | — | @whzzt | 0 | — |
| [33444](https://github.com/sgl-project/sglang/pull/33444) | [Spec] Fuse aux hidden state capture and support DFLASH with breakable/full prefill CUDA graphs | 进行中 | 2026-08-04 | 2026-08-05 | — | @Boreas618 | 1 | quant deepseek speculative-decodin… |
| [33513](https://github.com/sgl-project/sglang/pull/33513) | [bugfix] all reduce hidden_states from mlp in last decoder layer for Qwen3NextModel | 进行中 | 2026-08-04 | 2026-08-04 | — | @staugust | 1 | — |
| [32057](https://github.com/sgl-project/sglang/pull/32057) | [Fix] Preserve caller-owned hidden_states in DeepGemm standard pre-permute | 进行中 | 2026-07-22 | 2026-07-28 | — | @lcz1998 | 1 | — |
| [15614](https://github.com/sgl-project/sglang/pull/15614) | fix: main block return hidden states before norm for MiMo v2 flash | 进行中 | 2025-12-22 | 2025-12-25 | — | @ollybbmonster | 1 | run-ci |

真正的「功能新增」只有 #43449 / #43450 两条；其余是模型侧捕获修正（#38642 Qwen4-Exp、#39286 Gemma、#33513 Qwen3Next）与 PP 场景（#39290 DeepSeek-V4 跨 PP 前传 aux hidden states，draft，376 行），外加 #15614（MiMo v2 flash 在 norm 前返回 hidden states，2025-12-22 开出，**仅 2 行改动已挂近 10 个月**）。

### 4.4 已关闭未合入（此处列 2026-06 之后关闭的 28 条）

| PR | 标题 | 状态 | 创建 | 更新 | 合入/关闭 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|---|
| [32424](https://github.com/sgl-project/sglang/pull/32424) | [DSpark PD 3/3] Implement streaming hidden-state transfer for Mooncake | 已关闭 | 2026-07-26 | 2026-10-09 | 2026-10-09 | @zhangxiaolei123456 | 4 | deepseek |
| [32423](https://github.com/sgl-project/sglang/pull/32423) | [DSpark PD 2/3] Integrate hidden-state bootstrap into PD scheduling | 已关闭 | 2026-07-26 | 2026-10-09 | 2026-10-09 | @zhangxiaolei123456 | 2 | deepseek |
| [39644](https://github.com/sgl-project/sglang/pull/39644) | [Spec][PP] Reject EAGLE3/DFLASH aux hidden-state capture under pipeline parallelism | 已关闭 | 2026-09-15 | 2026-09-29 | 2026-09-29 | @YAMY1234 | 3 | — |
| [32103](https://github.com/sgl-project/sglang/pull/32103) | fix(hidden_states): keep prefill offset aligned for non-requesting reqs to stop cross-request leak | 已关闭 | 2026-07-22 | 2026-09-23 | 2026-09-23 | @Anai-Guo | 2 | — |
| [40332](https://github.com/sgl-project/sglang/pull/40332) | fix: transfer native hidden states from prefill to decode | 已关闭 | 2026-09-19 | 2026-09-19 | 2026-09-19 | @ishandhanani | 0 | — |
| [39776](https://github.com/sgl-project/sglang/pull/39776) | [NPU][Bugfix][MiMo-V2.5-Pro] Capture post-final-norm hidden state for the final DFlash aux layer | 已关闭 | 2026-09-16 | 2026-09-18 | 2026-09-18 | @iridiumine | 0 | — |
| [28261](https://github.com/sgl-project/sglang/pull/28261) | [Bugfix]Fix Qwen3-VL DFlash aux hidden state capture setup | 已关闭 | 2026-06-15 | 2026-09-16 | 2026-09-16 | @gq112 | 1 | — |
| [21434](https://github.com/sgl-project/sglang/pull/21434) | [spec decoding] avoid capturing hidden_states when not needed | 已关闭 | 2026-03-26 | 2026-09-14 | 2026-05-12 | @Qiaolin-Yu | 3 | run-ci |
| [28498](https://github.com/sgl-project/sglang/pull/28498) | [Spec] Add NGRAM hidden_states regression test (stacked on #28496) | 已关闭 | 2026-06-17 | 2026-09-14 | 2026-06-17 | @kpham-sgl | 0 | — |
| [39161](https://github.com/sgl-project/sglang/pull/39161) | [Bugfix] Preserve complete Gemma speculative hidden-state captures | 已关闭 | 2026-09-12 | 2026-09-12 | 2026-09-12 | @whzzt | 1 | speculative-decoding jit-kernel |
| [26315](https://github.com/sgl-project/sglang/pull/26315) | [Fix] Generalize spec-V2 hidden_states slicing (stacked on #26217) | 已关闭 | 2026-05-25 | 2026-09-12 | 2026-09-12 | @HumphreySun98 | 2 | — |
| [26652](https://github.com/sgl-project/sglang/pull/26652) | Add EAGLE3 + return_hidden_states regression test | 已关闭 | 2026-05-29 | 2026-09-08 | 2026-09-08 | @LIUYellowBlack | 2 | — |
| [26217](https://github.com/sgl-project/sglang/pull/26217) | [Fix] Generalize spec-V1 hidden_states slicing to all V1 algorithms (stacked on #26150) | 已关闭 | 2026-05-24 | 2026-09-06 | 2026-09-06 | @javierlimt6 | 3 | — |
| [25651](https://github.com/sgl-project/sglang/pull/25651) | [AMD] Guard Qwen3.5 layers against empty hidden states | 已关闭 | 2026-05-18 | 2026-09-05 | 2026-09-05 | @lishuoshuo-amd | 4 | amd |
| [25343](https://github.com/sgl-project/sglang/pull/25343) | [RFC] Pluggable hidden-states transport for spec-training callback (colocate) | 已关闭 | 2026-05-15 | 2026-09-04 | 2026-09-04 | @zhubohao911 | 3 | documentation |
| [28526](https://github.com/sgl-project/sglang/pull/28526) | [NPU] Quantize bfloat16 hidden_states before npu_grouped_matmul in W4… | 已关闭 | 2026-06-17 | 2026-08-26 | 2026-08-26 | @496041613 | 1 | npu |
| [20834](https://github.com/sgl-project/sglang/pull/20834) | [Bugfix] Fix PP proxy tensor corruption with TP+MoE+A2A (skip allgather on scattered hidden states) | 已关闭 | 2026-03-18 | 2026-08-25 | 2026-08-25 | @yu-depend | 4 | — |
| [12183](https://github.com/sgl-project/sglang/pull/12183) | Option to enable auxiliary hidden capture mode. | 已关闭 | 2025-10-27 | 2026-08-11 | 2026-08-11 | @w32zhong | 2 | — |
| [32422](https://github.com/sgl-project/sglang/pull/32422) | [DSpark PD 1/3] Add hidden-state transfer primitives and draft injection | 已关闭 | 2026-07-26 | 2026-08-06 | 2026-08-06 | @zhangxiaolei123456 | 4 | deepseek |
| [32680](https://github.com/sgl-project/sglang/pull/32680) | [Fix] Capture Eagle3/DFlash aux hidden states inside the two-batch-overlap region | 已关闭 | 2026-07-28 | 2026-07-29 | 2026-07-28 | @hnyls2002 | 1 | deepseek |
| [29015](https://github.com/sgl-project/sglang/pull/29015) | Fix EAGLE hidden-state sizing and text prefill CUDA graphs | 已关闭 | 2026-06-23 | 2026-06-24 | 2026-06-23 | @merrymercy | 1 | run-ci |
| [29018](https://github.com/sgl-project/sglang/pull/29018) | Fix Eagle draft hidden state shape handling | 已关闭 | 2026-06-23 | 2026-06-24 | 2026-06-24 | @merrymercy | 14 | run-ci |
| [26150](https://github.com/sgl-project/sglang/pull/26150) | [Fix] NGRAM spec decode: use `is not None` for hidden_states tensor check | 已关闭 | 2026-05-23 | 2026-06-17 | 2026-06-17 | @Nas01010101 | 5 | — |
| [24780](https://github.com/sgl-project/sglang/pull/24780) | [Spec] Introduce `EagleDraftExtendInput`; eliminate hidden_states phase-shift | 已关闭 | 2026-05-09 | 2026-06-14 | 2026-06-09 | @hnyls2002 | 1 | high priority run-ci bypass-fastfa… |
| [27540](https://github.com/sgl-project/sglang/pull/27540) | [gemma4] Compute RMSNorm in fp32 to fix deep-layer hidden-state divergence vs HF | 已关闭 | 2026-06-08 | 2026-06-12 | 2026-06-12 | @yihao-liang | 3 | — |
| [18542](https://github.com/sgl-project/sglang/pull/18542) | fix: fixed aux hidden state index out of range when using eagle3 with nsa cp | 已关闭 | 2026-02-10 | 2026-06-12 | 2026-06-12 | @echo-rain | 2 | deepseek |
| [13868](https://github.com/sgl-project/sglang/pull/13868) | [Feature] Add aux and last hidden state dumping for EAGLE | 已关闭 | 2025-11-25 | 2026-06-11 | 2026-06-11 | @ChangseokSong | 2 | documentation |
| [6605](https://github.com/sgl-project/sglang/pull/6605) | Add return_hidden_states for async generate | 已关闭 | 2025-05-26 | 2026-06-10 | 2026-06-10 | @kebe7jun | 7 | — |

模式很清楚：**关闭的绝大多数是「模型侧 aux 捕获的一次性修补」**（Gemma、Qwen3-VL、MiMo-V2.5-Pro、mHC、NSAC、各家 NPU/AMD），有的被后续 PR 取代，有的作者自行撤回。真正有分量的关闭只有两组：

- **#25343（RFC：为 spec-training callback 提供可插拔的 hidden-states 传输，2026-09-04 关闭）**——它提出的需求，正是 #43449 / #43450 现在试图以更靠上层的方式解决；
- **#32422 / #32423 / #32424（DSpark PD 三段式）**——见 4.1。

## 5. 跨仓库对比

| 维度 | vLLM | SGLang |
|---|---|---|
| 核心机制 | KV-connector 承载 hidden states → `ExtractHiddenStatesProposer` → drafter | 模型内 `capture_aux_hidden_states` ＋ `--return-hidden-states-mode` 双通道 |
| 高相关 PR / 已合入 / 进行中 / 已关闭 | 87 / 35 / 27 / 25（关闭率 29%） | 96 / 47 / 9 / 40（关闭率 42%） |
| 本周期合入 | #60763（10-09）、#57197（09-30）、#49819（09-17） | #40888（10-06）、#41080（09-24）、#40801（09-23） |
| 对请求直接返回 hidden states | **仍在建设**：#59543 / #57185 / #57736 / #57663 全部未合入，RFC #59542 刚开 | **已有** `--return-hidden-states-mode`（含 `"last"`），本周期只在修边界（#41448） |
| 「不挂 draft 也能采集」 | 没有统一开关，按模型逐个补（#59037 GLM-5.3、#52184 inkling、#60855 Granite…） | #43449 用两个 server 参数统一打开（**本周期新增，最重要**） |
| 跨进程 / PD 传输 | Mooncake EC connector（#47302）＋ P/D PoC 塞 KV slot（#60545） | DSpark PD 三段式全部关闭；#43450 转向 observer ＋ 异步 consume |
| 积压特征 | needs-rebase/stale 多、评审带宽紧、KV cache 计费类集中 | 大量模型逐个修、作者自行撤回、功能类仅 2 条 |
| 代码落点 | `vllm/model_executor/models/extract_hidden_states.py`（内核 06-30 后冻结）、`v1/spec_decode`、`v1/kv_cache_interface.py` | `srt/model_executor`（capture 钩子）、`srt/arg_groups/fields/exec_.py`（server 参数）、`srt/speculative` |

**一句话差异：vLLM 在把 hidden states 做成「引擎的一等输出与一等传输对象」（KV slot、PP/PD、计费），SGLang 在把「采集」做成可配置、可外接的公共能力（server args、observer）。** 两者都在服务 RL / 数据生产管线，落点不同：vLLM 偏基础设施，SGLang 偏接口与扩展点。

## 6. 建议跟踪清单

**P0 —— 直接决定 RL 数据生产可行性**

1. vLLM #57736 ＋ #57663（windowed hidden-state capture for rollout）：目前 GPUModelRunner 侧抽取尚未实现、窗口→step 的位置数学仍在征求设计意见；这是 rollout 侧按 token 窗口取隐状态的唯一入口。
2. SGLang #43449 ＋ #43450：若合并，可以不再 patch SGLang 就从纯 target server 拿到 draft 训练特征并异步落盘（Mooncake），对 SpecForge 类管线是数量级的维护成本差异。
3. vLLM #59543（＋ RFC #59542）：决定「每个生成 token 的 hidden state」是否成为正式 API 输出，影响所有把隐状态当奖励/特征的下游。

**P1 —— 正确性与稳定性风险**

4. vLLM #43869：CUDA graph padding token 的 `-1` slot 会静默污染 KV cache（根因见 3.1）；已 stale，需确认是否已被其他改动顺带修掉，否则应推动合并或显式关闭。
5. SGLang #41448：`--enable-decoder-swa-bounded-replay` ＋ hidden-state 请求会崩掉整个 server。
6. vLLM #56822 / #53074 / #59037：hidden-state 层的 KV cache 计费与布局（hybrid Mamba/MLA、DeepSeek-V4 MLA、packed BLHNC），直接决定服务能否起来。
7. vLLM #60477：PP 边界激活值错误，带 `ready`，收益明确（TP8PP2 对齐 TP16）。

**P2 —— 能力覆盖面**

8. 模型侧逐个补：vLLM #60855（Granite）、#50815（Kimi-K3）、#43671（MiniMax-M2）、#52184（inkling）、#55682（GLM-5.3-Flash）；SGLang #38642（Qwen4-Exp）、#39286（Gemma）、#39290（DeepSeek-V4 + PP）。
9. 长期挂起、建议二选一（推动或关闭）：SGLang #15614（MiMo v2 flash，2 行改动，挂 10 个月）；vLLM #46421（测试）、#52074（文档）、#43869（见 P1）。

## 7. 数据来源与方法（可复现）

- **PR 检索**：GitHub Search API（`/search/issues`）。对两个官方仓库分别用 6 组关键词交叉检索后按 `repo#number` 合并去重：标题含 `"hidden states"` / `"hidden state"`；正文含 `extract_hidden_states` / `return_hidden_states` / `aux_hidden_states` / `aux_hidden_state`。共得 **286 条唯一 PR**，再按标题正则（含 `hidden state(s)` / `hidden_state` / `HiddenStates` / `aux hidden` / `aux-state` / `hidden capture`，排除 `hidden_size` / `hidden dim` / `num_hidden_layers` / `hidden_act` 等无关词）筛出 **183 条高相关**（vLLM 87 / SGLang 96）。
- **提交检索**：GitHub Search API（`/search/commits`），关键词 `extract_hidden_states`、`return_hidden_states`、`auxiliary hidden states`、`aux hidden states`；另按文件查询 `vllm/model_executor/models/extract_hidden_states.py` 在 main 上的历史（结果：最后功能性变更 2026-06-30 #46301）。
- **本地仓库交叉核对**：`git log` 检索 `upstream/main`（sglang，含少量无 PR 号的 squash 前提交，如 `fix: relay requested hidden states across pipeline stages`）与 `origin/main`（vllm）；确认 vLLM 本地检出落后上游约 38 天，**故以 GitHub API 的远端状态为准**，本地仅用于代码定位。
- **重点 PR 详情**：对 13 条关键在途 PR 抽查了 `mergeable_state` / draft / changed_files / review_comments（结果已并入 3.1 与 4.1 的叙述）。
- **未做的事（避免误用）**：未逐条拉取 review 线程与 CI 结果；未做逐行代码审查；未复现 PR 描述中的性能/精度数字；`已关闭` 表只列 2026-06 之后关闭的条目，更早的见 JSON / CSV。表中的「标签」列做过截断（超过 34 字符加省略号），完整标签见 CSV。
