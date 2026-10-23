# sample mask 增量更新（2026-09-29 → 2026-10-10）

> 基线：同目录 `sample-mask-prs.md` / `.json` / `.csv`（归档日期 2026-09-29，核心相关 30 条 + 边缘 8 条，共 38 条记录）
> 本次：2026-10-10 用**同一组** GitHub Search 关键词重跑检索（sglang 122/3/104、vllm 79/10/106 命中，去重后 254 条 PR），逐条比对基线记录的 `state` / `merged_at` / `comments` / `updated_at`
> 产物：本文件、[sample-mask-delta-2026-10-10.csv](sample-mask-delta-2026-10-10.csv)（31 条增量记录）
> 术语沿用基线：`merged` = 已合入，`open` = 进行中，`closed` = 已关闭未合并

## 1. 一句话结论

**上次归档里两条压舱石级的 open PR 都在这 11 天合入了**：SGLang #34201（投机解码路径的 top-p mask 捕获 + DFlash/DSpark 实现）与 vLLM 侧 MRV2 MTP 的 mask replay（#59359）。同时该特性域的重心明显从「能不能拿到 mask」转向「拿 mask 的代价」与「过滤语义的一致性」。

- 归档内 30 条核心 PR：**3 条状态变化（全部 open → merged）**，3 条活动增加，其余无变化。
- 新增核心相关 **14 条**（2 已合入 / 3 已关闭 / 9 进行中），另补漏 8 条上次未收录但在改同一份代码的 PR（含 1 条更早开出、性质属同域的 PR）。
- 归档中仍有 6 条 open 老面孔，其中 3 条 11 天内零活动。

## 2. 归档内 PR 的状态变化（3 条，全部 open → merged）

| PR | 标题 | 状态 | 创建 | 合入/最后更新 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|
| [34201](https://github.com/sgl-project/sglang/pull/34201) | [RL, Spec] Introduce top-p mask capture for spec and add DFlash/DSpark impl | 已合入 | 2026-08-10 | 2026-09-30 | @jvmncs | 16 | run-ci jit-kernel |
| [54442](https://github.com/vllm-project/vllm/pull/54442) | [Bugfix] Don't let a structured-output request sample from an unmasked row | 已合入 | 2026-08-30 | 2026-10-01 | @ArcheyChen | 29 | bug speculative-decoding ready… |
| [55935](https://github.com/vllm-project/vllm/pull/55935) | [Bugfix] Preserve sampling masks in DELTA and TITO streaming outputs | 已合入 | 2026-09-08 | 2026-10-01 | @aoshen02 | 4 | bug documentation frontend rea… |

- **sglang #34201**（8/10 开出，挂 7 周，16 条评论）——上次归档中 SGLang 侧最重的未合并项。它把 #27408 引入的 mask 机制扩展到 DFlash 家族，为 DFlash 与 DSpark 各写 worker adapter，同时保持 #35205 的「top_k 必须有限」契约。动机原文明确以 RL 为准：sampling mask 让 trainer 更准确地重建推理策略、降低 rollout 与训练策略的错配。**它同时是 #41955「过滤顺序」讨论的语义前提**（见 §5）。
- **vllm #54442**（29 条评论）——结构化输出的 grammar bitmask 只在 `apply_bitmask` 持有期间约束该行；draft 窗口中的 `-1` 占位符会置位 `seen_padding`，使其后所有行被填成 `_full_mask`（全词表放开），即「本该被约束的请求采样到了未加 mask 的行」。属采样 mask 正确性缺陷。
- **vllm #55935**（@aoshen02）——`--return-sampling-mask` 此前只在 final output 带 mask，DELTA 消费者与 `/inference/v1/generate` 流式客户端拿不到，Rust 前端该路由完全不返回；现改为 mask 跟随 token 下发，只改 3 行 `output_processor.py` + 7 行 `outputs.py`，CUMULATIVE / FINAL_ONLY 语义不变。**归档时它标为 open、0 评论，现已合入。**

## 3. 归档内仍在 open 的老面孔（活动对比）

| PR | 标题 | 状态 | 创建 | 合入/最后更新 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|
| [57442](https://github.com/vllm-project/vllm/pull/57442) | [Frontend][Core] Sampled-token logprob fast path for /inference/v1/generate (return_token_logprobs) | 进行中 | 2026-09-17 | 2026-10-09 | @yinli-systems | 17 | documentation frontend needs-r… |
| [53826](https://github.com/vllm-project/vllm/pull/53826) | [Bugfix][Model Runner V2] Preserve sampling masks in batch-sharded sampling | 进行中 | 2026-08-26 | 2026-10-01 | @waizuichougou | 5 | bug needs-rebase mrv2 |
| [58231](https://github.com/vllm-project/vllm/pull/58231) | [Bugfix][Sampler] Fix split top-p masking every token for huge logits (thinking_token_budget + spec decode emits token 0) | 进行中 | 2026-09-23 | 2026-10-06 | @zamog | 3 | bug |
| [57540](https://github.com/vllm-project/vllm/pull/57540) | [Feature][Sampling] Add native GPU output no-repeat n-gram masking | 进行中 | 2026-09-18 | 2026-09-25 | @HuXinjing | 10 | documentation performance mrv2 |
| [41205](https://github.com/sgl-project/sglang/pull/41205) | [Sampling] Write /generate sampling masks from numpy rows | 进行中 | 2026-09-25 | 2026-09-25 | @nanjiangwill | 0 |  |
| [39934](https://github.com/sgl-project/sglang/pull/39934) | Feature/npu sampling mask max tokens | 进行中 | 2026-09-17 | 2026-09-17 | @wl55721 | 0 | npu |

- **#57442 是本组最活跃的**：评论 5 → 17，10-07 已按 #58181 的新输出形状（`logprobs.sampled` on `GenerateLogProbs`，取代独立的 `choices[].token_logprobs`）rebase，仅 tokens-mode、NaN 钳到 `-9999.0`；但作者注明 **live-server 测试自该改动后尚未在 GPU 上重跑**，且仍带 needs-rebase。
- **#53826** 10-01 有更新（4 → 5 评论）、仍 needs-rebase。它是 #49577 引入 mask replay 后、被 #50465 的 batch-sharded sampler-output gather 丢掉 `SamplingMaskTensors` 元数据造成的回归——即 TP + MRV2 分片采样下 mask 静默丢失。
- **#58231** 10-06 有更新（2 → 3 评论）。
- **11 天内零活动**：sglang #41205（09-25 后无动静，0 评论）、sglang #39934（09-17 后无动静）、vllm #57540（09-25 后无动静）。

## 4. 新增核心相关：已合入 / 已关闭（3 条）

| PR | 标题 | 状态 | 创建 | 合入/最后更新 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|
| [59359](https://github.com/vllm-project/vllm/pull/59359) | [Feature][Spec Decode] Support sampling mask replay for MRV2 MTP | 已合入 | 2026-09-30 | 2026-10-01 | @vx120 | 13 | documentation speculative-deco… |
| [59358](https://github.com/vllm-project/vllm/pull/59358) | [Feature][Spec Decode] Support sampling mask replay for MRV2 MTP | 已关闭 | 2026-09-30 | 2026-09-30 | @vx120 | 2 | documentation speculative-deco… |
| [59361](https://github.com/vllm-project/vllm/pull/59361) | [Doc] Score centering via top-k processed logprobs | 已合入 | 2026-09-30 | 2026-10-01 | @vx120 | 5 | documentation frontend ready m… |

- **vllm #59359 是本次最重要的合入**，也是上次归档中「closed 未合并」的 **#54166 的第三次尝试**：**#54166（09-11 关闭，27 评论）→ #59358（09-30 开、当日关闭，2 评论）→ #59359（09-30 开、10-01 合入，13 评论）**。实现要点：在紧凑 sampling-mask 表示之上做 replay，保留单 token 快路径与精确 packed-mask 回退；对每个 emitted 的 accepted / recovered / bonus token 各返回一份 processed-target support；输出 D2H 排在 postprocessing 与下一次 proposal 之前。**关键差异：合入版本不再把 replay 限制在 `method="mtp"` 或标准 rejection sampling。**
- **vllm #59361**（同一作者）是配套文档：把 score centering（arXiv 2609.20807）走「top-k processed logprobs」的路径写进 `docs/training/sampling_mask.md`，明确 `token_id:<id>` 编码、`--max-logprobs` 上限、support 之外 token 的 `-9999.0` 哨兵值，以及 sampled token 会重复出现。**vLLM 已经把 sampling mask 当训练侧正式接口在写契约文档。**

## 5. 新增核心相关：进行中（9 条，另附 2 条同日关闭的兄弟 PR）

| PR | 标题 | 状态 | 创建 | 合入/最后更新 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|
| [59567](https://github.com/vllm-project/vllm/pull/59567) | [Perf] Batch sampling-mask transport and skip mask re-validation | 进行中 | 2026-10-01 | 2026-10-03 | @aoshen02 | 1 | frontend needs-rebase mrv2 rus… |
| [43047](https://github.com/sgl-project/sglang/pull/43047) | [sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding | 进行中 | 2026-10-08 | 2026-10-08 | @nreHieW | 0 |  |
| [43044](https://github.com/sgl-project/sglang/pull/43044) | [sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding | 已关闭 | 2026-10-08 | 2026-10-08 | @nreHieW | 0 |  |
| [43346](https://github.com/sgl-project/sglang/pull/43346) | [Sampler] Sort only the max(top_k) leading columns in the torch sampler | 进行中 | 2026-10-09 | 2026-10-09 | @jiaqiang-dot-liu | 0 |  |
| [42802](https://github.com/sgl-project/sglang/pull/42802) | [Speculative] Capture reusable verification sampling in CUDA graphs | 进行中 | 2026-10-06 | 2026-10-06 | @dmtri35 | 0 | deepseek jit-kernel |
| [41955](https://github.com/sgl-project/sglang/pull/41955) | [Sampling] Default to `top_k_first` and expose filtering order | 进行中 | 2026-09-30 | 2026-10-02 | @hnyls2002 | 0 | jit-kernel |
| [41957](https://github.com/sgl-project/sglang/pull/41957) | [Perf] Fuse compact `top_k_first` probability construction | 进行中 | 2026-09-30 | 2026-09-30 | @hnyls2002 | 0 | jit-kernel |
| [41954](https://github.com/sgl-project/sglang/pull/41954) | [Sampling] Align regular decoding with `top_k_first` | 已关闭 | 2026-09-30 | 2026-09-30 | @hnyls2002 | 0 |  |
| [41916](https://github.com/sgl-project/sglang/pull/41916) | [Bugfix] Make top-k/top-p filtering consistent across sampling paths | 进行中 | 2026-09-30 | 2026-10-06 | @alvion427 | 5 |  |
| [50979](https://github.com/vllm-project/vllm/pull/50979) | [Bugfix][Sampler] Keep exactly k tokens in top-k with tied logits | 进行中 | 2026-08-04 | 2026-10-10 | @liuxiang-sudo | 6 | bug verified |
| [56724](https://github.com/vllm-project/vllm/pull/56724) | [Spec Decode] Draft within the requests' top-k / top-p support | 进行中 | 2026-09-13 | 2026-10-04 | @gf239 | 13 | speculative-decoding mrv2 dfla… |

按主题分三组看：

**（a）性能与代价 —— 本周期的新焦点**

- **vllm #59567 给出了目前最硬的成本数据**：`--return-sampling-mask` 在 Qwen3-8B / GB300 上，窄 mask 损失 **5–8%** 输出吞吐，宽 mask（`top_k=50, top_p=1.0` → 每 token 50 个 id）损失 **14–18%**；且几乎全部是 engine 与 API server 的 **CPU** 开销而非 GPU。逐项拆解中最贵的是 `EngineCoreOutputs` 的 msgspec 编码（每个请求的 ndarray 触发一次 `enc_hook`，**+320 µs/step**）。方向是批量传输 mask 并跳过 mask 的重复校验，目前 needs-rebase。
- **sglang #43346** 是采样器侧对应问题：`top_k_top_p_min_p_sampling_from_probs_torch` 每个 decode step 对**整个词表**排序、再把 rank ≥ top_k 的列清零；而 **ROCm 上这正是默认后端**（`is_flashinfer_available()` 要求 CUDA，`sampling_backend` 落到 `pytorch`），任何有限 top_k 的请求都会走这条路。

**（b）过滤语义统一 —— SGLang 正在做一次影响面较大的默认值变更**

- **#41955** 把常规解码、投机验证与 sampling-mask 捕获统一为 `top_k_first`（先 top-k → 重归一化 → 再 top-p），并新增 server 级 `--sampling-filter-order`（`top_k_first` 为默认，`joint` 保留常规解码从 #1457 起的旧语义），理由是与 EAGLE/DFlash/UNO 现有顺序一致、且与 HF Transformers 对齐。**这直接决定 mask 捕获出的 support 与实际采样分布是否描述同一件事。** 示例：概率 `[0.5,0.3,0.2]`、`top_k=2`、`top_p=0.55` 时，joint 保留 token 0 与 1，top-k-first 重归一化到 `[0.625,0.375]` 后只保留 token 0。
- **#41954** 是它的前身（当日关闭），**#41957** 是配套的融合 kernel。
- **#41916** 修的是相反方向的不一致：部分路径（FlashInfer min-p、共享 EAGLE 路径等）先做 top-k 重归一化、再对剩余概率质量套 top-p，使 top-p 的效果取决于 top-k 后剩多少质量。它取代了 #36173。
- 这三条（#41955 / #41957 / #41916）与已合入的 #34201 属同一语义族，建议合并评审口径。

**（c）tie / 边界与投机路径覆盖**

- **vllm #50979**（已标 `verified`）：PyTorch top-k 用严格阈值 `logits < k-th largest`，会把并列在 k-th 的 token **全部**保留（可能多于 k 个），而 Triton kernel 恰好保留 k 个；由于 `apply_top_k_top_p` 按 batch size 分派（≥8 走 Triton），**同一请求单独跑与凑批跑会采出不同 token**。
- **vllm #56724**：V2 drafter 在 `draft_sample_method="probabilistic"` 下从全词表采样，而 target 只在请求的 top-k/top-p 内采样，导致 draft token 必被拒；改为在 drafter 的 CUDA graph 内套用请求的 top-k/top-p（输出分布不变、接受率提高）。原文指出 main 上 `NOTE(woosuk)` 是有意不做这件事的（简单性与性能）。
- **sglang #43047 / #43044**：sglang-miles 分支上 `return_sampling_mask` 会**直接拒绝所有投机解码请求**，导致用 EAGLE/NEXTN 时拿不到 support 与归一化 logprob；同题两次提交，前者 open。
- **sglang #42802**：把验证采样逻辑在 EAGLE chain、DSpark、DFlash、UNO 与 sampling-mask consumer 之间复用并捕获进 CUDA graph，保持截断 tie 行为。

## 6. 补漏：上次归档未收录、但在改同一份代码（8 条）

上次归档按打分（score ≥ 14）截取，以下落在阈值之下或未被当时检索覆盖，但确实在动采样 mask / top-p 掩码本体，建议并入跟踪：

| PR | 标题 | 状态 | 创建 | 合入/最后更新 | 作者 | 评论 | 标签 |
|---|---|---|---|---|---|---|---|
| [54651](https://github.com/vllm-project/vllm/pull/54651) | [Core] Triton kernel for small-batch top-p only masking | 已合入 | 2026-09-01 | 2026-09-03 | @njhill | 15 | ready |
| [55414](https://github.com/vllm-project/vllm/pull/55414) | [Perf] Precompile split top-p kernel variants to avoid runtime JIT | 进行中 | 2026-09-04 | 2026-09-13 | @njhill | 4 | ready needs-rebase |
| [59696](https://github.com/vllm-project/vllm/pull/59696) | [Perf] Reduce Triton top-k/top-p value specializations | 已关闭 | 2026-10-01 | 2026-10-01 | @WoosukKwon | 0 |  |
| [58542](https://github.com/vllm-project/vllm/pull/58542) | [Perf][PP] Skip sampled-token broadcasts whose requests leave the engine | 已合入 | 2026-09-24 | 2026-10-01 | @LostFox11 | 18 | ready mrv2 |
| [55708](https://github.com/vllm-project/vllm/pull/55708) | [Bugfix][Core] Preserve speculative metrics when coalescing outputs | 已合入 | 2026-09-07 | 2026-10-07 | @git-jxj | 11 | bug ready |
| [38565](https://github.com/sgl-project/sglang/pull/38565) | [Sampling] Add `SGLANG_RENORM_DETERMINISTIC` for bit-identical top-p / top-k renorm | 进行中 | 2026-09-08 | 2026-09-29 | @gilfordting | 16 | sgl-kernel run-ci |
| [37531](https://github.com/sgl-project/sglang/pull/37531) | [Spec][Fix] Fall back to eager DFlash sampler above graph batch limit | 进行中 | 2026-09-02 | 2026-09-29 | @LingZ315 | 5 |  |
| [40953](https://github.com/sgl-project/sglang/pull/40953) | [PD] Transfer EAGLE draft probabilities for rejection sampling | 进行中 | 2026-09-23 | 2026-09-23 | @roborluo | 0 | documentation |

- **vllm #54651**（njhill，09-03 合入）小批量 top-p-only masking 的 Triton kernel，正是 #59567 所测 `--return-sampling-mask` 路径的基础设施；#55414 / #59696 继续收敛该 kernel 的 JIT 与特化开销。
- **sglang #38565** 与 mask 直接相关：`SGLANG_RENORM_DETERMINISTIC`（默认关）把 top-p/top-k 重归一化路由到逐次调用逐位一致的 kernel，默认仍用快速 flashinfer kernel（确定性 top-k 回退慢 4–12 倍）。原文明确给出适用场景：**「当 renorm 输出本身需要跨 TP rank 一致或可复现时开启——logprobs、sampling masks」**。16 条评论仍 open。
- **sglang #40953**（09-23 开出，早于上次归档日期故未收录）属同一族但方向相反：PD 分离下需要把 EAGLE draft 的**完整 FP32 提议分布**传给 decode 侧，因为 top-k 候选集合本身不足以支撑 verifier 的拒绝采样——又一次说明「support 集合」与「支撑拒绝采样的概率分布」是两件事。

## 7. 与上次归档相比的三个趋势

1. **「推翻重做」仍在继续，但这轮两条主线都收敛合入了。** vllm #54166 → #59358 → #59359 是同一功能的第三次提交并最终合入；sglang #41954 → #41955 同理。上次归档给出的判断（该特性经历明显的方案重做）依旧成立，只是这 11 天里两条主线都落到了 main 上。
2. **关注点从「暴露」转向「代价 + 语义」。** 新增项里没有一条是在讨论「要不要支持 mask 输出」，而全是：#59567（5–18% 吞吐代价、CPU 编码瓶颈）、#43346（每步全词表排序）、#41955/#41916/#41957（过滤顺序）、#50979（并列 tie 的跨路径不一致）、#56724（draft 与 target 的 support 不一致）。
3. **RL 训练侧的正式化。** vllm #59361 把 score centering 的用法写进 `docs/training/sampling_mask.md`（含哨兵值、`token_id:` 编码、`--max-logprobs` 上限等契约细节）；sglang #34201 以「RL trainer 重建推理策略」为动机合入。mask 正在从调试/实验功能变成有文档、有契约的训练侧接口。

## 8. 建议动作

- **优先评审**：sglang #41955（改默认过滤顺序，影响面最大，且会改变默认采样行为）＞ vllm #59567（mask 回传性能，已量化，needs-rebase）＞ vllm #50979（跨路径采样不一致，已 `verified`）。
- **统一口径**：sglang #41916 / #41957 与 #41955 是同一主题的三个切面，建议合并评审，避免再次「先合一半」。
- **可以清理或明确关闭**：sglang #39934、#41205 已 2 周以上零活动；vllm #57540 自 09-25 起零活动。
- **长尾但重要，别忘**：vllm #53826（batch-sharded 下 mask 静默丢失的回归，needs-rebase）、#58231（split top-p 在超大 logits 下每个 token 都 mask）、#57442（logprob 快路径，17 条评论但缺 GPU 复测）。

## 9. 检索口径（与上次保持一致，便于对比）

- **关键词**：`"sampling mask"` / `sampling_mask` / `"sample mask"`，限定 `is:pr` 与 `vllm-project/vllm`、`sgl-project/sglang` 两个官方仓库，按 `updated` 倒序取前 100。
- **本次命中**：sglang 122 / 3 / 104，vllm 79 / 10 / 106；去重后 **254 条 PR**，其中 **216 条不在上次归档的 38 条记录内**（含大量 grammar mask / vocab mask / logit mask 等无关命中，以及上次打分低于阈值的边缘项）。命中数超过 100 的查询已按更新时间截断，但因为按 `updated` 倒序，**最新条目一定在结果内**，不影响增量判定。
- **增量判定**：以上次归档的 38 条记录为基线，逐条比对 `state` / `merged_at` / `comments` / `updated_at`；再在 254 条中人工筛出 2026-09-29 之后有实质更新、且与 sampling mask 本体相关的新增条目（§4、§5），以及虽然更早但属同一代码路径的补漏项（§6）。
- **未做的事**：未逐条拉 review 线程与 CI 结果；未验证 PR 描述中的性能/精度数字；未对 `.gh/raw/all_prs.json`（上次归档的原始检索结果）做逐条复算，因此 §6 的「补漏」是基于本次标题/正文判读，可能仍有少量未覆盖条目。
