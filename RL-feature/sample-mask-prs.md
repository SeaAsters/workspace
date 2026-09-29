# sample mask 相关 PR 归档（vLLM / SGLang）

> 归档日期：2026-09-29　|　数据来源：GitHub Search API（官方仓库 vllm-project/vllm、sgl-project/sglang）
>
> 归档产物：本文件（索引 + 核心表）、`sample-mask-prs.json`（结构化数据）、`sample-mask-prs.csv`（表格数据）

## 1. 结论速览

- **两个仓库都有 sample/sampling mask 相关 PR，且都在近期（2026-07 ~ 2026-09）形成密集迭代。**
- **SGLang 是主战场**：高相关 PR 21 个（已合并 13 个），从最初「返回 top-p/top-k mask」一路演进到「per-request 数组流式返回 / PD 分离回放 / overlap scheduling / NPU 适配」。
- **vLLM 侧随后跟进**：高相关 PR 9 个（已合并 4 个），重点是 Model Runner V2（MRV2）的 mask 保留与 GPU 端压缩、gRPC/HTTP 对外暴露、以及 Streaming（DELTA/TITO）下的 mask 保留问题。
- **RL 场景是最强驱动力**：SGLang 的 #33593「[RL] Expose top-p-only sampling masks」、#34201「[RL, Spec] top-p mask capture」直接以 RL rollout 为需求背景；mask 的用途是让 RL 训练侧拿到「该步实际可采样 token 集合」以重算/对齐 logprob。
- **跨仓库共性与差异**：两边都围绕 top-p/top-k 采样产生的 mask 做「捕获 → 传输 → 回放」三件事；差异在于 SGLang 走 per-request 数组 + PD 分离链路，vLLM 走 MRV2 bitmask 压缩 + 多前端（HTTP/gRPC/Rust）暴露。

## 2. 主题分类（高相关 PR）

| 主题 | PR 分布 | 说明 |
|---|---|---|
| 核心能力：暴露/返回 sampling mask | sglang #27408 #33593；vllm #50520 #56777 | 把采样时的 top-p/top-k 有效 token 集合作为结果返回给调用方 |
| RL / Spec 场景扩展 | sglang #34201 #32108 #38279；vllm #54166 | 面向 RL rollout 与投机解码的 mask 捕获与回放 |
| 传输与接口形态 | sglang #40986 #41270 #41205；vllm #56777 #50520 | per-request 数组、numpy 行写入、gRPC/HTTP 暴露 |
| PD 分离 / 调度兼容 | sglang #41235 #41271 #36629 #36631；vllm #53826 | prefill-decode 分离、overlap scheduling、batch-sharded 下的 mask 保留 |
| 性能优化 | sglang #36518 #36540；vllm #54901 | 把 mask 物化/同步移出 decode 热路径、GPU 端压缩 bitmask |
| 流式输出兼容 | vllm #55935 #57442 | DELTA/TITO 流式返回下不丢 mask |
| 测试 / CI / 边界 | sglang #32410 #37873 #35205 #34037；vllm #55315 #55852 | top-k 边界 tie、mask 长度 = top_k+1、XPU 跳过等 |

## 3. vLLM（vllm-project/vllm）

高相关 PR 共 9 个：已合并 4、待处理 5。

### 3.1 已合并

| PR | 标题 | 状态 | 创建 | 合并 | 作者 | 评论 | 链接 |
|---|---|---|---|---|---|---|---|
| #54901 | [Perf][Model Runner V2] Compact sampling masks on GPU instead of unpacking the full-vocab bitmask on CPU | merged | 2026-09-02 | 2026-09-04 | @aoshen02 | 12 | [link](https://github.com/vllm-project/vllm/pull/54901) |
| #55315 | [Test] Split test_sampling_mask_preserves_top_k_boundary_ties to remove Triton-kernel-specific assumption Description | merged | 2026-09-04 | 2026-09-04 | @mayuyuace | 3 | [link](https://github.com/vllm-project/vllm/pull/55315) |
| #55852 | [CI][XPU] skip test_sampling_mask_tensors_match_finite_support on XPU | merged | 2026-09-08 | 2026-09-09 | @mayuyuace | 4 | [link](https://github.com/vllm-project/vllm/pull/55852) |
| #56777 | [Rust Frontend] Return sampling masks over gRPC | merged | 2026-09-14 | 2026-09-18 | @biswapanda | 8 | [link](https://github.com/vllm-project/vllm/pull/56777) |

### 3.2 未合并 / 进行中

| PR | 标题 | 状态 | 创建 | 合并 | 作者 | 评论 | 链接 |
|---|---|---|---|---|---|---|---|
| #50520 | [Feature] Expose sampling mask via HTTP endpoints | closed | 2026-07-31 |  | @aoshen02 | 0 | [link](https://github.com/vllm-project/vllm/pull/50520) |
| #53826 | [Bugfix][Model Runner V2] Preserve sampling masks in batch-sharded sampling | open | 2026-08-26 |  | @waizuichougou | 4 | [link](https://github.com/vllm-project/vllm/pull/53826) |
| #54166 | [Feature][Spec Decode] Support sampling mask replay for MRV2 MTP | closed | 2026-08-28 |  | @chengcuiping | 27 | [link](https://github.com/vllm-project/vllm/pull/54166) |
| #55935 | [Bugfix] Preserve sampling masks in DELTA and TITO streaming outputs | open | 2026-09-09 |  | @aoshen02 | 0 | [link](https://github.com/vllm-project/vllm/pull/55935) |
| #57442 | [Frontend][Core] Sampled-token logprob fast path for /inference/v1/generate (return_token_logprobs) | open | 2026-09-18 |  | @yinli-systems | 5 | [link](https://github.com/vllm-project/vllm/pull/57442) |

## 4. SGLang（sgl-project/sglang）

高相关 PR 共 21 个：已合并 13、待处理 8。

### 4.1 已合并

| PR | 标题 | 状态 | 创建 | 合并 | 作者 | 评论 | 链接 |
|---|---|---|---|---|---|---|---|
| #27408 | Return top-p/top-k sampling mask/nucleas | merged | 2026-06-06 | 2026-07-14 | @hari-harmonic | 11 | [link](https://github.com/sgl-project/sglang/pull/27408) |
| #32108 | Cherry-pick sampling mask support to sglang-miles | merged | 2026-07-23 | 2026-07-23 | @hari-harmonic | 1 | [link](https://github.com/sgl-project/sglang/pull/32108) |
| #32410 | Fix flaky test_sampling_mask: mask length can legitimately be top_k + 1 | merged | 2026-07-26 | 2026-07-26 | @merrymercy | 4 | [link](https://github.com/sgl-project/sglang/pull/32410) |
| #33593 | [RL] Expose top-p-only sampling masks | merged | 2026-08-05 | 2026-08-15 | @nanjiangwill | 3 | [link](https://github.com/sgl-project/sglang/pull/33593) |
| #35205 | [Sampling] Restore finite top-k requirement for sampling masks | merged | 2026-08-18 | 2026-08-21 | @nanjiangwill | 2 | [link](https://github.com/sgl-project/sglang/pull/35205) |
| #36630 | [Sampling] Capture masks from sampler support | merged | 2026-08-27 | 2026-09-03 | @ByronHsu | 2 | [link](https://github.com/sgl-project/sglang/pull/36630) |
| #36631 | [Sampling] Support sampling masks with overlap scheduling | merged | 2026-08-27 | 2026-09-11 | @ByronHsu | 14 | [link](https://github.com/sgl-project/sglang/pull/36631) |
| #37873 | [Test] Allow top-k cutoff ties in `test_sampling_mask_matches_topk_logprobs` | merged | 2026-09-04 | 2026-09-04 | @hnyls2002 | 0 | [link](https://github.com/sgl-project/sglang/pull/37873) |
| #38279 | [Sampling] Allow sampling-mask replay with DisallowedTokensLogitsProcessor | merged | 2026-09-07 | 2026-09-07 | @ByronHsu | 5 | [link](https://github.com/sgl-project/sglang/pull/38279) |
| #40986 | [Sampling] Stream sampling masks as per-request arrays | merged | 2026-09-24 | 2026-09-26 | @nanjiangwill | 4 | [link](https://github.com/sgl-project/sglang/pull/40986) |
| #41235 | [PD] Keep the sampling mask of a replayed rebootstrap token | merged | 2026-09-25 | 2026-09-29 | @nanjiangwill | 2 | [link](https://github.com/sgl-project/sglang/pull/41235) |
| #41270 | [cherrypick from #40986] [Sampling] Stream sampling masks as per-request arrays | merged | 2026-09-26 | 2026-09-26 | @nanjiangwill | 0 | [link](https://github.com/sgl-project/sglang/pull/41270) |
| #41271 | [cherrypick from #41235] [PD] Keep the sampling mask of a replayed rebootstrap token | merged | 2026-09-26 | 2026-09-26 | @nanjiangwill | 0 | [link](https://github.com/sgl-project/sglang/pull/41271) |

### 4.2 未合并 / 进行中

| PR | 标题 | 状态 | 创建 | 合并 | 作者 | 评论 | 链接 |
|---|---|---|---|---|---|---|---|
| #34037 | [Sampling] Add explicit bounded sampling-mask mode | closed | 2026-08-08 |  | @nanjiangwill | 1 | [link](https://github.com/sgl-project/sglang/pull/34037) |
| #34201 | [RL, Spec] Introduce top-p mask capture for spec and add DFlash/DSpark impl | open | 2026-08-10 |  | @jvmncs | 12 | [link](https://github.com/sgl-project/sglang/pull/34201) |
| #35966 | [Sampling] Capture masks from sampler support | closed | 2026-08-22 |  | @ByronHsu | 1 | [link](https://github.com/sgl-project/sglang/pull/35966) |
| #36518 | [Sampling] Defer sampling-mask materialization off the decode hot path | closed | 2026-08-26 |  | @joev-harmonic | 1 | [link](https://github.com/sgl-project/sglang/pull/36518) |
| #36540 | Avoid sampling-mask synchronization in decode | closed | 2026-08-27 |  | @joev-harmonic | 1 | [link](https://github.com/sgl-project/sglang/pull/36540) |
| #36629 | [Sampling] Support sampling masks with overlap scheduling | closed | 2026-08-27 |  | @ByronHsu | 1 | [link](https://github.com/sgl-project/sglang/pull/36629) |
| #39934 | Feature/npu sampling mask max tokens | open | 2026-09-17 |  | @wl55721 | 0 | [link](https://github.com/sgl-project/sglang/pull/39934) |
| #41205 | [Sampling] Write /generate sampling masks from numpy rows | open | 2026-09-25 |  | @nanjiangwill | 0 | [link](https://github.com/sgl-project/sglang/pull/41205) |

## 5. 关键 PR 说明

### SGLang

- **#27408 Return top-p/top-k sampling mask/nucleas**（已合并 2026-07-13）：最早的落地 PR，让服务端把 top-p/top-k 采样对应的 mask（nucleus 集合）返回，是后续所有 mask 工作的基础。
- **#33593 [RL] Expose top-p-only sampling masks**（已合并 2026-08-14）：明确以 RL 为动机，只暴露 top-p 维度产生的 mask。
- **#34201 [RL, Spec] Introduce top-p mask capture for spec and add DFlash/DSpark impl**（open）：把 top-p mask 捕获扩展到投机解码路径（DFlash/DSpark），与 RL + Spec 结合。
- **#36630 / #36631 Capture masks from sampler support + overlap scheduling**（已合并 2026-09）：让 mask 捕获适配 overlap scheduling 调度模式。
- **#40986 Stream sampling masks as per-request arrays**（已合并 2026-09-25）：把 mask 从整批张量改为按请求的数组流式返回，是接口形态的关键重构；#41270 为其 cherry-pick。
- **#41235 [PD] Keep the sampling mask of a replayed rebootstrap token**（已合并 2026-09-28）：PD 分离下重放 rebootstrap token 时保留采样 mask；#41271 为其 cherry-pick。
- **#32108 Cherry-pick sampling mask support to sglang-miles**（已合并 2026-07-22）：把 mask 能力回移到 sglang-miles 分支，说明该能力被下游产品线采用。
- **#38279、#34037、#35205、#32410、#37873**：语义收敛与测试稳定性（mask 回放与 DisallowedTokensLogitsProcessor 共存、bounded mask 模式、top-k 必须有限、mask 长度可为 top_k+1、top-k 截断 tie）。

### vLLM

- **#50520 [Feature] Expose sampling mask via HTTP endpoints**（已关闭，未合并）：最早尝试把 sampling mask 通过 HTTP 接口暴露，虽未合并但确立了接口方向。
- **#54901 [Perf][Model Runner V2] Compact sampling masks on GPU**（已合并 2026-09-04）：针对 MRV2，把 full-vocab bitmask 的 CPU 解包改为 GPU 端压缩，是性能关键改动。
- **#56777 [Rust Frontend] Return sampling masks over gRPC**（已合并 2026-09-18）：把 sampling mask 通过 gRPC 从 Rust 前端返回，是 vLLM 侧对外暴露的落地实现。
- **#54166 [Feature][Spec Decode] Support sampling mask replay for MRV2 MTP**（已关闭）：MRV2 MTP 投机解码下的 mask 回放支持。
- **#53826 [Bugfix][Model Runner V2] Preserve sampling masks in batch-sharded sampling**（open）：batch 分片采样下 mask 丢失的修复。
- **#55935 [Bugfix] Preserve sampling masks in DELTA and TITO streaming outputs**（open）：流式输出（DELTA/TITO）下保留 mask。
- **#55315、#55852**：测试拆分（去除 Triton kernel 假设）与 XPU 上的 CI 跳过，反映 mask 测试在不同后端上的适配成本。

## 6. 未合并原因观察

- 多个高相关 PR 最终以「closed 未合并」收场（sglang #34037 #35966 #36518 #36540 #36629；vllm #50520 #54166），而同一主题往往随后以新 PR 重新提交并合并（例如 sglang #36629 → #36631、#35966 → #36630；vllm #50520 → #56777）。这说明该特性在 2026-08 ~ 2026-09 期间经历了明显的**方案推翻与重做**，评审对接口形态（整批张量 vs 按请求数组）和性能影响（decode 热路径同步）非常敏感。
- 未合并 PR 中有相当比例是「先把 mask 物化/同步移出 decode 热路径」的性能类尝试，最终通过 CPU 侧延迟物化、GPU 端压缩等方案解决。

## 7. 检索方法与口径

- 检索源：GitHub Search API `search/issues`，限定 `is:pr` 与官方仓库，关键词为 `"sample mask"`、`"sampling mask"`、`"sampling_mask"`（`"sample_mask"` 两个仓库均为 0 命中）。
- 各查询命中量：vllm 90 / 61 / 6，sglang 89 / 105 / 3，去重后共 225 条 PR 进入打分。
- 相关度打分：标题含 mask +6、含 sampl +3、标题同时含 sampl+mask +8；正文出现 sampling/sample mask 每处 +4（上限 +12）；涉及 RL +5、正文含 RL/rollout +3；涉及 spec/eagle/draft +3；语法/结构化输出/vocab/logit-bias 类通用 mask 扣 6。
- 归档口径：**score >= 14 为核心相关（共 30 个 PR，即上表）**；score 9~13 的 8 个边缘相关未列入上表但保留在 `sample-mask-prs.json`；低于 9 分的 187 个（多为 grammar/vocab/logit mask 等无关命中）仅保留在 `.gh/raw/all_prs.json`。
- 状态说明：`merged` 表示已合并；`open` 表示仍在讨论；`closed` 表示已关闭但未合并。

## 8. 附件

- `sample-mask-prs.json`：本表全部 PR 的结构化数据（含仓库、编号、状态、时间、作者、标签、评分与命中原因）。
- `sample-mask-prs.csv`：同上，CSV 格式便于表格工具打开。

