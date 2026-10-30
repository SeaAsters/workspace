# -*- coding: utf-8 -*-
"""Curated editorial content for the 2026-10-30 delta reports."""

# classification values: 旧PR变化 / 前期补漏续跟 / 本期新建 / 前期新增续跟
SGL_HIDDEN = [
    dict(
        num=43449, kind="pr", cls="旧PR变化", prio="P0", activity="合入 + 末轮修复",
        analysis=(
            "本期合入（Qiaolin-Yu，2026-10-26T23:18:44Z，head 25303b32b5，+942/-10，23 文件）。新增 "
            "--enable-return-aux-hidden-states 与 --return-aux-hidden-state-layer-ids：该模式下 "
            "--speculative-algorithm 只用于选择捕获约定（EAGLE3/DFLASH/DSPARK），随后被复位为 None，因此不加载 draft、"
            "不预留 draft 资源、speculative 栈保持关闭。校验要求 pp_size==1、禁止 --speculative-draft-model-path、"
            "DFLASH/DSPARK 必须显式给出层号且非负不重复；返回体的 hidden_states 改为拼接后的 aux 捕获，"
            "末层 hidden 另存 last_hidden_states 供进程内消费者，不进响应。合入前最后三个提交分别修正 aux 与最终行对齐、"
            "Inkling 捕获状态和文档层号约定。"
        ),
        npu=(
            "部分覆盖：合入同时给 NPU graph runner 增加了 last_hidden_states 切片，说明 Ascend 图路径已能传播该字段，"
            "上一期记的『NPU 缺 last_hidden_states』已随本次合入变化。仍为缺失/需实测：pp_size==1 之外的阶段切分受限，"
            "各模型 capture hook 的 Ascend 等价性、EAGLE3/DFLASH/DSPARK 三种约定在 NPU 的实际层序与数值都未验证。"
        ),
        validation=(
            "作者侧新增 test_aux_hidden_state_capture.py（394 行）、aux/最终 hidden 行对齐测试（104 行）与 Inkling 专项测试；"
            "本期未运行 GPU/NPU。NPU 验收建议：eager 与图两路径的 aux 层序与维度、last hidden 经 lm_head 得 logits、"
            "padding 裁剪、禁用路径零开销、pp>1 走拒绝分支。"
        ),
    ),
    dict(
        num=43450, kind="pr", cls="旧PR变化", prio="P0", activity="合入 + 标题/接口收敛",
        analysis=(
            "本期合入（Qiaolin-Yu，2026-10-27T20:37:13Z，head 9939f78d6f，+650/-7，13 文件）。新增 "
            "ModelRunner.forward_observer 协议：after_forward(batch, forward_batch, logits_output, *, can_run_graph) "
            "在 target forward 之后、采样清理张量之前回调，可返回 DeviceAuxiliaryOutput 交 scheduler 侧拷贝；"
            "并新增 Scheduler.defer_response(req, completion) 与 ResponseCompletion(done/result/cancel) 支持异步发布响应。"
            "文档明确边界：PD disaggregation 不支持；持有响应不影响 is_fully_idle，且只有 output rank 持有；"
            "图重放可能覆盖保留的 logits 输出视图，消费者必须自行复制。10/26 标题由 out-of-tree consumers 改为 plugins。"
        ),
        npu=(
            "缺失/需实测：复用公共 scheduler completion 生命周期，但 NPU 的 stream/event、owned buffer 与图重放语义需独立验证；"
            "不能把 graph buffer view 交给延后消费者，也不能用 GPU 侧测试推断 Ascend 跨节点路径。"
        ),
        validation=(
            "作者侧 test_forward_observer.py 与 test_deferred_output.py；NPU 验收应覆盖连续 replay、延迟发布、abort、"
            "共享 future 取消以及 PD 组合被拒绝。"
        ),
    ),
    dict(
        num=46320, kind="pr", cls="本期新建", prio="P0", activity="实质提案 + 实现",
        analysis=(
            "让 EAGLE3 在 PP>1 下可用：把 target 各阶段捕获的 aux hidden states 跨 PP relay（18 文件，+281/-73）。"
            "当前 main 的 --enable-return-aux-hidden-states 明确要求 pp_size==1，因此本条与已合入能力互补，"
            "不能把两者叠加当作『PP 下的 aux 捕获已支持』。作者称其为 #39634 的 P2（『EAGLE3/DFLASH ship aux across stages』）"
            "中的 EAGLE3 部分。"
        ),
        npu=(
            "缺失/需实测：PP relay 涉及阶段切分、send/recv 缓冲与层号约定；Ascend 需独立验证 relay 顺序、"
            "PP2/PP4 下列表长度与数值，以及是否额外占用 draft KV。"
        ),
        validation=(
            "作者报告在 Llama-3.1-8B 与 Qwen3-8B 上 PP2/PP4 自测通过；本期未复测。NPU 验收需逐阶段比对 aux 层序与数值、"
            "prefill/decode 两条路径，并确认与 #43449 的捕获约定不冲突。"
        ),
    ),
    dict(
        num=46678, kind="issue", cls="本期新建", prio="P1", activity="RFC 讨论",
        analysis=(
            "[RFC] Activation monitors in SGLang：提出在引擎内提供激活监视钩子，供外部在 forward 期间读取/拦截激活。"
            "与已合入的 forward observer 相比，差异在所有权与生命周期——RFC 若采纳必须回答张量何时拷贝、"
            "图重放缓冲区如何避免被覆盖、以及是否沿用 deferred response 的终结语义。"
        ),
        npu=(
            "缺失/需实测：NPU 上 hook 的 stream 语义与 buffer 生命周期需单独定义；RFC 文本不构成 Ascend 已支持。"
        ),
        validation="本期为设计讨论，无代码与设备验证。",
    ),
    dict(
        num=46290, kind="pr", cls="本期新建", prio="P1", activity="合入（重构）",
        analysis=(
            "把 draft 的并行放置从隐式推断改为模型声明解析（新增 configs/draft_parallel.py、draft_kv_budget.py 等，"
            "head 08a8d91dd6，+597/-158，29 文件），并重写 model_runner_components/spec_aux_hidden_state.py 的解析入口。"
            "对捕获特性的意义是：aux 捕获约定与 draft 放置解耦，draft 放置变化不再改写 target 的捕获层号；"
            "但各 family（EAGLE3/DFLASH/DSPARK）的层号语义仍各自解释。"
        ),
        npu=(
            "需实测：解析层是设备无关的，但 Ascend 若另有 draft 放置或 KV 预算实现，必须与声明式解析一致，"
            "否则会出现 aux 层号与 draft 放置错配。"
        ),
        validation="作者侧以测试替换推断分支；NPU 需在 draft 放置变化时验证捕获层号稳定与 aux 维度一致。",
    ),
    dict(
        num=47072, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "customized_info 从『每请求单值 dict + 逐提交 clone』改为带类型的 per-token / per-request 值收集"
            "（head 8254d685ce，+104/-4），明确生产者如何发布 device 张量以及读取时的拷贝语义。"
            "对 observer/aux 消费者是契约修复：捕获值不再依赖 clone 规避图 buffer 覆盖，而是由类型与拷贝规则保证。"
        ),
        npu=(
            "需实测：NPU 图重放下的 buffer 生命周期与 stream 可见性需与 CUDA 等价验证，尤其是 per-token device 值在 replay 前后不串值。"
        ),
        validation="作者侧更新 test_batch_result_processor_hidden_states.py 等；NPU 验收持续重放、abort 与取消路径。",
    ),
    dict(
        num=46176, kind="pr", cls="本期新建", prio="P2", activity="合入（性能/相邻）",
        analysis=(
            "DSv4.1 DSpark 解码路径微优化（head 05f9602021，+201/-37）：新增 mhc_stream_mean Triton 内核按 torch 顺序求四路 "
            "hc 流均值供 DSpark 捕获与 replay tail 使用；aux 捕获直接写入 AuxHiddenStatePacker 的 slot，不再 torch.cat。"
            "属捕获链路的直接相邻开销优化，不改变对外接口。"
        ),
        npu=(
            "需实测：Ascend 是否有等价内核；回退到 x.mean(dim=1) 时需确认 -0 行与 dtype 行为与 GPU 一致，"
            "否则捕获均值会与参考位级不一致。"
        ),
        validation="作者附 registered/kernels/ops/layernorm/test_mhc_stream_mean.py 与 dspark 采样测试；无 Ascend 证据。",
    ),
    dict(
        num=47047, kind="pr", cls="本期新建", prio="P2", activity="轻微更新（仅 cross-reference）",
        analysis=(
            "把 GLM-5.3-Flash 的 DFlash aux 捕获 gather 条件由枚举 deepep/mooncake/deepep_v2 改为 "
            "is_moe_input_scattered_across_dp_ranks()，覆盖 FlashInfer FP4 all-gather 与 DWDP 等所有 scattered MoE 输入路径"
            "（+2/-8，单文件）。本期只有 1 次 cross-reference，没有新提交或评审。"
        ),
        npu=(
            "需实测：若 Ascend 的 MoE 切分/通信后端不在该谓词覆盖范围内，aux 行数会与 attention 层边界行不一致。"
        ),
        validation="作者未附 NPU 证据；建议以 GLM-5.3-Flash 在两种 MoE 后端下做 aux 行对齐对照。",
    ),
    dict(
        num=45144, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（插件基础设施）",
        analysis=(
            "允许 out-of-tree 平台提供自己的 speculative worker（platform interface、spec_info、docs/plugin.mdx，head 78cbea036a）。"
            "与 #43450 的 out-of-tree 消费者方向配套，属 speculative 侧可替换点。"
        ),
        npu=(
            "需实测：Ascend 若通过该接口替换 speculative worker，需保证 aux 捕获约定与 target 层号仍由公共解析决定，"
            "而不是由插件自行解释。"
        ),
        validation="作者附 test_oot_eagle_hooks.py；本期未复测。",
    ),
    dict(
        num=45145, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（插件基础设施）",
        analysis=(
            "允许 out-of-tree 平台提供 EAGLE draft graph runner（platform interface、eagle_worker_v2、docs，head f8e4e2b2e7）。"
            "与 #45144 同一波合入，构成 draft 图侧可替换点。"
        ),
        npu=(
            "需实测：Ascend 图 runner 若由插件提供，需验证 aux hidden 输出裁剪与 last_hidden_states 传播不被绕过。"
        ),
        validation="作者附 test_oot_eagle_hooks.py；本期未复测。",
    ),
    dict(
        num=47216, kind="pr", cls="本期新建", prio="P2", activity="合入（NPU 修复）",
        analysis=(
            "NPU DSpark 图重放中 verify KV 长度被重复累计的修复（head 45c09e78ff，+5/-4，仅 npu_graph_runner.py）。"
            "与捕获特性的交集在于：NPU graph runner 现在是 aux/last hidden 输出裁剪与 DSpark 元数据的共同落点，"
            "长度记账错误会直接表现为捕获行错位或越界。"
        ),
        npu="本身就是 NPU 修复；建议在 Ascend DSpark 图重放 + aux 捕获组合下验证行数与 KV 长度一致。",
        validation="sglang-npu-bot 合入；无独立复测，建议按 min/max 图尺寸与多 batch 组合回归。",
    ),
    dict(
        num=42078, kind="pr", cls="前期补漏续跟", prio="P2", activity="关闭（superseded）",
        analysis=(
            "本期以 superseded 关闭（2026-10-30T01:50:34Z）：作者说明 main 已由 #43736 提供 bounded replay 支持，"
            "并由 #44850、#45301、#45407 补齐 tail buffer 修复。属跟踪清理，不能把关闭本身当修复。"
        ),
        npu="需实测：Ascend 若支持该模型与 bounded replay，仍需按 prompt logprob / hidden-state 读取路径单独验证。",
        validation="本期唯一事件为关闭评论。",
    ),
    dict(
        num=45925, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（模型基线，体量大）",
        analysis=(
            "MiniMax-M3.1-Flash 模型支持（+6257/-324，63 文件，创建于本期前 1 小时的 10/23T01:03Z，上期未纳入），"
            "其中包含 model_runner_components 的 aux 捕获测试更新与 speculative 相关改动。"
            "不是捕获接口变更，但改变了该模型在 aux 捕获路径上的可运行基线。"
        ),
        npu="需实测：该模型在 Ascend 的可运行性与 aux 捕获一致性需单独验证，不能由主仓合入推断。",
        validation="作者侧经多轮 CI rerun 后合入；本期未复测。",
    ),
]

SGL_SAMPLING = [
    dict(
        num=43517, kind="pr", cls="前期补漏续跟", prio="P0", activity="持续迭代 + LGTM",
        analysis=(
            "Ascend 侧 sampling mask 支持仍为 open（base main，head 22ca97cc63）。本期新增 10/24 提交 c1cc2213db"
            "（重排 Ascend mask 测试并保留参考实现）与 10/29 提交 22ca97cc63（[NPU] Unify Ascend sampler returns），"
            "10/24 两次 force-push 重写分支；10/26 floatlibai 给出 LGTM。测试已重组为 "
            "test/registered/npu/basic_function/backends/ 与 npu/rl/sampling_mask/ 两处。"
        ),
        npu=(
            "缺失：官方 main 仍在 scheduler 中以『ascend 采样后端直接从 logits 采样、不构造 top-k/top-p support』为由拒绝 "
            "return_sampling_mask，helper 也只返回 token ids。本条未合入前不能宣称 Ascend 已支持。"
        ),
        validation=(
            "作者侧为 NPU/CPU mock 验证。验收需覆盖 top_k=1025 回退、min_p、greedy、support/selected 两模式、混合 opted-in 行、"
            "seeded mask on/off 与连续图重放。"
        ),
    ),
    dict(
        num=38565, kind="pr", cls="旧PR变化", prio="P0", activity="冲突修复 + 重命名 + 新提交",
        analysis=(
            "确定性 top-p/top-k 归一化 PR 本期继续收敛：10/29 评论要求修冲突并请 claude review，10/30 两次 force-push，"
            "新增 c33aeb0b18 与 b8d4668008（[Test] sampling: Publish a runtime config），标题由 attn_tp_size>1 改为 tp_size>1，"
            "并新增 kernels/ops/sampling/renorm.py 与 tests/registered/sampling/test_sampling_mask.py 触点。"
        ),
        npu=(
            "需实测：不要移植 FlashInfer CUDA kernel 到 NPU；应在 Ascend 实际 renorm 算子上检查跨 rank 位级一致性、"
            "共享随机数与 token 决策，并与 #41916 的过滤语义一起验收。"
        ),
        validation=(
            "作者早前报告 H100/B200 上 fast 版本挂起、deterministic 版本完成；本期未复测。需限制 top_p<1，"
            "跟踪 target_logits→target_probs→accept_len/bonus→KV 提交的首个分歧。"
        ),
    ),
    dict(
        num=41916, kind="pr", cls="旧PR变化", prio="P0", activity="合 main + 后续修复提交",
        analysis=(
            "top-k/top-p 过滤一致性 PR 本期把 main 合入分支（c49d5a09，2026-10-28）并追加 3d1e9fddf3、cb72322eab"
            "（dummy tree cache / KVLayerRouter 修复），10/28 起由 guapisolo 接手评审协调，10/30 重跑 e2e 测试。"
            "语义方向仍是『保留 top-p nucleus 再与 top-k support 相交』，不是 #41955 的 top_k_first 默认切换。"
        ),
        npu=(
            "需实测：Ascend fused helper 与 torch fallback 的过滤顺序需逐分布比对；先定语义再迁移 kernel，"
            "不能据 PR open 状态宣称默认策略已改变。"
        ),
        validation=(
            "作者早前报告五路径 18 passed + 6 subtests、seeded 位级比较与 CUDA Graph 验证；本期为冲突合并与重跑，"
            "未提供新的 Ascend 数据。"
        ),
    ),
    dict(
        num=46080, kind="pr", cls="本期新建", prio="P1", activity="新建（无本期评审事件）",
        analysis=(
            "DFLASH 在 TP>1 + softcap lm_head 下，对采样批次改为在 all-gather 的 top-k support 上做 verify（opt-in，"
            "新增 kernels/ops/speculative/dflash_sparse_verify.py 与 env 开关）。"
            "本期 3 个提交均在 10/23 推送，PR 级时间线无本期评审事件，属新建未评审状态。"
        ),
        npu="需实测：稀疏 verify 只在 support 上重算概率，Ascend 需确认 support 与实际采样 support 完全一致，否则会改变 support 语义。",
        validation="作者附 test_dflash_sparse_verify.py 与 graph pool borrow 测试；NPU 需覆盖 top_k 边界与 support 一致性。",
    ),
    dict(
        num=46186, kind="pr", cls="本期新建", prio="P1", activity="新建 + 交叉引用",
        analysis=(
            "把 spec verify 的目标概率构造改为 radix top-k（touches kernels/ops/sampling/topk.py、dflash_utils.py），"
            "与 sampling-mask 捕获共用 build_speculative_verify_target_probs 路径。本期仅有 2 次 cross-reference（10/24）。"
        ),
        npu="需实测：radix top-k 的分块与归约在 Ascend 上需与 torch 版逐位或按容差比对，否则 support/mask 行会偏移。",
        validation="作者称组件 3/3，附 test_verify_target_probs.py；本期未复测。",
    ),
    dict(
        num=39934, kind="pr", cls="前期补漏续跟", prio="P2", activity="修 CI + 重命名 + 新提交",
        analysis=(
            "NPU 侧 --sampling-mask-max-tokens 的 e2e 回归测试（单文件，+266）。本期 10/27 两个 fix 提交、"
            "一次 bot 检查失败与一次重命名，仍 open。"
        ),
        npu="直接是 NPU 测试补齐：覆盖 top_k 上限触发 abort 的行为，是 #43517 合入前的验收护栏之一。",
        validation="作者侧测试未通过最新 CI；本期未复测。",
    ),
    dict(
        num=41205, kind="pr", cls="旧PR变化", prio="P2", activity="关闭（stale draft）",
        analysis=(
            "本期被 bot 以『仍是 draft 且 28 天未更新』关闭（2026-10-24T01:24:49Z），未合入。"
            "其思路（/generate 直接从 numpy 行写 sampling mask，避免逐元素 Python list 构造）仍是 mask 返回路径的性能参考。"
        ),
        npu="需实测：与 #43517 的 host 物化路径相关，但关闭状态不代表能力可用。",
        validation="关闭评论为唯一本期事件。",
    ),
    dict(
        num=42647, kind="pr", cls="旧PR变化", prio="P1", activity="合入 sglang-miles",
        analysis=(
            "radix transfer plugins 于 2026-10-23T06:17:53Z 合入 base 分支 sglang-miles（非 main），"
            "把 sampling masks / routed experts / indexer top-k 从 inline meta_info 改为共享 OutputStore 引用，"
            "selected logprobs 仍留在 meta_info。"
        ),
        npu=(
            "缺失/需实测：框架与 host 侧逻辑可复用，但 Ascend 输出需先接通 #43517，再核对 store 引用对齐、"
            "buffer 拷贝所有权与真实 NPU→host→Mooncake 路径。"
        ),
        validation="作者侧为 macOS fake Mooncake 与 CPU 测试；本期未复测，也未运行真实 Mooncake。",
    ),
    dict(
        num=43047, kind="pr", cls="旧PR变化", prio="P2", activity="无本期事件",
        analysis=(
            "[sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding 仍为 draft，base 为 sglang-miles，"
            "分支相对 main 有 218 文件（含整分支 diff）。本期时间线无实质事件，仅作为 branch 背景列出。"
        ),
        npu="不适用：目标分支不是 main，不能据此判断 main 或 Ascend 能力。",
        validation="无本期验证证据。",
    ),
]

VLLM_HIDDEN = [
    dict(
        num=63436, kind="pr", cls="本期新建", prio="P0", activity="合入（bugfix）",
        analysis=(
            "DFlash/DFlash2 drafter 在 target 给出 FP32 aux hidden states 时（Muse Spark 的残差流保持 FP32）会让 bf16 的 fc 收到 "
            "FP32 输入并启动失败；本 PR 按 draft dtype 转换 fc 输入（vllm/model_executor/models/qwen3_dflash.py，+44/-4），"
            "2026-10-26T20:58:42Z 由 mgoin 合入。"
        ),
        npu=(
            "需实测：Ascend 上若 target 残差流/aux 捕获为 FP32 而 draft 为 bf16，需要同样的 dtype 门禁；"
            "同时要确认 NPU fused fc 的 dtype 提升行为与 CUDA 一致。"
        ),
        validation="作者附 tests/v1/spec_decode/test_dflash2.py 的 FC dtype 用例（同题 CI 失败另见 #63972）；本期未复测。",
    ),
    dict(
        num=63427, kind="pr", cls="本期新建", prio="P1", activity="合入（内存/正确性）",
        analysis=(
            "CUDA Graph 捕获的嵌套输出用弱引用保存（vllm/utils/torch_utils.py，+33/-4，2026-10-27T03:04:50Z 合入）。"
            "按 PR 描述，此前每个捕获尺寸都保留 num_aux × num_tokens × hidden_size × dtype 字节的 aux hidden states"
            "（EAGLE3/DSpark/DFlash/extract_hidden_states），属捕获特性的内存泄漏修复。"
        ),
        npu=(
            "需实测：Ascend 图捕获若同样持有 aux 输出引用，需要等价弱引用或显式释放；NPU 图池复用下泄漏表现形式可能不同。"
        ),
        validation="作者附 tests/utils_/test_torch_utils.py 用例；建议在 Ascend 上按捕获尺寸数量测 RSS/NPU 显存曲线。",
    ),
    dict(
        num=64320, kind="pr", cls="本期新建", prio="P0", activity="新建 + 冲突待 rebase",
        analysis=(
            "为 Qwen3-MoE target 在 PP>1 下 relay EAGLE3 aux hidden states（qwen3_moe.py、qwen3_omni_moe_thinker.py、qwen3_vl_moe.py，+175/-13），"
            "沿用 Qwen2（#50514）与 MiniMax-M3（#57197）的模式；Qwen3.5 的对应实现是 #61180。"
            "本期 1 个提交（10/29）、1 次 force-push，mergify 报冲突待 rebase，无评审结论。"
        ),
        npu="缺失/需实测：PP relay 依赖模型内 aux 槽位与阶段层号；Ascend 需按模型逐项验证 relay 顺序与数值。",
        validation="作者附 tests/v1/e2e/spec_decode/eagle/test_eagle3_pp.py 与 test_eagle3_aux_hidden_states_pp.py；本期未复测。",
    ),
    dict(
        num=61180, kind="pr", cls="前期补漏续跟", prio="P1", activity="评审推进 + 冲突待 rebase",
        analysis=(
            "让 Qwen3.5 在 PP 下 relay 有序的 EAGLE3 与 DFlash aux hidden states（qwen3_5.py、qwen3_next.py，head d6ade47102），"
            "通过 EagleModelMixin._set_aux_hidden_state_layers() 按阶段配置槽位，修复 #60765。"
            "本期无新提交；10/29 作者请求评审并说明带 CPU 单测，mergify 同时报冲突待 rebase。"
        ),
        npu="缺失/需实测：同为模型内 relay 机制，Ascend 需验证阶段槽位、层序与 dtype。",
        validation="作者附 tests/v1/worker/test_eagle3_aux_hidden_states_pp.py；本期未复测。",
    ),
    dict(
        num=57917, kind="pr", cls="前期补漏续跟", prio="P1", activity="冲突待 rebase",
        analysis=(
            "把 relay_aux_hidden_states 机制接到 DeepSeek-V3.2 backbone（vllm/models/deepseek_v32/nvidia/model.py，+16/-5），"
            "使 GLM-5.2 DSpark 可在 PP 下运行；依赖 #50514 引入的 supports_aux_hidden_states_over_pp 约定。"
            "本期唯一事件是 10/29 mergify 的冲突提示。"
        ),
        npu="缺失/需实测：DeepSeek 系模型在 Ascend 上的 PP 组合与 aux relay 需独立验证。",
        validation="本期无新增设备证据。",
    ),
    dict(
        num=60477, kind="pr", cls="前期补漏续跟", prio="P1", activity="合入（正确性）",
        analysis=(
            "DSA（DeepSeek-V3.2/Dots3）在 PP 下的中间张量只应发送完整残差流：先对 partial hidden_states 做 TP all-reduce，"
            "再把 IntermediateTensors 的键从 [\"hidden_states\",\"residual\"] 收敛为 [\"hidden_states\"]，"
            "并修正 aux_hidden_states 的 hidden_states + residual 拼接（2026-10-29T05:14:24Z 合入）。"
            "属 aux hidden 跨阶段还原的正确性修复。"
        ),
        npu="需实测：Ascend 的 TP all-reduce 与 PP send 语义需等价验证，尤其是 partial hidden 未归约就发送的场景。",
        validation="作者附 test_sequence_parallel.py；本期未复测。",
    ),
    dict(
        num=64458, kind="pr", cls="本期新建", prio="P1", activity="合入（MRV2 PP）",
        analysis=(
            "MRV2 下规范 PP 的 draft / sampled-token 广播，并把 EAGLE3 aux-hidden-state 的 PP 支持检查移入 "
            "set_eagle3_aux_hidden_state_layers（pp_utils.py、model_runner.py、eagle3_utils.py，2026-10-29T16:41:39Z 合入）。"
            "对捕获特性是能力判定收敛：不支持 PP 的模型在设置阶段就失败，而不是运行期静默错位。"
        ),
        npu="需实测：Ascend MRV2 路径若存在，需验证同一判定与广播顺序。",
        validation="作者附 tests/v1/worker/test_gpu_model_runner_v2.py 更新；本期未复测。",
    ),
    dict(
        num=61699, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（配置）",
        analysis=(
            "standalone draft 加载配置未被遵守的问题修复，涉及 vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py "
            "与 standalone_ar/speculator.py（2026-10-28T22:13:50Z 合入）；测试用到 "
            "tests/v1/worker/test_gpu_extract_hidden_states_speculator.py。"
        ),
        npu="需实测：extract_hidden_states 方法在 Ascend 的实现路径需与 standalone draft 配置保持一致。",
        validation="作者附 gpu_model_runner_v2 与 extract_hidden_states speculator 测试；本期未复测。",
    ),
    dict(
        num=57736, kind="pr", cls="前期补漏续跟", prio="P0", activity="本期实质实现推进（6 个新提交）",
        analysis=(
            "为 rollout 训练增加响应窗口 hidden-state 捕获：调用方给出 [start,end) 响应区间，仅返回该区间行，"
            "并可对齐 raw top-k logprobs；窗口在生成过程中经 AuxOutput Connector 收集，以减少 GPU→CPU 传输，"
            "同时支持最终 hidden states 与配置的投机辅助输出（对应上游需求 #56916）。"
            "本期新增 6 个提交（10/28–10/29），落地 `vllm/distributed/aux_output_connector/`（connector.py、hidden_states.py、worker.py）、"
            "`vllm/config/aux_output.py`、`docs/features/speculative_decoding/windowed_hidden_states.md` 与专项 benchmark，"
            "并处理了 preemption 已交付行、直接/流式准入与请求校验。10/29 两次 force-push，mergify 报冲突待 rebase。"
        ),
        npu=(
            "缺失：vLLM 侧仍无公共 return_hidden_states 响应字段，本 PR 也未合入；Ascend 需在接口落地后再评估 "
            "AuxOutput Connector 与 D2H 路径，不能把 CUDA 上的传输量对比当作 NPU 结论。"
        ),
        validation="作者侧新增 tests/distributed/aux_output_connector/test_hidden_states.py 与 tests/v1/worker/test_aux_hidden_states.py、以及 benchmark；本期未复测。",
    ),
    dict(
        num=56822, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进",
        analysis=(
            "hybrid Mamba/MLA 模型上 extract_hidden_states 的 KV-cache 准入按最宽 peer group 计费（1 层 hidden-state group 被按 24 层 MLA "
            "宽度收费，约 5.4 GiB 被多算），本条改为按自身 group 宽度计费。属 RL 抽特征场景的显存预算正确性。"
        ),
        npu="需实测：Ascend KV 池分组与计费逻辑不同，需按实际 group 配置复算。",
        validation="作者附准入测试；本期未复测。",
    ),
    dict(
        num=50187, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进",
        analysis=(
            "修正 EAGLE3 aux-hidden-state 层方法 override 的拼写错误——拼错会导致 override 静默失效、捕获层回落默认值。"
            "属易被忽略的静默错误类型。"
        ),
        npu="需实测：任何依赖同名 override 的 Ascend 模型实现都会受影响，需检查是否有同类拼写。",
        validation="本期无新增设备证据。",
    ),
    dict(
        num=63406, kind="pr", cls="本期新建", prio="P2", activity="关联更新",
        analysis=(
            "把 prefill（token）与 decode（request）的 CUDA Graph 捕获尺寸分开；PR 描述与 #63427 的内存分解互相引用，"
            "列出『每个 PIECEWISE 尺寸保留 aux hidden states』的占用。属捕获内存治理的相邻改动。"
        ),
        npu="需实测：Ascend 图捕获尺寸策略不同，需按实际图池评估 aux 占用。",
        validation="本期未复测。",
    ),
    dict(
        num=59542, kind="issue", cls="前期补漏续跟", prio="P0", activity="RFC 讨论升温（3 条本期评论）",
        analysis=(
            "[RFC] Opt-in return of the last hidden state at sampled positions, per request (Model Runner V2)："
            "让带小头的部署在一次 forward 里同时拿到 token 分数与其对应 hidden state，避免再起一个 pooling 引擎或二次前向。"
            "本期讨论明显升温：10/26 有用户给出 RAG 证据门控 + serving 插件的实际用法，10/29 有维护者汇总『三个独立团队" 
            "遇到同一问题』并按 popularity 口径推进。这是 vLLM 侧最接近『RL/小头需要最终 hidden state』的接口提案，"
            "但本期仍无实现合入，也没有公共响应字段。"
        ),
        npu="缺失：无实现即无 Ascend 路径；若采纳需同时定义 NPU 的 D2H 布局、采样位置对齐与图重放下的 buffer 语义。",
        validation="设计讨论与用户场景；本期无代码与设备验证。",
    ),
    dict(
        num=48743, kind="issue", cls="前期补漏续跟", prio="P1", activity="RFC 讨论",
        analysis=(
            "[RFC] Return extracted hidden states in the generation response：主张把 extract_hidden_states 的产物通过生成响应返回，"
            "而不是仅用于 draft。与 #59542 的差别在『响应体契约』与『按请求 opt-in』的范围。"
        ),
        npu="缺失：需先有实现；Ascend 侧还要确认响应序列化与张量裁剪。",
        validation="设计讨论；本期无代码。",
    ),
    dict(
        num=36998, kind="issue", cls="前期补漏续跟", prio="P2", activity="仅 stale 标记（90 天无活动）",
        analysis=(
            "[RFC] Observation Plugin for Intercepting & Routing on Activations：提出可注册插件在 forward 中拦截激活。"
            "对应实现 PR #37002（Phase 1）同样在 10/29 被 stale 标记，本期无技术讨论，处于即将自动关闭的状态。"
            "与 SGLang 已合入的 forward observer 属同类需求的两条不同工程路径，不能把 RFC 存在当作能力可用。"
        ),
        npu="缺失/需实测：插件在 Ascend 上的 stream 语义与 buffer 所有权未定义。",
        validation="本期无技术评论与设备验证。",
    ),
    dict(
        num=37002, kind="pr", cls="前期补漏续跟", prio="P2", activity="仅 stale 标记（90 天无活动）",
        analysis=(
            "Observation Plugin Phase 1 实现（17 文件，+1257，含 docs/design/observation_plugins.md 与 e2e 测试），"
            "最后一次提交为 2026-04-26，本期仅被 stale bot 标记。"
        ),
        npu="缺失：无 Ascend 实现。",
        validation="本期无新增证据。",
    ),
    dict(
        num=61539, kind="issue", cls="前期补漏续跟", prio="P2", activity="RFC 讨论",
        analysis=(
            "[RFC] Explicit row layouts for routed-expert capture：routed-expert 捕获目前按张量行数与 dp_metadata 反推行归属，"
            "对 out-of-tree MoE 后端不成立，提议显式声明行布局。与 hidden-state 捕获同属『捕获所有权/布局』问题。"
        ),
        npu="需实测：Ascend MoE 后端若不在已知布局内，需要显式布局声明，否则捕获行错位。",
        validation="设计讨论；本期无代码。",
    ),
    dict(
        num=63818, kind="issue", cls="本期新建", prio="P2", activity="RFC + 实测跟进",
        analysis=(
            "[RFC] Pluggable pipeline-parallel transport, and a path to out-of-process (elastic) PP stages：把 PP 阶段做成独立生命周期，"
            "并允许替换传输实现。作者在 10/27 给出 4×A40（PCIe）+ 100Gb IB 上的实测与故障切换结果，配套草案 PR #63897/#63898/#63899。"
            "对 aux hidden 跨 PP relay 是基础设施层改造：一旦落地，relay 的缓冲、失败与重启语义都要重新定义。"
        ),
        npu="需实测：Ascend HCCL 传输不在该 RFC 的默认假设内，跨进程 PP 与故障重启需要独立验证。",
        validation="作者侧为 GPU + IB 实测；本期无 NPU 数据。",
    ),
    dict(
        num=63157, kind="issue", cls="前期补漏续跟", prio="P1", activity="Issue 讨论（维护者回应）",
        analysis=(
            "[Spec Decode][DeepSeek-V4.1] Support DSpark with pipeline parallelism：模型能力检查 "
            "verify_supports_aux_hidden_states_over_pp 在 worker 初始化阶段直接拒绝，权重可加载但引擎起不来。"
            "10/23 维护者回复认为 PP + DSpark 值得支持（尤其是 PD 分离下用 PP+DSpark 做 prefill），"
            "但本期没有修复合入。这是 vLLM 侧 PP × aux 捕获缺口最常见的用户可见形态。"
        ),
        npu="缺失/需实测：Ascend 上同一检查会给出相同拒绝；是否支持取决于模型是否实现 relay。",
        validation="用户环境报告 + 维护者表态；本期无修复合入。",
    ),
]
