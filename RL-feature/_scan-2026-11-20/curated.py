# -*- coding: utf-8 -*-
"""Curated editorial content for the 2026-11-20 delta reports (sglang + vLLM)."""

SGL_HIDDEN = [
    dict(
        num=51475, kind="pr", cls="本期新建", prio="P0", activity="新建（修公共 API）",
        analysis=(
            "[Fix] api: Fix `return_hidden_states` on `/generate`（Fixes #50879、#51470）。一次修三个问题："
            "① `\"last\"` 自 v0.5.22 起返回 400——api.v1 schema 把该字段声明为 `BoolOrList`，契约检查因此拒绝字符串；"
            "② 响应形状不对；③ 单 prompt 时返回该是 list 而不是裸张量。这是**公共接口层**的回归修复，直接决定 RL 侧还能不能按文档取到 last hidden。"
        ),
        npu="需实测：Ascend 走同一 `/generate` 与 tokenizer 契约检查，修复后需要复核 `\"last\"` 在 NPU 上返回的形状与 dtype 一致。",
        validation="作者附 32 文件改动与 13 个提交；本期未复测。",
    ),
    dict(
        num=50879, kind="issue", cls="本期新建", prio="P0", activity="新建 bug 报告",
        analysis=(
            "[Bug] `/generate` 自 **0.5.22** 起拒绝 `return_hidden_states: \"last\"`（0.5.21 可用）。这是版本级回归的明确证据："
            "同一个字段在 0.5.21 与 0.5.22 之间从「接受字符串」变成「只接受 bool/list」。对下游是静默破坏——调用方不改代码就 400。"
        ),
        npu="缺失：版本回归同样影响 Ascend 用户（同一 schema）；升级到 0.5.22+ 后需要该修复才能用 `\"last\"`。",
        validation="复现报告；修复见 #51475。",
    ),
    dict(
        num=51566, kind="pr", cls="本期新建", prio="P1", activity="新建（图内 buffer 所有权）",
        analysis=(
            "[Fix] cuda-graph: Pack prefill aux hidden states into one runner-owned buffer。把 prefill 图捕获的 aux hidden states 收进单个 runner 持有的缓冲，"
            "而不是每次捕获都新申请/保留。这直接影响两件事：图捕获尺寸增加时的 aux 常驻占用（与上游 vLLM #63427/#63406 同类问题），以及 aux 视图的生命周期。"
        ),
        npu="需实测：Ascend 图路径的 aux 缓冲同样需要「runner 持有 + 按 token 裁剪」；否则图重捕获会累积常驻。",
        validation="作者侧图捕获测试；本期未复测。",
    ),
    dict(
        num=50555, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Fix] rust-server: Refuse return_hidden_states above the server's mode（2026-11-16T09:43-08:00 合入，10 文件）。"
            "请求的 hidden 模式超过服务端上限时直接拒绝，而不是按较低模式静默返回。属于「宁可报错也不要悄悄降级」的契约收紧，"
            "与 tokenizer 侧已有的模式检查一致。"
        ),
        npu="需实测：Ascend 若走 rust-server 或同一模式检查，需确认上层拒绝语义一致（返回 400 而非空 hidden）。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=50039, kind="pr", cls="前期补漏续跟", prio="P1", activity="合入（窗口起点）",
        analysis=(
            "[Fix] deepseek-v4.1: Preserve compact DSpark outputs with CP bounded replay（2026-11-13T23:28+08:00 合入，2 文件）。"
            "上期跟踪的「bounded replay 覆盖 compact 输出」已修：Prefill CP + DSpark + decoder SWA bounded replay 组合下保证 compact 输出（含 DSpark 需要的 hidden 片段）不被后续步骤覆盖。"
        ),
        npu="需实测：Ascend 若支持该组合，需同样的输出保留语义。",
        validation="合入提交自带用例；本期未复测。",
    ),
    dict(
        num=47047, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进（11-18 更新）",
        analysis=(
            "[Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE all-to-all backend。本期新增 2 个提交（更新到 11-18），"
            "仍 open。gather 条件改为 `is_moe_input_scattered_across_dp_ranks()`，覆盖所有 scattered MoE 输入路径。"
        ),
        npu="需实测：Ascend 的 MoE 切分必须在谓词覆盖内，否则 aux 行数与 attention 边界行不一致。",
        validation="本期未复测。",
    ),
    dict(
        num=48188, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-14 更新）",
        analysis=(
            "[Fix] spec: Keep chain MTP prefill hidden states under DP attention。DP attention 下多层 EAGLE + chain MTP draft 在第一步后用全零 hidden 构造 draft KV；"
            "本期更新到 11-14，仍 open。"
        ),
        npu="需实测：Ascend 的 DP attention + chain MTP 是否命中同一分支；外部表征是接受率异常。",
        validation="本期未复测。",
    ),
]

SGL_SAMPLING = [
    dict(
        num=50859, kind="pr", cls="本期新建", prio="P0", activity="新建（abort 路径 mask）",
        analysis=(
            "[Fix] tokenizer: Return sampling masks on requests aborted from the waiting queue。当要求 `return_sampling_mask` 的请求从等待队列被 abort（retracted 后排队、"
            "或直接取消）时，响应现在仍带上它**已生成 token** 的 mask（按请求选定的嵌套或扁平格式）。"
            "这是 mask 语义的边界补齐：abort 路径此前不返回 mask，RL 侧会看到「请求失败但有部分输出」的不一致。"
        ),
        npu="需实测：Ascend 的 abort/retract 路径（含图重放下的 buffer 回收）需要通过同一验证，否则 NPU 上会出现「有 token 无 mask」。",
        validation="作者侧 6 个提交、38 条时间线事件；本期未复测。",
    ),
    dict(
        num=50126, kind="pr", cls="前期补漏续跟", prio="P1", activity="本期合入",
        analysis=(
            "[Perf] tokenizer: Return output sampling masks as flat arrays on request（2026-11-17T11:45-08:00 合入，10 文件）。"
            "上期新建的扁平数组返回正式落地：mask 不再按 token 嵌套 JSON，而是按请求返回扁平数组，显著降低大 top-k 下的序列化/反序列化成本。"
        ),
        npu="需实测：Ascend 走同一 tokenizer 返回路径，扁平化后 mask 行与 token 的对齐关系必须不变。",
        validation="合入提交自带基准与测试；本期未复测。",
    ),
    dict(
        num=50155, kind="pr", cls="前期补漏续跟", prio="P2", activity="关闭（未合入 main）",
        analysis=(
            "[Feat] sampling: Return sampling logprobs without the sampling mask 本期以 `closed`（未合入）结束，但被 cherry-pick 到发布分支（#51377，同样关闭）。"
            "也就是说「只取采样 logprob、不返回整张 mask」这条路径**没有**进入 main；RL 侧仍需按 mask 方案取概率，或等待新提案。"
        ),
        npu="不适用：无 main 实现，不能作为 Ascend 的替代路径。",
        validation="关闭态；无实现。",
    ),
    dict(
        num=51378, kind="pr", cls="本期新建", prio="P2", activity="新建（发布分支 cherry-pick）",
        analysis=(
            "[cherrypick from #50859] 把 abort 路径返回 mask 的修复搬到发布分支（open）。同期 #51376（#50126 的 cherry-pick）已于 11-18 合入发布分支。"
            "这说明维护者把 mask 的两条修复当作需要回移到稳定版的能力，而不是仅 main 的实验。"
        ),
        npu="不适用：发布分支条目，只作版本可用性参考。",
        validation="cherry-pick 由作者提交；本期未复测。",
    ),
]

VLLM_HIDDEN = [
    dict(
        num=68813, kind="issue", cls="本期新建", prio="P0", activity="新建 bug 报告",
        analysis=(
            "[Bug] DeepSeek-V3.2 / GLM-5.x **丢掉最后一层 aux hidden state**，导致 `extract_hidden_states` 返回的层数比请求少一个。"
            "报告环境：vLLM 0.33.0、8×H200、TP=8、target GLM-5.2（`GlmMoeDsaForCausalLM`，78 层）。"
            "这直接破坏 EAGLE-3/DFlash 的训练数据生成——少一层意味着 draft 输入维度与 checkpoint 约定不符。"
        ),
        npu="缺失/需实测：Ascend 的 DeepSeek-V3.x/GLM 实现若同一循环边界，也会少一层；必须用「返回层数 == 请求层数」断言验证。",
        validation="用户复现；修复见 #68824。",
    ),
    dict(
        num=68824, kind="pr", cls="本期新建", prio="P0", activity="新建（根因修复）",
        analysis=(
            "[Bugfix] Capture final-layer aux hidden state in DeepSeek-V3.2。根因：`DeepseekV32Model.forward` 只在 decoder layer 循环内 `idx in [start_layer, end_layer)` 记录 aux，"
            "当 `end_layer == num_hidden_layers` 时最后一层落在循环外，因此永不捕获。修复把最终层显式纳入。"
            "这是本期 vLLM 侧最关键的正确性修复：属于「静默少一层」而不是崩溃。"
        ),
        npu="需逐模型核对：Ascend 的 V4/V4.1 与 GLM 实现是否也只在循环内捕获；这是最容易在 NPU 上复现的一类缺口。",
        validation="作者侧单测（3 文件）；本期未复测。",
    ),
    dict(
        num=68769, kind="pr", cls="本期新建", prio="P1", activity="新建（约定与解析）",
        analysis=(
            "[Bugfix][SpecDecode] 读取嵌套的 `eagle_aux_hidden_state_layer_ids` 并写明 aux 层号约定：参考实现转换出的 EAGLE-3 checkpoint 把层号放在 "
            "`eagle_config['eagle_aux_hidden_state_layer_ids']`，而 `get_eagle3_aux_layers_from_config()` 只查 drafter 风格的 `layer_ids` 与顶层属性，"
            "于是这类 checkpoint 的层号被静默忽略。"
        ),
        npu="需实测：Ascend 侧解析同一配置，嵌套键与零基/一基约定必须一致（参考上期 Ascend 的早抓一层问题）。",
        validation="作者侧 3 个提交；本期未复测。",
    ),
    dict(
        num=67648, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Bugfix][Transformers] Preserve received hidden states on later pipeline stages（2026-11-20T00:38:48Z 合入，4 文件）。"
            "Transformers 后端在后续 PP 阶段不再重复应用 decoder 对输入的预处理（learned position embeddings、embedding scaling），"
            "修掉 GPTBigCode 的 PP 缺陷（#66290）。属「PP 阶段间 hidden 语义一致」的正确性修复。"
        ),
        npu="需实测：Ascend 若使用 Transformers 后端 PP，需要同样的「只做一次预处理」语义。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=66323, kind="pr", cls="前期补漏续跟", prio="P1", activity="本期合入",
        analysis=(
            "[Model] 为 Lfm3ForCausalLM 增加 EAGLE-3 / DSpark aux hidden-state 支持（2026-11-16T03:56:51Z 合入，单文件）。"
            "上期新建、本期合入；LFM3 现在可作为 EAGLE-3/DFlash/DSpark 与 `extract_hidden_states` 的 target。"
        ),
        npu="缺失：Ascend 的 LFM3 支持与 aux 接口需单独接。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=68289, kind="pr", cls="本期新建", prio="P2", activity="新建（模型扩展）",
        analysis=(
            "[Model] 为 EXAONE 4 与 DeciLM 增加 EAGLE-3 aux hidden states 支持。延续「逐模型补 target 侧 aux 接口」的节奏，"
            "目前 vLLM 的 aux-over-PP/aux-capture 仍是模型逐个声明，不是全局能力。"
        ),
        npu="缺失：Ascend 上这两个模型无 aux 支持。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=66785, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-16 更新）",
        analysis=(
            "extract_hidden_states 的 `hs_store_device=cpu`（CPU pages + token-hash 前缀回填）本期更新到 11-16，仍 open。"
            "Ascend 侧对应实现 #21792 也仍未合入（更新到 11-17）。"
        ),
        npu="需实测：Ascend 的 D2H 带宽与分页语义、请求结束回收，都还没有设备数据。",
        validation="本期未复测。",
    ),
    dict(
        num=65991, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进（11-19 更新）",
        analysis=(
            "[Spec Decode] `disable_extract_hidden_states_decode_drafts=True`（K=0）本期更新到 11-19，仍 open。"
            "上期由 #65250 的「37%–39% 吞吐损失」驱动，是目前唯一能关掉提取方法 decode 开销的开关。"
        ),
        npu="需实测：Ascend 使用同一调度；K=0 时 aux 仍需按请求返回。",
        validation="本期未复测。",
    ),
    dict(
        num=68634, kind="pr", cls="本期新建", prio="P2", activity="新建（mask 相邻）",
        analysis=(
            "[Feature][RL] Prototype independent sampling-mask opt-out：允许请求**独立**选择不返回 mask（与 logprob 解耦），"
            "避免为了 logprob 被迫承担 mask 打包成本。属采样 mask 主题，仅在此相邻提及。"
        ),
        npu="需实测：Ascend 的 mask 打包路径也要支持该 opt-out，否则 NPU 上仍被迫打包。",
        validation="作者侧原型；本期未复测。",
    ),
]
