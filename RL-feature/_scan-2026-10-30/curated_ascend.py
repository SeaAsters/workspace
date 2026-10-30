# -*- coding: utf-8 -*-
"""Curated editorial content for the vllm-ascend half of the 2026-10-30 joint report."""

VLLM_ASCEND = [
    dict(
        num=20075, kind="pr", cls="本期新建", prio="P0", activity="新建（draft）+ 冲突待解",
        analysis=(
            "[Feature][MRV2] Preserve SP token shards across pipeline stages（draft，head 8850a6f867，+1742/-61，25 文件）。"
            "在 MRV2 跨 PP 传输时保持 SP token 分片，同时把 **auxiliary hidden states 与 Top-K payload 保持为 full-token**，"
            "只在既有 send 边界恢复额外 TP padding，并在图重捕获时复用常驻 PP 缓冲。PR 明确说明 #16511 是 Kimi 专用的 V1/V2+DSpark 实现，"
            "本 PR 提供通用 MRV2 buffer/graph 支持。本期 3 个提交（10/28）、1 次 force-push，10/29 报冲突。"
        ),
        npu=(
            "这就是 Ascend 自己的 aux hidden 跨 PP 传输实现，属直接相关。风险点：full-token 语义与 TP padding 恢复是否在所有阶段一致，"
            "以及常驻 buffer 在图重捕获后是否被覆盖（与上游 #43450/#63427 的 buffer 所有权问题同源）。"
        ),
        validation="作者附 tests/ut/worker/v2/test_pp_transport.py、test_pp_token_shard_graph.py 与 A3 多卡 SP+PP e2e；draft 状态，未合入。",
    ),
    dict(
        num=19284, kind="pr", cls="前期补漏续跟", prio="P0", activity="持续迭代（5 次 force-push，冲突）",
        analysis=(
            "[Feature][Model] Enable DeepSeek V4.1 MRV2 PP with arbitrary layer partitions（head 0eec97fb58，+1903/-76，21 文件）。"
            "把 HC 激活、FP32 mixing state 与 **DSpark auxiliary states** 一并放进 PP intermediate tensors 传输，并保留 payload 校验、"
            "快照与不透明运行时导入/导出。本期 10/26–10/29 共 5 次 force-push、多轮冲突提示；评论把增量 source-cache 传输与 "
            "stage-local mirror 视为后续方向，另开 #19892 跟踪 A2 双节点 RoCE 验证范围。"
        ),
        npu=(
            "直接决定 Ascend 上 PP>1 时 aux 状态能否用；需验证任意层切分下 DSpark aux 槽位、HC 激活维度与 FP32 mixing 精度一致。"
        ),
        validation="作者附 test_deepseek_v41_{pp_cache,dspark,hyper_connection,layer_plan,cache}.py 与 PP cache transfer e2e；本期未复测。",
    ),
    dict(
        num=19537, kind="pr", cls="前期补漏续跟", prio="P1", activity="合入（多轮 force-push 后）",
        analysis=(
            "[Feature] Add MiniMax-M3 DSpark draft FlashAttn support on A5（head 4d58a50493，+442/-38，13 文件，"
            "2026-10-29T07:36:54Z 由 ZT-AIA 合入）。为 DSpark 提供 target 的 **auxiliary hidden states**，"
            "并修正混合 FP8 target / BF16 draft 的 KV page 尺寸与 indexer kernel page 视图。"
            "本期 8 次 force-push、两轮 rerun 后才合入。"
        ),
        npu=(
            "这是 Ascend 上 DSpark 消费 aux hidden 的落地证据；仍需逐层比对 aux 层序与数值，并验证 FP8 target / BF16 draft 混用时 "
            "捕获 dtype 与 cast 位置（对应上游 #63436 的同类问题）。"
        ),
        validation="作者附 test_dspark_graph_contract.py、minimax_m3 单测与 flash-attn 精度 e2e。",
    ),
    dict(
        num=17557, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（冲突）",
        analysis=(
            "[Feature][Model] Support GLM-5.3-Flash DFlash2 speculative decoding on model runner V2（head c7ca3b9a1f，+759/-35，15 文件）。"
            "DFlash drafter 消费 target 的逐层 hidden states，并按 `hc_contract(hidden_states + residual)` 收缩；"
            "同时补 rope/NoPE 缓存跳过、rope is_neox_style 透传与 QKNorm-Rope 融合条件。测试含 test_glm5next_dflash_capture.py 与 test_glm5next_pp.py。"
            "本期只有 1 次 force-push 与两次冲突提示。"
        ),
        npu=(
            "需验证 hc_contract 在 Ascend 上与目标模型残差流语义一致（含 hc_mult 维），以及 PP 下 aux 层号与捕获顺序；"
            "新增模型支持不等于硬件层已验证。"
        ),
        validation="作者侧单测与 e2e；本期未合入、未复测。",
    ),
    dict(
        num=19801, kind="pr", cls="本期新建", prio="P2", activity="大量重构提交 + 重跑",
        analysis=(
            "[Refactor][MRV2][PCP][SpecDecode] Align PCP speculative decoding with upstream draft routes（head 0d8b9fc163，+615/-1524，18 文件）。"
            "把 PCP 下的投机解码路由与上游 draft route 对齐，并在 10/29 的提交里明确 PCP 采样的 auxiliary-state 路由；"
            "净删除代码为主，触及 model_runner、aclgraph 与 dspark/mtp speculator 测试。本期 9 次 force-push。"
        ),
        npu=(
            "需验证 PCP 分片下 aux 状态的路由顺序与 full-token 还原，避免与 #20075 的 SP 分片语义冲突。"
        ),
        validation="作者侧单测为主；未合入。",
    ),
    dict(
        num=19150, kind="pr", cls="前期补漏续跟", prio="P2", activity="仅冲突提示",
        analysis=(
            "[Feature] Expose EAGLE3 support for Qwen3-ASR（head 8403b12063，+97，3 文件）：为 Qwen3-ASR 注册 Ascend 子类并继承 "
            "`SupportsEagle3`，否则能力检查会拒绝 EAGLE3 draft（PR 说明当前验证的 vLLM ref 缺少该声明）。"
            "属能力注册而非捕获实现，但决定了该模型能否走 aux hidden 路径。"
        ),
        npu="需验证注册后 EAGLE3 在 Ascend 上真的能起到 aux capture 与 draft 加载；本期仅一次冲突提示。",
        validation="作者附 tests/ut/models/test_qwen3_asr_eagle3.py；未合入。",
    ),
    dict(
        num=20231, kind="issue", cls="本期新建", prio="P2", activity="RFC 新建",
        analysis=(
            "[RFC]: Support xPress Causal Refiner for DFlash on Ascend NPU（10/29 创建）：复用 DFlash proposer 已有的输入准备"
            "（hidden states、token IDs、上下文元数据），把精炼头作为后置步骤挂载，不替代 DFlash 本身。本期无评论。"
        ),
        npu="设计阶段；若采纳需定义 refiner 读取的 hidden state 层、dtype，以及图捕获下的 buffer 所有权。",
        validation="无代码与设备验证。",
    ),
    dict(
        num=19845, kind="issue", cls="本期新建", prio="P2", activity="RFC 挂到父议题",
        analysis=(
            "[RFC]: Extend adaptive verification support across ascend attention backends（10/26 创建）：把自适应验证（DFlash 系列）"
            "从当前已支持的后端扩展到更多 Ascend attention 后端。自适应验证依赖 verify 阶段的 target 概率与 support，"
            "与采样 mask 捕获同源，属相邻特性。"
        ),
        npu="设计阶段；需明确各 attention 后端的 verify 语义与 support 一致性，否则 mask/概率会静默偏移。",
        validation="无代码；本期仅 parent_issue_added 事件。",
    ),
]
