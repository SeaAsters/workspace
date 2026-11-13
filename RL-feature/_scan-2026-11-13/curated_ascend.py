# -*- coding: utf-8 -*-
"""Curated editorial content for the vllm-ascend half of the 2026-11-13 report."""

VLLM_ASCEND = [
    dict(
        num=21439, kind="pr", cls="本期新建", prio="P0", activity="合入",
        analysis=(
            "[Feature][SpecDecode] Support GLM-5.3-Flash DFlash2（2026-11-12T16:21+08:00 合入，12 文件）。"
            "此前 GLM-5.3-Flash 起不了 DFlash2，因为 target 没有暴露 Eagle3 auxiliary hidden-state 接口；本条在 text 与多模态模型上实现该接口，"
            "并处理 mHC 层「延后的 `hc_post` 残差补全」与 SP 安全捕获。同题更早的 #21431 已关闭，以本条为准。"
        ),
        npu="这是 Ascend 上「模型侧 aux 接口」的标准形态：mHC 的残差补全必须延后到 aux 捕获之后再合并，否则 hidden 语义与上游不一致；"
        "SP 下的捕获还需 all-gather 还原 full-token。",
        validation="作者侧 DFlash2 e2e 与 mHC/SP 用例；本期未复测。",
    ),
    dict(
        num=21792, kind="pr", cls="本期新建", prio="P0", activity="新建（跨仓同步）",
        analysis=(
            "[SpecDecode][Feature] 为 MRV2 上的 `extract_hidden_states` 支持 `hs_store_device=cpu`：把上游 #66785 的 hidden-states CPU-pages "
            "特性适配到 vllm-ascend。抽取的 hidden states 不再占用 NPU 上的 cache-only KV 池，而由 KV connector 每步流式搬到主机侧"
            "「每请求 CPU 分页」，请求结束再回收。"
        ),
        npu="这是本期 Ascend 侧最有价值的新方向：把 aux 从 NPU 显存搬到 host，直接缓解 RL 抽特征时的缓存压力；"
        "代价是 D2H 带宽与分页管理，需测吞吐与显存曲线。",
        validation="作者侧 MRV2 用例；本期未复测。",
    ),
    dict(
        num=17557, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-12 更新）",
        analysis=(
            "[Feature][Model] GLM-5.3-Flash DFlash2 on MRV2：DFlash drafter 消费 target 逐层 hidden states，并按 `hc_contract(hidden_states + residual)` 收缩。"
            "本期更新到 11-12（73 条时间线事件、15 文件），仍 open；与已合入的 #21439 属同一模型的两条路径，需注意不要重复计算覆盖范围。"
        ),
        npu="需实测：`hc_contract` 在 Ascend 上与目标残差流语义一致（含 hc_mult 维），以及 PP 下 aux 层号与捕获顺序。",
        validation="作者侧单测与 e2e；本期未复测。",
    ),
    dict(
        num=11066, kind="pr", cls="前期补漏续跟", prio="P1", activity="长期迭代（125 条事件）",
        analysis=(
            "[Feature][SpecDecode] GLM-5.3-Flash 的 DSpark target states 与 draft caches：auxiliary hidden-state export、draft 滑窗缓存与"
            "hybrid cache 布局。PR 说明 GLM5Next 用 hyper-connections + hybrid cache，中间张量不能直接当普通 decoder hidden 导出。"
            "这是 Ascend 上「aux 导出语义与上游不同」的明确案例。"
        ),
        npu="需实测：导出必须与上游 `hidden_states.mean(dim=1)` 的语义一致（hc 流取均），并验证 draft 滑窗缓存不改变 aux 行数。",
        validation="作者侧迭代中；本期未复测。",
    ),
    dict(
        num=21735, kind="pr", cls="本期新建", prio="P2", activity="新建（清理双轨）",
        analysis=(
            "[Refactor] Remove legacy speculative decoding compatibility paths：Ascend 已强制 MRV2，删除不可达的 V1 AUX payload 分支，"
            "统一走 MRV2 的 `EagleModelMixin` AUX slot relay。对 aux 特性的意义是**消除双轨**："
            "配合上游 #66177 的集中化，Ascend 的 relay 只剩一条路径。"
        ),
        npu="需实测：删除 V1 分支后，仍在用旧路径的模型（若有）会直接失效，需要在 admission/测试里提前暴露。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=21023, kind="pr", cls="前期补漏续跟", prio="P0", activity="持续跟进（11-12 更新）",
        analysis=(
            "DeepSeek V4.1 DSpark aux 捕获边界修复（Fixes #21022）：checkpoint 的 `dspark_target_layer_ids` 是零基 attention-input 边界，"
            "MRV2 透传后 Ascend 又多减一层导致早抓一层。本期更新到 11-12，仍 open（19 条时间线事件）。"
        ),
        npu="这是「层号约定错一层」的典型样本，必须逐模型核对零基/一基；同类问题在 GLM/DFlash 模型中也可能存在。",
        validation="作者侧复现；本期未复测。",
    ),
    dict(
        num=21501, kind="issue", cls="本期新建", prio="P2", activity="新建 RFC",
        analysis=(
            "[RFC] GLM-5.3-Flash target-side architecture for speculative decoding（draft-method-agnostic）："
            "提出 target 侧 aux 接口应做成与 draft 方法无关的统一接口（当前 dflash 走 Eagle3 共享接口、dspark 另有路径）。"
            "这是把模型侧自定义 aux 导出收敛成公共契约的设计讨论。"
        ),
        npu="需实测：若采纳，Ascend 各模型的自定义 `combine_hidden_states`/mHC 处理都要迁到统一接口。",
        validation="设计讨论；本期无代码。",
    ),
]
