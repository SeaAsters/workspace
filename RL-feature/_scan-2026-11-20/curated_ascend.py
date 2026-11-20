# -*- coding: utf-8 -*-
"""Curated editorial content for the vllm-ascend half of the 2026-11-20 report."""

VLLM_ASCEND = [
    dict(
        num=22122, kind="pr", cls="本期新建", prio="P0", activity="合入（层号根因修复）",
        analysis=(
            "[Bugfix] Correct DeepSeek V4.1 DSpark auxiliary layer capture（2026-11-17T19:21+08:00 合入，4 文件）。"
            "根因写得很清楚：MRV2 的 resolver 保留 **零基** checkpoint 层号，而 target 侧此前用了 **一基**比较，"
            "于是 `[37,38,39]` 实际选到的是第 36/37/38 层的输入——整整早一层。修复统一为零基比较。"
            "上期 #21022/#21023 提出的问题在本期由本条真正修掉。"
        ),
        npu="这是「层号约定错一层」的标准修复样本。同类风险：GLM/DFlash2 等模型若各自实现捕获点，也可能有一基/零基混用，应按模型逐个核对，"
        "并用「捕获层号 == checkpoint 声明层号」断言固化。",
        validation="作者侧 DSpark 用例；本期未复测。",
    ),
    dict(
        num=22074, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[BugFix] Preserve draft LM-head TP for DFlash2 and DSpark（2026-11-18T20:09+08:00 合入，15 文件）。"
            "DFlash2 在 `finegrained_tp_config.lmhead_tensor_parallel_size > 1` 时构造失败：它的 candidate selector 绕过常规 draft-logits hook，"
            "导致 DP rank 上的不同请求没有对齐。修复保留 draft LM-head 的 TP 语义，使 DFlash2 与 DSpark 在该配置下可用。"
        ),
        npu="与 aux 捕获的关联：candidate selector 直接消费 target 的 aux/logits，绕过 hook 就会让「aux 行 → 请求」映射错位；"
        "修复后需在图模式下复核 DP rank 对齐。",
        validation="作者侧单测与 e2e；本期未复测。",
    ),
    dict(
        num=21792, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-17 更新）",
        analysis=(
            "[SpecDecode][Feature] `hs_store_device=cpu` for extract_hidden_states on MRV2：把上游 #66785 的 CPU-pages 路径适配到 Ascend，"
            "让抽取的 hidden states 不占 NPU cache-only KV 池、改为每请求主机 CPU 分页。本期更新到 11-17，仍 open。"
        ),
        npu="这是 Ascend 侧最直接的显存缓解路径；需要 D2H 带宽、分页回收与 dtype 往返三项设备数据，目前都还没有。",
        validation="本期未复测。",
    ),
    dict(
        num=21023, kind="pr", cls="前期补漏续跟", prio="P2", activity="仍未合入（已被 #22122 替代）",
        analysis=(
            "上期的 DSpark aux 捕获边界修复本期仍 open（更新到 11-17），但合入的同题修复是 #22122（11-17）。"
            "两条应视为同一问题的不同实现：跟踪能力时以已合入的 #22122 为准，#21023 若长期不合入应关闭或标注 superseded。"
        ),
        npu="不适用：落点已在 #22122。",
        validation="状态核对；以 #22122 为准。",
    ),
    dict(
        num=21469, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（main2main）",
        analysis=(
            "[Misc] main2main 1113（2026-11-19T16:46+08:00 合入，123 文件）：把 vllm-ascend 对齐到上游 1113 版本。"
            "本期上游发生了 aux relay 集中化（#66177）与 CPU-pages（#66785）等结构性改动，"
            "main2main 是对齐这些接口的常规通道——需要确认对齐后 Ascend 的自有 aux 传输是否仍与集中化后的 PPHandler 一致。"
        ),
        npu="需实测：main2main 之后要重跑 aux/PP 相关用例；这类升级最典型的故障是「接口还在但语义变了」。",
        validation="合入提交；本期未复测。",
    ),
    dict(
        num=17557, kind="pr", cls="前期补漏续跟", prio="P2", activity="关闭（未合入）",
        analysis=(
            "[Feature][Model] GLM-5.3-Flash DFlash2 on MRV2 本期以 `closed`（未合入）结束：同模型的 DFlash2 支持已由 #21439 合入，"
            "本条作为重复/替代路径关闭。跟踪覆盖时应以 #21439 为准，避免把两条算作两次能力新增。"
        ),
        npu="不适用：落点已在 #21439。",
        validation="状态核对；以 #21439 为准。",
    ),
]
