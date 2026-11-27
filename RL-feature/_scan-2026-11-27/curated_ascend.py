# -*- coding: utf-8 -*-
"""Curated editorial content for the vllm-ascend half of the 2026-11-27 report."""

VLLM_ASCEND = [
    dict(
        num=21831, kind="pr", cls="前期补漏续跟", prio="P0", activity="合入（GLM-5.3 DFlash2 + PP）",
        analysis=(
            "[Feature][SpecDecode] Support GLM-5.3-Flash DFlash2 and PP，`2026-11-23T15:39+08:00` 合入（`9316dd80`）。"
            "在已合入的 DFlash2 基础上把 **PP** 也打通：target 侧 aux 接口 + mHC 延后残差补全 + SP 安全捕获，"
            "并进入 PP 传输。是本期 Ascend 侧唯一把「aux 捕获 + PP」同时合入的能力项。"
        ),
        npu="需实测：PP>1 时 aux 槽位顺序、full-token 还原与 mHC 残差补全的次序必须与单卡一致；"
        "结合本期上游 #69875（接收缓冲 dtype）一起验，避免 bf16/fp32 混用。",
        validation="合入提交自带用例；本期未复测。",
    ),
    dict(
        num=22485, kind="pr", cls="前期补漏续跟", prio="P1", activity="合入（跳过无用 HC buffer）",
        analysis=(
            "[BugFix][Model] Skip unused DeepSeek V4.1 HC buffer for DSpark，`2026-11-20T16:27+08:00` 合入（`a285b50d`）。"
            "DSpark 不需要的 HC buffer 操作被跳过，属 aux/HC 链路的开销清理（延续上期 #16621 的思路）。"
        ),
        npu="需实测：删除后需确认 DSpark 实际消费的 aux 槽位仍被正确初始化，否则是脏数据而非崩溃。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=22198, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（Main2Main v0.34.0）",
        analysis=(
            "[Misc] Main2Main v0.34.0，`2026-11-24T18:32+08:00` 合入（`9b8c54e0`）：对齐上游 v0.34.0。"
            "上游本期把 aux relay 集中到 `PPHandler`（#66177）、加 CPU-pages（#66785）、改 draft 为独立 PP 段（#66063），"
            "main2main 之后必须重跑 aux/PP 组合用例。"
        ),
        npu="需实测：这类升级最典型的故障是「接口还在但语义变了」，尤其 aux 槽位与 dtype。",
        validation="合入提交；本期未复测。",
    ),
]
