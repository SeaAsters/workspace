# -*- coding: utf-8 -*-
"""Curated editorial content for the vllm-ascend half of the 2026-11-06 joint report."""

VLLM_ASCEND = [
    dict(
        num=21023, kind="pr", cls="本期新建", prio="P0", activity="新建（DSpark aux 边界修复）",
        analysis=(
            "[BugFix][SpecDecode] Fix DeepSeek V4.1 DSpark aux capture boundaries（Fixes #21022）。DeepSeek V4.1 的 checkpoint 里 "
            "`dspark_target_layer_ids` 是**零基 attention-input 边界**，MRV2 上 `set_eagle3_aux_hidden_state_layers()` 原样透传后，"
            "Ascend 侧又多减了一层，导致 aux 比预期早抓一层。修复只动边界换算，不改变捕获机制。"
        ),
        npu="这是 Ascend 侧最典型的 aux 层号偏差：不是崩溃、而是**早一层**，会直接让 DSpark 训练数据与推理不一致；需按 checkpoint 的零基/一基约定逐模型核对。",
        validation="作者侧复现 #21022（A3、1P1D、DP8/TP2、DSpark 5 tokens）；本期未在本地复测。",
    ),
    dict(
        num=21022, kind="issue", cls="本期新建", prio="P0", activity="新建 bug 报告",
        analysis=(
            "[Bug] DeepSeek V4.1 在 MRV2 上 DSpark 的 auxiliary states **早抓一层**。报告给出环境（vllm-ascend main @40b3842d3、"
            "vLLM v0.32.0、A3 910_9392、CANN 9.2.0）与 `dspark_target_layer_ids=[37,38,39]` 的零基边界说明，"
            "是本期最有价值的 Ascend 正确性问题定位。"
        ),
        npu="需实测：同类零基/一基换算问题可能存在于其它 DSpark/DFlash 模型，应按模型逐个核对层号约定。",
        validation="用户环境复现 + 关联修复 PR #21023。",
    ),
    dict(
        num=16621, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（性能）",
        analysis=(
            "[Performance] 跳过 DeepSeek V4 MTP 里未使用的 hidden-state buffer 操作。PR 描述明确指出 Ascend 的 DeepSeek V4 实现"
            "已在配置层收集 aux hidden states 并通过 `PPTransportDataType.AUX_HIDDEN_STATES` 跨 PP 传输，但 draft 侧仍会对"
            "**不会被消费**的 buffer 做拷贝/清零。属 aux 传输链路的直接开销清理。"
        ),
        npu="需实测：删除操作后必须确认 aux 槽位仍被正确初始化（尤其 PP 中段），否则会出现脏数据而非崩溃。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=16993, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（功能+性能）",
        analysis=(
            "[Feature][Performance][DeepSeek V4.1] Reapply #16544 并把 DSpark auxiliary hidden-state 的 ReduceMean 融进 HCPost"
            "（`npu_hc_post_with_mean`）：只有配置的 aux 目标层请求 mean 输出，非目标层仍走 `npu_hc_post`；PR 注明 SP 顺序需保持。"
            "这正是「hc 流取均值」的 Ascend 融合实现（对应上游 `hidden_states.mean(dim=1)`）。"
        ),
        npu="需实测：融合后的 mean 必须与 torch 顺序位级一致（含 -0 行、dtype 提升），否则 aux 数值偏移会静默传播。",
        validation="作者侧精度用例；本期未复测。",
    ),
    dict(
        num=19157, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（PP 支持）",
        analysis=(
            "[Feature][Model] 在 MRV2 上支持 DeepSeek V4.1 Flash 的 PP：stage-local Engram、mHC/auxiliary-state transport、"
            "投影后的 KV-cache 布局与 DSpark+PP 一起打通，并为旧 vLLM 版本通过 model-state hook 恢复原始 token ids。"
            "属「PP 下 aux 状态可搬」的端到端模型支持。"
        ),
        npu="需实测：任意层切分下 aux 槽位、mHC 维度与 Engram 的 stage-local 语义需逐项验证。",
        validation="作者侧 PP 用例；本期未复测。",
    ),
    dict(
        num=8513, kind="issue", cls="前期补漏续跟", prio="P1", activity="Issue 讨论（长期未解）",
        analysis=(
            "[Bug][Upstream] `ValueError: Unknown speculative decoding method: extract_hidden_states`：用户按上游文档用 "
            "`extract_hidden_states` 跑 KV connector 集成测试时，Ascend 侧方法分派不认识该方法。"
            "说明「Ascend 认识 extract_hidden_states 这个方法名」与「该方法在本环境可用」之间仍有差距。"
        ),
        npu="缺失：需要把该方法纳入 Ascend 的 spec 方法分派与配置校验（`ascend_config.py` 只在校验层出现该名字）。",
        validation="用户报告；本期无修复合入。",
    ),
    dict(
        num=17907, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进（mask 相邻）",
        analysis=(
            "[Bugfix][MRV2] 适配 vLLM #59359 改过的 `SamplingMaskTensors.from_logits` 新签名（新增 `cu_num_logits`、`rows_per_request`），"
            "让 Ascend 侧 request-major 的采样 mask 打包继续工作。属采样 mask 主题，但直接决定 Ascend 能否产出 mask 行。"
        ),
        npu="需实测：打包槽位与 support 行必须与主仓语义一致，否则 mask 行错位；这也是 #43517 之外 Ascend 侧 mask 能力的另一块拼图。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=21056, kind="pr", cls="本期新建", prio="P2", activity="新建（草稿 KV 窗口）",
        analysis=(
            "[Feature][SpecDecode] 为 MRV2 target-dependent 投机解码加 `additional_config[\"draft_kv_window\"]`，把 EAGLE/EAGLE3/MTP 的"
            "draft attention KV 限制在最近 W 个 token。属 draft 侧显存治理，与 aux 捕获共用 draft 元数据。"
        ),
        npu="需实测：窗口裁剪后 aux 行与 draft KV 行仍须一一对应，否则接受率异常。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=20617, kind="pr", cls="本期新建", prio="P2", activity="合入",
        analysis=(
            "[BugFix][Attention] 用上游 model tag 识别 V2 EAGLE MLA drafts（2026-11-03T17:43+08:00 合入，4 文件）。"
            "属 draft 能力识别修正：识别错会把 MLA draft 当普通 draft，进而影响 aux hidden 的读取路径。"
        ),
        npu="本身就是 Ascend 修复；需在图路径下确认识别结果与 aux 捕获分支一致。",
        validation="合入提交自带测试；本期未复测。",
    ),
]
