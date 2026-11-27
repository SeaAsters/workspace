# -*- coding: utf-8 -*-
"""Curated editorial content for the 2026-11-27 delta reports (sglang + vLLM)."""

SGL_HIDDEN = [
    dict(
        num=52507, kind="pr", cls="本期新建", prio="P0", activity="新建（深读：图重放按输出裁剪）",
        analysis=(
            "[Fix] cuda-graph: Keep full aux hidden states in breakable prefill replays with SP。**本期深读对象**。"
            "`BreakableCudaGraphBackend._output_rows()` 原先对 list/tuple 输出返回一个**标量** `min(cap, *rows)`，"
            "于是 `_slice_output()` / `_copy_output_to_buffer()` 把所有输出都按「最小行数」裁剪："
            "SP 分片后的 `hidden_states` 行数少于 full-token 的 aux hidden states 时，aux 会被截断成 SP 行数。"
            "修复让 `_output_rows()` 返回**逐输出**行数（`None` 项保留 `None`），并新增 `_rows_per_item()` 在切片/拷贝时按项应用；"
            "新单测构造 `hidden=(2,4)`、`aux=(8,6)`，断言 replay 后各自形状不变。"
        ),
        npu="**高风险同类项**：Ascend 的图重放同样在 SP/CP 下裁剪输出，只要 aux 与 hidden_states 行数不同就会按最小值截断；"
        "建议直接用这个断言（aux 行数 == full-token 行数）做 NPU 回归，而不是只看 replay 不报错。",
        validation="作者侧 CPU 单测 test_breakable_cuda_graph_backend.py（63 行，新增）；本期未复测。",
    ),
    dict(
        num=53074, kind="pr", cls="本期新建", prio="P1", activity="新建（TBO aux）",
        analysis=(
            "[Fix] tbo: Rebase child extend offsets and capture aux hidden states in overlapped layers。"
            "上期 #48660 报的 TBO 两个问题（子偏移仍父级相对、overlap 层里的 aux 捕获丢失）在本 PR 一并处理："
            "重设子 extend offset，并在重叠层里补 aux 捕获。"
        ),
        npu="需实测：Ascend 的 TBO 与 aux 组合要单独验证，错位表现为接受率异常而非崩溃。",
        validation="作者侧用例；本期未复测。",
    ),
    dict(
        num=47047, kind="pr", cls="前期补漏续跟", prio="P1", activity="本期合入",
        analysis=(
            "[Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE backend，`2026-11-21T20:59-08:00` 合入（3 文件）："
            "`glm5_next.py` 的 gather 条件改为 `is_moe_input_scattered_across_dp_ranks()`，并补 `test_glm5_next_dflash_capture.py`(+47) 与 "
            "`test_aux_capture_deferred_allreduce.py`。跟踪三期的条目收口。"
        ),
        npu="需实测：Ascend 的 MoE 切分需在该谓词覆盖内，否则 aux 行数与 attention 边界行不一致。",
        validation="合入提交自带单测；本期未复测。",
    ),
    dict(
        num=51566, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进（11-23 更新）",
        analysis=(
            "[Fix] cuda-graph: Pack prefill aux hidden states into one runner-owned buffer。本期更新到 11-23，仍 open；"
            "与 #52507 属同一片区域（图/重放下的 aux 缓冲所有权与裁剪）。"
        ),
        npu="需实测：Ascend 图路径同样需要 runner 持有 + 按 token 裁剪。",
        validation="本期未复测。",
    ),
    dict(
        num=51475, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（未合入）",
        analysis=(
            "[Fix] api: Fix `return_hidden_states` on `/generate`（\"last\"、响应形状、单 prompt list）。上期新建、本期仍 open（11-20 更新）。"
            "回归仍未修完：0.5.22+ 上 `\"last\"` 依旧 400。"
        ),
        npu="缺失：Ascend 用户按文档调用同样会 400，直到该 PR 合入。",
        validation="本期未复测。",
    ),
    dict(
        num=51926, kind="pr", cls="本期新建", prio="P2", activity="合入（新模型接入 aux 捕获）",
        analysis=(
            "[Feat] inkling2: Support Inkling-2-Mini，`2026-11-26` 合入（`4817721258`）。除模型与算子外，它改到了 "
            "`model_runner_components/spec_aux_hidden_state.py`（新增的 `_forward`/`forward` 形态），属新模型接入 aux 捕获面。"
            "大 PR 中的一个小接线点，仅作背景。"
        ),
        npu="需实测：Ascend 侧该模型若不在支持列表内，aux 捕获同样不成立。",
        validation="合入提交自带测试；本期未复测。",
    ),
]

SGL_SAMPLING = [
    dict(
        num=53418, kind="pr", cls="本期新建", prio="P0", activity="新建（深读：PD 下 mask 与发出 token 对齐）",
        analysis=(
            "[Fix] Match prefill sampling masks to emitted tokens。**本期深读对象**。PD 分离下 `return_sampling_mask` 有三个错位点被一起修："
            "① `disaggregation/utils.py` 只在 `self.enable_sampling_mask` 时才申请可选 wire buffer——standalone prefill 直接把采样行返回客户端，"
            "不需要这套缓冲，之前无条件申请让「没有真正 handoff」的路径也走两槽缓冲；"
            "② scheduler 对 fake PD decode（首 token 由合成产生、不过 sampler）显式拒绝："
            "`return_sampling_mask requires a real prefill result.`；"
            "③ 在 tokenizer 会返回空行的场景（无输出 token 的捕获）直接关掉该请求的 mask，避免「有 token 无 mask / mask 行数不一致」。"
            "测试覆盖 standalone prefill、无真实 handoff、以及 prefill+decode 混合批。"
        ),
        npu="需实测：Ascend 的 PD 路径同样有「合成首 token」与 standalone prefill 两种形态；关键是让 mask 行与**实际发出的 token**一一对应，"
        "否则 trainer 侧会出现长度不匹配。",
        validation="作者侧单测与 PD 集成测试；本期未复测。",
    ),
    dict(
        num=52270, kind="pr", cls="本期新建", prio="P1", activity="合入（每步构建 mask 行）",
        analysis=(
            "[Refactor] sampling: Build seeds and sampling-mask rows per step; drop SamplingBatchInfo filter/merge，"
            "`2026-11-22T03:43-08:00` 合入（12 文件）。把 seed 与 mask 行的构建从批级 filter/merge 改成逐 step 构建，"
            "去掉一层批内合并逻辑。属 mask 行归属的结构性简化。"
        ),
        npu="需实测：逐 step 构建后，Ascend 的 mask 行归属与 token 顺序需重新对齐验证。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=50859, kind="pr", cls="前期补漏续跟", prio="P2", activity="本期合入",
        analysis=(
            "[Fix] tokenizer: Return sampling masks on requests aborted from the waiting queue，`2026-11-20T18:43-08:00` 合入。"
            "上期新建、本期合入 main；其发布分支 cherry-pick（#51378）本期关闭，说明主干修复已足够。"
        ),
        npu="需实测：Ascend 的 abort/retract 路径需同样验证「有 token 必有 mask」。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=53153, kind="pr", cls="本期新建", prio="P1", activity="合入（PD 单 token 请求由 prefill 直答）",
        analysis=(
            "[Feat] pd: Answer a one-token request on the prefill server without a decode，`2026-11-26` 合入（`221713687e`）。"
            "prefill 直接回答只需一个 token 的请求，不再经 decode；改到 `disaggregation/utils.py`、`scheduler.py`、`tokenizer_manager.py`。"
            "这与 #53418 是同一片区域：一旦 prefill 自己发 token，mask 行就必须由 prefill 侧产出并对齐。"
        ),
        npu="需实测：Ascend 的 PD 路径若启用该直答，mask 产出点也随之改变。",
        validation="合入提交自带 PD 测试；本期未复测。",
    ),
]

VLLM_HIDDEN = [
    dict(
        num=69875, kind="pr", cls="本期新建", prio="P0", activity="新建（深读：PP aux 接收缓冲 dtype）",
        analysis=(
            "[Bugfix][Spec Decode][PP] Allocate aux hidden-state receive buffers in residual dtype。**本期深读对象**。"
            "`reserve_aux_intermediate_tensor_slots()` 原先用**模型 dtype**（如 bf16）为下游 PP 阶段预分配 aux hidden-state 槽位，"
            "但 aux 取自**残差流**，部分模型（如 Muse Spark）的残差流是 fp32——接收端缓冲比发送端窄，PP 传递时被静默降精度。"
            "修复改为：若 `IntermediateTensors` 里有 `hidden_states`，aux 槽位就用它的 dtype，否则回落到模型 dtype；"
            "新测试对 fp32 / bf16 / 无 hidden_states 三种情况分别断言槽位 dtype。"
        ),
        npu="**与 Ascend 直接相关**：Ascend 的 DeepSeek-V4 系列残差流保留 fp32（`FP32 mixing state` 就在 PP intermediate tensors 里），"
        "接收端 aux 槽位若按 bf16 分配会静默降精度。验收断言应是「aux 槽位 dtype == hidden_states dtype」，而不是「能跑通」。",
        validation="作者侧 test_eagle3_pp_aux_dtype.py（61 行，新增）；本期未复测。",
    ),
    dict(
        num=69064, kind="issue", cls="本期新建", prio="P1", activity="新建 bug 报告",
        analysis=(
            "[Bug][Spec Decode] Enable VLM hidden states extraction：VLM（多模态）走 `extract_hidden_states` 时提取不到或形状不对。"
            "说明 hidden-state 提取的模型侧覆盖仍偏文本模型，多模态 target 的 aux 路径尚未打通。"
        ),
        npu="缺失：Ascend 的多模态 target 同样没有该路径。",
        validation="用户报告；本期无修复。",
    ),
    dict(
        num=53243, kind="issue", cls="前期补漏续跟", prio="P1", activity="Issue 讨论",
        analysis=(
            "[Bug] `extract_hidden_states` 高估 DeepSeek-V4 的 KV cache 内存：提取用的 cache-only KV 组被按最大宽度计费（与 #56822 同族问题），"
            "同样显存只能起更小配置。本期有更新，仍 open。"
        ),
        npu="需实测：Ascend 的 KV 池计费需按实际 group 宽度复算，否则可用 batch 被低估。",
        validation="用户报告；本期无修复。",
    ),
    dict(
        num=52073, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进",
        analysis="[Spec Decode] Test extract_hidden_states on NemotronH hybrid models：为混合 Mamba 模型补提取测试，属覆盖补齐。",
        npu="需实测：Ascend 的 hybrid 模型若走同一路径，测试需在 NPU 上重跑。",
        validation="本期未复测。",
    ),
    dict(
        num=68824, kind="pr", cls="前期补漏续跟", prio="P2", activity="仍 open（上期新建）",
        analysis=(
            "上期的「DeepSeek-V3.2 丢最后一层 aux」修复本期仍未合入（11-20 之后无更新），问题 #68813 也仍 open——"
            "也就是说**少一层的问题在 main 上还没修**。"
        ),
        npu="缺失：Ascend 实现同样需要核对循环边界。",
        validation="本期未复测。",
    ),
    dict(
        num=65991, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进",
        analysis="[Spec Decode] 关闭 extract_hidden_states decode draft 的开关（K=0），本期更新到 11-24，仍 open。",
        npu="需实测：Ascend 同一调度下 K=0 时 aux 仍需按请求返回。",
        validation="本期未复测。",
    ),
]
