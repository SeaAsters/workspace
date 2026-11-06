# -*- coding: utf-8 -*-
"""Curated editorial content for the 2026-11-06 delta reports (sglang + vLLM)."""

SGL_HIDDEN = [
    dict(
        num=48569, kind="pr", cls="本期新建", prio="P0", activity="新建（NPU 测试补齐）",
        analysis=(
            "[Test] [NPU] return-aux-hidden-states：为 `--enable-return-aux-hidden-states` 补 NPU 端到端测试。"
            "PR 描述明确该开关复用投机解码的 aux 捕获：服务端在 target 上抓取中间层 hidden states 打包返回，而不加载 draft。"
            "这是上期合入的 #43449 在 Ascend 侧的第一个专门测试补丁，决定「NPU 上到底能不能跑」这一层的证据强度。"
        ),
        npu="直接是 NPU 证据链：若合入，Ascend 的 aux 捕获从「代码路径存在」推进到「有 e2e 用例」；仍需看它覆盖 eager 还是图路径、层号如何断言。",
        validation="作者侧为 NPU e2e；本期未在本地复测，也未看到 CI 结果。",
    ),
    dict(
        num=48659, kind="pr", cls="本期新建", prio="P1", activity="新建 + 关联 bug",
        analysis=(
            "[Fix] tbo: Preserve child offsets and auxiliary hidden states（对应 issue #48660）。TBO 把父 batch 的 token buffer 切成 microbatch 时保留了父级相对的 "
            "`extend_start_loc`，第二个 microbatch 会读错 KV 槽位；同时 TBO 路径会丢掉投机 aux hidden states。修复保留子偏移并把 aux 捕获带过 TBO 边界。"
        ),
        npu="需实测：Ascend 上 TBO（two-batch-overlap）与 aux 捕获的组合是否同样丢状态，取决于 NPU 侧是否复用同一 TBO 逻辑。",
        validation="作者侧以 #48660 的复现为准；本期未复测。",
    ),
    dict(
        num=48660, kind="issue", cls="本期新建", prio="P1", activity="新建 bug 报告",
        analysis=(
            "[Bug] TBO 会改变确定性 prefill 输出并丢掉投机 auxiliary hidden states。报告给出两处问题：microbatch 的 `extend_start_loc` 仍是父级相对、"
            "以及 aux 捕获在 TBO 下丢失。它把「aux 捕获正确性」从单 batch 扩展到 overlap 调度场景，是本期最有价值的问题定位之一。"
        ),
        npu="缺失/需实测：Ascend 是否启用 TBO 与 aux 捕获的组合需要单独验证；不能默认 NPU 无此问题。",
        validation="用户报告 + 关联修复 PR #48659。",
    ),
    dict(
        num=48188, kind="pr", cls="本期新建", prio="P1", activity="新建（DP attention 修复）",
        analysis=(
            "[Fix] spec: Keep chain MTP prefill hidden states under DP attention。DP attention 下，多层 EAGLE + chain MTP draft"
            "（`chain_mtp_hidden_states`：Step3p5MTP、Inkling MTP、GigaChat35NextN）在第一步之后用**全零 hidden** 构造 prompt 的 draft KV，"
            "属 hidden-state 传错而非崩溃型错误。"
        ),
        npu="需实测：Ascend 的 DP attention 与 chain MTP 组合是否命中同一分支；该类错误不会报错，只会让接受率异常。",
        validation="作者侧单测/复现已附；本期未复测。",
    ),
    dict(
        num=47493, kind="pr", cls="本期新建", prio="P1", activity="新建（DP attention 修复）",
        analysis=(
            "[Fix] dp-attention: Trim MLP-sync pad rows from extend outputs before the draft extend（Fixes #47490）。DP attention + EAGLE、target 用 "
            "flashinfer A2A、draft 用 DeepEP 时，prefill 图会按最大 rank 的 DP bucket 补齐，draft extend 却按未补齐的 token 数消费，导致 draft 侧行错位。"
            "属「图补齐行数 vs 消费者行数」这类 hidden/aux 行对齐问题。"
        ),
        npu="需实测：Ascend 图路径同样存在 bucket 补齐，需要确认 aux/draft 侧消费的是裁剪后的行。",
        validation="作者侧复现 #47490；本期未复测。",
    ),
    dict(
        num=36829, kind="issue", cls="前期补漏续跟", prio="P1", activity="关闭（有结论）",
        analysis=(
            "[Bug] GLM-5.3-Flash（glm5_next、mHC）EAGLE3/MTP：`fused_eh_norm` 宽度与 mHC 展平后的 aux hidden 不匹配，"
            "自行绕过后接受率约 1.0。这是「aux 捕获层/宽度与 draft 期望不一致」的典型现象：不是 shape 崩溃，而是接受率异常。本期关闭，"
            "说明结论已收敛，但关闭本身不等于代码已修。"
        ),
        npu="需实测：Ascend 上 mHC 展平顺序、`fused_eh_norm` 宽度与 GPU 是否一致，决定 GLM-5.3 在 NPU 上 EAGLE3 是否可用。",
        validation="用户复现与绕法；本期无设备验证。",
    ),
    dict(
        num=47750, kind="pr", cls="本期新建", prio="P2", activity="新建（校验）",
        analysis=(
            "[Fix] score: Validate logical item and MIS hidden-state row counts。为评分路径加显式行数校验：逻辑 item 数与 MIS（multi-item scoring）"
            "hidden-state 行数不一致时早失败，而不是让错位数据进入 logprob 计算。属把「静默错位」变成「显式报错」的防护。"
        ),
        npu="需实测：Ascend 上 MIS 路径同样需要该断言，尤其是 CP/SP 下 shard 后的行数。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=45142, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进（待 GPU 验证）",
        analysis=(
            "[Fix] qwen4_exp: Capture DFlash aux hidden states on Qwen3.8-Flash-Next（Fixes #38589）。上期已跟踪；本期更新为"
            "「CPU 单测通过、尚未在 GPU 上跑，等 GB300 验收数字后再标 ready」。说明该模型的 aux 捕获仍缺设备验证，不能按合入处理。"
        ),
        npu="缺失：连 CUDA 侧都待验证，Ascend 更无依据；NPU 需先确认官方 Ascend 模型实现是否走同一 capture hook。",
        validation="作者明示 pending GPU validation。",
    ),
    dict(
        num=48412, kind="pr", cls="本期新建", prio="P2", activity="新建（PP relay 基础设施）",
        analysis=(
            "[Feat] pp: Relay routed experts through pipeline-parallel stages。`--enable-return-routed-experts` 在 PP>1 时只有第一段保留宿主缓存，"
            "本条把 routed experts 跨阶段 relay。与 aux hidden states 的 PP relay 是同一类「每段只抓自己层、需合并」问题，属相邻基础设施。"
        ),
        npu="需实测：Ascend 的 PP relay 实现（`pp_transport_aux_hidden_states_`）是否可复用同一缓冲/校验约定。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=47445, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Fix] ipc: 让 multimodal payload、hidden states 与 weight-update 的 dtype 在 msgpack IPC 下正确往返（2026-10-30T13:46-07:00 合入，"
            "10 文件，含 `msgpack_utils.py` +98、`io_struct.py` +47、`schedule_batch.py` +39）。"
            "这是 hidden states 从前端/engine 到 tokenizer manager 的序列化路径修复：dtype 丢失会把 bf16/fp32 静默变成别的类型，"
            "进而让训练侧数据与参考不一致。"
        ),
        npu="需实测：Ascend 侧同一 IPC 路径（msgpack + numpy 张量）也要验证 dtype 往返，尤其是 bf16 与 fp32 mixing。",
        validation="作者附 `test_io_struct.py`（157 行改动）与 `test_data_parallel_controller.py`；本期未复测。",
    ),
    dict(
        num=42248, kind="pr", cls="前期补漏续跟", prio="P1", activity="合入",
        analysis=(
            "[Perf] srt: Native tensor output and prefill scoring optimizations（2026-10-30T15:50-07:00 合入，21 文件）。"
            "把 `pooled_hidden_states` 等输出以原生 tensor 传递（`batch_result_processor`、`output_streamer`、`io_struct`、`engine` 同步改），"
            "并优化 prefill 评分。对 extract-hidden-status 的意义是返回路径少了 list/tensor 转换，形状与 dtype 契约随之收紧。"
        ),
        npu="需实测：Ascend 输出路径同样走 batch_result_processor，需确认 tensor 直传后 D2H 与裁剪行为一致。",
        validation="作者侧单测（含 hidden-state 相关用例）；本期未复测。",
    ),
    dict(
        num=43902, kind="pr", cls="前期补漏续跟", prio="P2", activity="合入（#43450 后续重构）",
        analysis=(
            "[Refactor] scheduler: 把 deferred responses 收进对象，并把 PD prefill inflight 队列改名为 transfer queue（2026-11-01T21:42-08:00 合入，30 文件）。"
            "新增 `scheduler_components/deferred_responses.py`，是上期合入的 #43450（forward observer + deferred responses）的直接后续："
            "接口不变、所有权与状态机集中，便于后续在 PP/PD 下复用。"
        ),
        npu="需实测：Ascend 复用同一 scheduler 组件，需验证对象化后 resume/abort 路径与图重放 buffer 生命周期。",
        validation="合入提交自带单测调整；本期未复测。",
    ),
]

SGL_SAMPLING = [
    dict(
        num=41916, kind="pr", cls="前期补漏续跟", prio="P0", activity="本期合入",
        analysis=(
            "上期的 P0 在本期合入（2026-11-03T12:06-08:00，7 文件，+228/-44）：`sampler.py`、`scheduler.py`（+15，新增校验）、"
            "`dflash_utils.py`、`eagle_utils.py`、`uno_utils.py` 统一为 joint 语义（保留 top-p nucleus 再与 top-k support 相交），"
            "测试加在 `test_speculative_sampling.py`(+89) 与 `test_sampling_mask.py`(+28)。"
        ),
        npu="需实测：Ascend fused helper 与 torch fallback 必须逐分布比对；合入只代表语义定案，不代表 NPU 已对齐。",
        validation="作者侧五路径 + seeded 位级比较；本期未复测。",
    ),
    dict(
        num=38565, kind="pr", cls="前期补漏续跟", prio="P0", activity="本期合入",
        analysis=(
            "上期的 P0 在本期合入（2026-10-30T17:07-07:00，9 文件，+237/-15）：新增 `srt/layers/sampling_renorm.py`(98 行) 作为确定性归一化实现，"
            "`sampler.py` 按 TP 组大小选择路径，并在 `test_renorm.py`(+68)、`test_renorm_policy.py`(+49)、`test_sampling_mask.py`(+7) 补测。"
            "标题口径为 `tp_size > 1`，比原 `attn_tp_size` 覆盖面更大。"
        ),
        npu="需实测：Ascend 上确定性 renorm 需与 torch 路径逐 rank 位级对齐，尤其 bf16 累加顺序；不能仅凭合入推断 NPU 一致。",
        validation="作者侧 H100/B200 数据；本期未复测。",
    ),
    dict(
        num=43517, kind="pr", cls="前期补漏续跟", prio="P0", activity="持续迭代（仍在 open）",
        analysis=(
            "Ascend 采样 mask 仍未合入，本期继续更新到 11-06。主仓仍以「ascend 后端直接从 logits 采样、不构造 support」拒绝 "
            "`return_sampling_mask`，因此本条仍是 NPU 侧唯一通路。"
        ),
        npu="缺失：不能宣称 Ascend 已支持；需等该 PR 合入后按公共契约验收 top_k 上限、min_p、greedy、support/selected 与图重放。",
        validation="作者侧 NPU/mock 对照；本期未复测。",
    ),
    dict(
        num=47967, kind="pr", cls="本期新建", prio="P2", activity="新建（XPU 跟进）",
        analysis=(
            "[Fix] [XPU] sampling: Make deterministic top-k renorm reproducible on XPU。说明 #38565 的确定性策略在 XPU 上会回退到 sgl-kernel 之外的路径，"
            "需要单独保证可复现。属同一语义在不同后端的落地差异，对 Ascend 有直接参考价值。"
        ),
        npu="需实测：Ascend 的等价问题是「确定性策略在 NPU 算子上是否真的确定」，需要按该思路补 NPU 版验证。",
        validation="作者侧 XPU 测试；本期未复测。",
    ),
    dict(
        num=48487, kind="pr", cls="本期新建", prio="P2", activity="新建（采样同步优化）",
        analysis=(
            "[Fix] sampler: 当 TP sync group 只有一个 rank（`tp_size=1` 或 attn TP 分组）时跳过 token-id all-reduce。"
            "属采样同步路径的性能修复，与 mask 捕获的「selected token 一致性」相邻：同步被跳过时，mask 行的选中集也必须自洽。"
        ),
        npu="需实测：Ascend 上该分组判断与 HCCL all-reduce 的等价性；跳过同步后 mask 选中行需保持确定。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=43047, kind="pr", cls="前期补漏续跟", prio="P2", activity="仅状态更新",
        analysis=(
            "[sglang-miles] Return sampling masks for chain EAGLE/NEXTN decoding 仍为 draft，本期仅状态更新（11-01）。"
            "目标分支不是 main，不能据此判断 main 或 Ascend 能力。"
        ),
        npu="不适用：分支不是 main。",
        validation="无本期技术证据。",
    ),
]

VLLM_HIDDEN = [
    dict(
        num=57736, kind="pr", cls="前期补漏续跟", prio="P0", activity="标题与范围改写（实质推进）",
        analysis=(
            "上期的窗口化 hidden-state 捕获在本期把标题与范围改成「capture selected hidden-state rows for rollout windows and **teacher scoring**」，"
            "即同一原语同时服务两个负载：按绝对位置区间抓 rollout 的窗口行（训练 drafter），以及教师评分需要的行。"
            "仍通过 AuxOutput Connector 返回，包含最终 hidden states 与配置的投机辅助输出。"
        ),
        npu="缺失/需实测：仍未合入；Ascend 侧要先有接口，再评估 connector 的 D2H 与图重放 buffer 所有权。",
        validation="作者侧 connector 测试与 benchmark；本期未复测。",
    ),
    dict(
        num=66050, kind="pr", cls="本期新建", prio="P1", activity="新建（模型扩展）",
        analysis=(
            "[Model] 为 Ming-3.0-ASR 打开「音频条件下的 decoder hidden-state 提取」，走既有离线 `LLM` API，供 draft 训练使用；"
            "PR 明示 blocked on #54174（MM 输入相关）。说明 hidden-state 提取正在从纯文本扩到多模态模型。"
        ),
        npu="缺失：多模态 + hidden-state 提取在 Ascend 上无路径；需先有上游接口。",
        validation="作者侧为 draft PR，待依赖合入。",
    ),
    dict(
        num=65991, kind="pr", cls="本期新建", prio="P1", activity="新建（性能开关）",
        analysis=(
            "[Spec Decode] 增加 `disable_extract_hidden_states_decode_drafts=True`：把所有 batch size 的动态 SD 调度设为 K=0（`[(1,1,0)]`），"
            "即不再为 `extract_hidden_states` 排 decode draft（约 45 行非测试改动）。"
            "这是对「extract_hidden_states 方法在 decode 阶段几乎不被接受」的直接回应（见 #65250）。"
        ),
        npu="需实测：Ascend 若使用同一动态 SD 调度，该开关应同样生效；但需确认 K=0 时 aux 捕获仍按请求返回。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=65250, kind="issue", cls="本期新建", prio="P1", activity="新建性能议题",
        analysis=(
            "[Performance] `extract_hidden_states` 每步为每个请求排一个 draft token，接受率约 0.03%，在 L4 上损失 37%–39% 饱和吞吐。"
            "把「用 spec 方法做纯提取」的开销定量化，是本期最重要的成本证据。"
        ),
        npu="需实测：Ascend 上同一方法的 decode draft 开销可能更显著（draft 前向更贵）；需自行测接受率与吞吐损失。",
        validation="用户报告（L4）；本期未复测。",
    ),
    dict(
        num=60765, kind="issue", cls="前期补漏续跟", prio="P1", activity="持续跟进（有实现 PR）",
        analysis=(
            "[Bug] Qwen3.5（hybrid GDN）无法在 PP 下跑 dflash 投机解码：缺少 `supports_aux_hidden_states_over_pp` opt-in，"
            "引擎初始化即被拒。本期由 #61180 提供实现，issue 仍在 open 跟踪。"
        ),
        npu="缺失/需实测：Ascend 侧同一 opt-in 名单（DeepSeek-V4/Kimi-K3/MiniMax-M3 + DeepseekV2 补丁）不含 Qwen3.5，需要单独接。",
        validation="用户环境复现 + 上游实现 PR。",
    ),
    dict(
        num=64320, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（冲突待解）",
        analysis=(
            "[Bugfix][PP][Spec Decode] Qwen3-MoE 的 EAGLE3 aux hidden states 跨 PP relay（沿用 Qwen2 #50514 与 MiniMax-M3 #57197 的模式）。"
            "本期更新到 11-04，仍 open。"
        ),
        npu="缺失/需实测：Ascend 上 Qwen3-MoE 若不在 opt-in 名单内，PP 下同样被拒。",
        validation="作者侧 PP e2e 与 aux 单测；本期未复测。",
    ),
    dict(
        num=61180, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-06 更新）",
        analysis=(
            "[Bugfix][Spec Decode] Qwen3.5 的 aux states 跨 PP relay，修复 #60765；通过共享 EAGLE3 接口配置有序 aux 槽位，含 NVIDIA 侧实现。"
            "本期更新到 11-06，仍 open。"
        ),
        npu="缺失/需实测：Ascend 的 Qwen3.5 需要自己的 relay 实现（当前 opt-in 名单不含）。",
        validation="作者侧 CPU 测试；本期未复测。",
    ),
    dict(
        num=63205, kind="pr", cls="前期补漏续跟", prio="P1", activity="合入",
        analysis=(
            "[Model Runner V2][Spec Decode] Speculator 配置类与自定义 speculator（2026-10-30T10:52-07:00 合入，67 文件），"
            "新增 `vllm/v1/worker/gpu/spec_decode/extract_hidden_states_config.py` 与 `test_gpu_extract_hidden_states_speculator.py` 等。"
            "把 `extract_hidden_states` 从「方法名 + 散落参数」收敛成配置对象，是后续扩模型的接口基础。"
        ),
        npu="需实测：Ascend 的 speculator 实现需对齐该配置对象，否则自定义 speculator 无法挂载。",
        validation="合入提交自带单测；本期未复测。",
    ),
    dict(
        num=63406, kind="pr", cls="前期补漏续跟", prio="P2", activity="本期合入",
        analysis=(
            "上期跟踪的 CUDA Graph 捕获尺寸拆分在本期合入（2026-11-03T22:50-08:00，17 文件）：prefill（token）与 decode（request）"
            "分别使用捕获尺寸，减少每尺寸保留 aux hidden states 的常驻开销（与 #63427 的弱引用修复同一目标）。"
        ),
        npu="需实测：Ascend 图池按什么维度分配尺寸需对照；否则 aux 常驻占用可能不降反升。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=61539, kind="issue", cls="前期补漏续跟", prio="P2", activity="仅状态更新（10-30）",
        analysis=(
            "[RFC] 为 routed-expert 捕获引入显式行布局：当前按张量行数与 `dp_metadata` 反推行归属，对 out-of-tree MoE 后端不成立。"
            "本期有更新（10-30），仍 open。与 hidden-state 捕获属同一「行布局必须显式」的问题族。"
        ),
        npu="需实测：Ascend MoE 后端的行布局若不在已知模式内，需显式声明，否则捕获行错位。",
        validation="设计讨论；本期无代码。",
    ),
]
