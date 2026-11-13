# -*- coding: utf-8 -*-
"""Curated editorial content for the 2026-11-13 delta reports (sglang + vLLM)."""

SGL_HIDDEN = [
    dict(
        num=49664, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Perf] dflash: Keep a sliding-window MLA draft's KV in a per-request ring（2026-11-11T19:09-08:00 合入，20 文件）。"
            "把滑窗 MLA draft 的 KV 从共享池改成每请求环形缓冲，减少 draft KV 占用。与 aux 捕获的关联是：draft KV 布局变化会改变 "
            "draft 侧读取 target hidden 的行映射，属「draft 侧行/槽位」类改动。"
        ),
        npu="需实测：Ascend 的 draft KV 分配与环形缓冲语义需对照；draft 读取 aux 行的映射必须同步改。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=47493, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Fix] dp-attention: Trim MLP-sync pad rows from extend outputs before the draft extend（2026-11-09T15:38+08:00 合入，2 文件）。"
            "上期跟踪的「图补齐行 vs draft 消费行」错位已修：DP attention + EAGLE、target 用 flashinfer A2A、draft 用 DeepEP 时，"
            "prefill 图按最大 rank 的 DP bucket 补齐，进入 draft extend 前裁掉补齐行。"
        ),
        npu="需实测：Ascend 图路径同样有 bucket 补齐，必须确认 draft/aux 消费的是裁剪后的行——这是最容易在 NPU 复现的一类错位。",
        validation="作者侧复现 #47490；本期未复测。",
    ),
    dict(
        num=47750, kind="pr", cls="本期新建", prio="P2", activity="合入",
        analysis=(
            "[Fix] score: Validate logical item and MIS hidden-state row counts（2026-11-09T12:12-08:00 合入）。"
            "为评分路径加显式断言：逻辑 item 数与 MIS（multi-item scoring）hidden-state 行数不一致时早失败，把静默错位变成显式报错。"
        ),
        npu="需实测：Ascend 上 CP/SP shard 后的行数需与断言一致，否则会从「静默错」变成「误报」。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=48412, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Feat] pp: Relay routed experts through pipeline-parallel stages（2026-11-12T19:48+08:00 合入，14 文件）。"
            "`--enable-return-routed-experts` 在 PP>1 时各段只抓自己层，本条把 routed experts 跨阶段 relay 并合并。"
            "与 aux hidden states 的 PP relay 是同一类问题，是本期 PP 侧最有参考价值的一条。"
        ),
        npu="需实测：Ascend 的 `pp_transport_aux_hidden_states_` 约定是否与这条 relay 的缓冲/校验方式兼容。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=38589, kind="issue", cls="前期补漏续跟", prio="P1", activity="关闭（未修）",
        analysis=(
            "Qwen4Exp（Qwen3.8-Flash-Next）**无法产出 DFlash aux hidden states**：`set_dflash_layers_to_capture` 继承自基类、调用「看起来成功」，"
            "但捕获从未真正发生，导致该 target 上任何 DFlash/DSpark drafter 都无法服务。本期关闭（未合入修复），对应 PR #45142 仍停在「待 GPU 验证」。"
        ),
        npu="缺失：这是「捕获开关成功但不出数据」的静默失败形态；Ascend 上同类模型必须用「aux 是否非空 + 层数正确」两条断言验证，不能只看不报错。",
        validation="用户复现；本期无修复合入。",
    ),
    dict(
        num=48188, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-12 更新）",
        analysis=(
            "[Fix] spec: Keep chain MTP prefill hidden states under DP attention。DP attention 下多层 EAGLE + chain MTP draft 在第一步后"
            "用**全零 hidden** 构造 prompt 的 draft KV；本期更新到 11-12，仍 open。"
        ),
        npu="需实测：Ascend 的 DP attention + chain MTP 是否命中同一分支；外部表征是接受率异常。",
        validation="作者侧复现；本期未复测。",
    ),
    dict(
        num=47047, kind="pr", cls="前期补漏续跟", prio="P2", activity="持续跟进（无新提交）",
        analysis=(
            "[Fix] glm5_next: Gather DFlash aux hidden states for every scattered MoE all-to-all backend。"
            "把 gather 条件从枚举 deepep/mooncake/deepep_v2 改为 `is_moe_input_scattered_across_dp_ranks()`；本期仅状态更新（11-11），无新提交。"
        ),
        npu="需实测：Ascend 的 MoE 切分若不在该谓词覆盖内，aux 行数会与 attention 边界行不一致。",
        validation="本期无新增设备证据。",
    ),
    dict(
        num=50039, kind="pr", cls="本期新建", prio="P1", activity="新建（CP bounded replay）",
        analysis=(
            "[Fix] deepseek-v4.1: Preserve compact DSpark outputs with CP bounded replay。接续 #47406 的 compact-output 处理："
            "Prefill CP + DSpark + decoder SWA bounded replay 组合下，compact 输出（含 DSpark 需要的 hidden 片段）会被后续步骤覆盖或截断，"
            "本条保证其保留。属「bounded replay 与 aux 输出生命周期」交叉问题。"
        ),
        npu="需实测：Ascend 若支持 prefill CP + bounded replay，需要同样的输出保留语义；否则 DSpark 读到的是被覆盖的 buffer。",
        validation="作者侧复现/单测；本期未复测。",
    ),
]

SGL_SAMPLING = [
    dict(
        num=43517, kind="pr", cls="前期补漏续跟", prio="P0", activity="本期合入",
        analysis=(
            "跟踪四期的 Ascend 采样 mask 支持**本期合入**（2026-11-12T09:52+08:00，4 文件）：导出 Ascend 实际采样的 post-filter 权重、"
            "接入公共 mask 捕获与打包。这意味着「main 拒绝 ascend 后端返回 mask」的分支应随之变化，Ascend 用户首次有官方通路。"
        ),
        npu="这是本期 NPU 侧最重要的能力变化。合入后仍需按公共契约验收：`top_k=1025` 回退、`min_p`、greedy、support/selected 两模式、"
        "混合 opted-in 行、seeded mask on/off 与连续图重放；合入本身只证明代码进 main。",
        validation="作者侧 NPU/mock 对照；本期未复测。",
    ),
    dict(
        num=50126, kind="pr", cls="本期新建", prio="P1", activity="新建（序列化性能）",
        analysis=(
            "[Perf] tokenizer: Return output sampling masks as flat arrays on request。`return_sampling_mask` 目前把每个输出 token 的 support "
            "返回成嵌套 JSON（每 token 一个 list + 对应 logprob list），本条改为按请求返回扁平数组，降低序列化与反序列化开销。"
        ),
        npu="需实测：Ascend 侧同样走 tokenizer manager 的返回路径，扁平化后要确认 mask 行与 token 的对齐关系不变。",
        validation="作者侧基准；本期未复测。",
    ),
    dict(
        num=50155, kind="pr", cls="本期新建", prio="P1", activity="新建（API 拆分）",
        analysis=(
            "[Feat] sampling: Return sampling logprobs without the sampling mask。RL trainer 需要的是「采样后概率」本身（温度与 top-k/top-p/min-p "
            "之后、重新归一化），而 `output_token_logprobs` 是温度后的 logprob；本条允许只取 logprob 而不返回整张 mask。"
            "对 Ascend 的现实意义是：出现了一条不依赖 mask 打包的近似替代路径。"
        ),
        npu="需实测：Ascend 上该 logprob 必须与 mask 模式下 selected 概率位级一致，否则 trainer 的 off-policy 校正会漂移。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=48487, kind="pr", cls="前期补漏续跟", prio="P2", activity="本期合入（窗口起点）",
        analysis=(
            "[Fix] sampler: 当 TP sync group 只有一个 rank 时跳过 token-id all-reduce（2026-11-06T13:58-08:00 合入）。"
            "属采样同步路径优化；对 mask 的影响是「跳过同步后，选中集仍要自洽」。"
        ),
        npu="需实测：Ascend 上分组判断与 HCCL all-reduce 的等价性，以及跳过同步后 mask 选中行是否仍确定。",
        validation="合入提交自带测试；本期未复测。",
    ),
]

VLLM_HIDDEN = [
    dict(
        num=66177, kind="pr", cls="本期新建", prio="P0", activity="合入（PP aux relay 集中化）",
        analysis=(
            "[Spec Decode][PP] Centralize auxiliary hidden-state relay（2026-11-09T14:27+08:00 合入，23 文件）。"
            "把 EAGLE3 / DFlash / DSpark 的 aux hidden-state relay 与组装集中到 MRV2 的 `PPHandler`，PR 说明这使 gpt-oss、Qwen3-VL-MoE、"
            "Gemma4 可以在 PP 下运行，并修掉若干 relay 缺陷。这是本期 vLLM 侧最重要的结构性变化：aux relay 从「每模型自己搬」"
            "变成「公共层搬运 + 模型声明」。"
        ),
        npu="需实测：Ascend 的模型目前**自己**搬 aux（`pp_transport_aux_hidden_states_`），集中化后需确认 Ascend 是否改走公共 PPHandler，"
        "还是双轨并存；双轨时两侧的槽位顺序必须一致。",
        validation="合入提交自带 PP 用例；本期未复测。",
    ),
    dict(
        num=66063, kind="pr", cls="本期新建", prio="P0", activity="合入（draft 独立 PP 段）",
        analysis=(
            "[Spec Decode][PP] Run draft models as a single pipeline stage（2026-11-12T13:40-08:00 合入，57 文件）。"
            "把 draft 模型作为单个 PP 阶段运行，改变 target/draft 的阶段拓扑。对 aux 捕获的意义是：draft 不再与 target 共享阶段切分，"
            "aux 的产出与消费跨越不同阶段集合。"
        ),
        npu="需实测：Ascend 的 PP 拓扑与 `supports_aux_hidden_states_over_pp` 判定按新拓扑是否仍成立。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=66546, kind="pr", cls="本期新建", prio="P1", activity="合入",
        analysis=(
            "[Transformers backend] 支持 PP 下的 EAGLE3 aux hidden states 与 layer 0（2026-11-09T19:52Z 合入，6 文件）。"
            "把 aux-over-PP 的覆盖面从原生模型扩到 Transformers 后端，并处理 layer 0 作为捕获层的边界。"
        ),
        npu="需实测：Ascend 若走 Transformers 后端路径，需要同样的 layer-0 边界与 relay 支持。",
        validation="合入提交自带测试；本期未复测。",
    ),
    dict(
        num=66785, kind="pr", cls="本期新建", prio="P0", activity="新建（CPU pages 存储）",
        analysis=(
            "[Feature] extract_hidden_states: stream hidden states to CPU pages with token-hash prefix backfill。"
            "为 `extract_hidden_states` 增加可选 `hs_store_device=cpu`：hidden states 不再占 cache-only KV 池，而是经 KV connector 流式写入"
            "主机侧的每请求 CPU 分页，并用 token-hash 前缀回填处理缓存命中。RFC 见 #66794。"
            "这是「提取路径从 KV 池里解放出来」的关键改动，直接对 RL 抽特征的显存瓶颈。"
        ),
        npu="需实测：Ascend 的对应实现是同一天出现的 #21792，说明这是跨仓同步推进的方向；NPU 的 D2H 带宽与分页语义需单独验证。",
        validation="作者侧测试与 benchmark；本期未复测。",
    ),
    dict(
        num=66794, kind="issue", cls="本期新建", prio="P1", activity="新建 RFC",
        analysis=(
            "[RFC] extract_hidden_states — stream hidden states to CPU pages with token-hash prefix backfill，是 #66785 的设计文档："
            "存储位置、每请求分页、前缀哈希回填与失败/取消语义。"
        ),
        npu="需实测：Ascend 的 host 分页与 KV connector 路径需要同等设计，不能只移植开关。",
        validation="设计讨论；本期无设备验证。",
    ),
    dict(
        num=66323, kind="pr", cls="本期新建", prio="P1", activity="新建（模型扩展）",
        analysis=(
            "[Model] 为 Lfm3ForCausalLM 增加 EAGLE-3 aux hidden-state 支持，使其可作为 EAGLE-3/DFlash/DSpark 与 `extract_hidden_states` 的 target"
            "（修 #66311）。与 #66050（Ming ASR）一起，说明「哪些模型能当 target」仍在逐个补。"
        ),
        npu="缺失：Ascend 的 LFM3 支持与 aux 接口需单独接。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=56442, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-11 更新）",
        analysis=(
            "[Bugfix][Spec Decode] `ExtractHiddenStatesProposer.prepare_next_token_ids_padded()` 只读 `sampled_token_ids` 第一列来确定下一步 token，"
            "在多 token 输出（`num_speculative_tokens > 1`）时读到错的 token；修复改用最后一个有效 token。"
            "这是「多 token 时用错行」类 bug，直接影响提取数据的 token 对齐。"
        ),
        npu="需实测：Ascend 的 proposer 若沿用同一实现，同样会在多 token 场景错位。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=65991, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-12 更新）",
        analysis=(
            "[Spec Decode] `disable_extract_hidden_states_decode_drafts=True`：把动态 SD 调度设为 K=0，不再为提取方法排 decode draft。"
            "上期由 #65250 的成本量化驱动，本期更新到 11-12，仍 open。"
        ),
        npu="需实测：Ascend 使用同一调度；需确认 K=0 时 aux 仍按请求返回。",
        validation="作者侧单测；本期未复测。",
    ),
    dict(
        num=57736, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-10 更新）",
        analysis=(
            "窗口化 hidden-state 捕获（rollout 窗口 + teacher scoring）：本期更新到 11-10，19 个提交、30 文件、118 条时间线事件，仍 open。"
            "与 #66785 功能相邻：一个是「只返回某段行」，一个是「把行存到 CPU 分页」，两者都在降低返回路径传输成本。"
        ),
        npu="缺失：主仓仍未合入；Ascend 需等接口稳定。",
        validation="作者侧测试；本期未复测。",
    ),
    dict(
        num=64320, kind="pr", cls="前期补漏续跟", prio="P1", activity="持续跟进（11-09 更新）",
        analysis=(
            "Qwen3-MoE 的 EAGLE3 aux hidden states 跨 PP relay；本期更新到 11-09，仍 open。"
            "注意 #66177 已把 relay 集中到 PPHandler，这条的形态可能需要随之调整。"
        ),
        npu="缺失/需实测：Ascend 的 Qwen3-MoE 不在 opt-in 名单内。",
        validation="作者侧 PP 用例；本期未复测。",
    ),
    dict(
        num=61539, kind="issue", cls="前期补漏续跟", prio="P2", activity="关闭（未合入）",
        analysis=(
            "[RFC] Explicit row layouts for routed-expert capture 本期以关闭收尾（未合入实现）。结论应按「未采纳」处理："
            "行布局仍靠反推，Ascend 的 out-of-tree MoE 后端若布局不同需自行保证。"
        ),
        npu="缺失：该 RFC 未落地，Ascend 侧仍需自行校验行归属。",
        validation="关闭态；无实现。",
    ),
]
