# 混元实现级深挖 · Agent 与推理的代码级案例（2026-11-09）

> **定位**：07/08 号报告是**元数据级**（README/模型卡/commit/官网口径）。本报告是对"具体实现"的补充取证——把 6 个关键仓库浅克隆/更新到 `_probe/`，直接读代码，回答两个问题：**Agent 是怎么实现的？推理是怎么实现的？**
>
> **取证范围**（`_probe/`，浅克隆 + `GIT_LFS_SKIP_SMUDGE=1`；Hy-SimulMT 用 `--filter=blob:none --sparse` 只取代码目录）：
> UniRL `5fe89b3`（更新）、Hy-Memory `9737431`（更新）、FlashTopK / AppCIF-Bench / HyperLogic（新克隆）、Hy-SimulMT（稀疏克隆）；本地商发分支 `D:\WorkSpace\vllm-ascend-hy4\repo`（a74b08e3f）。
>
> **来源标记**：以下所有内容均为**仓库实测**（本机读文件），行号与常量照抄源码；仅 §5.4 配置为 YAML 读数。

---

## 0. 一页结论

| 层 | 案例 | 一句话实现事实 |
|---|---|---|
| Agent 训练 | UniRL `AgenticTrainer` | **barrier 式多轮 RL**：整批轨迹收齐才进训练步；失败轨迹以 NaN 剔除但**留在组内做基线**；advantage 是**轨迹级标量**（同轨迹所有 turn 共享），GRPO 风格组归一化 |
| Agent 训练 | UniRL `ToolAgentHarness` | 60 行的 env 循环：`reset → [generate → step → observe] × max_turns`；被截断的终轮判 `overflow` 而非 `completed`；异常只烧掉单条轨迹 |
| Agent 记忆 | Hy-Memory SDK（2.6 万行） | 冲突处理是**结构化 ops 协议**（ADD/supersedes/SHADOW 三场景）；working 与 coding 共链路但**变体隔离**（独立 SQLite/向量集合 + 不同的抽取规范提示词） |
| Agent 评测 | HyperLogic `hard_agent` | 给 agent 的"逻辑建模库"是 **4 个 solver 后端**（Z3 878 行 / BDD 863 行 / SAT-CNF 561 行 / Ganak）+ 734 行 DSL——库的形态是"建模语言 + 求解器桥" |
| Agent 评测 | AppCIF 引擎 | 墓穴探索引擎 = 纯标准库的 **3D 网格状态机**（GridPoint/ExplorerState/TombSystem），gold 答案由 `regenerate_gold.py` 从引擎重导出 |
| 推理内核 | FlashTopK | **五阶段单流无回读**流水线（采样→校准→scan/prune→精确选择→recovery 兜底）；gate 起点 2×topk；候选缓冲 max(4×topk, 16384)；bit 级确定性（平局取最小 index，-0.0 平 +0.0） |
| 推理服务 | Hy-SimulMT SimulInfer | "边听边译" = **把每个音频块当一轮 user 消息的多轮对话累积**；VAD 静音默认**置零保时轴**（非丢弃）；token 预算预检失败则切分且不消耗新 chunk |
| 推理算子 | vllm-ascend-hy4 iHC 融合 | 小 batch（≤256 token）iHC 用两 stage Triton pre + post kernel 融合，门控条件精确到 `hidden==6144 && HC_MULT==4`；**注释注明 adapted from SGLang PR #36805** |

---

## 1. Agent 实现 A：UniRL 的 agentic RL 闭环（代码级）

### 1.1 工具循环：`ToolAgentHarness`（60 行，`unirl/rollout/harness/tool_agent.py`）

```python
sample = self.env.reset(request)
for _ in range(self.max_turns - turns_done):
    sample = context.generate("policy", sample.fork(1, sampling_params=self.sampling))
    observation, done, _ = self.env.step(sample)
    if done:
        status = sample.gen_parts()[-1].status
        cut_off = status is not None and bool((status == SegmentStatus.TRUNCATED).any())
        return HarnessOutcome(sample, "overflow" if cut_off else "completed")
    if observation is not None:
        sample = sample.observe(observation)
```

**值得抄的三个语义**：
1. **`overflow` vs `completed` 的判定是"终轮是否被 token 预算截断"**（`SegmentStatus.TRUNCATED`）——"到 done 了但最后一个 turn 没写出 EOS"与"正常结束"是两种状态，前者下游可选择性丢梯度。
2. **`ContextOverflowError` 单独捕获**：一条轨迹在任意轮撑爆上下文 → 有生成内容则 `overflow`（可进基线），**一个 turn 都没有则 `failed`**（prompt 就溢出，无可打分内容）。
3. **故障隔离写进每一层**：`Exception → failed`（"one bad trajectory must not sink the drain"），`env.close` 的 teardown 异常也吞掉（"teardown must not sink the drain"）；`AgenticRolloutEngine.generate` 还有最后一层兜底网——harness 自身抛异常只标记该轨迹 `failed`（引擎层 `try/except` 包住 `harness.run`）。

### 1.2 环境注入：`AgenticRolloutEngine`（`unirl/rollout/engine/agentic/engine.py`）

- 工具 schema **从环境来**：`chat_template_kwargs.setdefault("tools", env.tool_schemas())` —— inner engine（SGLang/vLLM）拿到的是环境的工具定义，模型侧无需另配。
- 一致性校验：`env.max_turns == config.max_turns` 硬校验。
- `harness_status` 写在轨迹最后一个 part 上（`completed/suspended/failed/overflow` 四态），供 trainer 消费。

### 1.3 训练步：`AgenticTrainer`（491 行，`unirl/trainer/agentic.py`）——"Service-scored barrier"

**一个 `train_step` 的完整时序**（`train_step` L205-227）：

```
requests → _collect_groups:                       # 屏障式收集
    rollout.wake_up() → sync_weights(version)     # 权重同步（colocated）
    backend.offload()                             # 训练引擎卸载到 CPU
    RolloutManager.submit(prompts × group_size)
    groups = collect(batch_size)                  # 必须收齐：assert manager 为空
    rollout.sleep() → backend.onload()            # 换位回训练引擎
→ _score_trajectories:                            # 远程打分服务
→ _group_advantages:                              # 组归一化
→ _train_and_log:                                 # 组装 Part、stack.train_track
```

**五个实现级事实**（每条都有行号锚点）：

1. **Barrier 语义是硬约束**：`collect` 后 `if not self._rollout_manager.empty: raise RuntimeError("agentic barrier rollout must leave RolloutManager empty")`（L198-199）——不收齐不训练，宁可整批等待。这与异步交错式（如 stream-RL）是两个流派。
2. **失败轨迹的处理是"留组内、去奖励、零梯度"**：`_score_trajectories` 把 `failed` 轨迹 reward 置 NaN 但**不剔除出组**；`_group_advantages` 用 `finite_mean_std`（只对有限值算 mean/std，L293-306），NaN 行 advantage 置 0。**失败样本参与组基线统计**（拉低 mean，等效于抬高正常样本的相对优势），但不产生梯度——比"直接丢弃失败样本"更稳。
3. **`overflow` 的梯度开关**：`mask_overflow_loss=True` 时超长轨迹"stays in its group's baseline; only its gradient is dropped"（L325-327 注释原文）——overflow 轨迹同样参与基线。
4. **Advantage 是轨迹级标量**：`for generated in trajectory.gen_parts(): advantages ← 同一个标量`（L329-334）——**同一轨迹的全部 turn 共享一个 advantage**（terminal-answer reward 上游决定的 credit assignment 粒度），不是逐 turn 的过程奖励。
5. **打分是独立服务且带 DP 对齐**：reward 走 `RewardService.score_and_attach`（`reward_dp_size`），不足 DP 整数倍时复制首行 padding（L250-256）；返回 NaN 直接 `raise RuntimeError`（L279-280）——服务级故障不静默。

**默认可跑示例**（`examples/deep_research/deep_research_search_judge.yaml`，YAML 读数）：Qwen3-4B-Instruct + `ToolEnvironment` + 两个工具（`SearchTool`/`VisitTool`）+ `max_turns: 8` + `samples_per_prompt: 2`（组大小 2）+ `LLMJudgeRewardScorer` + `batch_size: 4` × `num_rollouts: 200`；奖励从 `<answer>...</answer>` 提取（取**最后一个** match，L39-43）。

> **与本系列前几轮结论的接驳**：10-12 报告的机制描述（Sample/组归一化/失败剔除/overflow 语义）全部得到代码证实；本轮新增的硬事实是——**barrier 收集、轨迹级 advantage、失败/溢出"留基线去梯度"、reward 服务化**。WorkForge（上轮报告）的"可验证环境"落进这个框架的位置就是 `env` 插槽：默认示例是手写沙箱（Search/Visit），WorkForge 环境是它的规模化替代。

---

## 2. Agent 实现 B：Hy-Memory SDK 的两个核心机制（2.6 万行实测）

### 2.1 scene 路由：一条链路、两个变体（`hy_memory/client.py` L861-911）

```python
# _normalize_scene: "working" 和 "coding" 归一成同一个返回值 "coding"：
#   它们走的是同一种生产力 pipeline，但使用按变体隔离的 store / writer / reader
if s in ("working", "coding"): return "coding"
```

- **chat 与（working|coding）是两条 pipeline**；working 与 coding 是**同一条生产力链路的两个变体**：独立 SQLite 文件 + 独立向量集合，行为差异仅在 ① 抽取规范——coding 用 `EXTRACT_PROMPT_CODING`（注释原文：`HARD CONSTRAINT 头部 + CODE WORK 节：分支名逐字、哪一版赢了、精确坐标、verified vs written`）② 是否挂版本链层（coding 额外挂 `coding.chain_store`）。
- `scene=None` 的自动路由由**三分类 judge**回填（LLM 一次调用），不固定吞成 working。
- 消息规范化层同时吃 OpenAI 与 Anthropic 两种协议（`tool_use`/`tool_result` blocks 拆为独立 tool 消息），并兼容第三方客户端把 `tool_calls` 编码成 JSON 字符串的情况（注释点名 `codebuddy`，L968-979）。

### 2.2 冲突处理：`agent/reconciler.py` 的三场景 ops 协议

模块头注释定义了精确的写入语义（L18-27）：

| 场景 | 操作 | 语义 |
|---|---|---|
| A 全新信息 | `ADD(supersedes=[])` | 与任何旧记忆无关，独立创建 |
| B 矛盾/取代 | `ADD(supersedes=[id,...])` | 旧节点 → `SHADOW` + `is_latest=False`，**进演化链**；`supersede_reason` 必填矛盾描述 |
| C 合并吸收 | `ADD(supersedes=[])` + `DELETE(id)` | 新节点吸收多个旧节点但不进演化链 |

**三个防幻觉细节**（代码注释原文）：
1. 演化链查询时"supersede 时只暴露 head 的 id+content，**history 不提供，防止 LLM 误操作**"（L638）——LLM 只看到最新版，改写历史的机会被结构性地拿走。
2. 每条 ADD 的输出预算 **200-300 tokens**（content + supersedes + reason，L333）——用 token 预算约束抽取粒度。
3. `reconcile()` 前先 `_generate_search_queries` 检索候选旧记忆（L360），冲突判定基于检索命中而非全库扫描——成本可控的"先查后写"。
4. `enforce_time_order`（L201）——时间排序在结构层强制，不靠 LLM 自觉。

---

## 3. Agent 实现 C：两个"程序化 ground truth"基准的实现形态

### 3.1 HyperLogic 的 `hard_agent`：给模型的"逻辑建模库"长什么样

`hard_agent/library/translation/` 共 **7 个模块约 4,900 行**，本质是"**建模 DSL + 求解器桥**"：

| 模块 | 行数 | 作用 |
|---|---:|---|
| `dsl.py` | 734 | 逻辑建模 DSL（题目 → 形式结构） |
| `backend.py` | 820 | 后端抽象与分发 |
| `backend_z3.py` | 878 | Z3 SMT 后端 |
| `backend_bdd.py` | 863 | BDD（二元决策图）后端 |
| `backend_sat.py` + `cnf_bridge.py` | 334+561 | SAT 后端 + CNF 编码桥 |
| `backend_ganak.py` | 179 | Ganak（#SAT 计数）后端 |
| `query.py` / `solve.py` / `eval_core.py` | 604/453/192 | 查询构造 / 求解调度 / 核心求值 |

这印证了 README 的发现"**模型倾向 plain Python 枚举，而非 solver**"——库给了 4 条通向形式化求解器的路，7 个前沿模型里 6 个仍然自己写枚举；**工具的可用性 ≠ 工具被使用**，这是 agent 工具设计的一手反例数据。评测器 `evaluate.py` 的判定链：`extract_matches`（从原始输出提取答案匹配）→ `judge_payload`（语义等价 LLM judge）→ `validate_answers`（gold 校验）。

### 3.2 AppCIF 的 Code Simulation Engine：一个可执行的"环境"

以墓穴探索引擎为例（`spatial_multi_rule/tomb_exploration/`）：`GridPoint`（3D 坐标）/ `Direction`(Enum) / `ExplorerState` / `TombSystem`（`_initialize_global_state` / `update_global_state` / `format_global_state`——**全局状态快照可格式化为文本喂给模型**）。纯标准库、可单跑（`python -m ...tomb_exploration_engine tomb_18`）、可重导（`regenerate_gold.py` 把 240 个仿真样本的 rubric 从引擎重新算一遍与发布数据比对）。**"Prompt-as-Environment"的实质 = 系统提示(14k 词) + 一个可执行的状态机模拟器**，二者共同定义任务真值。

---

## 4. 推理实现 A：FlashTopK 的五阶段内核（`flashtopk/topk.py`，455 行调度层）

### 4.1 单次 `select` 调用的完整流水线（`_run_rows` L289-338）

```
1. launch.sample_scores   共享采样：seq_len/128 个 key（钳制在 [2048, 8192]）打分入 sample 表
2. launch.seed_prep       校准：每行 256-bucket 映射 + gate（起点 ≈ GATE_SLACK(2.0)×topk 候选，
                          下限 K_MIN=32，右边界 SEED_MARGIN=1）
3a. launch.scan           无剪枝：全窗口扫，过 gate 者入候选缓冲
3b. launch.prune_scan     剪枝：上界掩码先筛，再对保留 key 全打分
    （pruning="adaptive"：由 planner 估计"界保留的 key 比例"决定走 3a 还是 3b）
4. launch.select          每行候选的精确 top-k（canonical 顺序）
5. self._recovery.run     兜底：候选溢出/不足的行稠密重打分重选（RECOVERY_ROWS 预留）
```

**关键常量与语义**（`topk.py` L36-51）：
- 候选缓冲 `cap = max(4×topk, 16384)`，**KV 超过 1M keys 时提到 32768**；行数 <128 的小调用反而**加宽** cap（采样 gate 保持更久，需要更多槽位兜底）。
- **确定性三重保障**：① 排序规则固定（分数降序 → key index 升序）；② `-0.0 ties +0.0`；③ `chunk_size` 切片间"rows are independent, so the slices select the same bits"（L50 注释）。
- **健康可观测**：`Info.health[8]`——[0] 溢出行数、[1] 不足行数、[3] 最大候选数、[7] recovery 重打分行数；`selector.last_info.pruned / .estimate` 暴露本次是否剪枝与估计值。
- **零同步设计**：文档头"all on one stream and **without reading anything back**"（L11）→ `run()` 可被 CUDA graph 捕获（vLLM 的 FULL_DECODE_ONLY 图模式因此能包含 selection）；`torch.library.custom_op("flashtopk::select", mutates_args=())` + fake 实现，torch.compile 可把整次选择当作一个不透明算子。
- **workspace 可预算**：`workspace_bytes()` ≈ `8 × chunk_size × max(4×topk, 16384)` 字节 + 每行小缓冲——部署前可精确算显存。
- `Selector.plan(max_rows, max_kv)` **预热编译全部 CuTe 内核变体**，`run()` 永不触发编译（Triton planner 例外，磁盘缓存）。

> **读法**：论文说的"selection 的输出不是 activation 而是 indices"在代码里就是 `cand_val/cand_idx` 两个 arena 复用同一块 float32 工作区（L299-300）——中间不物化 [num_q, seq_len] 分数表，这正是省显存与带宽的来源。**Hy4/GLM-5.x 的 1M 上下文预填能跑得动，这个内核是底层原因之一**。

---

## 5. 推理实现 B：Hy-SimulMT 的 SimulInfer（`SimulInfer/simulinfer/`，约 5,800 行）

### 5.1 "边听边译"的实现机制：多轮对话累积（`translation.py` `VLLMSession.step` L386-507）

**每个音频 chunk（chunk_ms 毫秒）就是一轮 user 消息**：

```
turn = {'role': 'user', 'content': [指令文本(仅首轮)] + [audio_part(samples)]}
budget = api.budget(self.messages + [turn])          # token 预算预检
reply = await self.api.generate(self.messages + [turn])
self.messages.append(turn)                            # 音频块永久入上下文
self.messages.append({'role':'assistant','content': raw})  # 译文也入上下文
```

**"先开口、后修补"的载体就在这里**：模型每收到一个新音频块，都**重新看到全部历史音频块 + 全部历史译文**，输出当前确定的译文片段——后续块的上下文允许它修订之前的话。多说话人音色克隆依赖同一机制（历史音频里带音色信息）。

### 5.2 指令词直接编码行为（L24-29，原文照抄）

```python
INSTRUCTIONS = {
  "zh-en": "请完成下面的同声传译任务，将陆续收到的语音翻译成英文。请结合上下文边听边译，"
           "输出能够确定的译文，不必等待整句结束。",
  ...}
ST_INSTRUCTIONS = {"zh-en": "把这段语音翻译成英文，要全部输出，不要分块。"}
```

——simultaneous 与离线 ST 是**两套指令词**，同一个人工智能模型的两种调用姿态。

### 5.3 工程细节（都值得抄）

- **协议自检**：启动时对 `SRC/TGT/EOT/EOS` 四个特殊 token 逐一 `/tokenize` 校验"必须是单 token"（L107-130），并验证 chat 模板渲染与 simulst 协议逐 token 一致——部署期的协议漂移直接报错，而不是推理时静默错位。
- **VAD 三实现**（`vad.py`）：`SileroActivity`（双阈值 onset 0.5 / release 0.35）、`FireRedNonstreamActivity`（0.4，默认）、`EnergyActivity`（能量兜底）。
- **静音的两种处理策略**（`step` L403-479）：
  - `post_speech_silence_mode='zero_fill'`（默认）：非人声样本**置零但不丢弃**——时间轴保持连续，模型看到的是"这里有静音"；
  - `audio_mode='noise_resistant'` + `simulst_crop_enabled`：**裁掉非人声段**只送人声，`retained_spans` 记录时间映射用于回填字幕坐标；纯静音 chunk `model_invoked=False`（不调模型）。
  - 整块人声超阈值（`vad_full_chunk_speech_ms`）则整块放行不掩码——防"长句被误剪"。
- **token 预算预检**（L493-501）：`budget.fits(max_tokens)` 不通过 → `cut('token_budget_preflight')` 分段重来，**新 chunk 不被失败的生成消耗**（注释："The new source chunk is not consumed by a failed generation"）。
- 分段状态机：`push(pcm) → step() → flush(force, trigger='eof'|'trailing_silence') / cut(reason) / discard() / commit()`——`trailing_silence_seconds` 触发收尾，`time_window` 触发时间窗切分。

---

## 6. 推理实现 C：本地商发分支的 iHC 融合算子（`vllm_ascend/ops/triton/hy4_ihc.py`，322 行）

```python
def triton_ihc_supported(x) -> bool:
    return (HAS_TRITON and npu and x.dtype == torch.bfloat16 and x.is_contiguous()
            and x.ndim in (2, 3) and 0 < x.shape[0] <= _MAX_FUSED_TOKENS   # ≤256 token
            and x.shape[-1] == _HIDDEN_SIZE                                 # 6144
            and (x.ndim == 2 or x.shape[1] == _HC_MULT)                     # HC_MULT=4
            and not envs.VLLM_BATCH_INVARIANT
            and get_current_hardware_profile().device_adaptor_family
                == DeviceAdaptorFamily.FP8_OPTIMIZED)                       # A5 硬件族
```

- **门控条件精确到形状**：只有"小 batch（≤256 token）、hidden 6144、4 条残差流、BF16、A5 族"才走 Triton 融合路径；其余输入"retain native iHC"。`not envs.VLLM_BATCH_INVARIANT` 把 batch-invariant（可复现）模式排除在外——两个特性互斥。
- **结构**：`_ihc_pre_stage1`（分块算 partial，BLOCK_K=1024）→ `_ihc_pre_stage2`（归约）→ `_ihc_post_kernel`——iHC 的 pre-mixing/post-mixing 两方向各成一个 kernel。
- **来源注释**（文件头）：`Adapted from the SGLang HY V4 implementation: https://github.com/sgl-project/sglang/pull/36805`——**混元模型在 SGLang 与 vLLM 两栈的 iHC 实现同源**，商发分支做的是 Ascend 移植 + 小 batch 融合。
- `vllm_ascend/patch/hy4_vl/model.py`（336 行）则是 `HYV4VLForConditionalGeneration` 的组合层：文本模型 + 视觉塔（`vision.py` 373 行）+ 图像/视频输入校验（`TensorSchema` 强类型）+ `get_mrope_input_positions`（三维 MRoPE 位置，sections `[12,10,10]`）。

---

## 7. 横向观察（本轮新增判断）

> **⑰ "程序化 ground truth"是同一价值观在四层的落地，不是口号。**
> 训练层：UniRL 的 reward 是服务化打分 + 结构化字段（advantages/rewards 分离、NaN 契约）；记忆层：Hy-Memory 的写入是结构化 ops（supersedes/SHADOW）而非自由文本；评测层：AppCIF 引擎可重导 gold、HyperLogic 答案由求解器导出 + SHA-256 commitment；内核层：FlashTopK 的 bit 级确定性 + `reproduce/` 固定输入。**四层的共同选择是"让真值可计算、让结果可复算、让异常可观测"**。

> **⑱ 混元的开源粒度覆盖五层，且各层可独立运行。**
> 内核（FlashTopK，`pip install -e .`）→ 算子（iHC，vllm-ascend patch）→ 服务（SimulInfer，`deploy.sh` 一键）→ 训练栈（UniRL，hydra 配置即训练）→ SDK（Hy-Memory，`pip install hy-memory`）。每一层都有独立的入口、健康检查与失败语义——这不是"配套开源"，是**每层都按可运维标准交付**。

> **⑲ agent 工具设计的"可用性悖论"有了定量案例。**
> HyperLogic 给了 4 个 solver 后端共 4,900 行的建模库，7 个前沿模型里 6 个仍然写 plain Python 枚举——**结构化的工具并不自动被结构化地使用**。这与 Hy-Memory reconciler 的设计形成对照：后者把 LLM 的输出空间压到三场景 ops 协议（LLM 只能选 A/B/C），把"自由度"换成了"可靠性"。**给 agent 自由度还是给协议，是两条待验证的路线**。

---

## 8. 素材位置

| 仓库 | 位置 | 关键文件 |
|---|---|---|
| UniRL | `_probe/UniRL/`（更新至 5fe89b3） | `unirl/trainer/agentic.py`（491 行）、`unirl/rollout/harness/tool_agent.py`（60 行）、`unirl/rollout/engine/agentic/engine.py`、`examples/deep_research/deep_research_search_judge.yaml` |
| Hy-Memory | `_probe/Hy-Memory/`（更新至 9737431） | `hy_memory/client.py`（L855-1010）、`hy_memory/agent/reconciler.py`、`hy_memory/pipelines/writer.py`（3,858 行）、`hy_memory/coding/chain_store.py`（1,099 行） |
| HyperLogic | `_probe/HyperLogic/` | `hard_agent/library/translation/`（4,900 行）、`evaluate.py` |
| AppCIF-Bench | `_probe/AppCIF-Bench/` | `code_simulation_engines/spatial_multi_rule/tomb_exploration/` |
| FlashTopK | `_probe/FlashTopK/` | `flashtopk/topk.py`（455 行）、`flashtopk/kernels/{scan,prune,refine}_sm90.py` |
| Hy-SimulMT | `_probe/Hy-SimulMT/`（稀疏） | `SimulInfer/simulinfer/translation.py`、`vad.py`、`budget.py` |
| 商发分支 | `D:\WorkSpace\vllm-ascend-hy4\repo` | `vllm_ascend/ops/triton/hy4_ihc.py`、`vllm_ascend/patch/hy4_vl/model.py` |

> `_probe/` 按 `.gitignore` 约定不入库；本报告所有行号/常量以 2026-11-09 快照为准。
