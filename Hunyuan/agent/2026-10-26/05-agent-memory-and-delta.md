# 混元 Agent 侧增量扫描 · 2026-10-12 → 2026-10-26

> **本次任务**：在既有基线（[`INDEX.md`](../../INDEX.md) + [`agent/2026-10-12/`](../2026-10-12/)）之上，扫描 **2026-10-12 之后**混元在 Agent 方向的新增仓库、代码变更与公开信息。
>
> **窗口**：`2026-10-12T00:00Z` → `2026-10-26T10:30+08:00`。
> **基线**：`_research/repos-classification.csv` 的 **85 个仓库**快照（Agent 14 / Infer 56 / 其余 15）。
> **本次实测**：组织现有 **90 个公开仓库**（+5）；Agent 方向 **新增 1 个重量级仓库 `Hy-Memory`**、**2 个既有仓库被推送**（`UniRL`、`ExplorationBench`）；另有 **1 篇不进仓库的 Agent 论文**（WorkForge）与 **1 次平台级收缩**（元器/ QClaw 停服）。

**数据来源标注约定**（沿用 10-12 报告，新增公众号 / HF / arXiv / Release 四类）

| 标记 | 含义 | 可信度 |
|---|---|---|
| **仓库实测** | 本机浅克隆后直接读文件（`_probe/`）或 `raw.githubusercontent.com` 原文 | 一手 |
| **GitHub atom / Release atom** | 提交 feed、`releases.atom`（客观时间戳） | 一手 |
| **HF API / 模型卡** | `huggingface.co/api/models?author=tencent`、模型卡原文 | 一手 |
| **arXiv API / abs 页** | `<published>` / `<updated>` / 摘要原文 | 一手（**但 arXiv 无单位元数据**） |
| **官方 X（媒体逐字引用）** | X 帖原文取不到，靠多家媒体引用互证 | 中高 |
| **媒体原创报道** | 财经/科技媒体一手采访，含版本号与规模数字 | 中高 |
| **媒体转述 / 自媒体** | 转载与二手编译，可能错配时间线 | 中→低，**单独标注** |
| **公众号** | `mp.weixin.qq.com` 正文被验证码拦截，只有标题/URL 可证 | 仅"存在性"级 |
| `README 未说明` | 原文没有，不做推断补全 | — |

---

## 0. 一句话结论

10-12 报告的判断是"**评测先行、框架稀缺**"（6 个评测 vs 1 个通用 Agent 框架 `UniRL`）。**本轮把"框架稀缺"补了三块，但没有一块是"再做一个通用 Agent 运行时"**：

1. **`Hy-Memory`（10-15 建，1,282 stars）—— 混元首个以独立仓库开源的 Agent 记忆框架**（10-12 基线的 85 个仓库清单里没有同类项目）。三层同时交付：记忆写入模型 `Hy-Memory-30B-A3B`（30B 总参 / **~3B 激活** MoE，Apache-2.0，arXiv 2610.22196）、SDK `hy-memory==2.0.x`（PyPI，Python ≥3.9）、本地优先应用 `MemHub`（macOS 桌面 + Linux Runtime + npm CLI）。已接入 `Claude Code / Cursor / Codex / OpenCode / Qwen Code` 与两个插件（`hermes`、`openclaw`）。
2. **`WorkForge`（arXiv 2610.04906，v4 修订 10-21，官方对外发布 10-22）—— Agent 训练"环境侧"的基础设施。** 用"真实文件工作区 + 可核验事实锚点"合成 **16.7K 个可验证环境**、覆盖 **40 个专业领域 / 60 种文件类型**；后训练 `Qwen3.5-35B-A3B-Base` 把 **GDPVal 45.5→73.6、APEX 5.0→21.3**。**它不是仓库，是一篇论文 + 一次官方发布**，恰好补上 10-12 报告里"评测多、环境少"的缺口。
3. **`UniRL v0.2.0`（release tag 2026-10-19，窗口内）正式把 `train_agentic` 列为发布项。** 但窗口内 commit feed（上限 20 条）**全是训练效率/重构**，agentic 代码与 README 均未改 —— **能力是基线前就有了，本轮只是"打版"**。
4. **平台侧发生收缩而非扩张：元器（11-09 停服，迁腾讯云 ADP）+ QClaw（12-24 停运，迁 WorkBuddy）。** 混元的 Agent 出口收敛为 **WorkBuddy（办公 Agent）+ 企业微信 + 腾讯云 ADP**；10-14 企业微信 5.0.12 与 WorkBuddy 5.8.0 双向互通，是"混元 Agent 进内部业务"最实的一条落地。
5. **具身智能线本轮零增量**（`HY-Embodied` 系列 5 个仓库 `pushed_at` 全部停在 2026-05～08）。唯一"具身相关"动作在推理侧：`Hy-Nomas` 的 Robotics LoRA 用真实机械臂数据增强生成世界的机器人仿真能力。

---

## 1. 增量总览（Agent 相关）

| 项 | 类型 | 时间（窗口内） | 规模指标 | Agent 侧结论 |
|---|---|---|---|---|
| **Hy-Memory** | **新仓库** | created 10-15；发布 10-16 | 1,282★（10-26）、20/20 条提交在窗口 | ⭐ 记忆框架三层开源（模型 + SDK + MemHub） |
| **WorkForge** | **新论文/发布** | v4 修订 10-21；官方发布 10-22 | 16.7K 环境 / 40 领域 / 60 文件类型 | ⭐ 长程工作 Agent 的**可验证环境合成**（无代码仓库） |
| **UniRL v0.2.0** | **Release tag** | 2026-10-19T07:01Z | ~350 PR；12 个新 model package | 正式打版把 `train_agentic` 列为发布项；窗口内提交均为效率 |
| UniRL（提交） | 既有仓库 | 10-16 → 10-19 | 20+ 条提交 | 训练效率/重构；**agentic 路径零改动** |
| ExplorationBench | 既有仓库 | 10-14 | stars 7 → 27 | News 补登两次官方 X 曝光；代码未变 |
| CL-bench | 既有仓库 | 10-17 | stars 583 → 589 | judge prompt 措辞调整（非 Agent 主线） |
| DiagSafe | 新仓库（**非 Agent**） | created 10-15；论文 10-21 | 3,489 prompt / 84 标签 | 安全诊断基准；含 `Hy4-Preview` 实测（见 §5） |
| OmniCapBench / HYDRA / Hy-Nomas | 新仓库（**非 Agent**） | 10-12 / 10-09 / 10-21 | — | 见 [`../../infer/2026-10-26/06-infer-world-model-and-serving.md`](../../infer/2026-10-26/06-infer-world-model-and-serving.md) |
| 元器停服 / QClaw 停运 | **平台事件** | 报道 10-15 / 公告 09-24 | 元器：11-09 停服；QClaw：12-24 停运、MAU 225.9 万 | Agent 出口收敛到 WorkBuddy + 企业微信 + 腾讯云 ADP |
| 企业微信 × WorkBuddy | **落地事件** | 10-14 | 企微 5.0.12 / WorkBuddy 5.8.0；企微覆盖 1400 万组织 | 混元系 Agent 进内部业务最实的一条 |

**分类口径更新**（建议写回 `INDEX.md`）：Agent **14 → 15**（+`Hy-Memory`）；Infer **56 → 59**（+`Hy-Nomas`、`HYDRA`、`OmniCapBench`）；其余 **15 → 16**（+`DiagSafe`）。

---

## 2. 核心增量 A：`Hy-Memory` / Hy Memory 2.0

- 仓库：<https://github.com/Tencent-Hunyuan/Hy-Memory>（created **2026-10-15**，Apache-2.0）
- 官网：<https://memory.hunyuan.tencent.com/>
- 模型：<https://huggingface.co/tencent/Hy-Memory-30B-A3B>（10-15）· `-FP8`（10-16）· `-GGUF`（10-19）
- 论文：`arXiv:2610.22196`《Hy-Memory-30B-A3B: Attributed Reinforcement Learning for Long-Term Memory Writing》（**2026-10-15**）；`arXiv:2606.09483`《Memory Beyond Recall: A Dual-Process Cognitive Memory System for Self-Evolving LLM Agents》（**EMNLP 2026** 已接收）

### 2.1 时间线（仓库自述 + Release atom + HF API 三源对齐）

| 时间（UTC） | 事件 | 来源 |
|---|---|---|
| 2026-05 | Hy Memory **1.0** 初代发布（README 链公众号原文） | 仓库 README |
| 2026-06-08 | 双系统认知记忆论文上 arXiv（后 EMNLP 2026 接收） | arXiv API |
| 2026-09 | 编程记忆基准 **CMB** 发布（已投稿 ICLR 2027，仓库内含 PDF） | 仓库 README |
| 2026-10-15 09:52 | `tencent/Hy-Memory-30B-A3B` 上线 HF | HF API |
| 2026-10-15 16:46 | 技术报告 arXiv 2610.22196 v1 | arXiv abs 页 |
| **2026-10-16** | **对外宣布 Hy Memory 2.0"模型/框架/应用三层全开源"** | 证券时报·人民财讯（新浪财经转载） |
| 2026-10-16 07:57 | `-FP8` 上线 | HF API |
| 2026-10-16 13:53 | **MemHub `v1.0.0`** 首个 Release（已公证 macOS arm64 DMG + Linux x64/arm64 Runtime） | Release atom |
| 2026-10-16 16:26 / 18:49 | `v1.0.1` / `v1.0.2` | Release atom |
| 2026-10-17 | `hy-memory 2.0.2 / 2.0.3`、MemHub `v1.0.3` | Release atom / CHANGELOG |
| 2026-10-18 11:14 | **MemHub `v1.0.4`**（restore 一键日志导出、build 13） | Release atom |
| 2026-10-19 07:52 | **`-GGUF`** 上线（imatrix + MoE 混合精度，6 档） | HF API |
| **2026-10-22 12:59** | **`hy-memory 2.0.5`**（`get()` 跨四类记录、显式 raw 检索、embedding 服务注入、生命周期过滤、**默认关闭 Coding diff 注入**、物理删除 fail-closed） | Release atom / CHANGELOG |
| 2026-10-24 → 10-26 | MemHub 1.0.5 build 22→23 修复循环（SDK 权威迁移、记忆地图一致性） | commit feed |

> **注意版本号口径**：根 README 仍写 `pip install hy-memory==2.0.4`，`memhub/README.md` 与 Release 已到 **2.0.5**——**以 2.0.5 为准**（10-22）。

### 2.2 三层交付物

| 层 | 产物 | 形态 | 位置 |
|---|---|---|---|
| **模型** | `Hy-Memory-30B-A3B` | 30B 总参 / **~3B 激活** MoE，专做**记忆写入**；FP8 / GGUF 两个量化族 | HF `tencent/…` |
| **框架** | `hy-memory`（SDK） | Python 包（**≥3.9**，Apache-2.0）；**只需 LLM + Embedding 两个 OpenAI 兼容端点** | PyPI / 仓库根 |
| **应用** | `MemHub` + `memhub-cli` | 本地优先桌面端（macOS arm64/x64）+ Linux Runtime（x64/arm64）+ npm CLI；**Windows 不支持** | 仓库 `memhub/` |

### 2.3 SDK 架构：三场景 + 八层 + 三档模式

**场景路由**（`scene`）：

| scene | 面向任务 | 记忆形态 |
|---|---|---|
| `chat` | 闲聊、个人信息、偏好、一般问答 | 分层记忆 **L0–L7** |
| `working` | 文档/表格/研究/财经分析/内容创作/电脑操作 | **双层**：会话事实 + 可复用经验 |
| `coding` | 编码、调试、测试、构建、API/SDK、脚本、SQL | 同 working 双层 + **代码版本演进链（diff 链）** |

**自动路由的 LLM 成本（README §5，选型时最该看的一段）**：写入不传 `scene` 时，不含工具调用 → 直接 `chat`（**零 LLM**）；含工具调用 → **一次 LLM** 做三分类。检索 `search_mode="default"` 固定走 `chat`（零 LLM），仅 `search_mode="agentic"` 才做场景分类 + query 改写 + 时间解析。

**L0–L7 分层**（`chat`）：`L0 BASIC_INFO` / `L1 RAW`（append-only）/ `L2 FACT`（原子事实，**版本化、不可变**）/ `L3 SUMMARY` / `L4 IDENTITY`（保留）/ `L5 KNOWLEDGE`（保留）/ `L6 SCHEMA`（心智模型，图层）/ `L7 INTENTION`（前瞻意图，带有效期，过期转普通事实）。

**三档模式**：

| 模式 | 能力 | 图库 | 适用 |
|---|---|---|---|
| `lite` | 仅 embedding 写入/检索，不调 LLM；**忽略 `scene` 与 `layers`** | 否 | 纯向量 RAG |
| `pro` | + LLM 抽取 + **reconcile 冲突处理** | 可选（`MEMORY_ENABLE_GRAPH`，默认关） | 标准 |
| `ultra` | + **System 2** 图式归纳 + 跨域 sweeper（`evolve()` 触发） | 是 | 完整认知架构 |

**存储**：向量 Chroma（默认）/ Qdrant / FAISS；图 Kuzu（默认）/ Neo4j；Working/Coding 元数据本地 SQLite。**检索 Reader** 有 `legacy`（纯向量）与 `hybrid`（+BM25），但**在 Chroma 上 hybrid 会退化为纯向量**（Chroma 返回空关键词结果），未知 Reader 名会 warning 回退。

### 2.4 Working / Coding：把"做过的事"与"学到的东西"分开

| 层 | 记录什么 | 每次会话产出 | 结构 |
|---|---|---|---|
| **事实层** `WorkingFact` | 这次会话做了什么 | **恰好 1 条**，不设门槛 | `statement`、`subject`、`files`、`details`（原文留存） |
| **经验层** `Experience` | 下次可直接复用的做法 | **0–N 条**，需过价值门槛 | `task`、`search_keys`、`solution`、`boundary_envs`、`confidence` |

两层互链（`fact.derived_experience_ids` ↔ `experience.source_fact_ids`）；旧事实以 **superseded** 保留并可经 `fact_chain_summary` 回溯。

**Coding 的版本演进链（设计最精巧的一处）**：事实层**刻意不粘代码**（预算留给坐标），而被撤掉的版本既不在工作区也不在 HEAD —— 所以用**零 LLM** 的 diff 链把每一版真实改动存下来；**它是装饰器不是检索器**（向量层选事实，链只在事实下挂改动版本）；来源是本地 Git diff（`MEMORY_CODING_CHAIN_REPO_ROOT`）或工具轨迹里带 `path/old/new` 的确定性替换；粒度是"某分支/某文件/某函数的某一块"，按 `user_id` + `workspace_id` 隔离。**2.0.5 起默认不再注入**，需 `MEMORY_CODING_CHAIN_BLOCK_ENABLED=true`。

> ⚠️ 使用前提：Working/Coding 写入**必须带工具调用轨迹**（`tool_calls` 或 `role=tool`），且只在 `pro`/`ultra` 生效。

### 2.5 记忆写入模型：`Hy-Memory-30B-A3B`

| 项 | 事实 | 来源 |
|---|---|---|
| 规模 | **30B 总参 / ~3B 激活** MoE（HF 标签 `hy_v3`，即 Hy3 架构栈）；学生 = Hy3-30B-A3B 指令微调 ckpt | 论文 §4.2 / HF 标签 |
| 角色 | **直接替换** Mem0 / MemOS / Hy-Memory 里的通用 LLM，只训 writer、**reader 冻结** | HF 模型卡 / 论文 |
| 教师 | **Hy3 旗舰模型**，仅前向、返回 top-50 log-prob | 论文 §4.2 |
| 阶段 1 | **Memory OPD**：top-50 近似 reverse KL，lr `1e-6` 恒定，entropy `0.005`，prompt/response 上限 4,096 / 126,976 token，每行 2 rollout | 论文 §4 |
| 阶段 2 | **Memory RL**：每 window 8 条 rollout、chunk 级 GRPO；奖励含 per-chunk recall + attributed outcome + **bank-growth 惩罚**；RL 从 OPD **step 375** 接续 | 论文 §5–§6 |
| 覆盖框架 | Mem0 / MemOS / Hy-Memory（Chat·Working·Coding），**留出 EverMemOS** 做 OOD | 论文 §8.3 |
| 部署 | vLLM（tested **v0.31.0**）+ OpenAI 兼容；`reasoning_effort=no_think`、`temperature=0` | HF 模型卡 |
| 许可 | Apache-2.0 | 模型卡 |

**跨 harness 结果**（模型卡 Table 1；4 harness × 6 指标等权，固定 **20% 子采样**）：

| Memory writer | Mem0 | Hy-Mem. | MemOS | EverMem.（held-out） | 平均 |
|---|---:|---:|---:|---:|---:|
| **Hy-Memory-30B-A3B** | **54.02** | **60.55** | **50.51** | **56.89** | **55.49** |
| Claude Opus 5 | 53.43 | 60.53 | 50.50 | 54.04 | 54.62 |
| DeepSeek V4.1 Flash | 49.73 | 60.20 | 49.17 | 55.47 | 53.64 |
| Hy3 | 45.99 | 58.06 | 50.19 | 52.30 | 51.63 |
| Hy3-A3B Base | 40.20 | 50.39 | 49.05 | 50.98 | 47.66 |

> **必须看清**：① 固定 20% 子采样，不是全量；② `Shopping PS` 全体落在 16–24 分，对均值区分度低；③ 领先 **0.87 分**。**"记忆写入超 Opus 5"成立，"通用能力超 Opus 5"不成立。**

### 2.6 框架层基准（媒体转述官方自评，**未经第三方复测**）

IT之家 10-16 详版（<https://www.ithome.com/1/014/108.htm>）给出的自评数字，口径与 HF 模型卡不同（这是**框架层**指标，不是记忆写入模型）：

| 场景 | 指标 | 数字 |
|---|---|---|
| 工作记忆 | MemoryArena（旅行规划 + 组合购物）平均分 | **55.31%**，领先次优 EverOS **10.2 个百分点** |
| 代码记忆 | CodeMemoryBench 代码问答 / 任务解决 | **64.2%** / **35.6%** |
| 代码记忆 | SWE-ContextBench 任务解决率 | **28.3%**（MemOS / Mem0 为 25.3%） |
| 聊天记忆 | LoCoMo 准确率 | **72.6%**（EverOS 71.9%） |
| 聊天记忆 | 每题 token | **1,159 → 703（−39.3%）** |
| 对比对象 | Mem0、MemOS、EverOS、OpenViking、RepoMemory、Simple RAG | 宣称"三类场景 4 个 Benchmark 均第一" |

> **交叉校验**：模型卡里 Hy-Memory harness 的 LoCoMo 是 **73.28**，媒体这里写 **72.6** —— 两处是**不同评测口径**（前者是六指标矩阵的固定 20% 子采样，后者是框架层单基准），不要混用。
> **LongMemEval 无证据**：论文把 LongMemEval 只当动机/相关工作引用，**官方公布的 4 个基准是 MemoryArena / CodeMemoryBench / SWE-ContextBench / LoCoMo**；本窗口内**没有**任何 Hy Memory 2.0 的 LongMemEval 成绩。

### 2.7 `MemHub`：把记忆做成跨宿主产品

| 项 | 事实 | 来源 |
|---|---|---|
| 一句话 | **本地优先**的共享记忆：把用户**显式授权**的 AI 对话与编码记录汇入**一份本地记忆**，再由用户**分别授权**给各宿主召回 | memhub README |
| 设计原则 | 一次安装、一个 Runtime、一份数据、多种客户端；SDK 是 memory/raw/vector/graph/coding 数据面的**唯一权威** | memhub README |
| 三条独立授权边界 | ①来源读取 ②模型处理 ③宿主读取 | memhub README |
| **不做**的采集 | 屏幕、Accessibility/AXTree、OCR、键鼠、剪贴板、后台录音 | memhub README |
| 组件 | `apps/core`（FastAPI Kernel）、`apps/cli`（`one-memory`）、`apps/web`（React/TS）、`apps/desktop`（Electron）、`packages/host-bridge`（Skill/Hook/插件/MCP）、`packages/runtime-control`、`packages/contracts`、`packages/i18n` | memhub README |
| 运行形态 | Runtime 由系统服务托管（**关 UI 不停**）；SQLite 数据默认在 **`~/.one-memory`**；npm `memhub-cli` 只装**约 384 KB 的 Bootstrap CLI**，`memhub install` 才拉完整 Runtime | memhub README |
| 平台 | Linux x64/arm64 Runtime；macOS arm64/x64 App（Developer ID 签名 + 公证）；**Windows 不支持** | memhub README |
| 宿主覆盖 | README 列 `Claude Code / Cursor / Codex / OpenCode / Qwen Code`；IT之家称覆盖 **10 种**（含 `WorkBuddy / Pi / Hermes / DeepSeek Harness`）；插件目录实有 `hermes`、`openclaw` | ⚠️ 两处不一致，见 §7 |

**插件语义**：`hermes` 是**原生 MemoryProvider**（显式 owner/agent 隔离、独立 HTTP 服务、**自动采集与召回默认关闭**）；`openclaw` 提供 `openclaw.plugin.json` + TS provider/tools + CLI + Python 环境探测（含测试）。

### 2.8 治理与工程细节（`CHANGELOG.md` 里最值得抄的部分）

- **写入幂等 + 可恢复**：durable idempotent write outcomes；"写入后权威校验失败"进**只读对账**，**不会自动重调模型**（避免二次幻觉写入）。
- **删除 fail-closed**：物理删除若 VDB/Graph 只清一半，**报失败**。
- **迁移安全**：legacy → SDK 权威迁移**不复制记忆**、不重复抽取；raw 迁移只重建最近三天有界 turn 并排除 tool 消息。
- **快照/恢复**：quiesced 本地数据面 snapshot/restore；离线导入/导出。
- **时间戳坑**：Cursor 的 WAL change token 与文件 mtime 分离，避免"哈希被当成几个世纪后的会话时间"。
- **多语言**：中/英/西三套 catalog，界面语言与记忆内容语言解耦。

### 2.9 与 Agent 方向的关系

| 用法 | 说明 |
|---|---|
| 给**第三方编码 Agent** 加长期记忆 | 写入时用 `source` 标注宿主（`openclaw` / `cursor` / `codex`），检索时用 `include_sources` / `exclude_sources` 过滤；与 `agent_id` 正交实现宿主隔离 |
| 给**自己的 Agent 服务**加记忆 | 只要两组 OpenAI 兼容配置（LLM + Embedding）；示例里直接用 `deepseek-chat` + `text-embedding-v3`，**不绑定混元模型** |
| 复现"记忆写入"研究 | 模型 + 技术报告 + 四 harness 口径 + CMB 基准全部公开 |
| 直接当产品 | MemHub（macOS/Linux）；⚠️ 无 Windows |

> **战略观察**：混元把记忆做成**宿主无关**的 SDK + 桌面产品，接入的却是 Claude Code / Cursor / Codex / OpenCode / Qwen Code —— 即"**在别人的 Agent 上卖记忆**"，与 `UniRL`（训自己的策略）、`WorkForge`（造自己的环境）是三条不同的生态路径。

---

## 3. 核心增量 B：`WorkForge` —— 长程工作 Agent 的可验证环境合成

> **这是什么**：`arXiv:2610.04906`《Scaling Verifiable Environments for Long-horizon Work Agents》。**不是仓库**（`Tencent-Hunyuan` 组织下至今没有 `WorkForge` 仓库），而是论文 + 2026-10-22 的官方发布。

- 论文：<https://arxiv.org/abs/2610.04906>（v1 **10-04**、v2 10-08 撤稿、v3 **10-13**、v4 **10-21**）
- 官方口径：`Tencent Hunyuan` 的 X 帖（媒体逐字引用：<https://www.techflowpost.com/zh-TW/newsletter/140968> 给出原帖 `x.com/TencentHunyuan/status/2113195029233811511`）；公众号原文标题《大模型如何胜任职场？WorkForge 让大模型进入真实职场工作流》（<https://mp.weixin.qq.com/s/GYyQ5fS4HMBK8qoNcPJlng>，**正文不可读**）

**方法（摘要原文要点）**：

1. 从**专家工作流**出发，识别每个流程需要的资源、决策与交付物；
2. 检索**真实世界文件**并组织成 workspace；
3. 扫描 workspace 抽取**可核验事实锚点（factual anchors）**——锚点决定"这个 workspace 能支持什么任务类型、结果怎么验"；
4. 由锚点**派生**任务指令、解题计划，以及**程序化 + 语义双验证器**，使验证始终可追溯到可观测的 workspace 证据。

**规模与效果（摘要照抄）**：

| 项 | 数字 |
|---|---|
| 环境数 | **16.7K** 个可验证环境 |
| 领域 | **40** 个专业领域 |
| 文件类型 | workspace 合计覆盖 **60** 种 |
| 后训练收益 | `Qwen3.5-35B-A3B-Base`：**GDPVal 45.5 → 73.6**，**APEX Score 5.0 → 21.3** |
| 附带结论 | `Qwen3.5-27B` 取得"极具竞争力"表现并超过强对手；在**数据量**与**交互时长**两个维度都观察到一致的 scaling |

**归属与可信度（必须写清）**：

- arXiv **元数据没有作者单位**；作者 26 人，含复旦团队（**Tao Gui / Qi Zhang / Xuanjing Huang**）与 `Clarenceai`。
- "腾讯混元出品"的依据是**官方 X 帖 + 多家媒体表述**（深潮 TechFlow、C114），不是一手元数据 → 本报告按 **中高** 处理。
- ⚠️ 网上流传的"42.3 个文件 / 279KB / 真实工作区文件数是合成的 5.4 倍 / 27B GDPval 82.4 分 / 示范轨迹由 GLM 5.3 + Claude Code + OpenClaw 生成"等**加分细节，均不在 arXiv 摘要内**，且出自单一低可信内容农场（`renrendaima.com`），**本报告一律不采信**。

**为什么它属于 Agent 主线**：10-12 报告说混元"评测多、框架少"，但真正的瓶颈是**环境**——没有可验证的长期交互环境，agentic RL 只能在少数手写沙箱里打转（`UniRL` 的 `SandboxTool`/`SearchTool`/`VisitTool` 正是这种手写沙箱）。WorkForge 把"环境"变成**可合成的数据资产**，思路与 `ExplorationBench`（可验证异世界）、`DiagSafe`（rubric 化判定）同源：**先让结果可验，再谈训练**。

---

## 4. 既有 Agent 仓库与平台侧的增量

### 4.1 `UniRL`：v0.2.0 打版（10-19），但窗口内提交零 Agent 改动

**Release 事实**（`releases.atom`，一手）：

| tag | 时间（UTC） | 说明 |
|---|---|---|
| `v0.1.0` | 2026-10-07T03:47 | 首次投放（**窗口外**） |
| **`v0.2.0`** | **2026-10-19T07:01** | 首次打版，折叠 **~350 个已合并 PR** |

v0.2.0 发布说明的四个主题（原文照抄要点）：

1. **模型覆盖大扩**：**12 个新 model package**（Z-Image、Qwen-Image-Edit-2511、Boogu-Image、LTX-2 / LTX-2.3、WAN 2.2 V2V、MiniMax-H3、SenseNova-U1.5、Janus-Pro、Qwen3.5、Qwen3-MoE(VeOmni EP)、Qwen3-Omni Thinker、Cosmos3），外加 BAGEL / HunyuanImage 3.0 新模式；
2. **新增训练模式**：SFT（`train_sft`）、**多轮 agentic RL（`train_agentic`）**、AR 与 diffusion 的分离式异步 RL（`train_async_ar` / `train_async_diffusion`）；
3. **算法**：CPPO、**AR DPPO**、GSPO、带 GAE + value head 的 PPO、DiffusionOPD（on-policy 蒸馏）、DiffusionNFT 的 reference-model KL；
4. **rollout 栈与性能组件**：重建的 SGLang Diffusion 与 vLLM-Omni 引擎、新的直连 vLLM TP 引擎、FastVideo 引擎、全权重与 LoRA 的 TP/EP 权重同步、统一批式扩散 replay（`BatchedStepReplayMixin`）、跨阶段 text-embedding 去重 + CPU LRU、带守卫的 CUDA MPS reward worker、全链路 fail-closed 类型契约 —— **底座为 SGLang 0.5.19 / vLLM-Omni 0.28 / torch 2.13 + CUDA 13**。

**但窗口内的提交不含 agentic 改动**（`commits/main.atom`，10-16 → 10-19 共 20+ 条）：

| 主题 | 代表提交 |
|---|---|
| 扩散/PE replay 批处理 | `feat(diffusion): batch video replay timesteps (#394)`、`perf(bagel): pack FlowGRPO replay steps (#384)`、`perf(minimax-h3): batch replay timesteps (#500)`、`feat(flowgrpo): add bounded-memory chunked replay backward (#465)` |
| 文本条件缓存 | `perf(models): deduplicate in-batch prompts…`、`perf(models): share bounded CPU LRU text-embed cache…`、`feat(hunyuan_video15): in-batch text-embed dedup…` |
| 训练侧内存 | `perf(models): skip loading text encoders and VAEs on trainer side… (#619)` |
| 正确性/守卫 | `fix(lora): reject unsupported trainable LoRA bias settings (#599)`、`fix(vllm-omni): fail closed in runtime patches…` |

> **结论要点**：`train_agentic` 的**能力**在 10-12 基线时已可从 README 读到（当时 News 已写 v0.2.0）；窗口内发生的是**打版与发布叙事**，**不是能力新增**。要评估 agentic RL 效果，仍以 10-12 报告的机制描述（每 turn 一个 `Sample`、group-normalized advantage、失败轨迹剔除、`overflow`/`failed` 语义）为准。

### 4.2 `ExplorationBench`：News 补登 + 两次官方曝光

README diff 把原来一条笼统的 `2026.09` 新闻拆成四条：

```
- **2026.10.13**: 📣 Tencent AI features ExplorationBench on X.   (x.com/TencentAI_News/status/2109904850708512825)
- **2026.10.08**: 💻 This repository goes public with the evaluation code and the two public demo worlds.
- **2026.09.30**: 📣 Tencent Hy announces ExplorationBench on X.  (x.com/TencentHunyuan/status/2105288808929558891)
- **2026.09.24**: 🎉 The paper is released on arXiv — two verifiable Alien Worlds, 55 discovery targets, 140 held-out tasks, results for 10 frontier AI systems.
```

代码未变（仍是两个 demo world + `analysis/` 复现脚本），stars 7 → 27（+20）。**10-13 那条是本窗口内 Agent 评测仓库唯一的官方传播动作**。

### 4.3 `CL-bench`：只动评测口径

唯一提交 `2026-10-17 eval: reword judge prompt opening as impartial evaluator`（judge prompt 开头改成"公正评估者"）。属复现性维护；该仓库在 10-12 报告中被列为 out-of-scope。

### 4.4 平台侧：一个月内两款 Agent 产品退场，出口收敛

| 事件 | 时间 | 关键数字 | 来源 | 可信度 |
|---|---|---|---|---|
| **元器（Yuanqi）停服公告** | 通知 **10-08**（报道 10-15） | **11-09 正式停服**，存量智能体不可用；11-09→2027-01-09 仅可登录导出备份；**2027-01-09** 起依法删除数据；官方迁移落点 **腾讯云智能体开发平台（ADP）**（2025-05 上线，由大模型知识引擎升级而来）。元器 2024-05-17 发布，2025-08 升级 v1.0（多智能体协作、复杂任务规划） | 经济观察网转 21 世纪经济报道 | 中高（一手通知被逐条引述，**原件未取到**） |
| **QClaw 停运公告** | 09-24 公告（窗口外） | **2026-12-24 00:00** 停止运营，数据可迁 **WorkBuddy**；停运时月活约 **225.9 万** | 同上 | 中高 |
| **企业微信 × WorkBuddy 双向互通** | **10-14** | 企业微信 **5.0.12** / WorkBuddy **5.8.0**；WorkBuddy 进企微通讯录、打通账号体系，企业授权范围内连接消息/文档/会议/网盘等**十大能力**；企微覆盖 **1400 万+ 企业组织**、日服务约 **7.5 亿**微信用户；WorkBuddy 自 2026-03 发布以来 **50+ 版本**、落地 **50+ 行业**；腾讯 2026 Q1 财报口径"按日活计中国最受欢迎效率 AI 智能体"；易观：6 月 PC 端 AI 原生办公智能体月访问量破 **2000 万** | 上海证券报·中国证券网（搜狐转载） | 中高 |

> **怎么读**：这不是"混元 Agent 大扩张"，而是**内部赛马后的整合**——入口从"元器（C 端创造平台）/QClaw/WorkBuddy 三线"收敛为"**WorkBuddy + 企业微信 + 腾讯云 ADP**"。对技术选型的含义：**要做混元系 Agent 落地，ADP 与 WorkBuddy 是官方指定承载面**，元器上的存量智能体有明确迁移死线。
> **注意**：WorkBuddy 是混元 Hy 系列的主要分发渠道（`Hy4-preview` 于 2026-08-28 在 WorkBuddy 上线），但 10-12 报告里它不在 `Tencent-Hunyuan` 组织内 —— 属"业务侧 Agent 产品"，本报告只作落地证据，不计入仓库统计。

### 4.5 传闻（**不建议引用**）

"腾讯元宝在做 Personal Agent 独立 App（'元宝助手'），给独立运行空间与权限，承载长周期任务与个人长期记忆" —— 来源 IT之家 10-12 转《读佳》引"知情人士"，**腾讯未确认**，无可核产品页。仅作方向注脚：与 Hy Memory 2.0 的"长期记忆"叙事同频，但**不是事实**。

---

## 5. 同期新增但**不属于 Agent 主线**

| 仓库 | 一句话 | 与 Agent 的关系 |
|---|---|---|
| `DiagSafe`（10-15，CC-BY-4.0，arXiv **2610.29225**，10-21） | 诊断式 LLM 安全评测：**3,489 条** prompt（DiagBase 2,401 / DiagAttack 1,088）、11/34/84 三级标签（由 204 条中美欧公开安全条款提炼）、单轮 2,560 + 多轮 929；**prompt-level rubric**（平均 4.5 条 + 8 条有用性） | **不是 Agent 基准**，但它给了一张"新模型安全体检表"，且**用 Hy3 作 rubric 引导的 judge** |

`DiagSafe` 的新模型 unsafe rate（%）与有用性（0–3），照抄 README（不加 system prompt、相同解码配置）：

| 模型 | 整体 ↓ | DiagBase ↓ | DiagAttack ↓ | 有用性 ↑ |
|---|:---:|:---:|:---:|:---:|
| Qwen3.8-27B | 8.5 | 3.3 | 20.2 | 2.23 |
| **Hy4-Preview** | **11.9** | **3.9** | **30.6** | **2.27** |
| GLM-5.3（低推理强度） | 13.1 | 9.5 | 20.9 | 2.37 |
| DeepSeek-V4.1-Flash | 18.5 | 5.8 | 47.5 | 2.14 |
| Kimi-K3 | 26.8 | 14.9 | 53.5 | 2.31 |

可靠性：与人工标注一致率最高 **95.34%**（安全分类器基线 69%–87%），跨裁判一致性 **Cohen's κ 0.828–0.864**；DiagAttack 下**每个模型**不安全率升至 DiagBase 的 **1.9–8.4 倍（中位 5.3×）**，11 个模型里 **8 个超 40%**。

> **对 Agent 的直接含义**：把有害意图包进"正常任务外壳"（指令攻击）就能显著削弱拒答 —— **工具调用/指令遵循越强的 Agent 化模型，这个面越大**。这与 10-12 报告里 `C3-Benchmark` 关注的"工具链鲁棒性"是同一枚硬币的两面。

其余三个（`OmniCapBench`、`HYDRA`、`Hy-Nomas`）见 [`../../infer/2026-10-26/06-infer-world-model-and-serving.md`](../../infer/2026-10-26/06-infer-world-model-and-serving.md)。

---

## 6. 公开信息时间线（Agent 侧）

| 日期 | 事件 | 来源类型 | 链接 |
|---|---|---|---|
| 2026-10-12 | 传闻：元宝在做 Personal Agent App"元宝助手"（未证实） | 媒体转述 | <https://www.ithome.com/1/012/026.htm> |
| 2026-10-14 | 企业微信 5.0.12 与 WorkBuddy 5.8.0 双向互通（混元系 Agent 进内部业务） | 财经媒体原创 | <http://biznews.sohu.com/a/1087595267_122014422> |
| 2026-10-15 | 元器停服一事被报道（公告 10-08，停止服务 11-09，迁腾讯云 ADP） | 财经媒体原创 | <http://www.eeo.com.cn/2026/1015/1065002.shtml> |
| 2026-10-15 | `Hy-Memory-30B-A3B` 上线 HF；技术报告 arXiv（16:46 UTC） | HF / arXiv | <https://huggingface.co/tencent/Hy-Memory-30B-A3B> · <https://arxiv.org/abs/2610.22196> |
| **2026-10-16 17:39** | 混元宣布 **Hy Memory 2.0**：模型/框架/MemHub 三层全开源；已在金融、资讯、知识管理、办公 Agent、社交娱乐、游戏等内部业务使用 | 证券时报·人民财讯 → 新浪财经 | <https://finance.sina.com.cn/jjxw/2026-10-16/doc-inivmiif7708191.shtml> |
| 2026-10-16 18:18 | IT之家详版：三层解释 + MemoryArena/CodeMemoryBench/SWE-ContextBench/LoCoMo 自评数字 + MemHub 支持 10 种工具 | 科技媒体 | <https://www.ithome.com/1/014/108.htm> |
| 2026-10-16 | 微信公众号原文《Hy Memory 2.0 发布：模型、框架、应用三层全开源》 | **公众号**（正文验证码） | <https://mp.weixin.qq.com/s/TT70I3zzMSLAae5LSPeMzQ> |
| 2026-10-16 13:53 | MemHub `v1.0.0` 首个 Release（macOS 公证 DMG + Linux Runtime） | Release atom | <https://github.com/Tencent-Hunyuan/Hy-Memory/releases> |
| 2026-10-18 11:14 | MemHub `v1.0.4` | Release atom | 同上 |
| **2026-10-19 07:01** | **UniRL `v0.2.0`** 打版（~350 PR；含 `train_agentic` 等新模式） | Release atom | <https://github.com/Tencent-Hunyuan/UniRL/releases/tag/v0.2.0> |
| 2026-10-19 | `Hy-Memory-30B-A3B-GGUF`（imatrix + MoE 混合精度，6 档） | HF | <https://huggingface.co/tencent/Hy-Memory-30B-A3B-GGUF> |
| **2026-10-22** | **WorkForge 官方发布**（X 帖 + 深潮/C114 报道）；论文 v4 已于 10-21 修订 | 官方 X（媒体逐字引用）/ 媒体 | <https://www.techflowpost.com/zh-TW/newsletter/140968> · <https://www.c114.net.cn/ainews/137064.html> |
| 2026-10-22 | 公众号原文《大模型如何胜任职场？WorkForge 让大模型进入真实职场工作流》 | **公众号**（正文验证码） | <https://mp.weixin.qq.com/s/GYyQ5fS4HMBK8qoNcPJlng> |
| 2026-10-22 12:59 | `hy-memory 2.0.5` | Release atom / PyPI | <https://pypi.org/project/hy-memory/2.0.5/> |
| 2026-10-23 | 台湾媒体转述官方 X：Hy-Nomas 强调 "world memory" 与"面向机器人等领域的轻量 LoRA 适配" | 媒体 | <https://shiftdelete.net/tencent-hunyuan-hy-nomas> |
| 2026-10-24 | 沙丘社区把 Hy-Nomas 归入"模型架构 / 模型推理 / **智能体系统**"三标签 | 第三方社区 | <https://www.shaqiu.cn/tracking/model%3Atencent%3Ahy-nomas-public-weights-20261024> |
| 2026-10-24 | 日文报道称 Hy-Memory-30B-A3B "長期記憶で Opus 5 超え" | 媒体（日） | <https://pc.watch.impress.co.jp/docs/news/2148974.html> |

> **可信度分级**：可核的官方口径只有「HF 模型卡 / GitHub 仓库 / Release atom / arXiv」四类 + 证券时报与 IT之家的**短讯级转述**；**两条公众号正文与全部 X 帖原文都没取到**，本报告不基于它们的正文内容做任何结论。

---

## 7. 与 10-12 基线结论的对照

| 10-12 结论 | 本轮（10-26）证据 | 判定 |
|---|---|---|
| **"评测先行、框架稀缺"**（6 评测 vs 1 框架） | 评测仍 6 个（无新增 Agent 基准；`DiagSafe` 是安全基准）；框架 **+1 记忆框架**、**+1 环境合成方法（WorkForge）**；通用训练/运行时框架仍只有 `UniRL` | **部分修订** —— 缺口从"框架"变成"**运行时框架仍稀缺，记忆层与环境层已补**" |
| **"具身智能是独立重线"** | 5 个具身仓库窗口内**零推送**；唯一相关动作是 `Hy-Nomas` 的 **Robotics LoRA**（RealSource-World 真实机械臂数据，用于生成世界的机器人仿真） | **成立，本轮无增量**；勿把 Hy-Nomas 误归具身模型 |
| **"评测方法论向可验证演进"** | 新增 `DiagSafe`（prompt-level rubric + 逐条证据）、`OmniCapBench`（确定性验证 + 局部字段比对、不报总分）、**WorkForge（事实锚点派生程序化 + 语义验证器）** | **被强化** —— "可归因、可复算"已是混元评测/环境的默认范式 |
| **Agentic RL 细节（UniRL）** | 代码与 README 未变；只有 release 打版 | **继续有效** |
| **"Stars 与能力脱钩"**（infer 报告同源观察） | `ExplorationBench` +20★ / `Hy-Memory` 1,282★ 但仅 4 天；`UniRL` +14★ 且无 agentic 改动 | **继续有效** |

**本轮新增判断**：

> **⑤ 混元的 Agent 战略从"测量能力"扩到"提供基础件（记忆 + 环境）"，但刻意不碰"Agent 运行时"这个位置。**
> 记忆（Hy-Memory）是**跨宿主**的横切能力，环境（WorkForge）是**跨模型**的训练资产 —— 两者都不绑定具体 Agent 框架，因此可以同时服务第三方宿主（Claude Code/Cursor/Codex）与自家模型。代价是**放弃了运行时调度权**，收益是分发成本最低。
> **同时**，C 端 Agent 入口在收缩（元器/QClaw 退场），说明"做平台"这条路在内部已被判定性价比不足，资源转向"做能力 + 做业务落地（WorkBuddy/企微）"。

---

## 8. 风险与待核实

| # | 事项 | 现状 | 建议 |
|---|---|---|---|
| 1 | "记忆写入超 Opus 5" | 固定 20% 子采样 + 六指标等权 + 四 harness 等权，均值差 0.87 | 引用必带口径，勿外推为通用能力 |
| 2 | **框架层自评数字**（MemoryArena 55.31% / CMB 64.2%·35.6% / SWE-ContextBench 28.3% / LoCoMo 72.6% / token −39.3%） | 仅见于 IT之家转述官方，**无第三方复测**；且与模型卡口径（LoCoMo 73.28）不同 | 当作"官方自评"引用，标注来源层级 |
| 3 | **LongMemEval 缺失** | 官方 4 基准里没有 LongMemEval；论文只把它当动机引用 | 不要写"LongMemEval 超过 X" |
| 4 | **MemHub 宿主清单不一致** | README 列 5 个；IT之家称 10 个（+WorkBuddy/Pi/Hermes/DeepSeek Harness）；插件目录只有 `hermes`/`openclaw` | 以 `packages/host-bridge` 与 release manifest 为准，README 会滞后 |
| 5 | **SDK 版本号不一致** | 根 README `2.0.4`、memhub README / Release `2.0.5` | 以 **2.0.5**（10-22）为准 |
| 6 | **WorkForge 归属** | arXiv 无单位元数据；作者含复旦（Tao Gui/Qi Zhang/Xuanjing Huang）与 `Clarenceai`；"腾讯混元"来自官方 X + 媒体 | 引用时写"腾讯混元官方发布（arXiv 未标单位）" |
| 7 | **WorkForge 的"加分细节"** | "42.3 文件/279KB/5.4 倍/27B GDPval 82.4/GLM 5.3+Claude Code+OpenClaw 生成轨迹"等**仅出自内容农场** | **一律不采信**；以 arXiv 摘要与 HTML 版表格为准 |
| 8 | **元器停服公告原件** | 通知日期/停服日/数据删除日均引自 21 世纪经济报道，未取到官方公告页 | 迁移项目请以官方 ADP 迁移指引为准 |
| 9 | **"元宝助手"传闻** | 单一来源 + "知情人士" + 官方未确认 | 不写进结论 |
| 10 | **公众号正文** | 两条均被验证码拦截；`r.jina.ai` 代理亦失败 | 只能证明"存在这两篇公众号文章"，不能引正文 |
| 11 | **Windows 不支持** | MemHub 仅 macOS/Linux | 若团队用 Windows，需自建 Linux Runtime |
| 12 | **Coding diff 链默认关闭** | `2.0.5` 起需 `MEMORY_CODING_CHAIN_BLOCK_ENABLED=true` | 升级行为变化要进变更说明 |
| 13 | **GitHub API 限流** | 未认证 60 次/小时，本轮中途 403；PS 5.1 `ConvertFrom-Json` 亦因深度失败 | 已改走 atom/raw/HF API；全量元数据建议带 token |
| 14 | 时间错位纠错 | 网传"混元世界模型 1.1（WorldMirror）2026-10-22/23 开源"实为 **2025-10** 旧闻（IT之家原文标注 2025-10-22；HF `HunyuanWorld-Mirror` createdAt 2025-10-16） | **不要写进 2026-10 时间线**（详见 infer 报告 §9） |

---

## 9. 证据来源与采集脚本

| 产物 | 位置 |
|---|---|
| 组织全量快照（90 仓库） | [`../../_research/2026-10-26/org-repos-2026-10-26.csv`](../../_research/2026-10-26/org-repos-2026-10-26.csv) |
| 增量表 | [`../../_research/2026-10-26/delta-2026-10-26.csv`](../../_research/2026-10-26/delta-2026-10-26.csv) |
| 提交 feed（11 仓库） | [`../../_research/2026-10-26/commits/`](../../_research/2026-10-26/commits/) |
| Release feed | [`../../_research/2026-10-26/unirl-releases.atom`](../../_research/2026-10-26/unirl-releases.atom) |
| HF 模型清单 / 模型卡 | [`../../_research/2026-10-26/hf-tencent-models.json`](../../_research/2026-10-26/hf-tencent-models.json) · [`hf-cards/`](../../_research/2026-10-26/hf-cards/) |
| arXiv 元数据 | [`../../_research/2026-10-26/arxiv-batch.xml`](../../_research/2026-10-26/arxiv-batch.xml) |
| README 快照与 diff | [`readmes/`](../../_research/2026-10-26/readmes/) · [`readme-diff.txt`](../../_research/2026-10-26/readme-diff.txt) |
| 本机克隆 | `_probe/Hy-Memory/`（`README.zh.md`、`CHANGELOG.md`、`docs/`、`memhub/`、`plugins/`）、`_probe/UniRL/`、`_probe/ExplorationBench/` |

**关键链接**：<https://github.com/Tencent-Hunyuan/Hy-Memory> · <https://memory.hunyuan.tencent.com/> · <https://huggingface.co/tencent/Hy-Memory-30B-A3B> · <https://arxiv.org/abs/2610.22196> · <https://arxiv.org/abs/2606.09483> · <https://arxiv.org/abs/2610.04906>（WorkForge） · <https://github.com/Tencent-Hunyuan/UniRL/releases/tag/v0.2.0> · <https://github.com/Tencent-Hunyuan/DiagSafe> · <https://github.com/Tencent-Hunyuan/ExplorationBench>
