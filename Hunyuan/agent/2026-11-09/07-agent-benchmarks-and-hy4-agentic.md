# 混元 Agent 侧增量扫描 · 2026-10-26 → 2026-11-09

> **本次任务**：在既有基线（[`INDEX.md`](../../INDEX.md) + [`agent/2026-10-26/05-agent-memory-and-delta.md`](../2026-10-26/05-agent-memory-and-delta.md)）之上，扫描 **2026-10-26 之后**混元在 Agent 方向的新增仓库、代码变更、Hy4 正式版的 agent 能力与公开信息。
>
> **窗口**：`2026-10-26T10:30+08:00` → `2026-11-09T11:20+08:00`（约 14 天）。  
> **基线**：`_research/2026-10-26/org-repos-2026-10-26.csv` 的 **90 个仓库**快照（Agent 15 / Infer 59 / 其余 16）。  
> **本次实测**：组织现有 **95 个公开仓库**（+5）；窗口内被推送的既有仓库 **12 个**；**新仓库 4 个**（Agent 侧 2 个）+ **1 个补录**（Hy-SimulMT，内容在上轮扫描 2 小时后公开）；另有**本轮最重要的平台级事件：Hy4 正式版于 11-04 发布**（GitHub/HF 侧零动静，产品/API 侧全量上线）。

**数据来源标注约定**（沿用前两轮）

| 标记                    | 含义                                                                  | 可信度          |
| --------------------- | ------------------------------------------------------------------- | ------------ |
| **仓库实测**              | 本机读文件 / `raw.githubusercontent.com` 原文 / `gh api`（本轮起认证后不再受限流约束）    | 一手           |
| **GitHub API / atom** | commits / releases（客观时间戳；本轮用认证 API 取得 `published_at`）               | 一手           |
| **HF API / 模型·数据集卡**  | `huggingface.co/api/{models,datasets}?author=tencent`               | 一手           |
| **arXiv API**         | `<published>` / `<updated>` / 摘要原文                                  | 一手（无单位元数据）   |
| **本地代码实测**            | `D:\WorkSpace\vllm-ascend-hy4\repo`（商发分支 hy4-v0.30.0）内直接读文件与 git 历史 | 一手（内部制品）     |
| **官方公告（媒体逐字引用）**      | 人民网 / 央广网 / 财联社·科创板日报逐字转述发布口径                                       | 中高           |
| **媒体转述 / 自媒体**        | 二手汇总，可能加料                                                           | 中→低，**单独标注** |
| **公众号**               | `mp.weixin.qq.com` 正文被验证码拦截                                         | 仅"存在性"级      |
| `README 未说明`          | 原文没有，不做推断补全                                                         | —            |

---

## 0. 一句话结论

10-26 报告的判断是"**运行时框架仍稀缺，记忆层与环境层已补**"。本轮三件事：

1. **Hy4 正式版（11-04）把 agent 能力推向旗舰主线**：首次引入**视觉理解 + Computer Use**，官方称 WorkBuddy 场景平均 83.18 分 / 任务通过率 96.15% / 12 场景 9 个国内第一，幻觉率 12.95%→7.2%。**但这是 API/产品侧发布：截至 11-09，GitHub 仓库与 HF 权重均无增量**——"产品先行、权重后置"是新节奏（详见 infer 报告 §2）。
2. **评测侧一次补进两个"程序化 ground truth"基准**：`HyperLogic`（10-28 公开，答案由执行形式模型导出 + Hard agent sandbox 工具链）与 `AppCIF-Bench`（11-05，Prompt-as-Environment + 6 个可执行代码仿真引擎）。上轮"评测向可验证演进"的判断在本轮被**以最强的形式**强化——两个新基准的核心卖点都是"答案可复算"，且都采用了防污染设计（30% 公开子集 / v1.1 修订声明）。
3. **Hy-Memory 进入高频产品化迭代**：窗口内 295 条提交，SDK `2.0.5 → 2.0.14`（9 个版本），MemHub `1.0.5 → 1.0.6`；stars +730 且**本轮有真实的代码迭代支撑**（上轮"热度与代码脱钩"的判断在它身上反转）。

---

## 1. 增量总览（Agent 相关）

### 1.1 组织级增量（95 = 90 + 5）

| 仓库               | 类型     | created / 公开                             | stars（现） | License                             | 分类         | 一句话                                           |
| ---------------- | ------ | ---------------------------------------- | -------: | ----------------------------------- | ---------- | --------------------------------------------- |
| **HyperLogic**   | **新**  | 10-27 建 / 10-28 公开                       |        1 | Apache-2.0                          | ⭐ Agent 评测 | 执行导出答案的中文逻辑推理基准 + Hard agent 工具链              |
| **AppCIF-Bench** | **新**  | 11-05                                    |        1 | 代码 Apache-2.0 / **数据 CC-BY-NC-4.0** | ⭐ Agent 评测 | 应用级复杂指令遵循，Prompt-as-Environment（NeurIPS 2026） |
| Hy-SimulMT       | **补录** | 10-22 建 / **10-26 12:53 导入**（上轮扫描后 2 小时） |      263 | API 报 NOASSERTION                   | Infer      | 实时同传模型（见 infer 报告）                            |
| EmoReward        | **新**  | 11-02                                    |       38 | API 报 NOASSERTION                   | 其余         | 情感视频生成 RL 的奖励校准                               |
| FlashTopK        | **新**  | 11-05                                    |       10 | Apache-2.0                          | Infer      | 稀疏注意力 fused top-k 内核（见 infer 报告）              |

**分类口径更新**（建议写回 `INDEX.md`）：Agent **15 → 17**（+`HyperLogic`、`AppCIF-Bench`）；Infer **59 → 61**（+`FlashTopK`、`Hy-SimulMT`）；其余 **16 → 17**（+`EmoReward`）。合计 95。

### 1.2 既有仓库增量（Agent 侧 12 个 TOUCHED）

| 仓库                     | push  | 窗口内提交     | 实质变化                                                    |
| ---------------------- | ----- | --------- | ------------------------------------------------------- |
| **Hy-Memory**          | 11-06 | **295 条** | SDK 2.0.5→2.0.14、MemHub 1.0.5/1.0.6（见 §3）               |
| **UniRL**              | 11-06 | 8 条       | 全部为 fix（rollout/sampling/reward），**agentic 路径零改动**      |
| **Hyra-results**       | 11-07 | 18 条      | ⭐ AI4Science 批量放出 6 个模型 + 数学猜想结果（见 §4）                  |
| ExplorationBench       | 10-28 | 1 条       | README 补 HF 数据集与 Synced 报道链接                            |
| DiagSafe               | 11-02 | 6 条       | docs/修复（badge、恢复 89% 一致率数字、judge 参数校验）                  |
| UniCom                 | 11-05 | 7 条       | camera-ready 作者列表与致谢（论文定稿）                              |
| Hy-Embodied-RoboFusion | 11-03 | 1 条       | ⭐ **补 MIT `LICENSE.txt`**（上轮"无 LICENSE 12 个"清单修订为 11 个） |
| Hunyuan3D-1            | 11-02 | 2 条       | 本地 demo 默认值与权重加载加固（社区 PR #67）                           |
| Prism                  | 10-31 | 3 条       | 仅 README 作者署名调整；**+368★ 仍无实质代码变更**                      |
| OmniWeaving            | 10-30 | 2 条       | README 更新                                               |
| AuK                    | 10-27 | 2 条       | docs：社区微调结果共享 model cards                               |
| CL-bench 等             | —     | —         | 未在窗口内推送                                                 |

> **UniRL 版本口径修订**：认证 API 显示 v0.2.0 的 `published_at=2026-10-08T08:43:40Z`，而 `created_at=2026-10-19T07:00:35Z`。上轮报告写"v0.2.0 于 10-19 打版"引用的是 release 对象创建时间（atom feed 的 updated）——**实际发布时刻是 10-08**。上轮"窗口内（10-12 后）打版"的说法随之修正：v0.2.0 发布在基线窗口**外**，10-12→10-26 窗口内发生的只是 README News 与提交。不影响上轮"打版而非能力新增"的结论本身。

---

## 2. 平台级事件：Hy4 正式版发布（11-04）—— Agent 能力部分

> **事件级定位**：这是自 8-28 `Hy4-preview` 开源以来混元最重要的一次发布。**GitHub/HF 侧零增量**（`Hy4-preview` 仓库 `pushed_at` 仍停在 2026-08-28；HF 无新权重），全部发生在**产品与 API 侧**——因此本轮把它写成"平台级事件"而非"仓库增量"。  
> **多源交叉**：财联社/科创板日报（<https://www.cls.cn/detail/2525324>、<https://www.cls.cn/detail/2524739>）、人民网财经（<https://finance.people.com.cn/n1/2026/1104/c1004-40827754.html>）、央广网（<https://tech.cnr.cn/techgd/20261104/t20261104_527862108.shtml>）、腾讯新闻、今日头条（潮新闻）、AI产品库（aiproducthub.cn）。**以上均为官方口径的逐字转述，本报告未取到混元官方一手页面**（公众号正文不可读惯例）。

### 2.1 与 Agent 直接相关的官方口径（数字照抄，全部为官方自评）

| 能力                 | 官方口径                                                                                                                       | 来源      |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------- | ------- |
| **Computer Use**   | 首次支持电脑操作，"结合命令行和图形界面处理工作和生活中的各类任务"；针对 **Blender、CAD 等专业软件和跨应用长程任务**开展专项训练，提升专业任务的规划与执行                                     | 人民网/央广网 |
| **视觉理解**           | 首次引入；配合推理与工具调用完成细粒度图像处理、复杂图表分析、数学视觉推理                                                                                      | 同上      |
| **长程智能体**          | "大幅提升指令遵循、长文理解、长程智能体等综合效果"                                                                                                 | 同上      |
| **WorkBuddy 场景评测** | 平均效果 **83.18 分**，任务通过率 **96.15%**，**12 个场景中 9 个排名国内第一**；人机双写功能在文档美观度/稳定性/可用性提升                                             | 财联社     |
| **实测样例**           | 科创板日报实测：金融任务（腾讯/阿里/百度 2020 至今三表）41 分 51 秒自主完成，产出 12 页 PDF + 8 图表 + 附录财务表，Excel 底稿含 **183 个可复算公式**，下载清单逐条记录 URL/状态/字节数/失败原因 | 财联社     |
| **金融能力**           | 数千工作空间/上万任务训练；联合腾讯金融科技 FiT **213 位专家**（153 位外部头部机构、七成 5 年+经验）盲测 **579 个**真实金融任务，Hy4 综合 **3.04 分**，居开源第一                    | 财联社/央广网 |
| **幻觉治理**           | 幻觉率 **12.95% → 7.2%**，因猜测导致的错误率 **23.1% → 5.6%**（自建测试，preview → 正式版）                                                       | 财联社     |
| **上下文/个人助理**       | "上下文学习能力持续提升……成为用户的长期个人助理"；借 WorkBuddy 企业微信套件连通聊天/邮件/文档/日程/会议，跨项目理清进展、识别待办、提醒优先级；团队协作功能随新版"项目"上线                           | 人民网     |
| **游戏/前端**          | 相比 preview，GameDev **+19.1%**、GameLogic **+9.8%**                                                                          | 财联社     |

### 2.2 发布形态与节奏（口径要点）

- **已上线**：腾讯云 TokenHub、WorkBuddy/CodeBuddy、元宝、企业微信智能助理"大圆"、ima；API 同步接入 **OpenRouter、Hermes、OpenCode**；QQ、王者荣耀、和平精英、微信读书、腾讯新闻等腾讯系产品持续接入中。
- **定价不变**：输入 6 元 / 输出 18 元 / 缓存命中 0.3 元（每百万 tokens）；TokenHub 刊例价 5 折至 **2026-12-31**；WorkBuddy/CodeBuddy **11-04 ~ 11-15** 提供免费额度。
- **权重后置**："Hy4 将于**近期开源模型权重**，发布模型系统卡片，并推进更负责任的开源，重点关注网络和生物安全等风险。"——**截至 2026-11-09，HF `tencent` 组织无 Hy4 正式版权重**（最新混元 HF 模型仍是 10-24 的 `Hy-Nomas-Diffusers`；窗口内唯一新模型是 11-07 的 `Qwen3.8-27B-DSpine-b16`，非混元团队）。
- **节奏承诺**："自今年 2 月重建基础设施以来，平均**每两个月推出一个大版本**，先以  Preview 形式开放，再根据用户反馈优化发布正式版。"**Hy4 flash** 将近期发布（继承能力、大幅降低任务成本）；**Hy5** 规模将进一步扩大（媒体推算年底/明年初）。
- **安全口径**：训练中主动剔除漏洞利用、逆向、渗透等攻击性网络安全数据；生物安全能力持续评估中（AI产品库转述）。
- **同期开源联动**：AI产品库提到"模型同时开源了单细胞表征基础模型 **HyCell** 与物理场基础模型 **HyPDE**"——与 `Hyra-results` 窗口内 11-02/11-04 的提交完全对应（见 §4），说明 Hy4 发布稿把 AI4Science 线作为生态叙事的一部分。

> **⚠️ 口径警示**：  
> ① "770B 总参 / 49B 激活"来自 AI产品库对**正式版**的描述——与 preview 完全相同的数字，**无官方一手源确认正式版与 preview 同规格**（正式版是否加视觉塔、参数是否变化，README 未说明——因为根本还没发）。  
> ② 财联社标题"腾讯史上最强开源模型"是**超前叙事**：权重尚未开源，官方原文是"将于近期开源"。  
> ③ "首席 AI 科学家姚顺雨带队"一说出自单家第三方媒体（hqwc.cn），**本报告不采信、不引用**。  
> ④ 所有 83.18 / 96.15% / 3.04 / 7.2% 均为**官方自评**，无第三方复测。

---

## 3. 核心增量 A：`Hy-Memory` 高频产品化迭代（295 commits / SDK 2.0.5 → 2.0.14）

上轮报告写到 2.0.5。本轮 CHANGELOG 实测：

| 版本                          | 日期                   | 要点（CHANGELOG 原文归纳）                                                                                                                                   |
| --------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| SDK 2.0.6                   | 10-27                | `search(rerank=False)` 单次跳过全部 rerank（本地 cross-encoder / graph rerank / BM25 rerank），按检索分返回；MMR 多样性仅在 `limit < 20`（`MMR_MAX_TOP_K`）时执行                |
| SDK 2.0.7 / 2.0.8           | 10-28                | （稳定性与配置修复）                                                                                                                                           |
| SDK 2.0.9 / 2.0.10 / 2.0.11 | 11-01                | （连续三日发版）                                                                                                                                             |
| SDK 2.0.12 / 2.0.13         | 11-02                | （同上）                                                                                                                                                 |
| **SDK 2.0.14**              | **11-04**            | `search()/async_search()` 增加可选 `statuses` 参数，显式过滤记忆状态                                                                                                |
| MemHub 1.0.5                | 10-28                | （release tag 10-28）                                                                                                                                  |
| **MemHub 1.0.6**            | 11-04 发布 / 11-06 tag | Unreleased 段已预告下一批：**capture hook 不再直写数据库**（改为把 hook JSON POST 给运行中的 Core，1.5s 预算，Core 不在则静默丢弃）；**hook 与 live file-sync 双通道去重**（同一 turn 不再被写成两条 raw） |

**读法**：

- 迭代节奏是**每天 1~3 个 SDK 版本**——这不是"开源后维护"，是**产品级敏捷发布**。上轮风险表里"SDK 版本号不一致（README 2.0.4 vs Release 2.0.5）"的坑，现在放大为"根 README 版本号**必然滞后**于 CHANGELOG"——**以 CHANGELOG 为准**。
- Unreleased 段的两项改动（hook 不直写 DB、双通道去重）说明 MemHub 的**数据一致性架构还在收敛**：从"多写入方 + 事后对账"走向"Core 单写入方 + 只读 hook"。
- stars 1,282 → **2,012（+730）**，且**有 295 条提交支撑**——上轮"Stars 与代码变更脱钩"的判断在 Hy-Memory 身上**不成立**（热度与迭代同步），但在 Prism 身上继续成立（见 §5）。

### 3.1 官网补充（memory.hunyuan.tencent.com，11-09 抓取）与三条口径修订

官网（2026.10.16 上线，中英一致）带来三条上轮无法从仓库/模型卡取得的一手证据：

1. **LongMemEval 有官方数字了**。官网给出模型层**六指标明细**（六项等权均值恰为 55.49，自洽）：Hy-Memory-30B-A3B 为 LoCoMo 67.2 / **LongMemEval 77.7** / PersonaMem v2 52.09 / Travel 48.22 / Shopping 19.90 / CMB 67.74。**注意：LongMemEval 上 Opus5（78.87）更高**——"超过 Opus 5"只对六项均值成立，不适用于每一项。上轮（05 报告）"LongMemEval 无证据"的风险项**部分解除**：模型层有数字（引用需带"六指标等权、四系统平均"口径），框架层四基准仍不含 LongMemEval。
2. **MemHub 宿主名单官方确认为 10 种**：WorkBuddy / Cursor / Codex / Claude Code / OpenCode / OpenClaw / Qwen Code / Pi / Hermes / DeepSeek Harness。上轮"README 5 个 vs IT之家 10 个"的不一致**收敛为：官方 10 个，README 滞后**。**WorkBuddy 在列**——与判断⑫"Agent 出口收敛到 WorkBuddy"再添一层证据（记忆层也以 WorkBuddy 为一等宿主）。
3. **腾讯内部业务落地数据**（全新，全部官方自评）：**腾讯自选股** A 股投研记忆专项——综合 64.3→79.6（+23.8% 相对）、多跳召回 50→90（+40pt）、冲突消解 67→83；**注意生产上记忆模型用的是 DeepSeek V4 Flash**，不是自家的 30B-A3B（写入模型可插拔的活证据）。**iWiki** CTX 插件：251 题、同模型同提示词下"仅记忆 36.7 vs 全量原始对话 24.7"（相对 +48.6%）——**蒸馏后的记忆比喂全量历史更准**（"压缩即降噪"卖点）。**腾讯新闻**：长期偏好理解 91.68/100（10 用户 41 案例，87.8% 案例 >90）。接入领域：金融/资讯/知识管理/办公 Agent/社交娱乐/游戏。

> 素材：[`../../_research/2026-11-09/hy-memory-official-site.md`](../../_research/2026-11-09/hy-memory-official-site.md)（含完整六指标矩阵与全部数字）。

---

## 4. 核心增量 B：`HyperLogic` + `AppCIF-Bench` —— 两个"程序化 ground truth"基准

### 4.1 `HyperLogic`（10-28 公开，Apache-2.0，arXiv 2605.19597 v2）

- 仓库：<https://github.com/Tencent-Hunyuan/HyperLogic> · 数据集：<https://huggingface.co/datasets/tencent/HyperLogic>（10-27 上线）
- 论文：arXiv **2605.19597**（**2026-05-19 v1**，**10-02 v2**——本轮 API 核验；README News 只把 10-28 写成"仓库+数据集公开日"）

**方法一句话**：先写题、后导答案。逻辑训练背景的本科生写中文种子题 → 四角色 agent 流程**加固题目但禁止解题** → 两个不同模型家族的 agent 独立把题目翻译成可执行有限域形式模型 → 逻辑求解器执行导出答案 → 分层裁决，两份编码一致才发布。**"求解器做搜索，所以题目可以比造题的模型更难，而每个答案都可执行、可审计。"**

**三层结构**（全量 523 题，公开 30%）：

| 层    |                               全量 |         公开 | 设计目标                                                     |
| ---- | -------------------------------: | ---------: | -------------------------------------------------------- |
| Seed |                              228 |         69 | 常规逻辑题；7 个前沿模型差距 <5.9 分，**不区分**                           |
| Base | 195（922 子题，平均 4.7 题/题干约 3,300 字） | 59（274 子题） | 子题全对才算对；拉开模型差距（44.6~77.6%，散布 33 分）                       |
| Hard |          100（题干约 8,700 字，耦合搜索空间） |         30 | 直答**无模型超 16%**；另测 sandbox agent（±logic modeling library） |

**Leaderboard 关键行**（全量基准，3 rollouts，最高推理档）：

| 模型                  |     Seed | Base Item |   Hard 直答 | Hard Sandbox | Hard + Library |
| ------------------- | -------: | --------: | --------: | -----------: | -------------: |
| GPT-5.6 Sol         | **89.8** |  **77.6** |      13.7 |         40.3 |           45.7 |
| Claude Opus 5       |     86.5 |     74.2† | **15.6**† |     **55.7** |       **57.0** |
| Kimi K3             |     88.6 |      57.9 |       4.0 |         41.3 |           43.0 |
| DeepSeek V4 Pro     |     83.9 |      51.8 |       9.3 |         26.0 |           20.0 |
| GLM-5.3             |     87.3 |      45.6 |       3.0 |         20.7 |           31.3 |
| Doubao Seed 2.1 Pro |     85.8 |      44.6 |       5.0 |         26.0 |           33.7 |

†Claude 的数字剔除被 API 拒答的题。

**README 给出的四条 Key Findings（与 Agent 直接相关）**：

1. **Hard 直答基本失效**（全部 <16%），sandbox +16.7~40.1 分但天花板 55.7%——**"给 agent 一个代码沙箱"是必要非充分**。
2. **logic modeling library 不稳定**：加库后 −6.0 ~ +10.6 分，5 升 2 降——"用得好需要理解建模语言"，即**工具本身有学习成本**。
3. **模型倾向 plain Python 暴力枚举，而非 solver**：7 个模型里 6 个主要写"枚举候选赋值 + 手写规则过滤"——**当前 agent 使用形式化工具的默认习惯与工具设计者的预期错位**。
4. **更长的推理不等于更高准确率**：Base 上中位推理长度 15.2k~52.7k token 与准确率无关；GLM-5.3 在 131k 限制下 585 次运行中 99 次无解析答案。

**工程细节**：公开子集经 `build_public_subset.py` 均匀采样，`selection_manifest.json` 对全量语料做 **SHA-256 commitment**（可审计不可反推）；Hard agent 评测完整开源（两个条件的精确 payload、workspace builder、Docker runner、11 模块 `translation` 库 + `MANUAL.md`）。**70% 私有保留**——防污染设计（见 §6 判断⑩）。

> **⚠️ 注意**：7 个模型里**没有混元自己的模型**。混元造基准测别人，自家旗舰缺席——与 AppCIF 一致（见下）。

### 4.2 `AppCIF-Bench`（11-05，NeurIPS 2026，代码 Apache-2.0 / 数据 CC-BY-NC-4.0）

- 仓库：<https://github.com/Tencent-Hunyuan/AppCIF-Bench>（v1.1）
- 定位：**首个应用级复杂指令遵循（CIF）基准**。两轴分层：**Horizon Length（H，干扰累积跨度）× Logical Depth（L，嵌套规则深度）**。
- **Prompt-as-Environment 范式**：L4 级系统提示约 **14k 词**；配套 **6 个可执行 Code Simulation Engines**（约 3.5k 行 Python/个，纯标准库）产生 **ground truth**——演唱会预算审计、时空交易、规则裁决、中医诊断、电网调度、墓穴探索（对应 6 个 L4 archetype）。
- 规模：**466 样本 / 3,017 atomic rubrics**（中英双语同构，中文版附 ground-truth derivation）；5 场景 × 4 难度；Real-world Simulation 240 题（6 子场景 × 40）。
- 指标：**SSR**（rubric 满足率）+ **HSR**（整样本全 rubric 满足率），LLM-as-judge（v1.1 用 Doubao-Seed-2.0-lite；论文用 Doubao-Seed-1.6，已停服）。
- **12 个 LLM 的 HSR 只有 1.1% ~ 45.9%**：GPT-5.4-xhigh 45.9% 断层第一，Gemini-3.1-Pro 34.6%，Doubao-Seed-2.0-Pro-high 20.0%，GLM-5.1 10.9%；**L4 级除 GPT 外全部 ≤2.2%**（多数 0%）。
- **版本声明**：v1.1 修正了 1,084 条 rubric（审计发现 gold/措辞/推导错误）+ 删 1 增 20；**v1.1 分数与论文不可直接比较**（重评结果在论文附录 J）。
- ⚠️ **数据许可 CC-BY-NC-4.0**：**数据不可商用**（代码可）。与上轮 `ArtifactsBenchmark/C3/PlanningBench` 的 CC-BY-4.0 不同，多了一个 NC。


### 4.3 两个基准合起来的信号

上轮判断"评测向可验证演进"，本轮两个新基准把这条线推到顶：

| 维度              | HyperLogic                         | AppCIF-Bench           | 上轮同类                         |
| --------------- | ---------------------------------- | ---------------------- | ---------------------------- |
| ground truth 来源 | 形式模型**执行**导出                       | 代码仿真引擎**执行**导出         | WorkForge"事实锚点"派生            |
| 防污染             | 30% 公开 + SHA-256 commitment，70% 私有 | v1.1 修订声明 + judge 变更声明 | ExplorationBench 两世界公开       |
| agent 评测形态      | sandbox ± 库，精确 payload 开源          | 系统提示即环境（14k 词）         | ExplorationBench 异世界         |
| 判定器             | LLM judge 仅做语义等价检查                 | LLM judge 逐 rubric 比对  | DiagSafe prompt-level rubric |

**"LLM judge"没有被抛弃，但被降级为"最后一公里的语义比对"——事实与数值由程序产生。** 这是混元评测方法论四轮演进后的稳定形态。

---

## 5. 其他 Agent 侧增量（一句话 + 证据）

- **UniRL**：8 条提交全为 fix/perf（`fix(rollout): honor per-sample AR seeds in vLLM-Omni`、`fix(rollout): fail closed on invalid vLLM-Omni AR logprobs`、`feat(reward): add monitor_rewards for score-only remote components` 等）；README News 无新增；**agentic 训练路径零改动**——上轮"能力是基线前就有"的判断继续成立。
- **Hyra-results**：11-02~~11-04 批量放出 **6 个 AI4Science 模型**（均含权重+代码+技术报告）：`HyPDE`（323M 2D PDE 基础模型，13 数据集 9/13 中位 VRMSE 胜 Walrus 1.29B）、`hyqec`（165 个非 CSS 量子码精确距离证明）、`HyCell`（16.1M 单细胞嵌入，held-out 0.735 平 Stack 218M）、`HyFF`（5 个力场 2.3M~~1.2B，Andante 达 UMA-S-1.2 力误差、29× MD 速度）、`HyInv`（1.96M 逆折叠）+ `HyPLM-Flash`（650M 蛋白质 LM，**含 INT8 快速推理引擎**）；另有 **Hyra 2.0 博客**（11-02，hy.tencent.ai/research/hyra2）与两个数学猜想结果（Petty 五维、Yau 圆作用）。**与 Hy4 发布稿点名 HyCell/HyPDE 呼应**——AI4Science 是混元对外的"科研品牌线"，由 Hyra（递归自我改进系统）产出。
- **ExplorationBench**：HF 数据集 10-28 上线（`tencent/ExplorationBench`），README 补 Synced 报道；stars +1。
- **DiagSafe / UniCom / Hunyuan3D-1 / OmniWeaving / Prism / AuK**：均为 docs/修复级（详见 §1.2 表）；`Prism` +368★ 而 3 条提交全是作者署名——**"热度与代码脱钩"继续成立**。
- **元器停服日到期**：上轮报告的预告事件今天（11-09）正式生效——存量智能体不可用，2027-01-09 起依法删除数据，迁移落点腾讯云 ADP。
- **《智能体发展自律公约》**（11-04，每经晚参转财联社）：中国信通院、华为、阿里、**腾讯**、小米、百度、火山引擎等 20 家单位联合发布，"共同抵制利用智能体实施算法压榨、黑灰产等行为"——平台级行业治理事件，混元是签署方之一。

---

## 6. 与 10-26 基线结论的对照

| 10-26 结论                                             | 本轮证据                                                                                                   | 判定                                                                                                              |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| **"运行时框架仍稀缺，记忆层与环境层已补"**                             | 本轮无新运行时框架；记忆层高频迭代（2.0.14）；无新环境合成工作                                                                     | **成立**                                                                                                          |
| **"评测方法论向可验证演进"**                                    | HyperLogic（执行导出）+ AppCIF（可执行引擎）双双落地，程序化 ground truth 成为新基准标配                                           | **强化（本轮最强）**                                                                                                    |
| **"Stars 与能力脱钩"**                                    | Hy-Memory +730★ 有 295 commits 支撑；Prism +368★ 仍只有署名提交                                                   | **部分修订**——分仓库看：有迭代的产品热度与代码同步，论文仓库热度继续脱钩                                                                         |
| **"Hy-Memory SDK 版本号以 Release 为准"**                  | 根 README 滞后加剧（CHANGELOG 已到 2.0.14）                                                                     | **修订**——以 **CHANGELOG** 为唯一权威                                                                                   |
| **"无 LICENSE 12 个"**                                 | RoboFusion 补 MIT `LICENSE.txt`（11-03，注意文件名是 `LICENSE.txt`）                                             | **修订**——12 → 11                                                                                                 |
| **"UniRL v0.2.0 于 10-19 打版"**                        | 认证 API `published_at=10-08`（10-19 是 release 对象创建时间）                                                    | **部分修订**——发布时刻口径修正，结论不变                                                                                         |
| **"HunyuanOCR-1.5 只在仓库名里（社区量化）"**（上轮 infer 报告风险 #11） | README 标题即 HunyuanOCR-1.5，News 明确 **2026-07-07 发布 1.5 正式版**（arXiv 2607.04884）+ 07-24 统一推理环境与 verl RL 栈 | **推翻并修正**——1.5 是官方真实版本（同仓库内升级，不单独建 HF 权重库）；上轮的"社区量化"判断针对的是 `prithivMLmods/HunyuanOCR-1.5-GGUF-Updated`，那个仍为社区产物 |
| **"具身智能本轮零增量"**                                      | RoboFusion 仅补 license                                                                                  | **继续成立**                                                                                                        |

**本轮新增判断**：

> **⑩ 评测基准的"防污染设计"成为混元标配。**  
> HyperLogic 公开 30% + SHA-256 commitment、AppCIF v1.1 修订不可比声明、Hy-Nomas-Bench 只放题目不放分数——三代基准（10-12 前的 C3/Planning、10 月的 Exploration/DiagSafe、11 月的 HyperLogic/AppCIF）一路演进，"私有保留集 + 可审计公开子集"已是默认范式。**含义**：用混元基准刷分的外部模型，测的都是"审过的子集"；想要全量分数必须走官方通道。
>
> **⑪ 混元造基准，自家模型缺席。**  
> HyperLogic 的 7 个前沿模型与 AppCIF 的 12 个 LLM 里**都没有 Hy 系列**。这与"官方自评数字全走产品口径（WorkBuddy 83.18）"形成对照：**学术基准测别人，产品评测测自己**。引用混元基准排名时要注意"该排名不含混元模型"这一口径。
>
> **⑫ Agent 出口进一步收敛到 WorkBuddy，且开始"模型-产品 Co-Design"叙事。**  
> Hy4 发布稿的 agent 能力全部以 WorkBuddy 场景呈现（人机双写、企业微信套件、新版"项目"），叠加 11-04~11-15 免费期——上轮"出口收敛到 WorkBuddy + 企微 + ADP"的判断，本轮有了**模型代际配套**。

---

## 7. 公开信息时间线（Agent 侧）

| 日期                 | 事件                                                                         | 来源类型                | 链接                                                                                                     |
| ------------------ | -------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------ |
| 2026-10-26 12:53   | Hy-SimulMT 仓库内容公开（上轮扫描后 2 小时）                                              | 仓库实测（commit 时间戳）    | <https://github.com/Tencent-Hunyuan/Hy-SimulMT>                                                        |
| 2026-10-27         | Hy-Nomas 技术报告上 arXiv 2610.36729；Hy-Nomas-Bench 数据集卡                        | arXiv / HF          | <https://arxiv.org/abs/2610.36729>                                                                     |
| 2026-10-27 / 10-28 | `HyperLogic` 仓库建立 / 数据集与仓库公开                                               | 仓库实测                | <https://github.com/Tencent-Hunyuan/HyperLogic>                                                        |
| 2026-10-28         | ExplorationBench HF 数据集上线                                                  | HF                  | <https://huggingface.co/datasets/tencent/ExplorationBench>                                             |
| 2026-10-28 / 11-06 | MemHub 1.0.5 / 1.0.6                                                       | Release（GitHub API） | <https://github.com/Tencent-Hunyuan/Hy-Memory/releases>                                                |
| 2026-11-02~11-04   | Hyra-results 批量放出 HyCell/HyFF/HyInv/HyPLM-Flash/HyPDE/hyqec + Hyra 2.0 博客  | 仓库实测                | <https://github.com/Tencent-Hunyuan/Hyra-results>                                                      |
| **2026-11-04**     | **Hy4 正式版发布**（API/产品侧；权重承诺近期开源）                                            | 官方公告（媒体逐字引用）        | <https://www.cls.cn/detail/2525324> · <https://finance.people.com.cn/n1/2026/1104/c1004-40827754.html> |
| 2026-11-04         | 20 家单位《智能体发展自律公约》（腾讯在列）                                                    | 媒体（每经晚参）            | <https://new.qq.com/rain/a/20261104A0ER7100>                                                           |
| 2026-11-04         | `tencent/TailorRig` 数据集上线（3D/mesh，arXiv 2611.02902，**未核验**——arXiv API 503） | HF                  | <https://huggingface.co/datasets/tencent/TailorRig>                                                    |
| 2026-11-05         | AppCIF-Bench 公开（v1.1）；EmoReward 公开                                         | 仓库实测                | <https://github.com/Tencent-Hunyuan/AppCIF-Bench>                                                      |
| 2026-11-07         | `Qwen3.8-27B-DSpine-b16` 上线 HF（DSpine drafter 27B 版，非混元团队）                 | HF                  | <https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16>                                                |
| **2026-11-09**     | **元器正式停服**（上轮预告事件到期）                                                       | 上轮报告引用的财经媒体         | 见上轮 §6                                                                                                 |

> **验证边界**：混元官方公众号在本窗口的发布（Hy4 正式版公告等）正文均不可读，本报告只以可核的媒体逐字转述 + 仓库/HF/arXiv 一手数据为准；X 帖原文仍未取到。

---

## 8. 风险与待核实

| #  | 事项                                                | 现状                                                          | 建议                                                         |
| -- | ------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------------------------------- |
| 1  | **Hy4 正式版权重未开源**                                  | 官方口径"近期开源 + 模型系统卡片"；截至 11-09 HF 无                           | 不要引用任何"Hy4 正式版已开源"的说法；等 HF 上线后核对架构（是否有视觉塔、参数是否同 preview）   |
| 2  | **正式版规格**                                         | "770B/49B"仅 AI产品库转述（B 级），且与 preview 相同                      | 一手源出现前，写"官方未披露正式版参数"                                       |
| 3  | **WorkBuddy 83.18 / 96.15% / 金融 3.04 / 幻觉率 7.2%** | 全部官方自评，无第三方复测，评测集未公开                                        | 引用必带"官方自评"限定语                                              |
| 4  | **Hy4 的 Computer Use 评测**                         | 官方未公布 computer-use 专项基准数字                                   | 等 OSWorkerBench（HF 9 月已有 `tencent/OSWorkerBench` 数据集）或系统卡片 |
| 5  | **AppCIF 数据 NC 许可**                               | 数据 CC-BY-NC-4.0                                             | 商用产品不得用其数据训练/评测发布，代码可用                                     |
| 6  | **HyperLogic 全量分数**                               | 公开子集分数≠全量分数（README 明示）                                      | 引用 Leaderboard 必须注明"全量基准、官方测评"                             |
| 7  | **UniRL 发布时间口径**                                  | 上轮 10-19 → 本轮修正为 published 10-08                            | 引用 v0.2.0 用 10-08                                          |
| 8  | **TailorRig**                                     | 3D/mesh 数据集（HF tag 指向 arXiv 2611.02902），论文核验因 arXiv 503 未完成 | 归 3D 生成线，不影响本轮结论；下次扫描补核                                    |
| 9  | **内容农场**                                          | paooo.com 对 Hy-SimulMT 写"GEMBA 约 85.2 分"等 README 中不存在的精确数字  | 不采信；Hy-SimulMT 数字以 README 为准                               |
| 10 | **公众号正文**                                         | 全部被验证码拦截（惯例）                                                | 沿用"存在性证据"规则                                                |
| 11 | **元器存量迁移**                                        | 今天停服                                                        | 有元器智能体的项目确认已迁 ADP                                          |

---

## 9. 证据来源与采集脚本

| 产物                                  | 位置                                                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 组织全量快照（95 仓库）                       | [`../../_research/2026-11-09/org-repos-2026-11-09.csv`](../../_research/2026-11-09/org-repos-2026-11-09.csv)                          |
| 增量表（17 行：NEW 4 / TOUCHED 12 / 补录 1） | [`../../_research/2026-11-09/delta-2026-11-09.csv`](../../_research/2026-11-09/delta-2026-11-09.csv)                                  |
| 窗口内提交（17 仓库，认证 API）                 | [`../../_research/2026-11-09/commits-summary.json`](../../_research/2026-11-09/commits-summary.json)                                  |
| Release 记录                          | [`../../_research/2026-11-09/releases-summary.json`](../../_research/2026-11-09/releases-summary.json)                                |
| HF 模型/数据集清单                         | [`../../_research/2026-11-09/hf-tencent-models.json`](../../_research/2026-11-09/hf-tencent-models.json) · `hf-tencent-datasets.json` |
| README 快照（16 份）                     | [`../../_research/2026-11-09/readmes/`](../../_research/2026-11-09/readmes/)                                                          |
| 本地商发分支实测                            | `D:\WorkSpace\vllm-ascend-hy4\repo`（详见 infer 报告 §6）                                                                                   |
| 采集脚本                                | `scan_org.py` / `scan_detail.py` / `fetch_readmes.py`（同目录，可重跑）                                                                        |

**关键链接**：<https://github.com/Tencent-Hunyuan/HyperLogic> · <https://github.com/Tencent-Hunyuan/AppCIF-Bench> · <https://github.com/Tencent-Hunyuan/Hy-Memory> · <https://github.com/Tencent-Hunyuan/Hyra-results> · <https://huggingface.co/datasets/tencent/HyperLogic> · <https://arxiv.org/abs/2605.19597> · <https://www.cls.cn/detail/2525324>（Hy4 正式版）
