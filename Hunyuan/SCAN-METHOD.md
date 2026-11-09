# 增量扫描方法论（Runbook）

> 面向"**对某个组织/项目集合做定期增量扫描，并产出可溯源的中文调研报告**"这一类任务。
> 本文件是从一次真实执行中抽出来的通用方法：主体与具体项目无关（只保留占位符），附录 A 给出该次执行的完整参数与命令，附录 B 是踩坑台账。
> 任意项目按 §0 的参数表填空即可复用。

---

## 0. 参数化模板（换项目只改这张表）

| 占位符 | 含义 | 示例（本次） |
|---|---|---|
| `{ROOT}` | 工作根目录 | `D:\WorkSpace\workspace\Hunyuan` |
| `{ORG}` | 被扫描的组织 / 上游集合 | `Tencent-Hunyuan`（GitHub）+ `tencent`（HuggingFace） |
| `{CUTOFF}` | 上次基线的截止时刻（**ISO UTC**） | `2026-10-12T00:00:00Z` |
| `{NOW}` | 本次采集时刻 | `2026-10-26T10:30+08:00` |
| `{TRACKS}` | 划分的赛道（2–3 条，各自成目录） | `agent` / `infer` |
| `{OUT}` | 报告落盘目录模式 | `{ROOT}/{track}/{YYYY-MM-DD}/` |
| `{RESEARCH}` | 原始素材目录 | `{ROOT}/_research/{YYYY-MM-DD}/` |
| `{INDEX}` | 总索引文件 | `{ROOT}/INDEX.md` |
| 附加模型源 | 权重/模型卡托管方 | HuggingFace `api/models?author=` |
| 附加论文源 | 论文库 | arXiv API / abs 页 |

**先决条件**：上一次扫描必须留下三样东西——① 仓库清单 CSV（含每个仓库的 `last_push` 与 `stars` 数字快照）② README / LICENSE 原文快照 ③ 一份写了"结论 + 范围界定 + 分类计数"的 `{INDEX}`。缺任何一样，先补基线（见 §9）。

---

## 1. 阶段一：定窗口与口径（写进报告第 1 节）

1. **窗口** = `{CUTOFF} → {NOW}`，同时写清时区基准（`pushed_at` 是 UTC，报告里的日期是北京时间时要说）。
2. **"增量"的机器定义**（三个集合都要算，缺一个就会漏）：
   - `NEW` = `created_at >= {CUTOFF}`
   - `TOUCHED` = `pushed_at > {CUTOFF}`
   - `MISSED-IN-BASELINE` = 存在于当前全量、但基线清单里没有（**最有价值的一类**：说明上次漏采，如私有转公开、大小写/改名）
3. **分类口径**：把 `{TRACKS}` 的收录边界写成一句话，并预先声明**边界处理规则**：
   - 同组织但非目标团队的仓库 → 单列，不计入统计
   - 评测/基准类、安全类、训练框架类 → 归到主赛道还是"其余"要提前定
   - 边界仓库（本次：`DiagSafe`、`OmniCapBench`、`HYDRA`）单独开"同期新增但不属主线"小节，避免污染结论
4. **来源标注约定**（必须写进报告表头）：一手（仓库文件/模型卡/API/论文）＞ 官方发布（X/公告，常不可直读）＞ 财经媒体原创 ＞ 转述 ＞ 自媒体/内容农场（**一律不采信**）；缺失信息统一标 `README 未说明`。

---

## 2. 阶段二：组织级全量快照（**每次重抓，不要试图用"增量接口"**）

```bash
# 全量拉，一次分页到底（比"对比 API"可靠：能看到下架/改名/MISSED）
GET https://api.github.com/orgs/{ORG}/repos?per_page=100&page=N&type=public&sort=created&direction=desc
```

落 `{RESEARCH}/org-repos-{date}.csv`，字段固定为：
`name, created_at, pushed_at, updated_at, stargazers_count, forks_count, open_issues_count, language, size, default_branch, license.spdx_id, description, archived`

然后与基线清单做集合运算，输出 `{RESEARCH}/delta-{date}.csv`：
`repo, kind(NEW|TOUCHED|MISSED-IN-BASELINE), created_at, pushed_at, stars_now, stars_baseline, stars_delta, forks_now, language, size_kb, license, default_branch, description`

**附带产物（很有用）**：`star_delta.txt` —— 基线 stars 数字 → 现在，按 `Δ` 排序。
用途不是"谁火"，而是**检验"热度与代码变更是否脱钩"**：本次 8 个涨幅最大的仓库里只有 1 个有推送，`Prism` +971 但窗口内唯一提交是改了一行 README 日期 —— 这类必须单独写，防止把传播事件当技术更新。

---

## 3. 阶段三：逐仓库增量取证（在 API 限流下也能做完）

按"便宜 → 贵"的顺序取证，**先用零限流的端点判定，再决定要不要拉代码**：

| 目的 | 端点 / 命令 | 为什么用它 |
|---|---|---|
| 窗口内改了哪些提交 | `https://github.com/{ORG}/{repo}/commits/{branch}.atom` | **不走 REST API**（无 60 次/小时限制），返回最近 20 条，足够判定"只动文档/只改版本号/真干活" |
| 有没有发版 | `https://github.com/{ORG}/{repo}/releases.atom` | 区分"**打版**"与"**能力新增**"（本次 `UniRL v0.2.0` 就属于打版） |
| README 是否真变 | `https://raw.githubusercontent.com/{ORG}/{repo}/{branch}/README.md` → `difflib` 对比基线快照 | 去掉徽章/日期/"News 补登"噪音后，才能看出实质变更 |
| 需要读代码/子文档 | `git clone --depth 1` + `GIT_LFS_SKIP_SMUDGE=1`，落到 `_probe/{repo}` | 浅克隆 + 跳过 LFS；`_probe/` 必须 gitignore |
| 刷新旧克隆 | `git fetch --depth 1 origin HEAD && git reset --hard FETCH_HEAD` | 比重新克隆快，且保留目录 |

**判定规则（把每个 touched 仓库归到一类，写进总览表）**：
- `CAPABILITY`：新增能力/新算法/新模型（→ 深挖）
- `DELIVERY`：发布镜像、量化、蒸馏、部署脚本、许可变更（→ 深挖，这类最有实操价值）
- `MAINTENANCE`：仅效率/重构/文档/依赖（→ 一句话带过，但要给出证据，如"20 条提交全为 perf/refactor"）
- `TRIVIAL`：仅一行 README/日期（→ 明确写"无仓库侧支撑"）

> **注意 default branch**：可能不是 `main`（本次外围有 5 个仓库是 `master`）。按 `main` 抓 raw 会静默 404/空文件。
> **注意大小写**：用户口语 `hunyuan` ≠ 实际目录 `Hunyuan`；仓库名大小写也可能变。

---

## 4. 阶段四：模型 / 权重侧（组织 ≠ 代码组织时尤其关键）

```bash
GET https://huggingface.co/api/models?author={ORG}&sort=createdAt&direction=-1&limit=100
GET https://huggingface.co/api/datasets?author={ORG}&sort=createdAt&direction=-1&limit=20
GET https://huggingface.co/{repo_id}/raw/main/README.md   # 模型卡原文 → 存 hf-cards/
```

从模型卡里**只抽这几样**（其余都是宣传）：
1. 部署事实：引擎 + 版本（如 `vLLM v0.31.0`）、启动参数、服务化命令、上下文设置
2. 量化事实：档位、文件大小、**每档对应的评测分数**、校准方法（imatrix / MoE 混合精度）
3. 已知限制与"未说明"（是否有 NPU、是否支持 Windows、是否只支持单轮…）
4. 许可（**以 LICENSE 文件与模型卡为准，不看 README 徽章**）
5. 数据集卡往往比仓库 README 细得多（本次 `Hy-Nomas-Bench` 的分 split 数字只在数据集卡里）

---

## 5. 阶段五：论文侧（两个动作，别省）

1. **核验**：报告里出现的每个 arXiv ID，用 `http://export.arxiv.org/api/query?id_list=A,B,C` 批量核对 `标题 / published / updated / 作者`。**报错就说明 ID 抄错了**（429 限流时改用 `https://arxiv.org/abs/{id}` 页面）。
2. **归属**：arXiv **常常没有单位元数据**。凡"某团队出品"的说法，必须有第二个证据（官方账号发布 / 媒体逐字引用 / 仓库徽章），并在报告里标"中高可信"。

---

## 6. 阶段六：外部公开信息（公众号/媒体/社交）

**检索方式**：一次 `web_search` 给 3–4 个查询词（中文主体 + 英文主体 + 站点限定 + 事件词），便于交叉。
**每条信息必须落这 6 个字段**：`日期 / 标题 / 信源类型 / URL / 要点摘录（数字照抄）/ 可信度`。

**信源分级（写进报告表头）**

| 级别 | 例子 | 用法 |
|---|---|---|
| A 一手 | 仓库文件、模型卡、API、arxiv abs、Release atom | 可直接下结论 |
| B 财经/科技媒体原创 | 含版本号、规模数字的报道 | 可用，标注"媒体" |
| C 转述/编译 | 二手汇总、外媒引用官方帖 | 仅作旁证 |
| D 自媒体/内容农场 | 匿名"据知情人士"、加料细节 | **不采信**，只作反证 |

**公众号（微信）专用流程**：`mp.weixin.qq.com` 正文基本都会返回验证码页（`r.jina.ai` 之类代理也常失败）→
1. 只把"标题 + URL"作为**发布存在性**证据，绝不引正文；
2. 内容改用可读转载（新浪财经/搜狐/澎湃/腾讯云社区/IT之家/C114/TechFlow）并写明"转载页"；
3. 若必须引原文，标注"正文不可读，待人工核"，并把 URL 留给读者。

**反核查（本方法里最值钱的一步）**：对任何"某某于本月开源/发布 X"的病毒式说法，去查**一手时间戳**（HF `createdAt`、原文页面日期、仓库 `created_at`）。本次就识别出一起：网传"世界模型 1.1 于 2026-10-22 开源"，实为 **2025-10** 旧闻，源头是内容农场把旧发布写成新发布。

---

## 7. 阶段七：交叉验证与"基线结论修订"

1. **双源规则**：每个准备写进结论的事实，至少两个独立来源（如"模型卡 + 论文"、"release + 官方帖"）。
2. **口径规则**：官方自评数字必须写清口径 —— 采样比例（如"固定 20% 子采样"）、指标是否等权、领先幅度（如 55.49 vs 54.62 = 0.87 分）。**永远不要把"某子项领先"写成"整体超越"**。
3. **同指标不同口径要并存**：本次 `LoCoMo 72.6`（框架层单基准）与 `73.28`（六指标矩阵子采样）是两件事，混用即错。
4. **逐条对照基线结论**，只允许三种判定：`成立` / `部分修订` / `推翻`，每条都要给本轮证据。
5. **新增判断**单列一段（本次 5 条），并明确它是"观察"而非"官方口径"。

---

## 8. 阶段八：交付物与素材（结构固定，便于逐期对比）

```
{ROOT}/
├── {INDEX}                       ← 总索引：范围界定 + 分类计数 + 风险 + 增量章节
├── {track}/{YYYY-MM-DD}/NN-<slug>.md   ← 报告（NN 全局连续编号，便于排序阅读）
└── _research/{YYYY-MM-DD}/       ← 素材（gitignore，但路径写进报告便于溯源）
    ├── org-repos-{date}.csv      delta-{date}.csv      star-delta.txt
    ├── commits/*.atom            unirl-releases.atom 等
    ├── readmes/*.md              readme-diff.txt
    ├── hf-cards/*.md             hf-*-models.json
    ├── arxiv-*.xml
    └── scan_*.py  diff_readmes.py  star_delta.py  check_tables.py   ← 可重跑
```

**报告骨架（9 节，照抄即可）**
1. 本次口径：窗口 / 基线 / 实测范围 / **来源标注表**
2. 一句话结论（3–5 条，每条一个事实）
3. 增量总览表（NEW / TOUCHED / 分类变化 / star 变化 / 平台级事件）
4. 核心深挖（每个核心对象一节：定位 → 架构/参数 → 部署与工程细节 → 评测 → 许可 → 与你基线结论的关系）
5. 其他增量（`MAINTENANCE` / `TRIVIAL` 逐条一句 + 证据）
6. 公开信息时间线（日期 + 事件 + 信源级别 + URL；表尾写"验证边界"）
7. 与基线结论的对照（成立/修订/推翻 + 新增判断）
8. 风险与待核实（**表格式**：编号 / 事项 / 现状 / 建议动作；把"时间错位""同名非官方""版本口径冲突"都列进去）
9. 证据来源与采集脚本（路径表 + 关键链接）

**入库规则**：报告与索引入库；`_research/`、`_probe/`（含上游 `.git`）一律 gitignore，只留可重跑脚本并在 §9 指明路径。

**最后的质量门（本方法必跑）**
```bash
# 1) 表格完整性：每张表每行未转义 | 数量一致（裸 | 会把表拆坏）
python check_tables.py
# 2) 链接可达：报告里的相对路径逐个 open 一次（INDEX ↔ 报告 ↔ 素材）
# 3) 事实抽查：随机挑 5 个数字回原文核对
```

---

## 9. 首次接手 / 基线缺失时怎么办

如果目标项目没有基线，先做一次"基线扫描"（本方法 §2 + §3 的全量版），并**强制留下**：
- 仓库清单 CSV（**必须含 `last_push` 和 `stars` 数字**，否则下次无法算增量）
- README / LICENSE 原文快照（下次做 diff 用）
- `{INDEX}`：范围界定 + 分类计数（三者之和要等于总数）+ 许可证分布 + 跨方向结论 + 报告索引

然后才允许开始增量模式。

---

## 10. 人力分工（对本类任务的建议）

- **广度**交给并行子代理：公众号/媒体/arXiv/HF 的召回式检索，一个 agent 一条赛道，要求"每条给 URL + 日期 + 信源级别 + 不知道就说不知道"。
- **核验**必须自己做：子代理给的所有数字，回到一手源（模型卡/数据集卡/abs 页/配置文件）重读一遍。本次 GGUF 量化表、`Hy-Nomas-Bench` 分 split、`sp_size`、arXiv 日期全部是主代理复核后采用的。
- 把"官方自评 vs 第三方复测"的分界线写死在报告模板里，避免复核时被叙事带走。

---

## 附录 A：本次执行的具体参数与命令（Hunyuan 2026-10-26）

| 项 | 值 |
|---|---|
| `{ROOT}` | `D:\WorkSpace\workspace\Hunyuan` |
| `{ORG}` | `Tencent-Hunyuan`（GitHub，90 个公开仓库）/ `tencent`（HF） |
| `{CUTOFF}` → `{NOW}` | `2026-10-12T00:00:00Z` → `2026-10-26T10:30+08:00` |
| `{TRACKS}` | `agent`（框架/评测/具身/记忆）、`infer`（推理优化库/模型发布部署/世界模型） |
| 交付 | `agent/2026-10-26/05-agent-memory-and-delta.md`、`infer/2026-10-26/06-infer-world-model-and-serving.md`、`INDEX.md` 第八节 |
| 素材 | `_research/2026-10-26/`（50 个文件：4 CSV/JSON、11 atom、17 README、7 HF 卡、2 arXiv XML、6 脚本、diff/star 文本） |

关键命令：

```bash
# 组织全量（python 落 CSV，避免 PowerShell 5.1 的 ConvertFrom-Json 深度限制）
python scan_org.py          # → org-repos-2026-10-26.csv / delta-2026-10-26.csv
python scan_delta_detail.py # → commits/*.atom / commits-summary.json / hf-tencent-models.json
python diff_readmes.py      # → readme-diff.txt（与 _research/readmes 基线快照逐行 diff）
python star_delta.py        # → star-delta.txt
python parse_arxiv.py       # → 核验 6 个 arXiv ID 的标题/日期/作者
python check_tables.py      # → 交付前表格完整性检查
```

取证小抄（本次实际用过）：

```bash
curl.exe -s "https://github.com/Tencent-Hunyuan/{repo}/commits/main.atom"        # 提交
curl.exe -s "https://github.com/Tencent-Hunyuan/{repo}/releases.atom"            # 发版
curl.exe -s -L "https://raw.githubusercontent.com/.../main/README.md"            # 文档现况
curl.exe -s -L "https://huggingface.co/{repo_id}/raw/main/README.md"             # 模型卡
curl.exe -s -A "Mozilla/5.0" "https://export.arxiv.org/api/query?id_list=..."    # 论文核验
git clone --depth 1 https://github.com/Tencent-Hunyuan/{repo} _probe/{repo}      # 需读代码时
export GIT_LFS_SKIP_SMUDGE=1   # 上面的 clone 之前设置
```

---

## 附录 B：踩坑台账（下次直接规避）

| # | 坑 | 症状 | 处理 |
|---|---|---|---|
| 1 | GitHub 匿名 REST API 限流 | 抓十几个仓库后 `403 rate limit exceeded (60/hr)` | **优先 atom/raw**（不限流）；全量元数据要带 token |
| 2 | PowerShell 5.1 解析 JSON | `ConvertFrom-Json: ArgumentException`（深度/编码） | 全部改 `python`（`urllib` + `json`），或 Node |
| 3 | PowerShell 无 heredoc | `python - <<'PY'` 直接语法错误 | 写 `.py` 文件再跑；多行脚不内联 |
| 4 | 控制台编码 | `UnicodeEncodeError: 'gbk'` | `$env:PYTHONIOENCODING="utf-8"`，输出重定向到文件再读 |
| 5 | arXiv API 429 | 返回空/`HTTP 429` | 换 `https://arxiv.org/abs/{id}` 页面抓；或稍后重试 |
| 6 | 公众号 captcha | `mp.weixin.qq.com` 返回"环境异常／去验证"；`r.jina.ai` 也失败 | 只作存在性证据，内容走转载/镜像并注明 |
| 7 | 内容农场编数 | 出现摘要里没有的"42.3 文件/279KB/82.4 分" | 只认摘要与官方页面；该站全部标 D 级 |
| 8 | 时间错位 | "X 于本月开源"实为一年前 | 查 HF `createdAt` / 原文日期 / 仓库 `created_at` |
| 9 | README 徽章 ≠ LICENSE | 徽章写 Apache，`LICENSE` 是 "Tencent is pleased… + Apache 正文"，API 报 `NOASSERTION` | 以 LICENSE 文件 + 模型卡为准，并在报告里说明 |
| 10 | default branch 不是 main | 按 `main` 抓 raw 全 404/空 | 先读 `default_branch` 字段 |
| 11 | 官方沉默 ≠ 不存在 | README 未写"需要几张卡/是否支持 NPU" | 从配置文件推断（如 `sp_size: 8`），但必须标"据配置推断" |
| 12 | 指标口径混用 | 同一基准两处 72.6 / 73.28 | 分别标注口径，禁止合并 |
| 13 | 播放帧率当吞吐 | "24 FPS 能跑" 被读成"实时" | 明确 fps 是**输出播放帧率**，与生成速度无关 |
| 14 | 同名非官方制品 | `*-1.5-GGUF` 实为社区量化，`base_model` 才指向官方 | 查 HF `createdAt` + `base_model` 字段 |
| 15 | Markdown 表被拆 | 单元格内出现裸 `|`（如 `source="a" | "b"`） | 改写成 `/` 分隔，或 `\|` 转义；交付前跑表格检查 |
| 16 | 索引 stat 过期 | 内容无差异却显示 `M` | `git update-index --refresh` 后恢复；不要误提交 |
| 17 | 克隆物误入库 | `_probe/` 未 gitignore，`git status` 一堆上游文件 | `_probe/` 写进 `.gitignore`，只提交自撰报告 |
