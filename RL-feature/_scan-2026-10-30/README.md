# 2026-10-30 RL-feature 增量扫描工作区

本目录是可复现的扫描管道与原始证据缓存，对应三份报告（extract-hidden-status 自本期起按仓分文件夹）：

- `../extract-hidden-status/2026-10-30/sglang/sglang-extract-hidden-states-delta-2026-10-30.md`
- `../extract-hidden-status/2026-10-30/vllm&vllm-ascend/vllm-extract-hidden-states-delta-2026-10-30.md`（vLLM 主仓 + vllm-ascend 合并报告）
- `../sample mask/2026-10-30/sample-mask-delta-2026-10-30.md`

UTC 窗口：`[2026-10-23T02:05:41Z, 2026-10-30T02:34:03Z)`。

## 数据来源与配额策略

未使用 `gh` CLI 或 PAT。GitHub REST 采用未认证访问，`core` 60 次/小时、`search` 10 次/分钟。本机存在两条不同出口：
`direct`（绕过系统代理）与 `proxy`（`http://127.0.0.1:7897`），两条出口各自计费，`gh.py` 按剩余配额择一使用并把
每个响应先落盘到 `raw/`，重跑时命中缓存不再消耗配额。

`search/issues` 承担发现（全文匹配标题+正文+评论），`core` 只用于被选中条目的 timeline/PR 详情。
合入侧不依赖 API：直接用 `git clone --depth 900` 的 main 快照扫描窗口内提交（三个仓：sglang、vllm、vllm-ascend）。

vllm-ascend 的额外一步是 **pickaxe**（`git log -S`）：用它定位 `supports_aux_hidden_states_over_pp`、
`pp_transport_aux_hidden_states_`、`extract_hidden_states` 各自是哪个提交引入的，从而区分「本期新增」与「既有能力」。

## 文件

| 文件 | 作用 |
| --- | --- |
| `gh.py` | 双出口 GitHub 客户端：配额感知、6.3s/搜索节流、磁盘缓存 |
| `discover.py` | sglang / vllm 关键词 + `created:` 按日切分发现扫描 → `discovery.json` |
| `discover_ascend.py` | vllm-ascend 侧发现扫描（含 `"pipeline parallel" hidden` 等组合短语）→ `discovery-ascend.json` |
| `analyze.py` | 主题打分与基线归属 → `candidates.json` |
| `rank.py` / `rank_ascend.py` / `shortlist.py` | 候选排序与强信号短名单（人工筛选辅助） |
| `collect_commits.py` | 本地 main 快照的窗口提交抽取 → `commits-{sglang,vllm,vllm-ascend}.json` |
| `deepdive.py` | vllm/sglang 选中条目的 timeline / PR 详情 / `.patch` → `deepdive.json`，patch 存 `raw/patch/` |
| `deepdive_ascend.py` | vllm-ascend 选中条目，写入同一 `deepdive.json`（key 为 `repo#number`） |
| `summarize.py` | 单条证据的人工复核视图 |
| `curated.py` / `curated_ascend.py` | 三份报告的编辑判断（分类、优先级、机制、NPU、验证） |
| `build.py` | 生成各报告 JSON / CSV（支持单仓与多仓合并；schema 兼容历史期并新增 repository 字段） |
| `check.py` / `check_links.py` / `table.py` | 结构与链接校验、记录表打印 |
| `counts_ascend.py` / `meta_ascend.py` | vllm-ascend 条目的事件计数与元数据核对 |

## 已知边界

- `core` 配额用尽时 `deepdive*.py` 会等到重置；未取到 timeline 的条目在 JSON 里表现为 `events: []`。
- `.patch` 里只有当前 head 可达的提交，force-push 丢弃的提交不可恢复（vllm-ascend 多条条目有 10–12 次 force-push）。
- 未运行任何 GPU/NPU 或模型测试；报告中的性能与精度数字均为作者报告。
- 扫描结束后已清理可再生成的大文件：`repos/`（三个浅克隆，约 300 MB）与 `raw/*.json`（API 缓存）。
  本地仍保留 `raw/patch/`（56 份 PR diff，9.4 MB）。
- **入库范围**：本目录只提交脚本、README 与小型产物；`raw/`、`repos/` 以及大体积中间结果
  （`discovery*.json`、`candidates.json`、`deepdive.json`、`commits-*.json`）由 `.gitignore` 排除，仅存在于本地工作区。
  重跑时按上面顺序执行即可重新生成。
