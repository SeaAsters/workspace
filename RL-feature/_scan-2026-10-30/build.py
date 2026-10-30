# -*- coding: utf-8 -*-
"""Build JSON + CSV deliverables for the 2026-10-30 delta reports (multi-repo aware)."""
import csv
import glob
import json
import os
import re
import time

import curated
import curated_ascend

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
START = "2026-10-23T02:05:41Z"
SGLANG = "sgl-project/sglang"
VLLM = "vllm-project/vllm"
ASCEND = "vllm-project/vllm-ascend"
MAIN = {
    SGLANG: "0f3f619850f222b4f238ff3d336926e99a2a5c56",
    VLLM: "f81a414faed63b576768487236f98ea5e655fc57",
    ASCEND: "c9fc4538526ba9b04d7f85c4f2766620bc1c4049",
}
NOISE = {
    "review_requested", "labeled", "mentioned", "subscribed", "head_ref_deleted",
    "unlabeled", "review_dismissed", "milestoned", "added_to_project_v2",
    "project_v2_item_status_changed", "deployed", "auto_squash_enabled",
    "auto_merge_disabled", "automatic_base_change_succeeded", "parent_issue_added",
}

OUT = os.path.join(WS, "extract-hidden-status", "2026-10-30")
OUT_SM = os.path.join(WS, "sample mask", "2026-10-30")


def load(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8-sig"))


DEEP = load("deepdive.json")
DISC = load("discovery.json")
DISC_ASC = load("discovery-ascend.json")
CAND = load("candidates.json")
COMMITS = {
    "sglang": load("commits-sglang.json"),
    "vllm": load("commits-vllm.json"),
    "vllm-ascend": load("commits-vllm-ascend.json"),
}


def patch_commits(repo, number):
    tag = repo.split("/")[1]
    paths = glob.glob(os.path.join(ROOT, "raw", "patch", f"{tag}-{number}-*.patch"))
    if not paths:
        return []
    txt = open(paths[0], encoding="utf-8", errors="replace").read()
    out = []
    for m in re.finditer(r"^From ([0-9a-f]{40}) .*?\n(.*?)(?=^From [0-9a-f]{40} |\Z)", txt, flags=re.M | re.S):
        sha, block = m.group(1), m.group(2)
        subj, date = "", ""
        for ln in block.split("\n")[:10]:
            if ln.startswith("Subject:"):
                subj = ln[8:].strip()
            if ln.startswith("Date:"):
                date = ln[5:].strip()
        try:
            dt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.strptime(date.replace("+0000", "+00:00")[:25], "%a, %d %b %Y %H:%M:%S"))
        except Exception:  # noqa: BLE001
            dt = date
        out.append({"sha": sha, "subject": subj, "date": dt})
    return out


def patch_files(repo, number):
    tag = repo.split("/")[1]
    paths = glob.glob(os.path.join(ROOT, "raw", "patch", f"{tag}-{number}-*.patch"))
    if not paths:
        return []
    txt = open(paths[0], encoding="utf-8", errors="replace").read()
    files = []
    for f in re.findall(r"^diff --git a/(\S+) b/", txt, flags=re.M):
        if f not in files:
            files.append(f)
    return files


def discovery_index():
    idx = {}
    for it in DISC.get("items", []):
        idx[(it["repo"], it["number"])] = it
    for it in DISC_ASC.get("items", []):
        idx[(ASCEND, it["number"])] = dict(it, repo=ASCEND)
    return idx


DIDX = discovery_index()


def build_record(repo, spec, topic):
    num = spec["num"]
    key = f"{repo}#{num}"
    dd = DEEP.get(key, {})
    pr = dd.get("pr") or {}
    iss = dd.get("issue") or {}
    meta = DIDX.get((repo, num), {})
    merged = pr.get("merged_at") or meta.get("merged_at")
    state = pr.get("state") or iss.get("state") or meta.get("state")
    period_state = "merged" if merged and merged >= START else state
    events = []
    etypes = {}
    for e in dd.get("timeline") or []:
        if (e.get("created_at") or "") < START:
            continue
        if e.get("event") in NOISE:
            continue
        body = e.get("body")
        if body and len(body) > 900:
            body = body[:900] + "…"
        rec = {
            "date": e.get("created_at"),
            "type": e.get("event"),
            "url": e.get("url"),
            "body": body,
            "sha": e.get("sha") or e.get("commit_id"),
            "actor": e.get("actor"),
        }
        if e.get("state"):
            rec["review_state"] = e.get("state")
        events.append(rec)
        etypes[e.get("event")] = etypes.get(e.get("event"), 0) + 1
    events.sort(key=lambda x: x["date"] or "")
    commits = patch_commits(repo, num)
    files = patch_files(repo, num)
    title = pr.get("title") or iss.get("title") or meta.get("title")
    created = pr.get("created_at") or iss.get("created_at") or meta.get("created_at")
    return {
        "repository": repo,
        "number": num,
        "kind": spec["kind"],
        "title": title,
        "url": f"https://github.com/{repo}/{'pull' if spec['kind'] == 'pr' else 'issues'}/{num}",
        "topic": topic,
        "priority": spec["prio"],
        "classification": spec["cls"],
        "activity": spec["activity"],
        "created_at": created,
        "period_end_state": period_state,
        "current_state": state,
        "current_state_fetched_separately": True,
        "merged_at": merged,
        "closed_at": pr.get("closed_at") or iss.get("closed_at"),
        "base_ref": pr.get("base_ref"),
        "head_sha": pr.get("head_sha") or (commits[-1]["sha"] if commits else None),
        "merged_by": pr.get("merged_by"),
        "additions": pr.get("additions"),
        "deletions": pr.get("deletions"),
        "draft": pr.get("draft"),
        "labels": pr.get("labels") or iss.get("labels") or meta.get("labels"),
        "event_types": etypes,
        "events": events,
        "commits": commits,
        "changed_files_current_head": files,
        "analysis_current_design": spec["analysis"],
        "npu": spec["npu"],
        "validation": spec["validation"],
        "evidence": "本 JSON 的 events / commits / changed_files_current_head 为结构化证据；扫描脚本与说明见仓库内 RL-feature/_scan-2026-10-30/README.md，原始 patch 与 API 缓存只保留在本地工作区（未入库）。",
    }


def commit_query_index(tag):
    data = COMMITS.get(tag) or {}
    commits = data.get("commits") or []
    hits = []
    for c in commits:
        joined = " ".join([c["subject"], c.get("body") or ""] + c["files"])
        if re.search(r"hidden[ _-]?state|aux_hidden|aux hidden|sampling[ _-]?mask|sample[ _-]?mask|mask capture|forward observer|deferred (output|response)", joined, re.I):
            hits.append(c["sha"][:12])
    return {
        "kind": "local_git_commit_scan",
        "repo": tag,
        "window_start": START,
        "head_sha": data.get("head_sha"),
        "commits_in_window": data.get("commit_count"),
        "keyword_matched_commits": hits,
    }


ARCHIVES = [
    os.path.join(WS, "extract-hidden-status", "2026-10-23", "extract-hidden-states-delta-2026-10-23.json"),
    os.path.join(WS, "extract-hidden-status", "2026-10-16", "extract-hidden-states-delta-2026-10-16.json"),
    os.path.join(WS, "sample mask", "2026-10-23", "sample-mask-delta-2026-10-23.json"),
    os.path.join(WS, "sample mask", "2026-10-16", "sample-mask-delta-2026-10-16.json"),
    os.path.join(WS, "sample mask", "2026-10-10", "sample-mask-delta-2026-10-10.json"),
    os.path.join(WS, "sample mask", "2026-09-29", "sample-mask-prs.json"),
]


def baseline_audit(repo, topic):
    prev = {}
    if repo == SGLANG:
        for path in ARCHIVES:
            if not os.path.exists(path):
                continue
            d = json.load(open(path, encoding="utf-8-sig"))
            if not isinstance(d, dict):
                continue
            for a in d.get("tracked_baseline_audit", []) or []:
                prev.setdefault(a["number"], {"number": a["number"], "title": a.get("title"), "topics": set(a.get("baseline_topics") or []), "state": a.get("state"), "merged_at": a.get("merged_at"), "updated_at": a.get("updated_at")})
            for r in d.get("records", []) or []:
                prev.setdefault(r["number"], {"number": r["number"], "title": r.get("title"), "topics": set(), "state": r.get("current_state"), "merged_at": r.get("merged_at"), "updated_at": r.get("created_at")})
    elif repo == VLLM:
        for it in CAND.get("items", []):
            if it["repo"] != repo:
                continue
            if (DIDX.get((repo, it["number"]), {}) or {}).get("created_at", "") >= START:
                continue
            prev[it["number"]] = {"number": it["number"], "title": it["title"], "topics": {"hidden_states"}, "state": it["state"], "merged_at": it.get("merged_at"), "updated_at": it.get("updated_at")}
    else:
        for it in DISC_ASC.get("items", []):
            if (it.get("created_at") or "") >= START:
                continue
            prev[it["number"]] = {"number": it["number"], "title": it["title"], "topics": {"hidden_states"}, "state": it["state"], "merged_at": it.get("merged_at"), "updated_at": it.get("updated_at")}
    rows = []
    for num, info in sorted(prev.items()):
        live = DIDX.get((repo, num))
        rows.append(
            {
                "repository": repo,
                "number": num,
                "title": (live or {}).get("title") or info["title"],
                "baseline_topics": sorted(info["topics"]) or [topic],
                "state": (live or {}).get("state") or info["state"],
                "merged_at": (live or {}).get("merged_at") or info["merged_at"],
                "updated_at": (live or {}).get("updated_at") or info["updated_at"],
                "in_window_activity": bool(((live or {}).get("updated_at") or info["updated_at"] or "") >= START),
            }
        )
    return rows


def candidate_decisions(repo, featured, topic_filter=None):
    rows = []
    feat = set(featured)
    src = CAND.get("items", []) if repo in (SGLANG, VLLM) else []
    for it in src:
        if it["repo"] != repo:
            continue
        if topic_filter and not (it.get("match_hidden") if topic_filter == "hidden" else it.get("match_mask")):
            continue
        rows.append(
            {
                "repository": repo,
                "number": it["number"],
                "kind": "pr" if it["is_pr"] else "issue",
                "title": it["title"],
                "decision": "纳入主代理核心判断" if it["number"] in feat else "背景/边缘/无实质活动",
                "reason": "按实际代码相关性和本期事件筛选；不将通用模型内部 hidden 张量使用、池化/嵌入任务、仅引用或纯 CI/时序调整计为输出接口新增",
                "url": f"https://github.com/{repo}/{'pull' if it['is_pr'] else 'issues'}/{it['number']}",
                "state": it["state"],
                "updated_at": it["updated_at"],
            }
        )
    return rows


ASC_SIG = [
    (r"aux[ _-]?hidden|aux_hidden_state", 8),
    (r"extract_hidden_states", 8),
    (r"hidden[ _-]?states? (over|across|relay|transport|shard)", 8),
    (r"(pp|pipeline parallel|pipeline stage).*(hidden|aux|token shard)", 6),
    (r"return[ _-]hidden[ _-]states?", 7),
    (r"dflash|dspark", 3),
    (r"sampling[ _-]?mask", 4),
    (r"eagle3|mtp|speculat", 2),
]
ASC_NEG = [
    (r"quantiz|nvfp4|mxfp4|fp8|w4a8|w8a8", -2),
    (r"document|readme|release note|translat", -3),
    (r"ci|nightly|flaky", -1),
]


def _asc_score(text):
    s = 0
    for pat, w in ASC_SIG:
        if re.search(pat, text, re.I):
            s += w
    for pat, w in ASC_NEG:
        if re.search(pat, text, re.I):
            s += w
    return s


def ascend_decisions(featured):
    rows = []
    feat = set(featured)
    for it in DISC_ASC.get("items", []):
        text = (it["title"] or "") + "\n" + (it.get("body") or "")
        s = _asc_score(text)
        if s < 5 and it["number"] not in feat:
            continue
        rows.append(
            {
                "repository": ASCEND,
                "number": it["number"],
                "kind": "pr" if it["is_pr"] else "issue",
                "title": it["title"],
                "decision": "纳入主代理核心判断" if it["number"] in feat else "背景/边缘/无实质活动",
                "reason": "按 Ascend 侧代码路径与本期事件筛选；纯文档/翻译、量化精度、CI 时序与无 hidden-state 语义的算子改动不计入",
                "url": f"https://github.com/{ASCEND}/{'pull' if it['is_pr'] else 'issues'}/{it['number']}",
                "state": it["state"],
                "updated_at": it["updated_at"],
                "score": s,
            }
        )
    rows.sort(key=lambda r: (-r["score"], r["updated_at"] or ""))
    return rows


def query_indexes(repo):
    seen = set()
    rows = []
    src = DISC.get("queries", []) if repo in (SGLANG, VLLM) else [{"repo": ASCEND, **q} for q in DISC_ASC.get("index", [])]
    for q in src:
        if q.get("repo") not in (None, repo):
            continue
        query = q.get("query")
        if not query or query in seen:
            continue
        seen.add(query)
        rows.append({"kind": "issues_search", "repo": repo, "query": query, "total": q.get("total"), "returned": q.get("returned"), "cutoff_utc": START})
    return rows


def write_report(name, groups, topic, out_dir, baseline_label, non_sub, extra_limitations=None):
    records = []
    for repo, specs in groups:
        records.extend(build_record(repo, spec, topic) for spec in specs)
    counts = {}
    for r in records:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1

    audited = []
    decisions = []
    cover = {
        "baseline_prs": 0,
        "baseline_with_window_activity": 0,
        "candidate_total": 0,
        "candidate_prs": 0,
        "candidate_issues": 0,
        "featured_records": len(records),
        "discovered_items": 0,
        "discovered_prs": 0,
        "discovered_issues": 0,
        "search_queries": 0,
    }
    per_repo = {}
    for repo, specs in groups:
        aud = baseline_audit(repo, topic)
        audited.extend(aud)
        if repo == ASCEND:
            dec = ascend_decisions([s["num"] for s in specs])
        else:
            dec = candidate_decisions(repo, [s["num"] for s in specs], topic_filter=topic)
        decisions.extend(dec)
        if repo in (SGLANG, VLLM):
            items = [i for i in DISC.get("items", []) if i["repo"] == repo]
        else:
            items = DISC_ASC.get("items", [])
        per_repo[repo] = {
            "baseline_prs": len(aud),
            "baseline_with_window_activity": sum(1 for a in aud if a["in_window_activity"]),
            "candidate_total": len(dec),
            "discovered_items": len(items),
            "discovered_prs": sum(1 for i in items if i["is_pr"]),
            "discovered_issues": sum(1 for i in items if not i["is_pr"]),
            "search_queries": len(query_indexes(repo)),
            "commits_scanned_in_window": (COMMITS.get(repo.split("/")[1]) or {}).get("commit_count"),
            "main_sha": MAIN[repo],
        }
        cover["baseline_prs"] += len(aud)
        cover["baseline_with_window_activity"] += per_repo[repo]["baseline_with_window_activity"]
        cover["candidate_total"] += len(dec)
        cover["candidate_prs"] += sum(1 for d in dec if d["kind"] == "pr")
        cover["candidate_issues"] += sum(1 for d in dec if d["kind"] == "issue")
        cover["discovered_items"] += len(items)
        cover["discovered_prs"] += per_repo[repo]["discovered_prs"]
        cover["discovered_issues"] += per_repo[repo]["discovered_issues"]
        cover["search_queries"] += per_repo[repo]["search_queries"]
    cover["per_repository"] = per_repo

    limitations = [
        "当前 diff 与正文只描述采集时 head，不是完整源码快照",
        "force-push 可能移除旧提交；patch/commits 只反映当前 head 可达提交，timeline 保留可见事件",
        "评论中的测试与性能数字均为作者报告，本次未运行 GPU/NPU 或模型测试",
        "相关度由主代理判断，完整候选与排除记录见本 JSON 的 candidate_decisions",
    ]
    if extra_limitations:
        limitations.extend(extra_limitations)

    primary = groups[0][0]
    payload = {
        "repository": primary,
        "repositories": [repo for repo, _ in groups],
        "topic": topic,
        "window_semantics": f"UTC [{START}, cutoff)",
        "start_utc": START,
        "end_utc_exclusive": CUTOFF,
        "cutoff_utc": CUTOFF,
        "main_sha": MAIN[primary],
        "main_shas": {repo: MAIN[repo] for repo, _ in groups},
        "main_observed_utc": MAIN_OBSERVED,
        "baseline": baseline_label,
        "record_count": len(records),
        "classification_counts": counts,
        "records": records,
        "non_substantive_followup": non_sub,
        "limitations": limitations,
        "scan_coverage": cover,
        "query_indexes": [q for repo, _ in groups for q in query_indexes(repo)] + [commit_query_index(repo.split("/")[1]) for repo, _ in groups],
        "tracked_baseline_audit": audited,
        "candidate_decisions": decisions,
    }
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["number", "title", "classification", "activity", "period_end_state", "current_state", "created_at", "merged_at", "base_ref", "head_sha", "priority", "url", "npu", "kind", "repository"])
        for r in records:
            w.writerow([r["number"], r["title"], r["classification"], r["activity"], r["period_end_state"], r["current_state"], r["created_at"], r["merged_at"], r["base_ref"], r["head_sha"], r["priority"], r["url"], r["npu"], r["kind"], r["repository"]])
    print(name, "records", len(records), counts, "->", json_path)
    return payload


CUTOFF = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
MAIN_OBSERVED = CUTOFF

if __name__ == "__main__":
    out_sglang = os.path.join(OUT, "sglang")
    out_vllm = os.path.join(OUT, "vllm&vllm-ascend")
    write_report(
        "sglang-extract-hidden-states-delta-2026-10-30",
        [(SGLANG, curated.SGL_HIDDEN)],
        "hidden",
        out_sglang,
        "sglang 2026-10-10/10-16/10-23 归档（131 条 tracked PR）；vLLM 与 vllm-ascend 侧另立合并报告",
        [
            {"number": 43502, "title": "[Fix] DSv4.1: Size the encoder-replay SWA pool by the DSpark draft's stages", "state": "merged", "note": "无本期活动，不计入"},
            {"number": 45407, "title": "[Fix] [NVIDIA] deepseek-v4.1: Fix the breakable prefill CUDA graph in the late-layer tail", "state": "merged", "note": "10/23 合入，属 bounded replay 尾部修复，不改变 hidden-state 输出契约"},
            {"number": 46481, "title": "[Chore] ci: Update test est_time values", "state": "merged", "note": "仅 CI 时序调整，触及 hidden-state 测试文件但无功能变化"},
        ],
    )
    write_report(
        "sample-mask-delta-2026-10-30",
        [(SGLANG, curated.SGL_SAMPLING)],
        "sampling",
        OUT_SM,
        "sglang 2026-09-29/10-10/10-16/10-23 归档；采样 mask 仅跟踪 SGLang",
        [
            {"number": 41955, "title": "[Sampling] Default to top_k_first and expose filtering order", "state": "open", "note": "目标分支 sampling-top-k-first，不能当 main 独立落地"},
            {"number": 41957, "title": "[Perf] Fuse compact top_k_first probability construction", "state": "open", "note": "同属 top_k_first 分支线，本期仅背景"},
            {"number": 43346, "title": "[Sampler] Sort only the max(top_k) leading columns in the torch sampler", "state": "merged", "note": "本期无实质事件"},
            {"number": 46481, "title": "[Chore] ci: Update test est_time values", "state": "merged", "note": "唯一触及 test_sampling_mask.py 的 main 提交，仅时序"},
        ],
        extra_limitations=["main 在本窗口内没有 sampling-mask 功能合入；唯一触及 mask 测试文件的 main 提交仅调整 CI est_time"],
    )
    write_report(
        "vllm-extract-hidden-states-delta-2026-10-30",
        [(VLLM, curated.VLLM_HIDDEN), (ASCEND, curated_ascend.VLLM_ASCEND)],
        "hidden",
        out_vllm,
        "vLLM 与 vllm-ascend 均无历史归档，本期为两仓首版基线（关键词命中的既有 PR/Issue/RFC + Ascend 侧代码路径核对）",
        [
            {"number": 63972, "repository": VLLM, "title": "[CI Failure]: CPU-Spec Decode Tests - test_dflash2.py::test_dflash_fc_runs_fp32_target_hidden_states_in_draft_dtype", "state": "closed", "note": "对应 #63436 的 CI 失败记录，随修复关闭"},
            {"number": 63406, "repository": VLLM, "title": "[Core][CUDA Graph] Separate prefill (token) and decode (request) CUDA graph capture size", "state": "open", "note": "仅与 #63427 的内存分解互相引用，非捕获接口变更"},
            {"number": 19866, "repository": ASCEND, "title": "[BugFix] Fix request-major sampling masks for speculative decoding on Ascend", "state": "open", "note": "采样 mask 主题（本报告只作相邻提及）"},
            {"number": 20129, "repository": ASCEND, "title": "[BugFix][Sample] Adapt sampling-mask packing to the vLLM slot layout", "state": "open", "note": "采样 mask 主题（本报告只作相邻提及）"},
        ],
        extra_limitations=[
            "vLLM 多数隐藏状态能力以 spec-decode 方法（extract_hidden_states）和逐模型 PP relay 存在，尚无公共 return_hidden_states 响应字段",
            "vllm-ascend 的 aux-over-PP 是逐模型（DeepSeek-V4/V2 补丁、Kimi-K3、MiniMax-M3）打开，不是全局能力",
        ],
    )
