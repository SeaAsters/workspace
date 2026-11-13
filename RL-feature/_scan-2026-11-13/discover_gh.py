# -*- coding: utf-8 -*-
"""Discovery via authenticated gh CLI for the 2026-11-13 window."""
import json
import os
import sys

import ghcli

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-11-06T06:35:52Z"
START_DATE = "2026-11-06"

SGL = "sgl-project/sglang"
VLLM = "vllm-project/vllm"
ASC = "vllm-project/vllm-ascend"

SGL_HIDDEN = ['"hidden states"', '"hidden state"', '"hidden-state"', "hidden_states", "return_hidden_states",
              "aux_hidden_states", "extract_hidden_states", '"aux hidden"', '"forward observer"']
SGL_SAMPLING = ['"sampling mask"', "sampling_mask", '"sample mask"', "sample_mask", "return_sampling_mask",
                '"top-p mask"', '"top-k mask"', '"mask capture"']
VLLM_HIDDEN = SGL_HIDDEN
ASC_HIDDEN = ['"hidden states"', '"hidden state"', "hidden_states", "aux_hidden_states", '"aux hidden"',
              "extract_hidden_states", '"hidden-state"', '"pipeline parallel" hidden',
              '"spec decode" "pipeline parallel"', "return_hidden_states"]

SPLITS = [("2026-11-06", "2026-11-09"), ("2026-11-09", "2026-11-11"), ("2026-11-11", "2026-11-14")]


def build_queries(repo, phrases, extra_rfc=True, created=True):
    qs = []
    for p in phrases:
        qs.append(f"repo:{repo} {p} updated:>={START_DATE}")
    if extra_rfc:
        qs.append(f'repo:{repo} label:"RFC" updated:>={START_DATE}')
    if created:
        for a, b in SPLITS:
            qs.append(f"repo:{repo} is:pr created:{a}..{b}")
            qs.append(f"repo:{repo} is:issue created:{a}..{b}")
    return qs


def norm(it, repo):
    pr = it.get("pull_request") or {}
    return {
        "repo": repo,
        "number": it["number"],
        "title": it.get("title", ""),
        "state": it.get("state"),
        "created_at": it.get("created_at"),
        "updated_at": it.get("updated_at"),
        "closed_at": it.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "is_pr": "pull_request" in it,
        "draft": bool(it.get("draft")),
        "url": it.get("html_url"),
        "labels": [la.get("name") for la in it.get("labels", [])],
        "body": (it.get("body") or "")[:1500],
    }


def run(name, groups):
    store = {}
    index = []
    for repo, phrases in groups:
        for q in build_queries(repo, phrases):
            try:
                res = ghcli.search_issues(q, per_page=100, max_pages=10)
            except Exception as exc:  # noqa: BLE001
                print("FAIL", q, exc, flush=True)
                index.append({"repo": repo, "query": q, "error": str(exc)})
                continue
            items = [norm(i, repo) for i in res["items"]]
            for it in items:
                key = (repo, it["number"])
                rec = store.setdefault(key, dict(it, queries=[]))
                if q not in rec["queries"]:
                    rec["queries"].append(q)
                if len(it["body"]) > len(rec.get("body") or ""):
                    rec["body"] = it["body"]
            prs = sum(1 for i in items if i["is_pr"])
            index.append({"repo": repo, "query": q, "total": res["total"], "returned": len(items), "prs": prs, "issues": len(items) - prs})
            print(f"{str(res['total']):>6} {len(items):>4} {q}", flush=True)
    out = {"window_start": START, "queries": index, "items": sorted(store.values(), key=lambda r: (r["repo"], -r["number"]))}
    path = os.path.join(ROOT, name)
    json.dump(out, open(path, "w", encoding="utf-8"))
    print("unique", len(out["items"]), "prs", sum(1 for r in out["items"] if r["is_pr"]), "issues", sum(1 for r in out["items"] if not r["is_pr"]))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    if which == "main":
        run("discovery.json", [(SGL, SGL_HIDDEN + SGL_SAMPLING), (VLLM, VLLM_HIDDEN)])
    else:
        run("discovery-ascend.json", [(ASC, ASC_HIDDEN)])
