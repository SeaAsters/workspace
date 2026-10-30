# -*- coding: utf-8 -*-
"""Focused discovery for vllm-project/vllm-ascend hidden-state work."""
import json
import os

import gh

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = "vllm-project/vllm-ascend"
START_DATE = "2026-10-01"

PHRASES = [
    '"hidden states"',
    '"hidden state"',
    "hidden_states",
    "aux_hidden_states",
    '"aux hidden"',
    "extract_hidden_states",
    "return_hidden_states",
    '"hidden-state"',
    '"pipeline parallel" hidden',
    '"sampling mask"',
    '"spec decode" "pipeline parallel"',
]

QUERIES = []
for p in PHRASES:
    QUERIES.append(f"repo:{REPO} {p} updated:>={START_DATE}")
QUERIES.append(f'repo:{REPO} label:"RFC" updated:>={START_DATE}')
QUERIES.append(f"repo:{REPO} is:pr created:>=2026-10-23")
QUERIES.append(f"repo:{REPO} is:issue created:>=2026-10-23")


def main():
    items = {}
    index = []
    for q in QUERIES:
        try:
            res = gh.search(q, kind="issues", per_page=100, max_pages=10)
        except Exception as exc:  # noqa: BLE001
            print("FAIL", q, exc)
            continue
        total = res.get("total")
        for it in res["items"]:
            pr = it.get("pull_request") or {}
            rec = items.setdefault(
                it["number"],
                {
                    "number": it["number"],
                    "title": it["title"],
                    "state": it["state"],
                    "is_pr": "pull_request" in it,
                    "merged_at": pr.get("merged_at"),
                    "created_at": it.get("created_at"),
                    "updated_at": it.get("updated_at"),
                    "labels": [la["name"] for la in it.get("labels", [])],
                    "body": (it.get("body") or "")[:1500],
                    "queries": [],
                },
            )
            rec["queries"].append(q)
        index.append({"query": q, "total": total, "returned": len(res["items"])})
        print(f"{str(total):>6} {len(res['items']):>4} {q}")
    out = {"repo": REPO, "index": index, "items": sorted(items.values(), key=lambda r: -r["number"])}
    with open(os.path.join(ROOT, "discovery-ascend.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("unique", len(items), "prs", sum(1 for r in items.values() if r["is_pr"]), "issues", sum(1 for r in items.values() if not r["is_pr"]))


if __name__ == "__main__":
    main()
