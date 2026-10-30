"""Discovery pass: keyword + created-in-window searches over GitHub issues/PRs."""
import json
import os
import re
import sys

import gh

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-23T02:05:41Z"
START_DATE = "2026-10-23"
END_DATE = "2026-10-31"

SGLANG = "sgl-project/sglang"
VLLM = "vllm-project/vllm"
VLLM_ASCEND = "vllm-project/vllm-ascend"

HIDDEN_PHRASES = [
    '"hidden states"',
    '"hidden state"',
    '"hidden-state"',
    "hidden_states",
    "return_hidden_states",
    "aux_hidden_states",
    "extract_hidden_states",
    '"aux hidden"',
    '"forward observer"',
]

SAMPLING_PHRASES = [
    '"sampling mask"',
    "sampling_mask",
    '"sample mask"',
    "sample_mask",
    "return_sampling_mask",
    '"top-p mask"',
    '"top-k mask"',
    '"mask capture"',
]

QUERIES = []
for p in HIDDEN_PHRASES:
    QUERIES.append((SGLANG, "hidden", f"repo:{SGLANG} {p} updated:>={START_DATE}"))
for p in SAMPLING_PHRASES:
    QUERIES.append((SGLANG, "sampling", f"repo:{SGLANG} {p} updated:>={START_DATE}"))
for p in HIDDEN_PHRASES:
    QUERIES.append((VLLM, "hidden", f"repo:{VLLM} {p} updated:>={START_DATE}"))
for p in HIDDEN_PHRASES[:6]:
    QUERIES.append((VLLM_ASCEND, "hidden", f"repo:{VLLM_ASCEND} {p} updated:>={START_DATE}"))

# RFC / design labels
QUERIES.append((SGLANG, "hidden", f'repo:{SGLANG} label:"RFC" updated:>={START_DATE}'))
QUERIES.append((VLLM, "hidden", f'repo:{VLLM} label:"RFC" updated:>={START_DATE}'))
QUERIES.append((SGLANG, "hidden", f'repo:{SGLANG} label:"rfc" updated:>={START_DATE}'))

# created-in-window sweeps (completeness against phrase drift), date-split to stay under the 1000-result cap
CREATED_SPLITS = [
    ("2026-10-23", "2026-10-26"),
    ("2026-10-26", "2026-10-29"),
    ("2026-10-29", "2026-10-31"),
]
for repo, topic in [(SGLANG, "hidden"), (SGLANG, "sampling"), (VLLM, "hidden")]:
    for a, b in CREATED_SPLITS:
        QUERIES.append((repo, topic, f"repo:{repo} is:pr created:{a}..{b}"))
        QUERIES.append((repo, topic, f"repo:{repo} is:issue created:{a}..{b}"))


def norm(item):
    pr = item.get("pull_request") or {}
    return {
        "number": item["number"],
        "title": item.get("title", ""),
        "state": item.get("state"),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "closed_at": item.get("closed_at"),
        "merged_at": pr.get("merged_at"),
        "is_pr": "pull_request" in item,
        "draft": bool(item.get("draft")),
        "url": item.get("html_url"),
        "comments": item.get("comments"),
        "labels": [la.get("name") for la in item.get("labels", [])],
        "body": (item.get("body") or "")[:2000],
    }


def main():
    all_items = {}
    index = []
    for repo, topic, q in QUERIES:
        try:
            res = gh.search(q, kind="issues", per_page=100, max_pages=10)
        except Exception as exc:  # noqa: BLE001
            print("FAILED", q, exc)
            index.append({"repo": repo, "topic": topic, "query": q, "error": str(exc)})
            continue
        items = [norm(i) for i in res["items"]]
        total = res.get("total")
        for it in items:
            key = (repo, it["number"])
            rec = all_items.setdefault(
                key,
                {
                    "repo": repo,
                    "number": it["number"],
                    "url": it["url"],
                    "title": it["title"],
                    "is_pr": it["is_pr"],
                    "state": it["state"],
                    "merged_at": it["merged_at"],
                    "created_at": it["created_at"],
                    "updated_at": it["updated_at"],
                    "closed_at": it["closed_at"],
                    "draft": it["draft"],
                    "comments": it["comments"],
                    "labels": it["labels"],
                    "topics": [],
                    "queries": [],
                    "body": it["body"],
                },
            )
            if topic not in rec["topics"]:
                rec["topics"].append(topic)
            if q not in rec["queries"]:
                rec["queries"].append(q)
            # prefer newest body text
            if it["body"] and len(it["body"]) > len(rec.get("body") or ""):
                rec["body"] = it["body"]
        prs = sum(1 for i in items if i["is_pr"])
        index.append(
            {
                "repo": repo,
                "topic": topic,
                "query": q,
                "total": total,
                "returned": len(items),
                "prs": prs,
                "issues": len(items) - prs,
            }
        )
        print(f"{(total if total is not None else -1):>5} {len(items):>4} {q}")

    records = sorted(all_items.values(), key=lambda r: (r["repo"], -r["number"]))
    out = {"window_start": START, "queries": index, "items": records}
    with open(os.path.join(ROOT, "discovery.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("unique items", len(records), "prs", sum(1 for r in records if r["is_pr"]), "issues", sum(1 for r in records if not r["is_pr"]))


if __name__ == "__main__":
    main()
