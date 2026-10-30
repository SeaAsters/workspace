# -*- coding: utf-8 -*-
"""Deep-dive for vllm-ascend items, reusing the shared store (deepdive.json)."""
import json
import os
import sys

import deepdive as dd
import gh

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = "vllm-project/vllm-ascend"
OUT = os.path.join(ROOT, "deepdive.json")

NUMBERS = [20075, 19284, 17557, 19537, 19801, 19150, 20231, 19845, 19866, 20129, 20288, 19987, 19641, 19593]


def main():
    only = sys.argv[1:]
    store = json.load(open(OUT, encoding="utf-8"))
    for num in NUMBERS:
        if only and str(num) not in only:
            continue
        key = f"{REPO}#{num}"
        rec = store.get(key, {})
        rec.update({"repo": REPO, "number": num, "kind": "pr" if num not in (20231, 19845, 19987, 19641) else "issue"})
        if not rec.get("pr") and rec["kind"] == "pr":
            data = gh.get(f"/repos/{REPO}/pulls/{num}")
            if data:
                rec["pr"] = {
                    "state": data.get("state"),
                    "draft": data.get("draft"),
                    "title": data.get("title"),
                    "created_at": data.get("created_at"),
                    "closed_at": data.get("closed_at"),
                    "merged_at": data.get("merged_at"),
                    "merged_by": (data.get("merged_by") or {}).get("login"),
                    "base_ref": (data.get("base") or {}).get("ref"),
                    "head_ref": (data.get("head") or {}).get("ref"),
                    "head_sha": (data.get("head") or {}).get("sha"),
                    "additions": data.get("additions"),
                    "deletions": data.get("deletions"),
                    "changed_files": data.get("changed_files"),
                    "labels": [la["name"] for la in data.get("labels", [])],
                    "body": data.get("body"),
                }
        if not rec.get("issue") and rec["kind"] == "issue":
            data = gh.get(f"/repos/{REPO}/issues/{num}")
            if data:
                rec["issue"] = {
                    "state": data.get("state"),
                    "title": data.get("title"),
                    "created_at": data.get("created_at"),
                    "closed_at": data.get("closed_at"),
                    "labels": [la["name"] for la in data.get("labels", [])],
                    "body": data.get("body"),
                }
        if not rec.get("timeline"):
            events = gh.paginated(f"/repos/{REPO}/issues/{num}/timeline", max_pages=3)
            slim = []
            for ev in events:
                e = {
                    "event": ev.get("event"),
                    "created_at": ev.get("created_at"),
                    "actor": (ev.get("actor") or {}).get("login"),
                    "sha": ev.get("sha"),
                    "commit_id": ev.get("commit_id"),
                    "url": ev.get("html_url") or ev.get("url"),
                }
                body = ev.get("body")
                if body:
                    e["body"] = body[:1500]
                if ev.get("event") == "reviewed":
                    e["state"] = ev.get("state")
                slim.append(e)
            rec["timeline"] = slim
        if not rec.get("patch") or not rec["patch"].get("commits"):
            rec["patch"] = dd.parse_patch(dd.fetch_patch(REPO, num))
        store[key] = rec
        print(key, len(rec.get("timeline") or []), len(rec.get("patch", {}).get("commits") or []), len(rec.get("patch", {}).get("files") or []), flush=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(store, fh)


if __name__ == "__main__":
    main()
