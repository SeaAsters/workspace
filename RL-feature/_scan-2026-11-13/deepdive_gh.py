# -*- coding: utf-8 -*-
"""Deep-dive via authenticated gh CLI: PR detail, timeline, commits, files."""
import json
import os
import sys

import ghcli

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "deepdive.json")

SGL = "sgl-project/sglang"
VLLM = "vllm-project/vllm"
ASC = "vllm-project/vllm-ascend"

ITEMS = [
    (SGL, 38589, "issue"),
    (SGL, 47047, "pr"),
    (SGL, 48188, "pr"),
    (SGL, 50039, "pr"),
    (SGL, 50126, "pr"),
    (SGL, 50155, "pr"),
    (VLLM, 66785, "pr"),
    (VLLM, 66794, "issue"),
    (VLLM, 66323, "pr"),
    (VLLM, 56442, "pr"),
    (VLLM, 65991, "pr"),
    (VLLM, 57736, "pr"),
    (VLLM, 64320, "pr"),
    (VLLM, 61539, "issue"),
    (ASC, 21792, "pr"),
    (ASC, 17557, "pr"),
    (ASC, 11066, "pr"),
    (ASC, 21735, "pr"),
    (ASC, 21023, "pr"),
    (ASC, 21501, "issue"),
]

NOISE = {
    "review_requested", "labeled", "mentioned", "subscribed", "head_ref_deleted",
    "unlabeled", "review_dismissed", "milestoned", "added_to_project_v2",
    "project_v2_item_status_changed", "deployed", "auto_squash_enabled",
    "auto_merge_disabled", "automatic_base_change_succeeded", "parent_issue_added",
}


def timeline(repo, num):
    events = ghcli.api_items(f"repos/{repo}/issues/{num}/timeline")
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
        if ev.get("body"):
            e["body"] = ev["body"][:1500]
        if ev.get("event") == "reviewed":
            e["state"] = ev.get("state")
        slim.append(e)
    return slim


def main():
    store = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for repo, num, kind in ITEMS:
        key = f"{repo}#{num}"
        rec = store.get(key, {})
        rec.update({"repo": repo, "number": num, "kind": kind})
        if kind == "pr":
            pr = ghcli.api(f"repos/{repo}/pulls/{num}")
            rec["pr"] = {
                "state": pr.get("state"),
                "draft": pr.get("draft"),
                "title": pr.get("title"),
                "created_at": pr.get("created_at"),
                "closed_at": pr.get("closed_at"),
                "merged_at": pr.get("merged_at"),
                "merged_by": (pr.get("merged_by") or {}).get("login"),
                "base_ref": (pr.get("base") or {}).get("ref"),
                "head_ref": (pr.get("head") or {}).get("ref"),
                "head_sha": (pr.get("head") or {}).get("sha"),
                "additions": pr.get("additions"),
                "deletions": pr.get("deletions"),
                "changed_files": pr.get("changed_files"),
                "labels": [la["name"] for la in pr.get("labels", [])],
                "body": pr.get("body"),
            }
            commits = ghcli.api_items(f"repos/{repo}/pulls/{num}/commits")
            cl = []
            for c in commits[-25:]:
                cl.append({
                    "sha": c["sha"],
                    "subject": (c["commit"]["message"].split("\n")[0])[:200],
                    "date": c["commit"]["author"]["date"],
                })
            files = ghcli.api_items(f"repos/{repo}/pulls/{num}/files")
            rec["patch"] = {"commits": cl, "files": [f["filename"] for f in files][:80]}
        else:
            iss = ghcli.api(f"repos/{repo}/issues/{num}")
            rec["issue"] = {
                "state": iss.get("state"),
                "title": iss.get("title"),
                "created_at": iss.get("created_at"),
                "closed_at": iss.get("closed_at"),
                "labels": [la["name"] for la in iss.get("labels", [])],
                "body": iss.get("body"),
            }
        rec["timeline"] = timeline(repo, num)
        store[key] = rec
        print(key, "events", len(rec["timeline"]), "commits", len((rec.get("patch") or {}).get("commits") or []), "files", len((rec.get("patch") or {}).get("files") or []), flush=True)
        json.dump(store, open(OUT, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
