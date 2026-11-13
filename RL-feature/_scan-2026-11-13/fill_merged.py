# -*- coding: utf-8 -*-
"""Fill deep-dive evidence for merged items straight from the local shallow clones.

For each (repo, number) whose merge commit subject carries "(#number)", write a
synthetic record: pr state + merged_at (commit date) + commits/files of that commit.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEEP = os.path.join(ROOT, "deepdive.json")

REPOS = {
    "sgl-project/sglang": "sglang",
    "vllm-project/vllm": "vllm",
    "vllm-project/vllm-ascend": "vllm-ascend",
}

ITEMS = [
    ("sgl-project/sglang", 49664),
    ("sgl-project/sglang", 47750),
    ("sgl-project/sglang", 47493),
    ("sgl-project/sglang", 48412),
    ("sgl-project/sglang", 48487),
    ("sgl-project/sglang", 43517),
    ("vllm-project/vllm", 66177),
    ("vllm-project/vllm", 66063),
    ("vllm-project/vllm", 66546),
    ("vllm-project/vllm-ascend", 21439),
]


def main():
    store = json.load(open(DEEP, encoding="utf-8")) if os.path.exists(DEEP) else {}
    commits = {}
    for repo, tag in REPOS.items():
        path = os.path.join(ROOT, f"commits-{tag}.json")
        if not os.path.exists(path):
            continue
        data = json.load(open(path, encoding="utf-8"))
        commits[repo] = data["commits"]
    disc = {}
    for name in ("discovery.json",):
        if os.path.exists(os.path.join(ROOT, name)):
            for it in json.load(open(os.path.join(ROOT, name), encoding="utf-8"))["items"]:
                disc[(it["repo"], it["number"])] = it
    if os.path.exists(os.path.join(ROOT, "discovery-ascend.json")):
        for it in json.load(open(os.path.join(ROOT, "discovery-ascend.json"), encoding="utf-8"))["items"]:
            disc[("vllm-project/vllm-ascend", it["number"])] = it

    for repo, num in ITEMS:
        key = f"{repo}#{num}"
        hit = None
        for c in commits.get(repo, []):
            if re.search(rf"\(#{num}\)", c["subject"]):
                hit = c
                break
        if not hit:
            print("no merge commit for", key)
            continue
        meta = disc.get((repo, num), {})
        rec = store.get(key, {})
        rec.update({"repo": repo, "number": num, "kind": "pr"})
        rec["pr"] = {
            "state": "closed",
            "merged_at": hit["date"],
            "created_at": meta.get("created_at"),
            "title": hit["subject"],
            "base_ref": "main",
            "head_sha": hit["sha"],
            "draft": False,
        }
        rec["commits"] = [{"sha": hit["sha"], "subject": hit["subject"], "date": hit["date"]}]
        rec["files"] = hit["files"]
        rec["patch"] = {"commits": rec["commits"], "files": hit["files"]}
        rec.setdefault("timeline", [])
        store[key] = rec
        print("filled", key, hit["date"], hit["subject"][:70], len(hit["files"]), "files")
    json.dump(store, open(DEEP, "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
