# -*- coding: utf-8 -*-
"""Print created dates and auto classification for the selected items."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-30T02:34:03Z"
ASC = "vllm-project/vllm-ascend"
WANT = {
    "sgl-project/sglang": [47445, 42248, 43902, 48569, 48659, 48660, 48188, 47493, 36829, 47750, 45142, 48412, 41916, 38565, 43517, 47967, 48487, 43047],
    "vllm-project/vllm": [57736, 66050, 65991, 65250, 60765, 64320, 61180, 63205, 63406, 61539],
    "vllm-project/vllm-ascend": [21023, 21022, 16621, 16993, 19157, 8513, 17907, 21056, 20617],
}


def main():
    idx = {}
    for name, repo in (("discovery.json", None), ("discovery-ascend.json", ASC)):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        for it in json.load(open(path, encoding="utf-8"))["items"]:
            r = it.get("repo", repo)
            if r:
                idx[(r, it["number"])] = it
    for repo, nums in WANT.items():
        for n in nums:
            i = idx.get((repo, n))
            if not i:
                print(repo.split("/")[-1], n, "MISSING")
                continue
            created = i["created_at"]
            cls = "本期新建" if created >= START else "前期补漏续跟"
            kind = "PR" if i["is_pr"] else "IS"
            print("%-8s %6d created=%s %s %s %s %s" % (repo.split("/")[-1], n, created[:10], cls, kind, i["state"], (i.get("merged_at") or "")[:10]))


if __name__ == "__main__":
    main()
