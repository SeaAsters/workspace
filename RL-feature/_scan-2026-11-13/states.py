# -*- coding: utf-8 -*-
"""State lookup for tracked numbers in the 2026-11-13 window."""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-11-06T06:35:52Z"
ASC = "vllm-project/vllm-ascend"
WANT = {
    "sgl-project/sglang": [48569, 48659, 48188, 47493, 47750, 45142, 48412, 47445, 42248, 43902, 41916, 38565, 43517, 47967, 48487, 43047, 47047, 46320, 50155, 50126, 50039, 48948, 49664, 38589],
    "vllm-project/vllm": [57736, 66050, 65991, 65250, 60765, 64320, 61180, 63205, 63406, 61539, 66785, 66794, 66323, 66311, 56442, 52074, 66546, 66825, 66521, 66797],
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
                print("%-8s %6d NOT-IN-WINDOW" % (repo.split("/")[-1], n))
                continue
            cls = "NEW" if i["created_at"] >= START else "old"
            print("%-8s %6d %s %s m=%s u=%s %s" % (repo.split("/")[-1], n, cls, i["state"], (i.get("merged_at") or "-")[:10], i["updated_at"][:10], i["title"][:62]))


if __name__ == "__main__":
    main()
