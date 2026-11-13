# -*- coding: utf-8 -*-
"""Print compact title/body snippets for selected discovery items."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SGL = "sgl-project/sglang"
VLLM = "vllm-project/vllm"
ASC = "vllm-project/vllm-ascend"
BATCHES = {
    "sgl": (SGL, [48569, 48659, 48188, 47493, 47750, 45142, 48412, 47967, 48487, 48660, 36829]),
    "vllm": (VLLM, [57736, 66050, 65991, 65250, 60765, 64320, 61180]),
    "asc": (ASC, [21023, 16621, 16993, 19157, 17907, 21056, 21022, 8513]),
}


def main():
    which = sys.argv[1]
    repo, nums = BATCHES[which]
    idx = {}
    for name in ("discovery.json", "discovery-ascend.json"):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        fallback = ASC if "ascend" in name else None
        for it in json.load(open(path, encoding="utf-8"))["items"]:
            if it.get("repo", fallback) == repo:
                idx[it["number"]] = it
    for n in nums:
        i = idx.get(n)
        if not i:
            print(f"#{n} MISSING")
            continue
        kind = "PR" if i["is_pr"] else "IS"
        body = " ".join((i.get("body") or "").split())[:260]
        print(f"#{n} {kind} {i['state']} m={str(i['merged_at'])[:10]} u={i['updated_at'][:10]} | {i['title'][:88]}")
        print("    " + body)


if __name__ == "__main__":
    main()
