"""Tight shortlist: strong feature signals only."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-23T02:05:41Z"

STRONG_HID = re.compile(
    r"aux[ _-]?hidden"
    r"|hidden[- _]states? (capture|captured|capturing|output|return|forward|extract|gather)"
    r"|capture (the )?aux"
    r"|return[ _-]hidden[ _-]states?"
    r"|extract[ _-]hidden[ _-]states?"
    r"|forward observer"
    r"|deferred (output|response)"
    r"|relaying aux"
    r"|hidden[ _-]states?[ _-]mode",
    re.I,
)
STRONG_MASK = re.compile(
    r"sampling[ _-]?masks?|sample[ _-]?mask|top-[pk][ _-]?(mask|support)|mask capture|return_sampling_mask",
    re.I,
)


def main():
    cand = json.load(open(os.path.join(ROOT, "candidates.json"), encoding="utf-8"))
    items = cand["items"]
    for repo in ("sgl-project/sglang", "vllm-project/vllm"):
        sub = [i for i in items if i["repo"] == repo]
        strong_h = [i for i in sub if STRONG_HID.search(i["title"] or "")]
        strong_m = [i for i in sub if STRONG_MASK.search(i["title"] or "")]
        print(f"\n############ {repo}")
        print(f"strong hidden-title hits: {len(strong_h)}   strong mask-title hits: {len(strong_m)}")
        for tag, rows in (("HIDDEN", strong_h), ("MASK", strong_m)):
            print(f"-- {tag}")
            for i in sorted(rows, key=lambda x: x["updated_at"] or "", reverse=True):
                kind = "PR" if i["is_pr"] else "IS"
                mn = (i.get("merged_at") or "")[:10] or "-"
                base = "BASE" if i["in_baseline"] else ("NEW" if i["new"] else "old")
                print(
                    f"   {kind} {i['number']:>6} {i['state'][:6]:6s} m={mn:10s} u={i['updated_at'][:10]} {base:4s} {i['title']}"
                )


if __name__ == "__main__":
    main()
