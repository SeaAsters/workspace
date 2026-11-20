# -*- coding: utf-8 -*-
"""Rank vllm-ascend items for hidden-state feature relevance."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-23T02:05:41Z"

SIG = [
    (r"aux[ _-]?hidden|aux_hidden_state", 8),
    (r"extract_hidden_states", 8),
    (r"hidden[ _-]?states? (over|across|relay|transport).*(pp|pipeline)", 8),
    (r"(pp|pipeline parallel).*(hidden|aux)", 6),
    (r"return[ _-]hidden[ _-]states?", 7),
    (r"eagle3|eagle 3", 3),
    (r"dflash|dspark|mtp|speculat", 3),
    (r"sampling[ _-]?mask", 4),
    (r"rejection sampl", 2),
    (r"model runner v2|mrv2", 2),
    (r"pcp|dcp|sequence parallel", 2),
]
NEG = [
    (r"quantiz|nvfp4|mxfp4|fp8|w4a8|w8a8|awq|int4", -2),
    (r"document|readme|release note|cookbook|tutorial|translat", -3),
    (r"ci|workflow|pipeline|nightly|flaky|e2e test", -1),
]


def score(text):
    s = 0
    for pat, w in SIG:
        if re.search(pat, text, re.I):
            s += w
    for pat, w in NEG:
        if re.search(pat, text, re.I):
            s += w
    return s


def main():
    d = json.load(open(os.path.join(ROOT, "discovery-ascend.json"), encoding="utf-8"))
    rows = []
    for it in d["items"]:
        text = (it["title"] or "") + "\n" + (it.get("body") or "")
        if (it.get("updated_at") or "") < START:
            continue
        rows.append((score(text), it, text))
    rows.sort(key=lambda x: (-x[0], x[1]["updated_at"] or ""))
    print("scored", len(rows))
    for s, it, text in rows[:45]:
        if s < 5:
            break
        mn = (it.get("merged_at") or "")[:10] or "-"
        print(f"[{s:>2}] {'PR' if it['is_pr'] else 'IS'} {it['number']:>6} {it['state'][:6]:6s} m={mn:10s} u={it['updated_at'][:10]} {it['title'][:104]}")
        snip = [l.strip()[:170] for l in text.splitlines() if re.search(r"aux|hidden state|hidden_states|relay|sampling mask|extract_hidden", l, re.I)][:2]
        for x in snip:
            print("        ·", x)


if __name__ == "__main__":
    main()
