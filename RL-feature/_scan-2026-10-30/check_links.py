# -*- coding: utf-8 -*-
"""Validate relative links in the three reports."""
import os
import re

WS = r"D:\WorkSpace\workspace\RL-feature"
reports = [
    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.md"),
    os.path.join(WS, "extract-hidden-status", "2026-10-30", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-10-30.md"),
    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.md"),
]
ok = True
for p in reports:
    s = open(p, encoding="utf-8").read()
    d = os.path.dirname(p)
    links = set(re.findall(r"\]\(<([^>]+)>\)", s)) | set(re.findall(r"\]\((?!https?://|<)([^)#]+)\)", s))
    missing = [l for l in links if not os.path.exists(os.path.normpath(os.path.join(d, l)))]
    print(os.path.basename(p), "local links", len(links), "missing", missing)
    if missing:
        ok = False
print("LINKS OK" if ok else "BROKEN LINKS")
