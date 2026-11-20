# -*- coding: utf-8 -*-
"""Retarget copied scripts from the 2026-11-13 run to the 2026-11-20 run."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_START = "2026-11-06T06:35:52Z"
NEW_START = "2026-11-13T04:40:41Z"
OLD_DATE = "2026-11-13"
NEW_DATE = "2026-11-20"


def patch(name, subs):
    path = os.path.join(ROOT, name)
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if old not in s:
            print("MISS", name, old[:70].replace("\n", "\\n"))
            continue
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", name)


patch("collect_commits.py", [(OLD_START, NEW_START)])
patch("discover_gh.py", [(OLD_START, NEW_START), ('START_DATE = "2026-11-06"', 'START_DATE = "2026-11-13"')])
patch("ghcli.py", [("updated:>=2026-11-06", "updated:>=2026-11-13")])

patch("analyze.py", [
    (OLD_START, NEW_START),
    ('    os.path.join(WS, "extract-hidden-status", "2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.json"),',
     '    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-13", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-13.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.json"),'),
    ('    os.path.join(WS, "sample mask", "2026-11-06", "sample-mask-delta-2026-11-06.json"),',
     '    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),\n'
     '    os.path.join(WS, "sample mask", "2026-11-06", "sample-mask-delta-2026-11-06.json"),'),
])

patch("build.py", [
    ("for the 2026-11-13 delta reports", "for the 2026-11-20 delta reports"),
    (OLD_START, NEW_START),
    ('OUT = os.path.join(WS, "extract-hidden-status", "2026-11-13")', 'OUT = os.path.join(WS, "extract-hidden-status", "2026-11-20")'),
    ('OUT_SM = os.path.join(WS, "sample mask", "2026-11-13")', 'OUT_SM = os.path.join(WS, "sample mask", "2026-11-20")'),
    ("_scan-2026-11-13/README.md", "_scan-2026-11-20/README.md"),
    ('    os.path.join(WS, "extract-hidden-status", "2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.json"),',
     '    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-13", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-13.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.json"),'),
    ('    os.path.join(WS, "sample mask", "2026-11-06", "sample-mask-delta-2026-11-06.json"),',
     '    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),\n'
     '    os.path.join(WS, "sample mask", "2026-11-06", "sample-mask-delta-2026-11-06.json"),'),
    ('"sglang-extract-hidden-states-delta-2026-11-13"', '"sglang-extract-hidden-states-delta-2026-11-20"'),
    ('"sample-mask-delta-2026-11-13"', '"sample-mask-delta-2026-11-20"'),
    ('"vllm-extract-hidden-states-delta-2026-11-13"', '"vllm-extract-hidden-states-delta-2026-11-20"'),
])

for name in ("check.py", "check_links.py"):
    patch(name, [
        ('"2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13', '"2026-11-20", "sglang", "sglang-extract-hidden-states-delta-2026-11-20'),
        ('"2026-11-13", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-13', '"2026-11-20", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-20'),
        ('"2026-11-13", "sample-mask-delta-2026-11-13', '"2026-11-20", "sample-mask-delta-2026-11-20'),
    ])

patch("coverage.py", [(OLD_DATE, NEW_DATE)])
