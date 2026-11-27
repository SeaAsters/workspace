# -*- coding: utf-8 -*-
"""Retarget copied scripts from the 2026-11-20 run to the 2026-11-27 run."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_START = "2026-11-13T04:40:41Z"
NEW_START = "2026-11-20T06:54:42Z"
OLD_DATE = "2026-11-20"
NEW_DATE = "2026-11-27"


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
patch("discover_gh.py", [
    (OLD_START, NEW_START),
    ('START_DATE = "2026-11-13"', 'START_DATE = "2026-11-20"'),
    ('SPLITS = [("2026-11-13", "2026-11-16"), ("2026-11-16", "2026-11-18"), ("2026-11-18", "2026-11-21")]',
     'SPLITS = [("2026-11-20", "2026-11-23"), ("2026-11-23", "2026-11-25"), ("2026-11-25", "2026-11-28")]'),
])
patch("ghcli.py", [("updated:>=2026-11-13", "updated:>=2026-11-20")])

patch("analyze.py", [
    (OLD_START, NEW_START),
    ('    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),',
     '    os.path.join(WS, "extract-hidden-status", "2026-11-20", "sglang", "sglang-extract-hidden-states-delta-2026-11-20.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-20", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-20.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),'),
    ('    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),',
     '    os.path.join(WS, "sample mask", "2026-11-20", "sample-mask-delta-2026-11-20.json"),\n'
     '    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),'),
])

patch("build.py", [
    ("for the 2026-11-20 delta reports", "for the 2026-11-27 delta reports"),
    (OLD_START, NEW_START),
    ('OUT = os.path.join(WS, "extract-hidden-status", "2026-11-20")', 'OUT = os.path.join(WS, "extract-hidden-status", "2026-11-27")'),
    ('OUT_SM = os.path.join(WS, "sample mask", "2026-11-20")', 'OUT_SM = os.path.join(WS, "sample mask", "2026-11-27")'),
    ("_scan-2026-11-20/README.md", "_scan-2026-11-27/README.md"),
    ('    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),',
     '    os.path.join(WS, "extract-hidden-status", "2026-11-20", "sglang", "sglang-extract-hidden-states-delta-2026-11-20.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-20", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-20.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),'),
    ('    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),',
     '    os.path.join(WS, "sample mask", "2026-11-20", "sample-mask-delta-2026-11-20.json"),\n'
     '    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),'),
    ('"sglang-extract-hidden-states-delta-2026-11-20"', '"sglang-extract-hidden-states-delta-2026-11-27"'),
    ('"sample-mask-delta-2026-11-20"', '"sample-mask-delta-2026-11-27"'),
    ('"vllm-extract-hidden-states-delta-2026-11-20"', '"vllm-extract-hidden-states-delta-2026-11-27"'),
])

for name in ("check.py", "check_links.py"):
    patch(name, [
        ('"2026-11-20", "sglang", "sglang-extract-hidden-states-delta-2026-11-20', '"2026-11-27", "sglang", "sglang-extract-hidden-states-delta-2026-11-27'),
        ('"2026-11-20", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-20', '"2026-11-27", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-27'),
        ('"2026-11-20", "sample-mask-delta-2026-11-20', '"2026-11-27", "sample-mask-delta-2026-11-27'),
    ])

patch("coverage.py", [(OLD_DATE, NEW_DATE)])
