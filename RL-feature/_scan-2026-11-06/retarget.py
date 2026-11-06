# -*- coding: utf-8 -*-
"""Retarget the copied scan scripts from the 2026-10-30 run to the 2026-11-06 run."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_START = "2026-10-23T02:05:41Z"
NEW_START = "2026-10-30T02:34:03Z"


def patch(name, subs, required=True):
    path = os.path.join(ROOT, name)
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if old not in s:
            if required:
                print("MISS", name, old[:70].replace("\n", "\\n"))
            continue
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", name)


patch("collect_commits.py", [(OLD_START, NEW_START)])

patch(
    "discover.py",
    [
        (OLD_START, NEW_START),
        ('START_DATE = "2026-10-23"', 'START_DATE = "2026-10-30"'),
        ('END_DATE = "2026-10-31"', 'END_DATE = "2026-11-07"'),
        (
            'CREATED_SPLITS = [\n    ("2026-10-23", "2026-10-26"),\n    ("2026-10-26", "2026-10-29"),\n    ("2026-10-29", "2026-10-31"),\n]',
            'CREATED_SPLITS = [\n    ("2026-10-30", "2026-11-01"),\n    ("2026-11-01", "2026-11-03"),\n    ("2026-11-03", "2026-11-05"),\n    ("2026-11-05", "2026-11-07"),\n]',
        ),
    ],
)

patch(
    "discover_ascend.py",
    [
        ('START_DATE = "2026-10-01"', 'START_DATE = "2026-10-23"'),
        ('created:>=2026-10-23")', 'created:>=2026-10-30")'),
    ],
)

patch(
    "analyze.py",
    [
        (OLD_START, NEW_START),
        (
            '    os.path.join(WS, "extract-hidden-status", "2026-10-23", "extract-hidden-states-delta-2026-10-23.json"),',
            '    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.json"),\n'
            '    os.path.join(WS, "extract-hidden-status", "2026-10-30", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-10-30.json"),\n'
            '    os.path.join(WS, "extract-hidden-status", "2026-10-23", "extract-hidden-states-delta-2026-10-23.json"),',
        ),
        (
            '    os.path.join(WS, "sample mask", "2026-10-23", "sample-mask-delta-2026-10-23.json"),',
            '    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),\n'
            '    os.path.join(WS, "sample mask", "2026-10-23", "sample-mask-delta-2026-10-23.json"),',
        ),
    ],
)

patch(
    "build.py",
    [
        ("for the 2026-10-30 delta reports", "for the 2026-11-06 delta reports"),
        (OLD_START, NEW_START),
        ('OUT = os.path.join(WS, "extract-hidden-status", "2026-10-30")', 'OUT = os.path.join(WS, "extract-hidden-status", "2026-11-06")'),
        ('OUT_SM = os.path.join(WS, "sample mask", "2026-10-30")', 'OUT_SM = os.path.join(WS, "sample mask", "2026-11-06")'),
        ("_scan-2026-10-30/README.md", "_scan-2026-11-06/README.md"),
        (
            '    os.path.join(WS, "extract-hidden-status", "2026-10-23", "extract-hidden-states-delta-2026-10-23.json"),',
            '    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.json"),\n'
            '    os.path.join(WS, "extract-hidden-status", "2026-10-30", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-10-30.json"),\n'
            '    os.path.join(WS, "extract-hidden-status", "2026-10-23", "extract-hidden-states-delta-2026-10-23.json"),',
        ),
        (
            '    os.path.join(WS, "sample mask", "2026-10-23", "sample-mask-delta-2026-10-23.json"),',
            '    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),\n'
            '    os.path.join(WS, "sample mask", "2026-10-23", "sample-mask-delta-2026-10-23.json"),',
        ),
        ('"sglang-extract-hidden-states-delta-2026-10-30"', '"sglang-extract-hidden-states-delta-2026-11-06"'),
        ('"sample-mask-delta-2026-10-30"', '"sample-mask-delta-2026-11-06"'),
        ('"vllm-extract-hidden-states-delta-2026-10-30"', '"vllm-extract-hidden-states-delta-2026-11-06"'),
    ],
)

patch(
    "check.py",
    [
        ('"2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30"', '"2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06"'),
        ('"2026-10-30", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-10-30"', '"2026-11-06", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-06"'),
        ('"2026-10-30", "sample-mask-delta-2026-10-30"', '"2026-11-06", "sample-mask-delta-2026-11-06"'),
    ],
)

patch(
    "check_links.py",
    [
        ('"2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.md"', '"2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.md"'),
        ('"2026-10-30", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-10-30.md"', '"2026-11-06", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-06.md"'),
        ('"2026-10-30", "sample-mask-delta-2026-10-30.md"', '"2026-11-06", "sample-mask-delta-2026-11-06.md"'),
    ],
)
