# -*- coding: utf-8 -*-
"""Retarget copied scripts from the 2026-11-06 run to the 2026-11-13 run."""
import io
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_START = "2026-10-30T02:34:03Z"
NEW_START = "2026-11-06T06:35:52Z"
MAIN = {
    "0f3f619850f222b4f238ff3d336926e99a2a5c56": "4f8b6867534c8a6632845353997555fca857dd63",
    "de05fd1db93f712288e654bc28a8480a67a7a8e2": "c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15",
    "1c2efa8ac24ba14fd69f12609b850d4e6daf0865": "671292ee7ee2c6a620f307ce7930d84d772f4d5a",
}


def patch(name, subs):
    path = os.path.join(ROOT, name)
    s = io.open(path, encoding="utf-8").read()
    for old, new in subs:
        if old not in s:
            print("MISS", name, old[:60].replace("\n", "\\n"))
            continue
        s = s.replace(old, new)
    io.open(path, "w", encoding="utf-8", newline="\n").write(s)
    print("patched", name)


patch("analyze.py", [
    (OLD_START, NEW_START),
    ('    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.json"),',
     '    os.path.join(WS, "extract-hidden-status", "2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-06", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-06.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.json"),'),
    ('    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),',
     '    os.path.join(WS, "sample mask", "2026-11-06", "sample-mask-delta-2026-11-06.json"),\n'
     '    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),'),
])

patch("build.py", [
    ("for the 2026-11-06 delta reports", "for the 2026-11-13 delta reports"),
    (OLD_START, NEW_START),
    ('OUT = os.path.join(WS, "extract-hidden-status", "2026-11-06")', 'OUT = os.path.join(WS, "extract-hidden-status", "2026-11-13")'),
    ('OUT_SM = os.path.join(WS, "sample mask", "2026-11-06")', 'OUT_SM = os.path.join(WS, "sample mask", "2026-11-13")'),
    ("_scan-2026-11-06/README.md", "_scan-2026-11-13/README.md"),
    ('    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.json"),',
     '    os.path.join(WS, "extract-hidden-status", "2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-11-06", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-06.json"),\n'
     '    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30.json"),'),
    ('    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),',
     '    os.path.join(WS, "sample mask", "2026-11-06", "sample-mask-delta-2026-11-06.json"),\n'
     '    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),'),
    ('"sglang-extract-hidden-states-delta-2026-11-06"', '"sglang-extract-hidden-states-delta-2026-11-13"'),
    ('"sample-mask-delta-2026-11-06"', '"sample-mask-delta-2026-11-13"'),
    ('"vllm-extract-hidden-states-delta-2026-11-06"', '"vllm-extract-hidden-states-delta-2026-11-13"'),
] + [(k, v) for k, v in MAIN.items()])

patch("check.py", [
    ('"2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06"', '"2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13"'),
    ('"2026-11-06", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-06"', '"2026-11-13", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-13"'),
    ('"2026-11-06", "sample-mask-delta-2026-11-06"', '"2026-11-13", "sample-mask-delta-2026-11-13"'),
])

patch("check_links.py", [
    ('"2026-11-06", "sglang", "sglang-extract-hidden-states-delta-2026-11-06.md"', '"2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.md"'),
    ('"2026-11-06", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-06.md"', '"2026-11-13", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-13.md"'),
    ('"2026-11-06", "sample-mask-delta-2026-11-06.md"', '"2026-11-13", "sample-mask-delta-2026-11-13.md"'),
])
