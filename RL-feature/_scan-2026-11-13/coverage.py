# -*- coding: utf-8 -*-
"""Report evidence coverage per record for the 2026-11-13 build."""
import json
import os

WS = r"D:\WorkSpace\workspace\RL-feature"
paths = [
    os.path.join(WS, "extract-hidden-status", "2026-11-13", "sglang", "sglang-extract-hidden-states-delta-2026-11-13.json"),
    os.path.join(WS, "extract-hidden-status", "2026-11-13", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-11-13.json"),
    os.path.join(WS, "sample mask", "2026-11-13", "sample-mask-delta-2026-11-13.json"),
]
for p in paths:
    d = json.load(open(p, encoding="utf-8-sig"))
    print("==", os.path.basename(p), d["record_count"], d["classification_counts"])
    for r in d["records"]:
        print(
            "  %6d ev=%2d cm=%2d fl=%2d %-6s %s"
            % (r["number"], len(r["events"]), len(r["commits"]), len(r["changed_files_current_head"]), r["current_state"], r["classification"])
        )
