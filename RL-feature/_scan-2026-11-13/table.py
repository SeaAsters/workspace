# -*- coding: utf-8 -*-
import json
import os

WS = r"D:\WorkSpace\workspace\RL-feature"
paths = [
    os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang-extract-hidden-states-delta-2026-10-30.json"),
    os.path.join(WS, "extract-hidden-status", "2026-10-30", "vllm-extract-hidden-states-delta-2026-10-30.json"),
    os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30.json"),
]
for p in paths:
    d = json.load(open(p, encoding="utf-8-sig"))
    print("=====", os.path.basename(p))
    print("cutoff", d["cutoff_utc"], "main", d["main_sha"], "records", d["record_count"], d["classification_counts"])
    for r in d["records"]:
        print(
            "  %6d %-5s %-8s %s state=%-6s period=%-6s merged=%s created=%s head=%s"
            % (
                r["number"],
                r["kind"],
                r["classification"],
                r["priority"],
                r["current_state"],
                r["period_end_state"],
                r["merged_at"],
                r["created_at"],
                str(r["head_sha"])[:10],
            )
        )
