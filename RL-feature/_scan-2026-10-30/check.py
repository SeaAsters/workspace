# -*- coding: utf-8 -*-
"""Consistency checks across the three deliverables."""
import json
import os
import re

WS = r"D:\WorkSpace\workspace\RL-feature"
pairs = [
    (
        os.path.join(WS, "extract-hidden-status", "2026-10-30", "sglang", "sglang-extract-hidden-states-delta-2026-10-30"),
        "sglang",
    ),
    (
        os.path.join(WS, "extract-hidden-status", "2026-10-30", "vllm&vllm-ascend", "vllm-extract-hidden-states-delta-2026-10-30"),
        "vllm",
    ),
    (
        os.path.join(WS, "sample mask", "2026-10-30", "sample-mask-delta-2026-10-30"),
        "mask",
    ),
]
REQ = [
    "number", "kind", "title", "url", "topic", "priority", "classification", "activity",
    "created_at", "period_end_state", "current_state", "merged_at", "head_sha",
    "event_types", "events", "commits", "changed_files_current_head",
    "analysis_current_design", "npu", "validation",
]
ok = True
for base, tag in pairs:
    d = json.load(open(base + ".json", encoding="utf-8-sig"))
    md = open(base + ".md", encoding="utf-8").read()
    print("==", tag, "records", d["record_count"], "counts", d["classification_counts"])
    for k in ["repository", "topic", "start_utc", "cutoff_utc", "main_sha", "records", "scan_coverage", "tracked_baseline_audit", "candidate_decisions", "query_indexes", "limitations"]:
        if k not in d:
            print("  MISSING KEY", k)
            ok = False
    for r in d["records"]:
        for k in REQ:
            if k not in r:
                print("  record", r.get("number"), "missing", k)
                ok = False
        for k in ("analysis_current_design", "npu", "validation"):
            if not (r.get(k) or "").strip():
                print("  record", r["number"], "empty", k)
                ok = False
        if f"#{r['number']}" not in md:
            print("  record", r["number"], "not referenced in markdown")
            ok = False
        if r["current_state"] not in ("open", "closed", "merged"):
            print("  record", r["number"], "odd state", r["current_state"])
            ok = False
        for e in r["events"]:
            if (e.get("date") or "") < d["start_utc"]:
                print("  record", r["number"], "event before window", e["date"])
                ok = False
    # csv row count
    rows = open(base + ".csv", encoding="utf-8").read().strip().split("\n")
    print("   csv lines", len(rows), "(expect", d["record_count"] + 1, ")")
    if len(rows) != d["record_count"] + 1:
        ok = False
    print("   md lines", md.count("\n"))
print("ALL OK" if ok else "ISSUES FOUND")
