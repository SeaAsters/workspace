# -*- coding: utf-8 -*-
"""Count in-window force-pushes / renames for the ascend records."""
import collections
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-23T02:05:41Z"
d = json.load(open(os.path.join(ROOT, "deepdive.json"), encoding="utf-8"))
for n in [20075, 19284, 19537, 17557, 19801, 19150]:
    r = d.get(f"vllm-project/vllm-ascend#{n}", {})
    c = collections.Counter(
        e.get("event") for e in (r.get("timeline") or []) if (e.get("created_at") or "") >= START
    )
    print(n, dict(c), "commits", len(r.get("patch", {}).get("commits") or []))
