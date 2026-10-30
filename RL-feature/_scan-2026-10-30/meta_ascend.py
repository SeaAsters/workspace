# -*- coding: utf-8 -*-
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(ROOT, "deepdive.json"), encoding="utf-8"))
nums = [20075, 19284, 19537, 17557, 19801, 19150, 20231, 19845, 19866, 20129, 20288, 19987, 19641, 19593]
for n in nums:
    r = d.get(f"vllm-project/vllm-ascend#{n}", {})
    pr = r.get("pr") or r.get("issue") or {}
    kind = "pr" if r.get("pr") else "issue"
    print(
        n,
        kind,
        "state=" + str(pr.get("state")),
        "merged=" + str(pr.get("merged_at")),
        "created=" + str(pr.get("created_at")),
        "draft=" + str(pr.get("draft")),
        "head=" + str(pr.get("head_sha"))[:10],
        "+%s/-%s" % (pr.get("additions"), pr.get("deletions")),
        "files=" + str(pr.get("changed_files")),
    )
