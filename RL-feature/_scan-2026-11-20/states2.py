import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-11-13T04:40:41Z"
ASC = "vllm-project/vllm-ascend"
WANT = {
 "sgl-project/sglang": [51475,50879,51566,50555,47047,48188,51074,50126,50155,50859,51376,51377,51378,50039],
 "vllm-project/vllm": [68813,68824,68769,67648,66323,68289,66785,65991,68634,57736,64320,56442,68871,68624],
 "vllm-project/vllm-ascend": [22122,22074,21469,21005,20756,21792,21023,21735,11066,17557],
}
idx = {}
for name, repo in (("discovery.json", None), ("discovery-ascend.json", ASC)):
    for it in json.load(open(os.path.join(ROOT, name), encoding="utf-8"))["items"]:
        r = it.get("repo", repo)
        if r: idx[(r, it["number"])] = it
for repo, nums in WANT.items():
    for n in nums:
        i = idx.get((repo, n))
        if not i:
            print("%-8s %6d NOT-IN-WINDOW" % (repo.split("/")[-1], n)); continue
        cls = "NEW" if i["created_at"] >= START else "old"
        print("%-8s %6d %s %-6s m=%-10s u=%s %s" % (repo.split("/")[-1], n, cls, i["state"], (i.get("merged_at") or "-")[:10], i["updated_at"][:10], i["title"][:58]))
