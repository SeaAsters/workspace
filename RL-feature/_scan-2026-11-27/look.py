import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-11-20T06:54:42Z"
ASC = "vllm-project/vllm-ascend"
WANT = {
 "sgl-project/sglang": [52507,53074,47047,51566,51475,53418,52270,50859],
 "vllm-project/vllm": [69875,69064,53243,52073,68824,65991,50187],
 "vllm-project/vllm-ascend": [21831,22485,22307,22198],
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
        cls = "本期新建" if i["created_at"] >= START else "前期补漏续跟"
        print("%-8s %6d %s %-6s m=%-10s %s" % (repo.split("/")[-1], n, cls, i["state"], (i.get("merged_at") or "-")[:10], i["title"][:60]))
