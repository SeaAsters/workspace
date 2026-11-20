import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
idx = {}
for name, repo in (("discovery.json", None), ("discovery-ascend.json", "vllm-project/vllm-ascend")):
    for it in json.load(open(os.path.join(ROOT, name), encoding="utf-8"))["items"]:
        idx[(it.get("repo", repo), it["number"])] = it
for repo, num in [("sgl-project/sglang", 51475), ("sgl-project/sglang", 50879), ("sgl-project/sglang", 50859),
                  ("vllm-project/vllm", 68813), ("vllm-project/vllm", 68824), ("vllm-project/vllm", 68769),
                  ("vllm-project/vllm", 67648), ("vllm-project/vllm-ascend", 22122), ("vllm-project/vllm-ascend", 22074)]:
    i = idx.get((repo, num))
    print("#%d %s" % (num, "MISSING" if not i else i["title"][:90]))
    if i:
        print("   " + " ".join((i.get("body") or "").split())[:300])
