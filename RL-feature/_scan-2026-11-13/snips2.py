import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
idx = {}
for name, repo in (("discovery.json", None), ("discovery-ascend.json", "vllm-project/vllm-ascend")):
    for it in json.load(open(os.path.join(ROOT, name), encoding="utf-8"))["items"]:
        idx[(it.get("repo", repo), it["number"])] = it
for repo, num in [("sgl-project/sglang", 50039), ("sgl-project/sglang", 50126), ("sgl-project/sglang", 50155), ("sgl-project/sglang", 38589),
                  ("vllm-project/vllm", 56442), ("vllm-project/vllm", 66323), ("vllm-project/vllm", 66785), ("vllm-project/vllm", 66177),
                  ("vllm-project/vllm-ascend", 21439), ("vllm-project/vllm-ascend", 21792), ("vllm-project/vllm-ascend", 21735)]:
    i = idx.get((repo, num))
    print("#%d %s" % (num, "MISSING" if not i else i["title"][:88]))
    if i:
        print("   " + " ".join((i.get("body") or "").split())[:240])
