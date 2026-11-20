"""Rank discovered items by feature-relevance signals in title+body, with evidence snippets."""
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))

SIG = [
    (r"aux[ _-]?hidden|aux hidden", 6),
    (r"capture (the )?aux|aux .{0,20}captur", 5),
    (r"return[ _-]hidden[ _-]states?|extract[ _-]hidden[ _-]states?", 6),
    (r"forward observer", 5),
    (r"deferred (output|response)", 4),
    (r"hidden[ _-]states? .{0,15}(gather|relay|transport|over pp|across pp)", 5),
    (r"relay .{0,20}hidden", 5),
    (r"hidden[ _-]states? capture", 5),
    (r"return_sampling_mask|sampling[ _-]?masks?|sample[ _-]?mask", 5),
    (r"top-[pk][ _-]?(mask|support)", 4),
    (r"mask capture", 4),
    (r"dflash|dspark|eagle", 1),
    (r"npu|ascend", 1),
    (r"rl\b|reinforcement|training", 1),
]
NEG = [
    (r"diffusion|image|video|wan|flux|ltx|hunyuanimage|bria", -4),
    (r"quantiz|nvfp4|mxfp4|fp8|w4a8|w8a8|awq", -2),
    (r"lora|watermark|embedding|pooling", -2),
    (r"moe\b.{0,20}(router|marlin|topk)", -2),
    (r"grammar|tool[- ]call|reasoning", -1),
]


def score(text):
    s = 0
    for pat, w in SIG:
        if re.search(pat, text, re.I):
            s += w
    for pat, w in NEG:
        if re.search(pat, text, re.I):
            s += w
    return s


def snippets(text, limit=2):
    out = []
    for line in text.splitlines():
        if re.search(r"aux[ _-]?hidden|hidden_states|hidden state|sampling[ _-]?mask|top-[pk]|mask capture|observer|deferred", line, re.I):
            line = line.strip()
            if 8 < len(line) < 300 and line not in out:
                out.append(line[:220])
            if len(out) >= limit:
                break
    return out


def main():
    cand = json.load(open(os.path.join(ROOT, "candidates.json"), encoding="utf-8"))
    rows = []
    for i in cand["items"]:
        text = (i["title"] or "") + "\n" + (i.get("body") or "")
        s = score(text) + (2 if i["in_baseline"] else 0)
        rows.append((s, i, text))
    for repo in ("sgl-project/sglang", "vllm-project/vllm"):
        sub = sorted([r for r in rows if r[1]["repo"] == repo], key=lambda x: (-x[0], x[1]["updated_at"] or ""))
        print(f"\n######## {repo}")
        for s, i, text in sub:
            if s < 5:
                continue
            kind = "PR" if i["is_pr"] else "IS"
            mn = (i.get("merged_at") or "")[:10] or "-"
            base = "BASE" if i["in_baseline"] else ("NEW" if i["new"] else "old")
            print(f"[{s:>2}] {kind} {i['number']:>6} {i['state'][:6]:6s} m={mn:10s} u={i['updated_at'][:10]} {base:4s} {i['title'][:110]}")
            for sn in snippets(text):
                print(f"        · {sn}")


if __name__ == "__main__":
    main()
