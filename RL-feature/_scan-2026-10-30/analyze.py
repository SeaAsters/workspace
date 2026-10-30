"""Score discovered items for topical relevance and split into new / baseline / other."""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(ROOT)
START = "2026-10-23T02:05:41Z"

PREV = [
    os.path.join(WS, "extract-hidden-status", "2026-10-23", "extract-hidden-states-delta-2026-10-23.json"),
    os.path.join(WS, "extract-hidden-status", "2026-10-16", "extract-hidden-states-delta-2026-10-16.json"),
    os.path.join(WS, "extract-hidden-status", "2026-10-10", "extract-hidden-states-status.json"),
    os.path.join(WS, "sample mask", "2026-10-23", "sample-mask-delta-2026-10-23.json"),
    os.path.join(WS, "sample mask", "2026-10-16", "sample-mask-delta-2026-10-16.json"),
    os.path.join(WS, "sample mask", "2026-10-10", "sample-mask-delta-2026-10-10.json"),
    os.path.join(WS, "sample mask", "2026-09-29", "sample-mask-prs.json"),
]

HID = re.compile(
    r"hidden[ _-]?states?|aux[_ ]hidden|aux\.hidden|return_hidden|extract_hidden|capture_aux"
    r"|forward observer|hidden_states_mode|deferred output|return-hidden",
    re.I,
)
MASK = re.compile(
    r"sampling[ _-]?masks?|sample[ _-]?mask|top-[pk][ _-]?mask|mask capture|return_sampling_mask|samplingmask",
    re.I,
)
NEG = re.compile(
    r"quantiz|kv cache|fp8|nvfp4|mxfp8|w4a8|w8a8|lora|watermark|embedding|vision|vlm|mm\b",
    re.I,
)


def load_baseline():
    num = {}
    for path in PREV:
        if not os.path.exists(path):
            continue
        d = json.load(open(path, encoding="utf-8-sig"))
        if isinstance(d, list):
            for r in d:
                if isinstance(r, dict) and "number" in r:
                    num.setdefault(r["number"], {"topics": set(), "title": r.get("title"), "sources": set()})
                    num[r["number"]]["sources"].add(os.path.basename(path))
            continue
        for a in d.get("tracked_baseline_audit", []) or []:
            num.setdefault(a["number"], {"topics": set(), "title": a.get("title"), "sources": set()})
            num[a["number"]]["topics"].update(a.get("baseline_topics") or [])
            num[a["number"]]["sources"].add(os.path.basename(path))
        for r in d.get("records", []) or []:
            num.setdefault(r["number"], {"topics": set(), "title": r.get("title"), "sources": set()})
            num[r["number"]]["sources"].add(os.path.basename(path))
    return num


def main():
    disc = json.load(open(os.path.join(ROOT, "discovery.json"), encoding="utf-8"))
    base = load_baseline()
    items = []
    for it in disc["items"]:
        if it["repo"] != "sgl-project/sglang" and it["repo"] != "vllm-project/vllm":
            continue
        text = (it["title"] or "") + "\n" + (it.get("body") or "")
        th = bool(HID.search(text))
        tm = bool(MASK.search(text))
        if not (th or tm):
            continue
        it2 = dict(it)
        it2["match_hidden"] = th
        it2["match_mask"] = tm
        it2["in_baseline"] = it["number"] in base
        it2["baseline_topics"] = sorted(base.get(it["number"], {}).get("topics", []))
        it2["new"] = (it.get("created_at") or "") >= START
        it2["updated_in_window"] = (it.get("updated_at") or "") >= START
        it2["neg"] = bool(NEG.search(it["title"] or ""))
        items.append(it2)

    # add baseline items that had activity but were not caught by keyword searches
    seen = {(i["repo"], i["number"]) for i in items}
    missing_base = []
    for it in disc["items"]:
        if it["repo"] not in ("sgl-project/sglang", "vllm-project/vllm"):
            continue
        if it["number"] in base and (it["repo"], it["number"]) not in seen:
            if (it.get("updated_at") or "") < START:
                continue
            missing_base.append(
                {
                    "repo": it["repo"],
                    "number": it["number"],
                    "title": it["title"],
                    "state": it["state"],
                    "is_pr": it["is_pr"],
                    "created_at": it.get("created_at"),
                    "updated_at": it.get("updated_at"),
                    "merged_at": it.get("merged_at"),
                    "in_baseline": True,
                    "baseline_topics": sorted(base[it["number"]]["topics"]),
                    "new": False,
                    "updated_in_window": True,
                    "match_hidden": False,
                    "match_mask": False,
                    "neg": False,
                }
            )

    out = {
        "window_start": START,
        "baseline_numbers": sorted(base),
        "items": items,
        "baseline_activity_no_keyword": missing_base,
    }
    with open(os.path.join(ROOT, "candidates.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)

    for repo in ("sgl-project/sglang", "vllm-project/vllm"):
        sub = [i for i in items if i["repo"] == repo]
        print(f"\n########## {repo}: {len(sub)} keyword items (of {sum(1 for x in disc['items'] if x['repo']==repo)} discovered)")
        for grp, cond in (
            ("NEW", lambda i: i["new"]),
            ("BASELINE-ACTIVE", lambda i: (not i["new"]) and i["in_baseline"]),
            ("OTHER", lambda i: (not i["new"]) and not i["in_baseline"]),
        ):
            rows = [i for i in sub if cond(i)]
            print(f"-- {grp} ({len(rows)})")
            for i in sorted(rows, key=lambda x: x["updated_at"] or "", reverse=True):
                kind = "PR" if i["is_pr"] else "IS"
                tags = ("H" if i["match_hidden"] else "") + ("M" if i["match_mask"] else "")
                if i["neg"]:
                    tags += "~neg"
                print(
                    f"   {kind} {i['number']:>6} {i['state'][:6]:6s} m={str(i['merged_at'])[:10]:10s} c={i['created_at'][:10]} u={i['updated_at'][:10]} [{tags}] {i['title'][:96]}"
                )
    print(f"\nbaseline-activity-without-keyword: {len(missing_base)}")
    for i in missing_base:
        print(f"   {i['repo'].split('/')[1]:8s} {i['number']:>6} {i['state'][:6]:6s} u={i['updated_at'][:10]} {i['title'][:90]}")


if __name__ == "__main__":
    main()
