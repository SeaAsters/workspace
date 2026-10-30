"""Compact evidence summary per curated item."""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-23T02:05:41Z"


def patch_commits(repo, number):
    tag = repo.split("/")[1]
    paths = glob.glob(os.path.join(ROOT, "raw", "patch", f"{tag}-{number}-*.patch"))
    if not paths:
        return []
    txt = open(paths[0], encoding="utf-8", errors="replace").read()
    out = []
    for m in re.finditer(r"^From ([0-9a-f]{40}) .*?\n(.*?)(?=^From [0-9a-f]{40} |\Z)", txt, flags=re.M | re.S):
        sha, block = m.group(1), m.group(2)
        subj = ""
        date = ""
        for ln in block.split("\n")[:10]:
            if ln.startswith("Subject:"):
                subj = ln[8:].strip()
            if ln.startswith("Date:"):
                date = ln[5:].strip()
        out.append({"sha": sha, "subject": subj, "date": date})
    return out


def main():
    keys = sys.argv[1:]
    store = json.load(open(os.path.join(ROOT, "deepdive.json"), encoding="utf-8"))
    for key in keys:
        rec = store.get(key)
        if not rec:
            print("MISSING", key)
            continue
        pr = rec.get("pr") or rec.get("issue") or {}
        print(f"\n===== {key} [{rec.get('kind')}] {pr.get('state')} merged={pr.get('merged_at')} base={pr.get('base_ref')} draft={pr.get('draft')} head={str(pr.get('head_sha'))[:10]}")
        print(f"      title: {pr.get('title')}")
        if pr.get("labels"):
            print(f"      labels: {pr['labels']}")
        if rec.get("kind") == "pr":
            print(f"      +{pr.get('additions')}/-{pr.get('deletions')} files={pr.get('changed_files')} merged_by={pr.get('merged_by')}")
        evs = rec.get("timeline") or []
        noise = {"review_requested", "labeled", "mentioned", "subscribed", "head_ref_deleted", "unlabeled", "review_dismissed", "milestoned"}
        inwin = [e for e in evs if (e.get("created_at") or "") >= START and e.get("event") not in noise]
        print(f"      events total={len(evs)} in_window={len(inwin)} (+noise filtered)")
        for e in inwin:
            kind = e.get("event")
            extra = ""
            if kind in ("committed", "head_ref_force_pushed"):
                extra = str(e.get("sha"))[:10]
            if kind == "reviewed":
                extra = e.get("state") or ""
            if kind == "labeled":
                extra = ""
            body = (e.get("body") or "").replace("\n", " ")
            if len(body) > 160:
                body = body[:160] + "…"
            print(f"         {e.get('created_at','')[:19]} {kind:24s} {extra:12s} {e.get('actor') or '':16s} {body}")
        pc = patch_commits(rec["repo"], rec["number"])
        print(f"      patch commits={len(pc)}")
        for c in pc[:14]:
            print(f"         {c['date'][:31]:32s} {c['sha'][:10]} {c['subject'][:100]}")
        files = (rec.get("patch") or {}).get("files") or []
        print(f"      files({len(files)}): {', '.join(files[:10])}")


if __name__ == "__main__":
    main()
