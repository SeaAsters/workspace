"""Deep-dive fetch: timelines, PR details, patch text for the curated feature set."""
import json
import os
import re
import sys
import time
import urllib.request
import hashlib

import gh

ROOT = os.path.dirname(os.path.abspath(__file__))
RAWDIR = os.path.join(ROOT, "raw", "patch")
os.makedirs(RAWDIR, exist_ok=True)

SGL = "sgl-project/sglang"
VLLM = "vllm-project/vllm"

ITEMS = [
    # --- sglang / hidden states ---
    (SGL, 43449, "pr", "hidden"),
    (SGL, 43450, "pr", "hidden"),
    (SGL, 46290, "pr", "hidden"),
    (SGL, 46176, "pr", "hidden"),
    (SGL, 47072, "pr", "hidden"),
    (SGL, 47216, "pr", "hidden"),
    (SGL, 46320, "pr", "hidden"),
    (SGL, 47047, "pr", "hidden"),
    (SGL, 46678, "issue", "hidden"),
    (SGL, 45144, "pr", "hidden"),
    (SGL, 45145, "pr", "hidden"),
    (SGL, 42078, "pr", "hidden"),
    (SGL, 45925, "pr", "hidden"),
    # --- sglang / sampling mask ---
    (SGL, 43517, "pr", "sampling"),
    (SGL, 46080, "pr", "sampling"),
    (SGL, 46186, "pr", "sampling"),
    (SGL, 39934, "pr", "sampling"),
    (SGL, 41205, "pr", "sampling"),
    (SGL, 42647, "pr", "sampling"),
    (SGL, 38565, "pr", "sampling"),
    (SGL, 41916, "pr", "sampling"),
    (SGL, 43047, "pr", "sampling"),
    # --- vllm / hidden states ---
    (VLLM, 63436, "pr", "hidden"),
    (VLLM, 63427, "pr", "hidden"),
    (VLLM, 64458, "pr", "hidden"),
    (VLLM, 61699, "pr", "hidden"),
    (VLLM, 60477, "pr", "hidden"),
    (VLLM, 63406, "pr", "hidden"),
    (VLLM, 57736, "pr", "hidden"),
    (VLLM, 64320, "pr", "hidden"),
    (VLLM, 61180, "pr", "hidden"),
    (VLLM, 57917, "pr", "hidden"),
    (VLLM, 50187, "pr", "hidden"),
    (VLLM, 56822, "pr", "hidden"),
    (VLLM, 37002, "pr", "hidden"),
    (VLLM, 36998, "issue", "hidden"),
    (VLLM, 59542, "issue", "hidden"),
    (VLLM, 48743, "issue", "hidden"),
    (VLLM, 61539, "issue", "hidden"),
    (VLLM, 63818, "issue", "hidden"),
    (VLLM, 63157, "issue", "hidden"),
    (VLLM, 63972, "issue", "hidden"),
]


def fetch_patch(repo, number):
    key = hashlib.sha1(f"{repo}#{number}".encode()).hexdigest()[:16]
    path = os.path.join(RAWDIR, f"{repo.split('/')[1]}-{number}-{key}.patch")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://github.com/{repo}/pull/{number}.patch"
    for proxy in ("http://127.0.0.1:7897", None):
        try:
            handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})]
            opener = urllib.request.build_opener(*handlers)
            req = urllib.request.Request(url, headers={"User-Agent": "dsh-rl-feature-scan"})
            with opener.open(req, timeout=90) as resp:
                data = resp.read()
            if data:
                with open(path, "wb") as fh:
                    fh.write(data)
                return path
        except Exception as exc:  # noqa: BLE001
            print("patch fail", repo, number, proxy, exc)
            time.sleep(2)
    return None


def parse_patch(path):
    if not path or not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8", errors="replace").read()
    blocks = re.split(r"^From ([0-9a-f]{40}) ", txt, flags=re.M)
    commits = []
    for i in range(1, len(blocks), 2):
        sha = blocks[i]
        body = blocks[i + 1]
        subj = ""
        m = re.match(r".*?\n(?:From:.*\n)?(?:Date:.*\n)?(?:Subject: (.*))?", body)
        lines = body.split("\n")
        for idx, ln in enumerate(lines[:8]):
            if ln.startswith("Subject:"):
                subj = ln[len("Subject:") :].strip()
                break
        commits.append({"sha": sha, "subject": subj})
    files = re.findall(r"^diff --git a/(\S+) b/", txt, flags=re.M)
    seen = []
    for f in files:
        if f not in seen:
            seen.append(f)
    return {
        "commits": commits,
        "files": seen,
        "additions": len([l for l in txt.split("\n") if l.startswith("+") and not l.startswith("+++")]),
        "deletions": len([l for l in txt.split("\n") if l.startswith("-") and not l.startswith("---")]),
        "bytes": len(txt),
    }


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    out_path = os.path.join(ROOT, "deepdive.json")
    store = {}
    if os.path.exists(out_path):
        store = json.load(open(out_path, encoding="utf-8"))
    for repo, number, kind, topic in ITEMS:
        key = f"{repo}#{number}"
        if only and str(number) not in only:
            continue
        rec = store.get(key, {})
        rec.update({"repo": repo, "number": number, "kind": kind, "topic": topic})
        if not rec.get("issue") and kind == "issue":
            data = gh.get(f"/repos/{repo}/issues/{number}")
            if data:
                rec["issue"] = {
                    "state": data.get("state"),
                    "title": data.get("title"),
                    "created_at": data.get("created_at"),
                    "closed_at": data.get("closed_at"),
                    "labels": [la["name"] for la in data.get("labels", [])],
                    "comments": data.get("comments"),
                    "body": data.get("body"),
                }
        if not rec.get("pr") and kind == "pr":
            data = gh.get(f"/repos/{repo}/pulls/{number}")
            if data:
                rec["pr"] = {
                    "state": data.get("state"),
                    "draft": data.get("draft"),
                    "title": data.get("title"),
                    "created_at": data.get("created_at"),
                    "closed_at": data.get("closed_at"),
                    "merged_at": data.get("merged_at"),
                    "merged_by": (data.get("merged_by") or {}).get("login"),
                    "base_ref": (data.get("base") or {}).get("ref"),
                    "head_ref": (data.get("head") or {}).get("ref"),
                    "head_sha": (data.get("head") or {}).get("sha"),
                    "additions": data.get("additions"),
                    "deletions": data.get("deletions"),
                    "changed_files": data.get("changed_files"),
                    "commits": data.get("commits"),
                    "labels": [la["name"] for la in data.get("labels", [])],
                    "body": data.get("body"),
                }
        if not rec.get("timeline"):
            events = gh.paginated(f"/repos/{repo}/issues/{number}/timeline", max_pages=4)
            slim = []
            for ev in events:
                e = {
                    "event": ev.get("event"),
                    "created_at": ev.get("created_at"),
                    "actor": (ev.get("actor") or {}).get("login"),
                    "sha": ev.get("sha"),
                    "commit_id": ev.get("commit_id"),
                    "url": ev.get("html_url") or ev.get("url"),
                }
                body = ev.get("body")
                if body:
                    e["body"] = body[:1500]
                if ev.get("event") == "reviewed":
                    e["state"] = ev.get("state")
                if ev.get("event") == "renamed":
                    e["rename"] = ev.get("rename")
                if ev.get("event") == "committed" or ev.get("event") == "committed":
                    pass
                slim.append(e)
            rec["timeline"] = slim
        if not rec.get("patch") or not rec["patch"].get("commits"):
            rec["patch"] = parse_patch(fetch_patch(repo, number))
        store[key] = rec
        print(
            key,
            "pr" if rec.get("pr") else ("issue" if rec.get("issue") else "-"),
            len(rec.get("timeline") or []),
            len(rec.get("patch", {}).get("commits") or []),
            len(rec.get("patch", {}).get("files") or []),
            flush=True,
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(store, fh)
    print("done", len(store))


if __name__ == "__main__":
    main()
