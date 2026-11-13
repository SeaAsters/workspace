# -*- coding: utf-8 -*-
"""Thin wrapper over the authenticated `gh` CLI (5000 req/h) for this scan."""
import json
import subprocess

GH = r"C:\Program Files\GitHub CLI\gh.exe"


def _run(args, timeout=180):
    out = subprocess.run([GH] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args[:3])} failed: {out.stderr[:300]}")
    return out.stdout


def api(path, paginate=False, jq=None):
    args = ["api", path]
    if paginate:
        args.append("--paginate")
    if jq:
        args += ["--jq", jq]
    raw = _run(args)
    if jq:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    return json.loads(raw) if raw.strip() else None


def api_items(path, timeout=180):
    """Fetch a paginated list endpoint as a flat list of objects."""
    raw = _run(["api", path, "--paginate", "--jq", ".[]"], timeout=timeout)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


def search_issues(query, per_page=100, max_pages=10):
    """search/issues via gh api, paginated by our own page loop (gh caps at 100/p)."""
    items = []
    total = None
    for page in range(1, max_pages + 1):
        raw = _run(
            [
                "api",
                "-X", "GET", "search/issues",
                "-f", f"q={query}",
                "-f", f"per_page={per_page}",
                "-f", f"page={page}",
            ]
        )
        data = json.loads(raw)
        if total is None:
            total = data.get("total_count")
        batch = data.get("items", [])
        items.extend(batch)
        if len(batch) < per_page:
            break
    return {"total": total, "items": items}


if __name__ == "__main__":
    r = search_issues('repo:sgl-project/sglang "hidden states" updated:>=2026-11-06')
    print("total", r["total"], "returned", len(r["items"]))
    if r["items"]:
        it = r["items"][0]
        print(it["number"], it["state"], it["title"][:60])
