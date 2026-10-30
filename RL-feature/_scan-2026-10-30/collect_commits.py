"""Extract commits in the scan window from a local shallow clone."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
START = "2026-10-23T02:05:41Z"

REC = "\x1e"
FLD = "\x1f"


def git(repo, *args):
    out = subprocess.run(
        ["git", "-C", repo, "--no-pager"] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if out.returncode != 0:
        print("git err", out.stderr[:400])
    return out.stdout


def main(repo_path, tag):
    head = git(repo_path, "rev-parse", "HEAD").strip()
    head_dt = git(repo_path, "log", "-1", "--date=iso-strict", "--pretty=%cI").strip()

    fmt = FLD.join(["%H", "%cI", "%an", "%ae", "%s", "%b"])
    meta_raw = git(
        repo_path, "log", f"--since={START}", "--date=iso-strict", f"--pretty=format:{fmt}{REC}"
    )
    files_raw = git(
        repo_path,
        "log",
        f"--since={START}",
        "--pretty=format:%x1e%H",
        "--name-only",
    )

    file_map = {}
    for chunk in files_raw.split(REC):
        lines = [ln.strip() for ln in chunk.strip("\n").split("\n")]
        if not lines or not lines[0]:
            continue
        sha = lines[0]
        file_map[sha] = [ln for ln in lines[1:] if ln]

    commits = []
    for chunk in meta_raw.split(REC):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        parts = chunk.split(FLD)
        while len(parts) < 6:
            parts.append("")
        sha, dt, an, ae, subj, body = parts[:6]
        commits.append(
            {
                "sha": sha,
                "date": dt,
                "author": an,
                "email": ae,
                "subject": subj,
                "body": body,
                "files": file_map.get(sha, []),
            }
        )

    data = {
        "repo": tag,
        "head_sha": head,
        "head_date": head_dt,
        "window_start": START,
        "commit_count": len(commits),
        "commits": commits,
    }
    out_path = os.path.join(ROOT, f"commits-{tag}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    print(tag, "head", head, head_dt, "commits", len(commits), "->", out_path)


if __name__ == "__main__":
    main(os.path.join(ROOT, "repos", sys.argv[1]), sys.argv[1])
