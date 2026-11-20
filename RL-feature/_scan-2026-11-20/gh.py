"""Minimal GitHub REST client with dual egress pools, on-disk cache and pacing."""
import json
import os
import time
import hashlib
import threading
import urllib.parse
import urllib.request
import urllib.error

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "raw")
os.makedirs(CACHE, exist_ok=True)

PROXIES = [
    ("direct", None),
    ("proxy", "http://127.0.0.1:7897"),
]

_LOCK = threading.Lock()


def _now():
    return time.time()


class Pool:
    def __init__(self, name, proxy):
        self.name = name
        self.proxy = proxy
        handlers = []
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        else:
            handlers.append(urllib.request.ProxyHandler({}))
        self.opener = urllib.request.build_opener(*handlers)
        self.core_remaining = 60
        self.core_reset = 0
        self.search_remaining = 10
        self.search_reset = 0
        self.last_search = 0.0
        self.dead = False

    def refresh(self):
        try:
            r = self.fetch("https://api.github.com/rate_limit", kind="meta", use_cache=False)
            res = r["resources"]
            self.core_remaining = res["core"]["remaining"]
            self.core_reset = res["core"]["reset"]
            self.search_remaining = res["search"]["remaining"]
            self.search_reset = res["search"]["reset"]
        except Exception as exc:  # noqa: BLE001
            print("rate refresh failed", self.name, exc)

    def _open(self, url):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "dsh-rl-feature-scan",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        return self.opener.open(req, timeout=60)

    def fetch(self, url, kind="core", use_cache=True, page_note=None):
        key = hashlib.sha1(url.encode()).hexdigest()[:20]
        path = os.path.join(CACHE, f"{kind}-{key}.json")
        if use_cache and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload.get("data")

        for attempt in range(4):
            if kind == "search":
                with _LOCK:
                    wait = 6.3 - (_now() - self.last_search)
                if wait > 0:
                    time.sleep(wait)
                with _LOCK:
                    self.last_search = _now()
            try:
                with self._open(url) as resp:
                    body = json.load(resp)
                    rem = resp.headers.get("x-ratelimit-remaining")
                    reset = resp.headers.get("x-ratelimit-reset")
                if kind == "search":
                    self.search_remaining = int(rem or 0)
                    self.search_reset = int(reset or 0)
                else:
                    self.core_remaining = int(rem or 0)
                    self.core_reset = int(reset or 0)
                payload = {"url": url, "status": 200, "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data": body}
                if page_note:
                    payload["note"] = page_note
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                return body
            except urllib.error.HTTPError as exc:
                code = exc.code
                rem = exc.headers.get("x-ratelimit-remaining") if exc.headers else None
                reset = exc.headers.get("x-ratelimit-reset") if exc.headers else None
                if kind == "search":
                    self.search_remaining = int(rem or 0)
                    self.search_reset = int(reset or 0)
                else:
                    self.core_remaining = int(rem or 0)
                    self.core_reset = int(reset or 0)
                if code in (403, 429):
                    if self.core_remaining <= 0 or self.search_remaining <= 0:
                        if attempt == 3:
                            raise
                        continue
                    time.sleep(2 + attempt * 3)
                    continue
                if code == 404:
                    return None
                if 500 <= code < 600:
                    time.sleep(3 + attempt * 3)
                    continue
                raise
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(3 + attempt * 3)
        raise RuntimeError("unreachable")


POOLS = [Pool(n, p) for n, p in PROXIES]


def pick(kind):
    avail = [p for p in POOLS if not p.dead]
    if not avail:
        raise RuntimeError("no pool")
    if kind == "search":
        return min(avail, key=lambda p: (p.search_remaining, p.name != "direct"))
    return max(avail, key=lambda p: (p.core_remaining, p.name == "direct"))


def wait_for_quota(kind, need=1, max_wait=900):
    """Sleep until some pool has quota."""
    while True:
        avail = [p for p in POOLS if not p.dead]
        if not avail:
            raise RuntimeError("no pool")
        now = _now()
        for p in avail:
            rem = p.core_remaining if kind == "core" else p.search_remaining
            reset = p.core_reset if kind == "core" else p.search_reset
            if rem >= need:
                return
            if reset and reset - now > 0 and reset - now < max_wait:
                pass
        soonest = min(
            (p.core_reset if kind == "core" else p.search_reset) for p in avail
        )
        delay = max(5, min(max_wait, soonest - now + 3))
        print(f"[quota] {kind} exhausted, sleeping {delay:.0f}s (resets at {soonest:.0f}, now {now:.0f})")
        time.sleep(delay)
        for p in avail:
            p.refresh()


def get(path, kind="core", use_cache=True):
    url = path if path.startswith("http") else "https://api.github.com" + path
    for attempt in range(3):
        wait_for_quota(kind)
        pool = pick(kind)
        try:
            return pool.fetch(url, kind=kind, use_cache=use_cache)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                pool.refresh()
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            print("fetch error", url, exc)
            pool.dead = True
            if attempt == 2:
                raise
    return None


def search(q, kind="issues", per_page=100, max_pages=10):
    out = []
    page = 1
    total = None
    while page <= max_pages:
        enc = urllib.parse.quote(q, safe="")
        path = f"https://api.github.com/search/{kind}?q={enc}&per_page={per_page}&page={page}&sort=updated&order=desc"
        data = get(path, kind="search")
        if not data:
            break
        if total is None and data.get("total_count") is not None:
            total = data["total_count"]
        items = data.get("items", [])
        out.extend(items)
        if len(items) < per_page:
            break
        page += 1
    return {"total": total, "items": out}


def paginated(path, kind="core", per_page=100, max_pages=30):
    out = []
    page = 1
    while page <= max_pages:
        sep = "&" if "?" in path else "?"
        data = get(f"{path}{sep}per_page={per_page}&page={page}", kind=kind)
        if not data:
            break
        if isinstance(data, dict):
            data = data.get("items", [])
        out.extend(data)
        if len(data) < per_page:
            break
        page += 1
    return out


if __name__ == "__main__":
    for p in POOLS:
        p.refresh()
        print(p.name, "core", p.core_remaining, p.core_reset, "search", p.search_remaining, p.search_reset)
