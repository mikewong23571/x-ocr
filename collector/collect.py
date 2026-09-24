"""X Desktop Web collector: drives the user's Chrome via playwright-cli session.

Safe ops only: navigate, scroll, open/close menus & dialogs, resize, focus. No posting/liking.

Sample layout: data/raw/<session>/<seq:04d>/{shot.png, anno.json, meta.json}

Usage:
  .venv/bin/python collector/collect.py --session s01 --target 100
  .venv/bin/python collector/collect.py --session s01 --states menu_states --viewports 1440x900,1920x1080
"""
import argparse
import hashlib
import json
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PW = str(ROOT / "scripts" / "pw")
EXTRACT = (ROOT / "scripts/js/extract_annotated.js").read_text()

random.seed()


def pw(*args: str, timeout: int = 90, check: bool = True) -> str:
    try:
        r = subprocess.run([PW, *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        if check:
            raise
        return ""
    if check and (r.returncode != 0 or "### Error" in r.stdout):
        raise RuntimeError(f"pw {args[:2]} failed: {r.stdout[-400:]}")
    return r.stdout


def current_url() -> str:
    out = pw("eval", "() => location.href", timeout=30)
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("### Result"):
            for ln2 in lines[i + 1:]:
                s = ln2.strip()
                if s.startswith("-"):
                    return s.lstrip("- ").strip().strip('"')
    return ""


def goto(url: str, expect: str, tries: int = 3) -> bool:
    for _ in range(tries):
        try:
            pw("goto", url, check=False)
        except Exception:
            pass
        time.sleep(3.5)
        try:
            if expect in current_url():
                return True
        except Exception:
            pass
    return False


def scroll(dy: int) -> None:
    pw("eval", f"() => {{ window.scrollBy(0, {int(dy)}); return window.scrollY; }}", timeout=30, check=False)
    time.sleep(1.8)


def phash_png(path: Path) -> str:
    import imagehash
    from PIL import Image
    with Image.open(path) as im:
        return str(imagehash.phash(im))


def _layout_fingerprint(anno: dict) -> list:
    # geometry-only signature; survives aria/text churn, catches reflow shifts
    return sorted((e["role"], int(e["rect"]["x"]) // 8, int(e["rect"]["y"]) // 8,
                   int(e["rect"]["w"]) // 8, int(e["rect"]["h"]) // 8)
                  for e in anno["els"])


def capture(sample_dir: Path, route: str, viewport: str, extra: dict | None = None,
            extract_js: str = None) -> dict:
    sample_dir.mkdir(parents=True, exist_ok=True)
    anno_f = sample_dir / "anno.json"
    shot_f = sample_dir / "shot.png"
    # Screenshot first (pixels are label truth), then prove the layout is stable
    # with a second DOM extract — X reflows lazily (pill dismiss, compose move)
    # and a stale DOM mislabels every box by the shift amount.
    stable = False
    for _attempt in range(2):
        pw("screenshot", "--filename", str(shot_f), timeout=60, check=False)
        pw("eval", extract_js or EXTRACT, "--filename", str(anno_f), timeout=60, check=False)
        try:
            anno = json.loads(anno_f.read_text())
        except Exception:
            anno = None
        if not anno or not shot_f.exists():
            return {"ok": False}
        fp1 = _layout_fingerprint(anno)
        time.sleep(0.4)
        pw("eval", extract_js or EXTRACT, "--filename", str(anno_f), timeout=60, check=False)
        try:
            anno2 = json.loads(anno_f.read_text())
        except Exception:
            anno2 = None
        if anno2 and _layout_fingerprint(anno2) == fp1:
            anno, stable = anno2, True
            break
    if not stable:
        return {"ok": False, "unstable": True}
    meta = {
        "sample_id": sample_dir.name,
        "session": sample_dir.parent.name,
        "ts": datetime.now(timezone.utc).isoformat(),
        "url": anno["url"],
        "route": route,
        "viewport": viewport,
        "vw": anno["vw"],
        "vh": anno["vh"],
        "scrollY": anno["scrollY"],
        "dark": anno["dark"],
        "theme_bg": anno["theme_bg"],
        "n_els": anno["n"],
        "stable": True,
        "phash": phash_png(shot_f),
        "sha1_shot": hashlib.sha1(shot_f.read_bytes()).hexdigest()[:12],
        "extra": extra or {},
    }
    (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return {"ok": True, "meta": meta, "n": anno["n"]}


class Collector:
    def __init__(self, out_root: Path, session: str):
        self.out = out_root / session
        self.seq = 0
        self.recent_hashes: list[str] = []

    def take(self, route: str, viewport: str, extra: dict | None = None, dedup_retry: int = 2,
             extract_js: str | None = None) -> bool:
        for attempt in range(dedup_retry + 1):
            self.seq += 1
            sd = self.out / f"{self.seq:04d}"
            res = capture(sd, route, viewport, extra, extract_js=extract_js)
            if not res["ok"]:
                print(f"    capture failed @ {self.seq}", flush=True)
                return False
            h = res["meta"]["phash"]
            if any(self._hamming(h, p) < 8 for p in self.recent_hashes[-12:]) and attempt < dedup_retry:
                if extract_js:
                    break  # state capture: menu overlay may confuse phash; keep as-is
                scroll(700 + random.randint(0, 900))  # too similar → advance feed
                continue
            self.recent_hashes.append(h)
            print(f"    #{self.seq:04d} {route:14s} {res['meta']['vw']}x{res['meta']['vh']} "
                  f"dark={res['meta']['dark']} els={res['n']}", flush=True)
            return True
        return False

    @staticmethod
    def _hamming(a: str, b: str) -> int:
        n = max(len(a), len(b))
        return sum(x != y for x, y in zip(a.ljust(n), b.ljust(n)))


ROUTES = {
    "home": "https://x.com/home",
    "explore": "https://x.com/explore",
    "notifications": "https://x.com/notifications",
    "profile": "https://x.com/0xMikeWong",
    "grok": "https://x.com/i/grok",
    "search_people": "https://x.com/search?q=agents&src=typed_query&f=user",
    "search_live": "https://x.com/search?q=xai&f=live",
}


def feed_states(c: Collector, viewport: str, target: int = 10**9) -> None:
    """Route + scroll coverage of main feeds."""
    for route, url in ROUTES.items():
        if c.seq >= target:
            return
        if not goto(url, route):
            print(f"  ! goto failed: {route}", flush=True)
            continue
        time.sleep(3.5)
        pw("eval", "() => { window.scrollTo(0, 0); return 1; }", timeout=30, check=False)
        time.sleep(1.0)
        c.take(route, viewport)
        for k in range(random.randint(1, 3)):
            scroll(800 + random.randint(0, 1400))
            c.take(route, viewport, extra={"scroll_step": k + 1})


def _extract_body() -> str:
    """Body of the standard extraction function (without the `() => {` wrapper)."""
    s = EXTRACT.strip()
    assert s.startswith("() => {") and s.endswith("}")
    return s[len("() => {"):-1]


def state_extract_js(click_js: str, wait_ms: int = 1600) -> str:
    """Single async eval: click → wait → extract. Menu guaranteed open during extraction."""
    return ("(async () => { const el = " + click_js + "; if (el) el.click();"
            f" await new Promise(r => setTimeout(r, {wait_ms}));"
            + _extract_body() + "})()")


def menu_states(c: Collector, viewport: str) -> None:
    """Overlay states: open menu/dialog → capture (atomic) → close. All reversible."""
    if not goto(ROUTES["home"], "home"):
        return
    time.sleep(3)
    pw("eval", "() => { window.scrollTo(0, 0); return 1; }", timeout=30, check=False)
    time.sleep(1)
    states = [
        ("more_menu", "document.querySelector(\"button[data-testid='AppTabBar_More_Menu']\")"),
        ("account_menu", "document.querySelector(\"button[data-testid='SideNav_AccountSwitcher_Button']\")"),
        ("compose_dialog", "document.querySelector(\"a[data-testid='SideNav_NewTweet_Button']\")"),
        ("post_overflow_menu", "document.querySelector(\"article [data-testid='caret']\")"),
        ("search_focus", "document.querySelector(\"input[data-testid='SearchBox_Search_Input']\")"),
        ("repost_menu", "document.querySelector(\"article [data-testid='retweet']\")"),
    ]
    for name, click_js in states:
        try:
            js = state_extract_js(click_js)
            c.take(f"state_{name}", viewport, extra={"opened": True}, extract_js=js)
        except Exception as e:  # noqa: BLE001
            print(f"  ! state {name} failed: {e}", flush=True)
        finally:
            pw("eval", "() => { document.dispatchEvent(new KeyboardEvent('keydown', {key: 'Escape', keyCode: 27})); return 1; }",
               timeout=20, check=False)
            pw("eval", "() => { const d = document.querySelector(\"[role='dialog'] [aria-label='Close'], [aria-label='Close']\"); if (d && d.getBoundingClientRect().height > 0) { d.click(); } return 1; }",
               timeout=20, check=False)
            time.sleep(0.8)


def harvest_status_urls(n: int = 8) -> list[str]:
    """Collect post-permalink URLs from home + explore (feed may be exhausted/stale)."""
    urls: list[str] = []
    for route in ("home", "explore", "notifications"):
        goto(ROUTES[route], route)
        time.sleep(3.5)
        for _ in range(3):
            out = pw("eval", """() => {
              const links = Array.from(document.querySelectorAll("a[href*='/status/']"))
                .map(a => a.getAttribute('href'))
                .filter(h => h && h.split('/').length === 4);
              window.scrollBy(0, 1200);
              return [...new Set(links)].slice(0, 12).join('|');
            }""", timeout=30, check=False)
            for ln in out.splitlines():
                s = ln.strip()
                if "|" in s and s.startswith(("-", '"')):
                    raw = s.lstrip("- ").strip().strip('"')
                    for u in raw.split("|"):
                        if u.startswith("/") and u not in urls:
                            urls.append(u)
                    break
            time.sleep(1.6)
            if len(urls) >= n:
                break
        if len(urls) >= n:
            break
    return [f"https://x.com{u}" if u.startswith("/") else u for u in urls[:n]]


def thread_states(c: Collector, viewport: str, n: int = 6) -> None:
    """Post detail pages via direct navigation (clicking stale feeds is unreliable)."""
    urls = harvest_status_urls(n)
    print(f"    harvested {len(urls)} status urls", flush=True)
    for i, u in enumerate(urls):
        try:
            if not goto(u, "/status/"):
                continue
            time.sleep(3.2)
            c.take("status", viewport, extra={"status_idx": i})
            if i % 2 == 1:  # scroll inside every other thread
                scroll(900)
                c.take("status", viewport, extra={"status_idx": i, "scrolled": True})
        except Exception as e:  # noqa: BLE001
            print(f"  ! thread {i} failed: {e}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--states", default="feed,menus,threads")
    ap.add_argument("--viewports", default="1920x1080,1440x900,1280x720")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out_root = ROOT / args.out
    c = Collector(out_root, args.session)
    vps = args.viewports.split(",")
    plan = args.states.split(",")
    t0 = time.time()

    for vp in vps:
        w, h = vp.split("x")
        try:
            pw("resize", w, h, timeout=30, check=False)
        except Exception:
            pass
        time.sleep(1.5)
        if "feed" in plan:
            print(f"[feed @{vp}]", flush=True)
            feed_states(c, vp, args.target)
        if c.seq >= args.target:
            break

    if "menus" in plan and c.seq < args.target:
        print("[menus]", flush=True)
        menu_states(c, vps[0])
    if "threads" in plan and c.seq < args.target:
        print("[threads]", flush=True)
        thread_states(c, vps[-1])

    print(f"DONE session={args.session} samples={c.seq} elapsed={time.time()-t0:.0f}s → {c.out}", flush=True)


if __name__ == "__main__":
    main()
