"""Logged-out X collector: fresh browser, independent session (task_spec §十 empty/loading/error + §十一 独立 session).

Own playwright context (NOT the extension session): full API, no login state touched.
States captured: login wall, logged-out explore/search/trends, public status page,
skeleton (early capture), error page (nonexistent user), empty variants.

Usage: .venv/bin/python collector/collect_loggedout.py --session loggedout_s01 --out data/raw
"""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
EXTRACT_JS = (ROOT / "scripts/js/extract_annotated.js").read_text()

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")

VPS = [(1920, 1080), (1440, 900), (1280, 720)]

ROUTES = [
    ("loggedout_home", "https://x.com/"),                      # login wall / redirect
    ("loggedout_login", "https://x.com/i/flow/login"),         # login form
    ("loggedout_explore", "https://x.com/explore"),            # trends, no login required
    ("loggedout_search", "https://x.com/search?q=ai&src=typed_query"),
    ("loggedout_error_user", "https://x.com/this_user_does_not_exist_9x8y7z"),  # error state
]


def phash_png(path: Path) -> str:
    import imagehash
    from PIL import Image
    with Image.open(path) as im:
        return str(imagehash.phash(im))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="loggedout_s01")
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()

    out = ROOT / args.out / args.session
    profile = ROOT / "data/browser-loggedout"  # dedicated, disposable profile
    profile.mkdir(parents=True, exist_ok=True)
    seq = 0

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=False,
            viewport={"width": 1920, "height": 1080}, user_agent=UA,
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        def capture(route: str, vw: int, vh: int, extra: dict | None = None) -> None:
            nonlocal seq
            seq += 1
            sd = out / f"{seq:04d}"
            sd.mkdir(parents=True, exist_ok=True)
            anno = page.evaluate(EXTRACT_JS)
            (sd / "anno.json").write_text(json.dumps(anno, ensure_ascii=False))
            page.screenshot(path=str(sd / "shot.png"))
            meta = {
                "sample_id": sd.name, "session": args.session,
                "ts": datetime.now(timezone.utc).isoformat(),
                "url": anno["url"], "route": route,
                "viewport": f"{vw}x{vh}", "vw": anno["vw"], "vh": anno["vh"],
                "scrollY": anno["scrollY"], "dark": False,
                "theme_bg": anno["theme_bg"], "n_els": anno["n"],
                "phash": phash_png(sd / "shot.png"),
                "sha1_shot": hashlib.sha1((sd / "shot.png").read_bytes()).hexdigest()[:12],
                "extra": extra or {},
            }
            (sd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
            print(f"  #{seq:04d} {route:24s} {anno['vw']}x{anno['vh']} els={anno['n']} bg={anno['theme_bg']}")

        for vw, vh in VPS:
            page.set_viewport_size({"width": vw, "height": vh})
            for name, url in ROUTES:
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    # skeleton/loading state: capture early
                    time.sleep(0.6)
                    capture(f"{name}_skeleton", vw, vh, {"early": True})
                    time.sleep(3.5)
                    capture(name, vw, vh)
                    # one scrolled variant on content pages
                    if "explore" in name or "search" in name:
                        page.evaluate(f"window.scrollBy(0, {800 + random.randint(0, 600)})")
                        time.sleep(2)
                        capture(f"{name}_scrolled", vw, vh)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {name} failed: {e}")

        # public status page (harvest a link from explore)
        try:
            page.goto("https://x.com/explore", wait_until="domcontentloaded", timeout=30000)
            time.sleep(3)
            link = page.evaluate("""() => {
              const a = Array.from(document.querySelectorAll("a[href*='/status/']"))[0];
              return a ? a.getAttribute('href') : null;
            }""")
            if link:
                page.goto(f"https://x.com{link}" if link.startswith("/") else link,
                          wait_until="domcontentloaded", timeout=30000)
                time.sleep(3.5)
                capture("loggedout_status", 1920, 1080)
        except Exception as e:  # noqa: BLE001
            print(f"  ! status failed: {e}")

        ctx.close()
    print(f"DONE {args.session}: {seq} samples → {out}")


if __name__ == "__main__":
    main()
