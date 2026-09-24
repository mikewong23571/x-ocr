"""Observe X Desktop Web routes/states: dump DOM JSON + screenshot per state.

Safe operations only: navigation, scroll, open/close menus & dialogs, no posting/liking.
Usage: .venv/bin/python scripts/observe_routes.py [--out data/observation]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PW = str(ROOT / "scripts" / "pw")
EXTRACT = (ROOT / "scripts/js/extract_fn.js").read_text()


def pw(*args: str, timeout: int = 60) -> str:
    r = subprocess.run([PW, *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or "### Error" in r.stdout:
        raise RuntimeError(f"pw {args} failed: {r.stdout[-600:]} {r.stderr[-300:]}")
    return r.stdout


def current_url() -> str:
    out = pw("eval", "() => location.href", timeout=30)
    lines = out.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("### Result"):
            for ln2 in lines[i + 1 :]:
                s = ln2.strip()
                if s.startswith("-"):
                    return s.lstrip("- ").strip().strip('"')
    raise RuntimeError(f"no result in eval output: {out[:300]}")


def goto(url: str, expect: str, tries: int = 3) -> None:
    last = "?"
    for _ in range(tries):
        try:
            pw("goto", url, timeout=90)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(4)
        try:
            last = current_url()
            if expect in last:
                return
        except Exception:  # noqa: BLE001
            pass
    raise RuntimeError(f"goto {url} failed, still at {last}")


def pw_quiet(*args: str, timeout: int = 60) -> bool:
    """Run a command, return True if no '### Error' in output."""
    r = subprocess.run([PW, *args], capture_output=True, text=True, timeout=timeout)
    return "### Error" not in r.stdout


def dump(name: str, out: Path, meta: dict | None = None) -> dict:
    jf = out / f"{name}.json"
    pw("eval", EXTRACT, "--filename", str(jf))
    d = json.loads(jf.read_text())
    if meta:
        d["meta"] = meta
        jf.write_text(json.dumps(d, ensure_ascii=False))
    pw("screenshot", "--filename", str(out / f"{name}.png"))
    print(f"  {name}: {d['n']} els, {len(set(e['testid'] for e in d['els'] if e['testid']))} testids")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/observation")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    states: list[tuple[str, str, list[str]]] = [
        # name, action sequence (pw args)
        ("home_top", [("goto", "https://x.com/home"), ("sleep", "4")]),
        ("home_scrolled", [("scroll", "2000"), ("sleep", "3")]),
        ("explore", [("goto", "https://x.com/explore"), ("sleep", "4")]),
        ("notifications", [("goto", "https://x.com/notifications"), ("sleep", "4")]),
        ("bookmarks", [("goto", "https://x.com/i/bookmarks"), ("sleep", "4")]),
        ("profile", [("goto", "https://x.com/0xMikeWong"), ("sleep", "4")]),
        ("grok", [("goto", "https://x.com/i/grok"), ("sleep", "6")]),
        ("settings", [("goto", "https://x.com/settings/account"), ("sleep", "4")]),
    ]

    for name, actions in states:
        print(f"[{name}]", flush=True)
        try:
            for act, val in actions:
                if act == "sleep":
                    time.sleep(float(val))
                elif act == "scroll":
                    # page-context scroll: avoids CLI's post-action network wait on X
                    pw("eval", f"() => {{ window.scrollBy(0, {int(val)}); return window.scrollY; }}")
                    time.sleep(2)
                elif act == "goto":
                    goto(val, val.replace("https://x.com", "") or "x.com")
            dump(name, out, {"actions": [list(a) for a in actions]})
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}", flush=True)

    # --- interactive states (open then close) ---
    print("[state: more_menu]")
    pw_quiet("goto", "https://x.com/home")
    time.sleep(3)
    if pw_quiet("click", "button[data-testid='AppTabBar_More_Menu']"):
        time.sleep(1.5)
        try:
            dump("menu_more", out, {"kind": "menu_open"})
        except Exception as e:  # noqa: BLE001
            print(f"  dump failed: {e}")
        pw_quiet("press", "Escape")

    print("[state: account_switcher]")
    if pw_quiet("click", "button[data-testid='SideNav_AccountSwitcher_Button']"):
        time.sleep(1.5)
        try:
            dump("menu_account", out, {"kind": "menu_open"})
        except Exception as e:  # noqa: BLE001
            print(f"  dump failed: {e}")
        pw_quiet("press", "Escape")

    print("[state: compose_dialog]")
    if pw_quiet("click", "a[data-testid='SideNav_NewTweet_Button']"):
        time.sleep(2)
        try:
            dump("dialog_compose", out, {"kind": "dialog_open"})
        except Exception as e:  # noqa: BLE001
            print(f"  dump failed: {e}")
        pw_quiet("press", "Escape")

    print("done →", out)


if __name__ == "__main__":
    main()
