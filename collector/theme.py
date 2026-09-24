"""X display-theme switcher via /i/display settings dialog (works unfocused; dialogs don't auto-close).

Themes: 'default' (light), 'dim', 'lightsout'. Note: this X account's options are
'default' | 'lights out' | 'use system setting' (+ 'dim' when scrolled).

CLI: .venv/bin/python collector/theme.py status | set <theme>
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PW = str(ROOT / "scripts" / "pw")

DIRECT_JS = """(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const dlg = document.querySelector("[role='dialog']");
  if (!dlg) return 'no-dialog';
  const want = __WANT__;
  const themes = {default: 'default', dim: 'dim', lightsout: 'lights out', system: 'use system setting'};
  const target_text = themes[want];
  let target = null;
  for (const el of dlg.querySelectorAll('*')) {
    if (el.childElementCount !== 0) continue;
    const t = (el.textContent || '').trim().toLowerCase();
    // 'default' also appears in the Color section; Background is the last section → keep last match
    if (t === target_text && el.getBoundingClientRect().height > 5) { target = el; }
  }
  if (!target) return 'no-option:' + want;
  (target.closest('label') || target.parentElement).click();
  await sleep(900);
  const done = Array.from(dlg.querySelectorAll('button, [role=button]'))
    .find(b => (b.innerText || '').trim().toLowerCase() === 'done');
  if (done) done.click();
  await sleep(700);
  return JSON.stringify({picked: want, bg: getComputedStyle(document.body).backgroundColor});
})()"""


def pw(*args: str, timeout: int = 45, check: bool = False) -> str:
    r = subprocess.run([PW, *args], capture_output=True, text=True, timeout=timeout)
    if check and (r.returncode != 0 or "### Error" in r.stdout):
        raise RuntimeError(f"pw {args[:2]}: {r.stdout[-300:]}")
    return r.stdout


def current_theme() -> str:
    out = pw("eval", "() => getComputedStyle(document.body).backgroundColor", timeout=30)
    for ln in out.splitlines():
        s = ln.strip()
        if "rgb" in s and s.startswith(("-", '"')):
            rgb = s.lstrip("- ").strip().strip('"')
            r, g, b = (int(x) for x in rgb.replace("rgb(", "").replace(")", "").split(","))
            if (r, g, b) == (0, 0, 0):
                return "lightsout"
            if r == g == b and r < 120:
                return "dim"
            return "default"
    return "unknown"


def set_theme(theme: str) -> tuple[bool, str]:
    if theme not in ("default", "dim", "lightsout", "system"):
        raise ValueError(theme)
    pw("goto", "https://x.com/i/display", timeout=45)
    time.sleep(3.5)
    js = DIRECT_JS.replace("__WANT__", json.dumps(theme))
    out = pw("eval", js, "--filename", "/tmp/xocr_theme_out.json", timeout=45)
    try:
        res = json.loads(Path("/tmp/xocr_theme_out.json").read_text())
    except Exception:
        return False, out[-200:]
    return "picked" in res, current_theme()


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(current_theme())
    elif cmd == "set":
        ok, now = set_theme(sys.argv[2])
        print(ok, now)


if __name__ == "__main__":
    main()
