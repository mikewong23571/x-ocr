"""X display-theme switcher (light/dim/lightsout) via X's own Display settings dialog.

Safe: opens More menu → Display, selects a theme radio, closes dialog. Fully reversible;
remember(original) + restore() pattern. Usage from collector between captures.

CLI check: .venv/bin/python collector/theme.py status
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PW = str(ROOT / "scripts" / "pw")

# X theme radio labels (data-testid on the radio rows)
THEME_TESTIDS = {
    "default": "radioButton-Default",   # follows system (light on light systems)
    "dim": "radioButton-Dim",
    "lightsout": "radioButton-Lights out",
}


def pw(*args: str, timeout: int = 45, check: bool = False) -> str:
    r = subprocess.run([PW, *args], capture_output=True, text=True, timeout=timeout)
    if check and (r.returncode != 0 or "### Error" in r.stdout):
        raise RuntimeError(f"pw {args[:2]}: {r.stdout[-300:]}")
    return r.stdout


def current_theme() -> str:
    """Read body background color as theme proxy."""
    out = pw("eval", "() => getComputedStyle(document.body).backgroundColor", timeout=30)
    for ln in out.splitlines():
        if ln.strip().startswith("-") and "rgb" in ln:
            rgb = ln.strip().lstrip("- ").strip().strip('"')
            r, g, b = (int(x) for x in rgb.replace("rgb(", "").replace(")", "").split(","))
            if (r, g, b) == (0, 0, 0):
                return "lightsout"
            if r == g == b and r < 120:
                return "dim"
            return "default"
    return "unknown"


def set_theme(theme: str) -> bool:
    """Open Display settings and select the theme radio. Returns success."""
    if theme not in THEME_TESTIDS:
        raise ValueError(theme)
    pw("eval", "() => { window.scrollTo(0,0); return 1; }")
    pw("click", "button[data-testid='AppTabBar_More_Menu']")
    time.sleep(1.2)
    # the Display entry is a menu item with data-testid containing 'Dropdown-Dropdown'
    ok = pw("eval", f"""() => {{
      const items = Array.from(document.querySelectorAll("[data-testid^='Dropdown'] a, [role='menuitem']"));
      const disp = items.find(i => (i.innerText || '').trim().toLowerCase().startsWith('display'));
      if (!disp) return 'no-display-item';
      disp.click(); return 'ok';
    }}""")
    time.sleep(1.5)
    picked = pw("eval", f"""() => {{
      const radios = Array.from(document.querySelectorAll("[data-testid^='radioButton']"));
      const target = radios.find(r => r.getAttribute('data-testid') === {THEME_TESTIDS[theme]!r});
      if (!target) return 'no-radio:' + radios.map(r => r.getAttribute('data-testid')).join(',');
      target.click(); return 'ok';
    }}""")
    time.sleep(1.0)
    pw("press", "Escape")
    time.sleep(0.5)
    return "ok" in picked


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        print(current_theme())
    elif cmd == "set":
        print(set_theme(sys.argv[2]), current_theme())


if __name__ == "__main__":
    main()
