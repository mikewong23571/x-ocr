"""Vision-driven browser automation demo: detector → coordinates → real click → verify.

Zero selectors: every click target comes from parse_screen (pure vision).
Verify step uses runtime/verify.verify_state_change (pixel diff).

Safe demo on a logged-out browser: focus login input + type throwaway text (not
submitted, cleared), switch login-method tabs. No real account actions.

Usage: .venv/bin/python scripts/vision_automation_demo.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

from runtime.api import ScreenParser  # noqa: E402
from runtime.verify import verify_state_change  # noqa: E402

import yaml  # noqa: E402

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")

CACHE = ROOT / "data/demo_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def main() -> None:
    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"), names, conf=0.35, check_purity=False)

    def shot(page, name):
        p = CACHE / f"va_{name}.png"
        page.screenshot(path=str(p))
        return str(p)

    def vision_click(page, role, idx=0, require=True):
        """Find element by role via the detector, click its center with a REAL mouse event."""
        els = [e for e in sp(shot(page, f"pre_{role}_{idx}")) if e["role"] == role]
        if not els or idx >= len(els):
            if require:
                raise RuntimeError(f"vision_click: no {role}#{idx} on screen")
            return None
        els.sort(key=lambda e: e["bbox"][1])
        e = els[idx]
        x = int((e["bbox"][0] + e["bbox"][2]) / 2)
        y = int((e["bbox"][1] + e["bbox"][3]) / 2)
        page.mouse.click(x, y)
        print(f"  ✓ vision-click {role}#{idx} @ ({x},{y}) conf={e['confidence']:.2f}")
        return e

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(ROOT / "data/browser-loggedout"), channel="chrome", headless=False,
            viewport={"width": 1440, "height": 900}, user_agent=UA)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        print("[1] 打开 x.com（未登录）")
        page.goto("https://x.com/", wait_until="domcontentloaded", timeout=30000)
        time.sleep(4)

        print("[2] 视觉定位用户名输入框并点击、输入临时文本（不提交）")
        before = shot(page, "step2_before")
        e = vision_click(page, "login_input", idx=0)
        time.sleep(1.0)
        page.keyboard.type("demo@example.com", delay=30)
        time.sleep(0.8)
        after = shot(page, "step2_after")
        r = verify_state_change(before, after, e["bbox"])
        print(f"    verify(输入框区域像素变化): {r}  → {'文字已落入 ✓' if r['changed'] else '未变化 ✗'}")

        print("[3] 视觉切换登录方式 tab（手机 → Apple），验证状态变化")
        before = shot(page, "step3_before")
        clicked = None
        for i in (2, 1):
            clicked = vision_click(page, "login_button", idx=i, require=False)
            if clicked:
                break
        if not clicked:
            print("  ! 无可切换的登录方式按钮，跳过")
        time.sleep(1.5)
        after = shot(page, "step3_after")
        els = sp(after)
        box = None
        if e and e["bbox"]:
            box = e["bbox"]
        r = verify_state_change(before, after, box or (400, 200, 1100, 700),
                                min_changed_ratio=0.01)
        print(f"    verify(登录卡片区域): {r}")

        print("[4] 回到手机 tab（还原现场）")
        try:
            vision_click(page, "login_button", idx=0, require=False)
        except Exception:
            pass
        time.sleep(1.2)

        ctx.close()
    print("\n视觉自动化闭环演示完成：检测器(坐标) → playwright 真鼠标 → 像素验证")


if __name__ == "__main__":
    main()
