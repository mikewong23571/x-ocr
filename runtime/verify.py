"""Runtime verification primitives (pure vision, no DOM).

1. verify_parse(els, img_size)  — structural invariants every sane parse must satisfy
2. verify_state_change(img_before, img_after, bbox) — did the target region visually change?
   (post-click verification without a state head: liked heart fills red, menus open, etc.)

Usage examples in __main__ (self-test on dataset pairs).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def verify_parse(els: list[dict], img_w: int, img_h: int) -> dict:
    """Structural sanity checks. Returns {ok, failures[], warnings[]}."""
    failures, warnings = [], []

    def xyxy(e):
        x1, y1, x2, y2 = e["bbox"]
        return x1, y1, x2, y2

    for e in els:
        x1, y1, x2, y2 = xyxy(e)
        if not (0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h):
            failures.append(f"{e['role']} box out of image: {e['bbox']}")
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            warnings.append(f"{e['role']} tiny box (<4px): {e['bbox']}")

    # same-class near-identical overlaps (model double-fire)
    from collections import defaultdict
    by_role = defaultdict(list)
    for e in els:
        by_role[e["role"]].append(e)
    for role, es in by_role.items():
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = xyxy(es[i]), xyxy(es[j])
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                inter = ix * iy
                union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                if union > 0 and inter / union > 0.85:
                    warnings.append(f"{role} near-dup pair IoU={inter/union:.2f}")

    # action-row grouping sanity: action buttons should sit inside SOME post_container span
    containers = [xyxy(e) for e in els if e["role"] == "post_container"]
    if containers:
        orphans = []
        for e in els:
            if e["role"] in ("like_button", "reply_button", "repost_button", "bookmark_button", "share_button"):
                yc = (e["bbox"][1] + e["bbox"][3]) / 2
                if not any(c[1] - 20 <= yc <= c[3] + 20 for c in containers):
                    orphans.append(e["role"])
        if orphans:
            warnings.append(f"{len(orphans)} action buttons outside any post_container span: {orphans[:3]}")

    return {"ok": not failures, "failures": failures[:8], "warnings": warnings[:8]}


def verify_state_change(img_before, img_after, bbox, min_changed_ratio: float = 0.02) -> dict:
    """Pixel-diff the SAME bbox across before/after screenshots. Pure vision click-verification.

    Returns {changed, changed_ratio, mean_abs_diff}. changed=True ⇒ the region visually
    responded (state toggle, menu open, color fill), i.e. the action very likely took effect.
    """
    import numpy as np
    from PIL import Image

    def crop(img):
        im = Image.open(img).convert("RGB")
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox]
        x2, y2 = min(x2, im.width), min(y2, im.height)
        return np.asarray(im.crop((x1, y1, x2, y2)), dtype=np.int16)

    a, b = crop(img_before), crop(img_after)
    if a.shape != b.shape:
        return {"changed": True, "changed_ratio": 1.0, "mean_abs_diff": None,
                "note": "region size differs (layout moved) → treat as changed"}
    diff = np.abs(a - b).mean(axis=2)
    changed_ratio = float((diff > 24).mean())
    return {"changed": changed_ratio >= min_changed_ratio,
            "changed_ratio": round(changed_ratio, 4),
            "mean_abs_diff": round(float(diff.mean()), 2)}


if __name__ == "__main__":
    # Self-test 1: parse verification across the demo samples
    from runtime.api import parse_screen

    ok_n = warn_n = 0
    import glob
    for img in sorted(glob.glob(str(ROOT / "data/raw/s1*/00*/shot.png")))[:8]:
        els = parse_screen(img)
        v = verify_parse(els, 1920, 1080)
        ok_n += v["ok"] and not v["failures"]
        warn_n += len(v["warnings"])
        if v["failures"]:
            print("FAIL", img, v["failures"][:2])
    print(f"parse self-check: {ok_n}/8 张通过结构不变式，共 {warn_n} 条 warning（近重复/小框/孤立操作钮）")

    # Self-test 2: state-change verification on a REAL state pair (menu open vs closed)
    before = str(ROOT / "data/raw/s01/0001/shot.png")   # home
    after = str(ROOT / "data/raw/s02/0001/shot.png")    # more-menu open (same home base)
    from runtime.api import parse_screen as ps
    els = ps(after)
    # menu region: menu_item boxes bounding area
    menu_items = [e for e in els if e["role"] == "menu_item"]
    if menu_items:
        x1 = min(e["bbox"][0] for e in menu_items) - 10
        y1 = min(e["bbox"][1] for e in menu_items) - 10
        x2 = max(e["bbox"][2] for e in menu_items) + 10
        y2 = max(e["bbox"][3] for e in menu_items) + 10
        r = verify_state_change(before, after, (x1, y1, x2, y2))
        print(f"state-change verify (菜单区域 before/after): {r}")
        # negative control: a region that should NOT change (top nav)
        nav = [e for e in els if e["role"] == "nav_item"][0]
        r2 = verify_state_change(before, after, nav["bbox"])
        print(f"negative control (导航区域, 不该变): {r2}")
