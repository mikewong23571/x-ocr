"""P3: structured screen document assembler — screenshot → compact agent context.

Layers: detector (geometry+roles) → geometric grouping (posts/nav/rails) → on-demand
crop OCR (text) → stable schema (~400-700 tokens per screen).

Usage:
    from runtime.screen_doc import screen_document
    doc = screen_document("shot.png")            # vision-only
    doc = screen_document("shot.png", dom=payload)  # + DOM calibration/exact text

CLI: .venv/bin/python runtime/screen_doc.py shot.png [--dom anno.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.api import ScreenParser  # noqa: E402

# X left-nav layout prior: order is stable per account family
NAV_PRIOR = ["Home", "Explore", "Notifications", "Follow", "DM", "Grok",
             "Premium", "History", "CreatorStudio", "Articles", "Profile", "More"]

ACTION_ROLES = ("reply_button", "repost_button", "like_button", "bookmark_button", "share_button")


class _OCR:
    def __init__(self):
        self._o = None

    def __call__(self, im) -> str:
        if self._o is None:
            from rapidocr_onnxruntime import RapidOCR
            self._o = RapidOCR()
        import numpy as np
        res, _ = self._o(np.asarray(im))
        return " ".join(r[1] for r in res).strip() if res else ""


_ocr = _OCR()


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def screen_document(image, dom: dict | None = None, conf: float = 0.35,
                    max_text_chars: int = 140) -> dict:
    from PIL import Image
    from runtime.calibrate import calibrate, text_spans_in

    import yaml
    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"), names,
                      conf=conf, check_purity=False)
    im = Image.open(image).convert("RGB")
    els = sp(im)
    if dom:
        els = calibrate(els, dom)

    by = {}
    for e in els:
        by.setdefault(e["role"], []).append(e)

    def read_text(bbox):
        if dom:  # exact DOM text when available (priority)
            ts = text_spans_in(dom, bbox)
            if ts:
                return " ".join(t["text"] for t in ts)[:max_text_chars]
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2, y2 = min(x2, im.width), min(y2, im.height)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return ""
        return _ocr(im.crop((x1, y1, x2, y2)))[:max_text_chars]

    doc = {
        "viewport": [im.width, im.height],
        "theme": "dark" if _is_dark(im) else "light",
        "nav": [],
        "search": None,
        "compose": None,
        "posts": [],
        "menus": [],
        "dialog": None,
        "right_rail": [],
        "n_elements": len(els),
    }

    # nav column (left rail, sorted by y) with layout prior names
    navs = sorted(by.get("nav_item", []), key=lambda e: e["bbox"][1])
    doc["nav"] = [
        {"name": NAV_PRIOR[i] if i < len(NAV_PRIOR) else f"nav{i}",
         "click": [int(_center(e["bbox"])[0]), int(_center(e["bbox"])[1])]}
        for i, e in enumerate(navs)
    ]
    if by.get("search_input"):
        doc["search"] = {"click": [int(v) for v in _center(by["search_input"][0]["bbox"])]}
    if by.get("compose_button"):
        doc["compose"] = {"click": [int(v) for v in _center(by["compose_button"][0]["bbox"])],
                          "post_button": [int(v) for v in _center(by["post_button"][0]["bbox"])]
                          if by.get("post_button") else None}

    # posts: username-anchored grouping. A container box can span several posts
    # (merged detection) — pooling its children pairs the wrong author with the
    # wrong text, so we group by "each column-left-aligned username starts a post".
    uns = sorted(by.get("username_link", []), key=lambda e: e["bbox"][1])
    grouped = False
    if uns:
        ax = sorted(u["bbox"][0] for u in uns)[len(uns) // 2]  # median column left edge
        cards = by.get("link_card", []) + by.get("post_media", [])

        def in_card(e) -> bool:
            cx, cy = _center(e["bbox"])
            return any(k["bbox"][0] - 4 <= cx <= k["bbox"][2] + 4
                       and k["bbox"][1] - 4 <= cy <= k["bbox"][3] + 4 for k in cards)

        feed = sorted(
            (e for e in els
             if e["role"] in (*ACTION_ROLES, "username_link", "post_text",
                              "post_media", "video", "link_card", "overflow_button")
             and ax - 100 <= (e["bbox"][0] + e["bbox"][2]) / 2 <= ax + 620),
            key=lambda e: e["bbox"][1])
        cur = None
        for e in feed:
            r = e["role"]
            if r == "username_link":
                if e["bbox"][0] <= ax + 30 and not in_card(e):
                    cur = {"actions": {},
                           "author": (e.get("text") or read_text(e["bbox"]))[:48]}
                    doc["posts"].append(cur)
                elif cur is not None:
                    # indented username inside a quote/media card: annotate, don't take over
                    cur.setdefault("quoted_author",
                                   (e.get("text") or read_text(e["bbox"]))[:48])
                continue
            if cur is None:
                # cut-off post above the first visible username: keep its visible
                # text/actions as an anonymous post instead of dropping them
                if r == "overflow_button":
                    continue
                cur = {"actions": {}, "author": None}
                doc["posts"].append(cur)
            if r == "post_text":
                t = e.get("text") or read_text(e["bbox"])
                cur["text"] = (cur.get("text", "") + " " + t).strip()[:max_text_chars]
            elif r in ACTION_ROLES:
                cur["actions"].setdefault(r, [int(v) for v in _center(e["bbox"])])
            elif r == "video":
                cur["media"] = "video"
            elif r == "post_media":
                cur["media"] = (cur.get("media", "") + "+img").lstrip("+")
            elif r == "link_card":
                cur["link_card"] = read_text(e["bbox"])[:40]
        grouped = doc["posts"] != []

    if not grouped:
        # fallback: container-scoped grouping with action-row dedup
        seen_action_rows: set[tuple] = set()
        for c in sorted(by.get("post_container", []), key=lambda e: e["bbox"][1]):
            span = (c["bbox"][1] - 40, c["bbox"][3] + 40)
            own = [e for e in els
                   if e is not c and e["role"] in (*ACTION_ROLES, "username_link", "post_text",
                                                  "post_media", "video", "link_card", "overflow_button")
                   and span[0] <= (e["bbox"][1] + e["bbox"][3]) / 2 <= span[1]
                   and c["bbox"][0] - 20 <= (e["bbox"][0] + e["bbox"][2]) / 2]
            acts = {e["role"]: [int(v) for v in _center(e["bbox"])]
                    for e in own if e["role"] in ACTION_ROLES}
            key = tuple(tuple(v) for v in sorted(acts.values()))
            if acts and key in seen_action_rows:   # duplicated container on same action row
                continue
            if acts:
                seen_action_rows.add(key)
            p = {"actions": acts}
            for e in own:
                if e["role"] == "username_link":
                    p["author"] = (e.get("text") or read_text(e["bbox"]))[:48]
                elif e["role"] == "post_text" and "text" not in p:
                    p["text"] = (e.get("text") or read_text(e["bbox"]))[:max_text_chars]
                elif e["role"] == "video":
                    p["media"] = "video"
                elif e["role"] == "post_media":
                    p["media"] = (p.get("media", "") + "+img").lstrip("+")
                elif e["role"] == "link_card":
                    p["link_card"] = read_text(e["bbox"])[:40]
            doc["posts"].append(p)

    # menus / dialog states
    doc["menus"] = [{"item": (e.get("text") or read_text(e["bbox"]))[:24],
                     "click": [int(v) for v in _center(e["bbox"])]}
                    for e in by.get("menu_item", [])]
    if by.get("dialog"):
        d = by["dialog"][0]
        doc["dialog"] = {"bbox": [int(v) for v in d["bbox"]],
                         "close": [int(v) for v in _center(by["close_button"][0]["bbox"])]
                         if by.get("close_button") else None}
    # right rail
    for e in by.get("trend", []) + by.get("trend_item", []) + by.get("news_card", []):
        doc["right_rail"].append({"type": e["role"].replace("_item", ""),
                                  "text": read_text(e["bbox"])[:36]})
    return doc


def _is_dark(im) -> bool:
    import numpy as np
    a = np.asarray(im.convert("L").resize((64, 36)))
    return float(a.mean()) < 128


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--dom", default=None, help="extractor v2 anno.json for hybrid mode")
    a = ap.parse_args()
    dom = json.loads(Path(a.dom).read_text()) if a.dom else None
    out = screen_document(a.image, dom=dom)
    s = json.dumps(out, ensure_ascii=False, indent=1)
    print(s)
    print(f"# ≈{len(s)//3} tokens", file=sys.stderr)
