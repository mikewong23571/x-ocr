"""Structured screen perception: Atomic detector + Layout parser + Regional OCR.

Deterministic geometry fusion only (no learned fusion): containment / overlap /
y-alignment / nearest-parent, per the phase-2 spec. Output is a hierarchy:

    {"regions": [{"type": "post", "bbox": [...], "text": "...", "children":
                   [{"role": "like_button", "bbox": [...]}, ...]}, ...]}

Text comes from regional OCR (crop→rec) on layout text regions — never a default
full-screen OCR pass.

Usage:
    from runtime.structured import structured_screen
    doc = structured_screen("shot.png")            # pure vision, 3 models
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.api import ScreenParser, assert_pure_runtime  # noqa: E402

ACTION_ROLES = ("reply_button", "repost_button", "like_button", "bookmark_button", "share_button")
# atomic roles attached under layout regions (children)
CHILD_ROLES = (*ACTION_ROLES, "avatar", "username_link", "overflow_button",
               "post_button", "follow_button", "icon_button", "tab", "compose_textbox")
TEXT_ROLES_OCR = ("text_region", "post_header")  # layout regions we read text from


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def _contains(outer, inner, pad=0):
    cx, cy = _center(inner)
    return (outer[0] - pad <= cx <= outer[2] + pad
            and outer[1] - pad <= cy <= outer[3] + pad)


class LayoutParser(ScreenParser):
    """Same ONNX pipeline, layout class names, higher conf (regions are big)."""


class RegionalOCR:
    """Crop→rec-only OCR engine wrapper (rapidocr v3 PP-OCRv6-small rec)."""

    def __init__(self):
        from rapidocr import RapidOCR
        from rapidocr.utils.parse_parameters import OCRVersion, ModelType
        self._full_ctx = RapidOCR(params={
            "Rec.ocr_version": OCRVersion("PP-OCRv6"), "Rec.model_type": ModelType("small")})

    def read_lines(self, im, boxes) -> list[str]:
        import io
        out = []
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = im.crop((max(0, x1 - 4), max(0, y1 - 3), x2 + 4, y2 + 3))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            r = self._full_ctx(buf.getvalue())  # engine reused in det-off mode below
            out.append((r.txts[0] if r.txts is not None and len(r.txts) else "").strip())
        return out

    def read_region(self, im, bbox) -> str:
        """det+rec on the region crop — for element-level boxes without line splits."""
        import io
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 - x1 < 8 or y2 - y1 < 8:
            return ""
        crop = im.crop((max(0, x1 - 6), max(0, y1 - 6), x2 + 6, y2 + 6))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        r = self._full_ctx(buf.getvalue())
        if r.txts is None:
            return ""
        return " ".join(t for t in r.txts if t).strip()


_ocr: RegionalOCR | None = None


def _get_ocr() -> RegionalOCR:
    global _ocr
    if _ocr is None:
        _ocr = RegionalOCR()
    return _ocr


def structured_screen(image, layout_onnx: str | Path | None = None,
                      atomic_onnx: str | Path | None = None,
                      conf_atomic: float = 0.30, conf_layout: float = 0.35,
                      ocr: bool = True, timing: bool = False):
    """screenshot → structured screen representation (3 cooperating small models)."""
    from PIL import Image
    import yaml

    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    atomic_names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    ltax = yaml.safe_load((ROOT / "schema/layout_taxonomy.yaml").read_text())
    layout_names = {i: c for i, c in enumerate(sorted(ltax["classes"].keys()))}

    t0 = time.time()
    im = Image.open(image).convert("RGB")
    atomic = ScreenParser(str(atomic_onnx or ROOT / "experiments/latest/best_onnx_fp32.onnx"),
                          atomic_names, conf=conf_atomic, check_purity=False)(im)
    t1 = time.time()
    if layout_onnx:
        layout = LayoutParser(str(layout_onnx), layout_names, conf=conf_layout,
                              check_purity=False)(im)
    else:
        layout = []
    t2 = time.time()

    # --- deterministic geometry fusion ---
    # 1) attach atomic children to the smallest containing layout region
    regions = [{"type": e["role"], "bbox": e["bbox"]} for e in layout]
    els = [{"role": e["role"], "bbox": e["bbox"], "confidence": e["confidence"]}
           for e in atomic]
    for e in els:
        if e["role"] not in CHILD_ROLES:
            continue
        hosts = [r for r in regions if _contains(r["bbox"], e["bbox"])]
        host = min(hosts, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])) if hosts else None
        if host is not None:
            host.setdefault("children", []).append(
                {"role": e["role"], "bbox": e["bbox"], "confidence": e["confidence"]})

    # 2) y-alignment fallback: action buttons with no container → nearest post above
    posts = [r for r in regions if r["type"] == "post"]
    for e in els:
        if e["role"] not in ACTION_ROLES:
            continue
        if any(e in [c for c in r.get("children", [])] or
               any(c["bbox"] == e["bbox"] for c in r.get("children", [])) for r in regions):
            continue
        cx, cy = _center(e["bbox"])
        above = [p for p in posts if p["bbox"][1] <= cy <= p["bbox"][3] + 120
                 and _contains((p["bbox"][0], p["bbox"][1], p["bbox"][2], p["bbox"][3] + 120), e["bbox"])]
        if above:
            above[0].setdefault("children", []).append(
                {"role": e["role"], "bbox": e["bbox"], "confidence": e["confidence"]})

    # 3) regional OCR on text-bearing layout regions (never full screen)
    t3 = time.time()
    if ocr:
        eng = _get_ocr()
        for r in regions:
            if r["type"] in TEXT_ROLES_OCR:
                r["text"] = eng.read_region(im, r["bbox"])[:140]
    t4 = time.time()

    # 4) roll-up: nested regions' children/text surface on their post (the
    #    agent-facing contract is post → actions + text, not deep nesting)
    post_regions = [r for r in regions if r["type"] == "post"]
    for p in post_regions:
        inner = [r for r in regions if r is not p and r["type"] != "post"
                 and _contains(p["bbox"], r["bbox"])]
        for r in inner:
            for c in r.get("children", []):
                if not any(c["bbox"] == k["bbox"] for k in p.get("children", [])):
                    p.setdefault("children", []).append(c)
        texts = [r.get("text") for r in inner
                 if r["type"] == "text_region" and r.get("text")]
        if texts:
            p["text"] = " ".join(texts)[:140]
        elif ocr and "text" not in p:
            # post without a detected text_region: derive a text strip —
            # header bottom → first child action top (guarded, may be empty)
            kids = p.get("children", [])
            tops = [c["bbox"][1] for c in kids if c["role"] in ACTION_ROLES]
            if tops and p["bbox"][3] > min(tops) - 8:
                strip = (p["bbox"][0], p["bbox"][1] + 20, p["bbox"][2], min(tops) - 8)
                if strip[3] - strip[1] > 16:
                    p["text"] = eng.read_region(im, strip)[:140]
    t5 = time.time()

    doc = {
        "viewport": [im.width, im.height],
        "regions": regions,
        "n_atomic": len(els),
        "n_layout": len(regions),
        "_hybrid": False,
    }
    if timing:
        doc["_timing_ms"] = {"atomic": round((t1 - t0) * 1000), "layout": round((t2 - t1) * 1000),
                             "fusion": round((t3 - t2) * 1000), "ocr": round((t4 - t3) * 1000),
                             "rollup": round((t5 - t4) * 1000)}
    return doc


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--layout-onnx")
    ap.add_argument("--no-ocr", action="store_true")
    args = ap.parse_args()
    doc = structured_screen(args.image, layout_onnx=args.layout_onnx, ocr=not args.no_ocr)
    print(json.dumps(doc, ensure_ascii=False, indent=1)[:2400])
