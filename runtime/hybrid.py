"""Hybrid perception: ML detector + DOM, one call, in an automation browser.

    from runtime.hybrid import perceive
    doc = perceive(page)            # playwright Page → calibrated screen document
    els = perceive(page, raw=True)  # → calibrated element list

Pipeline (same frame): DOM extract + screenshot → detector → calibration
(snap / FP-filter / class-arbitration) → grouping → exact text → screen document.
Falls back to vision-only automatically when DOM extraction fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXTRACT_JS = (ROOT / "scripts/js/extract_annotated.js").read_text()


def _layout_fp(dom: dict) -> list:
    return sorted((e["role"], int(e["rect"]["x"]) // 8, int(e["rect"]["y"]) // 8,
                   int(e["rect"]["w"]) // 8, int(e["rect"]["h"]) // 8)
                  for e in dom["els"])


def extract_dom(page, stable: bool = True) -> dict | None:
    """Same-frame DOM payload (extractor v2: els + text_spans + hierarchy).

    stable=True re-extracts once and returns None when the layout churned between
    calls — calibrating against a reflowed DOM would snap every box off-target.
    """
    try:
        d1 = page.evaluate(EXTRACT_JS)
        if not stable or not d1:
            return d1
        page.wait_for_timeout(250)
        d2 = page.evaluate(EXTRACT_JS)
        return d2 if d2 and _layout_fp(d2) == _layout_fp(d1) else None
    except Exception:
        return None


def perceive(page, raw: bool = False, conf: float = 0.35):
    """One-call perception inside a playwright browser.

    page: playwright Page (any chromium context, e.g. vision_automation_demo's)
    raw=True returns the calibrated element list instead of the screen document.
    """
    import tempfile

    from runtime.screen_doc import screen_document
    from runtime.api import ScreenParser
    from runtime.calibrate import calibrate
    import yaml

    shot = page.screenshot()  # screenshot FIRST, then DOM in the same settled state
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(shot)
        img_path = f.name
    dom = extract_dom(page)

    if raw:
        tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
        names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
        sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"), names,
                          conf=conf, check_purity=False)
        els = sp(img_path)
        if dom:
            els = calibrate(els, dom)
        Path(img_path).unlink(missing_ok=True)
        return els, dom

    doc = screen_document(img_path, dom=dom, conf=conf)
    Path(img_path).unlink(missing_ok=True)
    doc["_hybrid"] = dom is not None  # marker: DOM calibration active?
    return doc
