"""P2: DOM calibration for runtime detections (hybrid mode).

When a DOM extraction is available (automation browser / collector), upgrade
detections with it: box snapping, FP removal, class arbitration.
Pure functions; works on parse_screen output + extractor v2 DOM payload.

Usage:
    from runtime.calibrate import calibrate
    els2 = calibrate(parse_screen(img), dom_payload)
"""
from __future__ import annotations


def _xyxy(rect: dict) -> list[float]:
    return [rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"]]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def calibrate(dets: list[dict], dom_payload: dict,
              snap_iou: float = 0.55, min_conf_keep: float = 0.90) -> list[dict]:
    """Return upgraded detections.

    - matched (IoU>=snap_iou, same class preferred): bbox snapped to DOM rect; role arbitrated by DOM
    - unmatched detection: kept only if confidence >= min_conf_keep (FP suppression)
    - each result carries calibrated: True/False and matched_role when a DOM pair existed
    """
    dom = []
    for e in dom_payload.get("els", []):
        dom.append({"role": e["role"], "xyxy": _xyxy(e["rect"]), "raw": e})
    used: set[int] = set()
    out = []
    for d in dets:
        best_iou, bi, bonus = 0.0, -1, 0.0
        for i, g in enumerate(dom):
            if i in used:
                continue
            v = _iou(d["bbox"], g["xyxy"])
            s = v + (0.2 if g["role"] == d["role"] else 0.0)
            if v >= snap_iou and s > bonus:
                best_iou, bi, bonus = v, i, s
        if bi >= 0:
            g = dom[bi]
            used.add(bi)
            nd = dict(d)
            nd["bbox"] = [round(v, 1) for v in g["xyxy"]]     # snap to DOM rect
            nd["role"] = g["role"]                              # DOM wins class arbitration
            nd["calibrated"] = True
            nd["dom_source"] = g["raw"].get("source")
            nd["text"] = g["raw"].get("text")
            out.append(nd)
        elif d["confidence"] >= min_conf_keep:
            nd = dict(d)
            nd["calibrated"] = False
            out.append(nd)
    return out


def text_spans_in(dom_payload: dict, bbox: list[float]) -> list[dict]:
    """DOM text spans (extractor v2) whose center falls in bbox — exact text for an element."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    hits = []
    for t in dom_payload.get("text_spans", []):
        r = t["rect"]
        if r["x"] <= cx <= r["x"] + r["w"] and r["y"] <= cy <= r["y"] + r["h"]:
            hits.append(t)
    return hits
