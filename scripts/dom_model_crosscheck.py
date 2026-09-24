"""Cross-check model detections against DOM ground truth, per sample.

Association: for each detection, find the DOM element (same role preferred) whose box
contains the detection center / max IoU. Reports:
  matched (class-agree) / matched (class-disagree) / DOM-only (model missed)
  / model-only (candidate FP or DOM label gap)  + box IoU distribution.

Two cleaning uses:
  a) disagreement samples → data cleaning candidates (model error OR DOM label noise)
  b) runtime hybrid: when DOM is available, snap/filter detections by DOM

Usage: .venv/bin/python scripts/dom_model_crosscheck.py --samples "data/raw/s05,data/raw/s19,data/raw/s17" --n 24
"""
import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.api import ScreenParser  # noqa: E402

import yaml  # noqa: E402


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def to_xyxy(rect) -> list[float]:
    return [rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"]]


def associate(dets, dom_els):
    """Greedy match each detection to best DOM element (IoU), preferring same role."""
    pairs = []
    used = set()
    for d in dets:
        best, bi, sc = 0.0, -1, 0.0
        for i, g in enumerate(dom_els):
            if i in used:
                continue
            v = iou(d["bbox"], g["_xyxy"])
            bonus = 0.15 if g["role"] == d["role"] else 0.0
            s = v + bonus
            if s > sc and v > 0.30:
                best, bi, sc = v, i, s
        if bi >= 0:
            used.add(bi)
            pairs.append((d, dom_els[bi], best))
        else:
            pairs.append((d, None, 0.0))
    return pairs, used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="data/raw/s05,data/raw/s17,data/raw/s19")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="experiments/dom_model_crosscheck.json")
    args = ap.parse_args()

    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"), names,
                      conf=0.30, check_purity=False)

    dirs = []
    for pat in args.samples.split(","):
        for d in glob.glob(pat):
            p = Path(d)
            sub = [x for x in p.iterdir() if x.is_dir() and x.name.isdigit()]
            dirs += [str(x) for x in sub] if sub else [d]
    dirs = sorted(dirs)[: args.n]

    stats = Counter()
    ious = []
    disagree_samples = []
    for d in dirs:
        anno_f, shot = Path(d) / "anno.json", Path(d) / "shot.png"
        if not (anno_f.exists() and shot.exists()):
            continue
        dom = json.loads(anno_f.read_text())["els"]
        for g in dom:
            g["_xyxy"] = to_xyxy(g["rect"])
        dets = sp(shot)
        pairs, used = associate(dets, dom)
        dis = []
        for det, g, v in pairs:
            if g is None:
                stats["model_only(candidate FP/DOM gap)"] += 1
                dis.append({"kind": "model_only", "role": det["role"], "conf": det["confidence"]})
            else:
                ious.append(v)
                if g["role"] == det["role"]:
                    stats["matched_class_agree"] += 1
                else:
                    stats["matched_class_disagree"] += 1
                    dis.append({"kind": "class_swap", "model": det["role"], "dom": g["role"],
                                "iou": round(v, 2)})
        for i, g in enumerate(dom):
            if i not in used:
                stats["dom_only(model missed)"] += 1
        if dis:
            disagree_samples.append({"sample": d, "n": len(dis), "detail": dis[:6]})

    n = len(ious)
    rep = {
        "n_samples": len(dirs),
        "stats": dict(stats),
        "box_iou": {
            "mean": round(sum(ious) / n, 3) if n else None,
            "p50": round(sorted(ious)[n // 2], 3) if n else None,
            "frac>=0.75": round(sum(1 for v in ious if v >= 0.75) / n, 3) if n else None,
        },
        "disagreement_samples": disagree_samples[:20],
    }
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "disagreement_samples"},
                     ensure_ascii=False, indent=1))
    print(f"disagreement samples: {len(disagree_samples)}/{len(dirs)} → {args.out}")


if __name__ == "__main__":
    main()
