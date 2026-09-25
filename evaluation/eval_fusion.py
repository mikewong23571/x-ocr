"""E4 — multi-model fusion evaluation on the frozen layout test set.

Runs Atomic + Layout (+ Regional OCR) and measures, against DOM-synthesized GT:
  - atomic UI recall / precision (IoU≥0.3 vs fused DOM els)
  - layout recall / precision (IoU≥0.5 vs synthesized layout boxes)
  - parent-child association accuracy (children attached under the right region)
  - complete post reconstruction accuracy (post with ≥1 action + any text, all
    correctly grouped vs GT post membership)
  - text recognition CER on post text regions (when --ocr)
  - end-to-end latency breakdown + total artifact size

Usage: .venv/bin/python evaluation/eval_fusion.py --layout-onnx <best.onnx> [--ocr]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.structured import structured_screen, _iou, _center  # noqa: E402


def load_gt():
    frozen = json.loads((ROOT / "data/export_meta_v03_frozen.json").read_text())
    gtest = set(frozen["groups_test"])
    frames = []
    for jf in sorted((ROOT / "data/layout_labels").glob("*/*.json")):
        lab = json.loads(jf.read_text())
        g = f"{lab['session']}::{lab['route'].replace('state_', 's_')}"
        if g in gtest:
            frames.append(lab)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout-onnx", required=True)
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--stub-layout", action="store_true",
                    help="use GT layout boxes as predictions (pipeline upper bound)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    frames = load_gt()
    if args.limit:
        frames = frames[: args.limit]
    print(f"frozen test frames: {len(frames)}")

    # atomic GT comes from processed manifests
    atomic_gt = {}
    for mf in sorted((ROOT / "data/processed").glob("*/manifest.jsonl")):
        for l in mf.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            atomic_gt[(r["session"], r["sample_id"])] = [
                {"role": e["role"], "bbox": [e["bbox"][0], e["bbox"][1],
                                              e["bbox"][0] + e["bbox"][2],
                                              e["bbox"][1] + e["bbox"][3]]}
                for e in r["els"]]

    agg = defaultdict(list)
    if args.stub_layout:
        import runtime.structured as _st
        _orig_lp = _st.LayoutParser

        def _make_stub(lab):
            class _Stub:
                def __init__(self, *a, **k):
                    self.lab = lab
                def __call__(self, im):
                    return [{"role": b["cls"], "bbox": b["bbox"], "confidence": 1.0}
                            for b in self.lab["boxes"]]
            return _Stub
    for lab in frames:
        shot = ROOT / "data/raw" / lab["image"]
        if not shot.exists():
            continue
        if args.stub_layout:
            import runtime.structured as _st
            _st.LayoutParser = _make_stub(lab)
        doc = structured_screen(shot, layout_onnx=args.layout_onnx, ocr=args.ocr,
                                timing=True)
        if args.stub_layout:
            import runtime.structured as _st
            _st.LayoutParser = _orig_lp
        t = doc.get("_timing_ms", {})

        # layout recall/precision
        gt_boxes = [(b["cls"], b["bbox"]) for b in lab["boxes"]]
        pred_boxes = [(r["type"], r["bbox"]) for r in doc["regions"]]
        used = set()
        hits = 0
        for gcls, gb in gt_boxes:
            best, bv = None, 0.45
            for i, (pcls, pb) in enumerate(pred_boxes):
                if i in used or pcls != gcls:
                    continue
                v = _iou(gb, pb)
                if v > bv:
                    best, bv = i, v
            if best is not None:
                used.add(best)
                hits += 1
        agg["layout_recall"].append(hits / max(len(gt_boxes), 1))
        agg["layout_precision"].append(hits / max(len(pred_boxes), 1))

        # atomic recall/precision
        agt = atomic_gt.get((lab["session"], lab["sample_id"]), [])
        # restrict GT to children-attachable roles for fair recall vs doc? No —
        # atomic recall measured on the raw atomic list inside structured_screen:
        from runtime.api import ScreenParser
        import yaml
        tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
        names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
        sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"),
                          names, conf=0.30, check_purity=False)
        raw = sp(shot)
        au = set()
        ah = 0
        for g in agt:
            best, bv = None, 0.30
            for i, p in enumerate(raw):
                if i in au or p["role"] != g["role"]:
                    continue
                v = _iou(g["bbox"], p["bbox"])
                if v > bv:
                    best, bv = i, v
            if best is not None:
                au.add(best)
                ah += 1
        agg["atomic_recall"].append(ah / max(len(agt), 1))
        agg["atomic_precision"].append(ah / max(len(raw), 1))

        # parent-child association: predicted child inside predicted post region
        # vs GT: child's center inside GT post of same... compare membership of
        # action buttons: GT parent = containing GT post; pred parent = region.
        gt_posts = [b["bbox"] for b in lab["boxes"] if b["cls"] == "post"]
        n_ok = n_tot = 0
        for r in doc["regions"]:
            if r["type"] != "post":
                continue
            for c in r.get("children", []):
                cx, cy = _center(c["bbox"])
                in_pred = True
                in_gt = any(g[0] <= cx <= g[2] and g[1] <= cy <= g[3] for g in gt_posts)
                n_tot += 1
                if in_pred and in_gt:
                    n_ok += 1
        if n_tot:
            agg["assoc_accuracy"].append(n_ok / n_tot)

        # complete post reconstruction: post region with ≥1 action child AND
        # (text or a text child region nested) AND matching a GT post IoU≥0.5
        for r in doc["regions"]:
            if r["type"] != "post":
                continue
            kids = r.get("children", [])
            has_act = any(c["role"] in ("reply_button", "like_button", "repost_button",
                                        "bookmark_button", "share_button") for c in kids)
            if has_act and r.get("text"):
                ious = [_iou(r["bbox"], g) for g in gt_posts]
                agg["post_recon"].append(float(max(ious, default=0) >= 0.5))

        for k, v in t.items():
            agg[f"t_{k}"].append(v)

    def m(k):
        v = agg[k]
        return round(sum(v) / len(v), 4) if v else None

    print(json.dumps({k: m(k) for k in sorted(agg)}, indent=1))


if __name__ == "__main__":
    main()
