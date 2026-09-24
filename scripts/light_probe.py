"""Light-theme generalization probe: parse_screen (ONNX) vs DOM GT on light samples.

Reports per-class recall/precision on any set of raw samples — used to quantify
zero-shot light performance of dark-trained models and the gain after light training.

Usage: .venv/bin/python scripts/light_probe.py --samples "data/raw/s19*,data/raw/s20*" --onnx experiments/latest/best_onnx_fp32.onnx
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


def load_names():
    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    return {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}


def match(gt_boxes, dets):
    hit = 0
    used = set()
    for gx, gy, gw, gh in gt_boxes:
        for i, (dx1, dy1, dx2, dy2, _) in enumerate(dets):
            if i in used:
                continue
            cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
            if gx <= cx <= gx + gw and gy <= cy <= gy + gh:
                hit += 1
                used.add(i)
                break
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="glob patterns comma-separated, e.g. 'data/raw/s19*,data/raw/s20*'")
    ap.add_argument("--onnx", default=str(ROOT / "experiments/latest/best_onnx_fp32.onnx"))
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    dirs = []
    for pat in args.samples.split(","):
        for d in glob.glob(pat):
            p = Path(d)
            if not p.is_dir():
                continue
            sub = [x for x in p.iterdir() if x.is_dir() and x.name.isdigit()]
            dirs += [str(x) for x in sub] if sub else [d]  # session dir → its sample dirs
    dirs = sorted(set(dirs))[: args.limit]
    if not dirs:
        raise SystemExit("no sample dirs")

    sp = ScreenParser(args.onnx, load_names(), conf=args.conf)
    cls_hit, cls_gt, cls_pred = Counter(), Counter(), Counter()
    n = 0
    for d in dirs:
        anno_f, shot = Path(d) / "anno.json", Path(d) / "shot.png"
        if not (anno_f.exists() and shot.exists()):
            continue
        anno = json.loads(anno_f.read_text())
        dets = sp(shot)
        n += 1
        for role in {e["role"] for e in anno["els"]} | {e["role"] for e in dets}:
            gts = [(e["rect"]["x"], e["rect"]["y"], e["rect"]["w"], e["rect"]["h"])
                   for e in anno["els"] if e["role"] == role]
            ds = [(e["bbox"][0], e["bbox"][1], e["bbox"][2], e["bbox"][3], e["confidence"])
                  for e in dets if e["role"] == role]
            cls_hit[role] += match(gts, ds)
            cls_gt[role] += len(gts)
            cls_pred[role] += len(ds)

    total_gt, total_hit = sum(cls_gt.values()), sum(cls_hit.values())
    rows = []
    for role, gt in cls_gt.most_common():
        rec = cls_hit[role] / gt if gt else 0
        prec = cls_hit[role] / cls_pred[role] if cls_pred[role] else 0
        rows.append((role, gt, round(rec, 3), round(prec, 3)))
    out = {
        "onnx": args.onnx, "n_images": n, "conf": args.conf,
        "micro_recall": round(total_hit / total_gt, 4) if total_gt else None,
        "total_gt": total_gt, "total_hit": total_hit,
        "per_class": [{"role": r, "gt": g, "recall": rec, "precision": p} for r, g, rec, p in rows],
    }
    print(json.dumps(out, indent=1)[:2500])
    tag = Path(args.onnx).parent.name
    (ROOT / "experiments" / f"light_probe_{tag}.json").write_text(json.dumps(out, indent=1))
    print(f"→ experiments/light_probe_{tag}.json")


if __name__ == "__main__":
    main()
