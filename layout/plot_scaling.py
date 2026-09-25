"""Scaling-study tables + curve from the layout matrix results.

Reads experiments/layout_matrix/results.jsonl → prints the prior×scale mAP50
matrix, per-class AP for full-data arms, and renders
experiments/layout_matrix/scaling_curve.png (mAP50 vs train samples per prior).

Usage: .venv/bin/python layout/plot_scaling.py [--results experiments/layout_matrix/results.jsonl]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ARMS = ["coco_nano", "coco_small", "gui_screenparser", "gui_omniparser"]
ARM_LABEL = {"coco_nano": "YOLO11n (COCO)", "coco_small": "YOLO11s (COCO)",
             "gui_screenparser": "ScreenParser (GUI, YOLO11-L)",
             "gui_omniparser": "OmniParser v2 (GUI, YOLOv8)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/layout_matrix/results.jsonl")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / args.results).read_text().splitlines() if l.strip()]
    by = {(r["arm"], str(r["scale"])): r for r in rows if r.get("ok")}
    scales = ["100", "250", "500", "full"]

    print("| prior \\ train | " + " | ".join(scales) + " |")
    print("|---" * (len(scales) + 1) + "|")
    for arm in ARMS:
        cells = []
        for s in scales:
            r = by.get((arm, s))
            cells.append(f"{r['mAP50']:.3f}" if r else "—")
        print(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")

    ok = [r for r in rows if r.get("ok")]
    print(f"\nruns ok: {len(ok)}/{len(rows)}")
    for r in sorted(ok, key=lambda x: -x["mAP50"]):
        print(f"  {r['exp']:36s} mAP50={r['mAP50']:.4f} mAP50-95={r.get('mAP50-95',0):.4f} "
              f"P={r.get('precision',0):.3f} R={r.get('recall',0):.3f} params={r.get('params')}")

    # per-class AP for full-data arms
    for arm in ARMS:
        r = by.get((arm, "full"))
        if r and r.get("per_class_ap50"):
            pc = sorted(r["per_class_ap50"].items(), key=lambda kv: kv[1])
            print(f"\n[{ARM_LABEL[arm]} @full] per-class AP50 (lowest first):")
            for k, v in pc:
                print(f"  {k:14s} {v:.3f}")

    # curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for arm in ARMS:
        xs, ys = [], []
        for s in scales:
            r = by.get((arm, s))
            if r:
                xs.append(100 if s == "100" else 250 if s == "250" else 500 if s == "500" else 729)
                ys.append(r["mAP50"])
        if xs:
            ax.plot(xs, ys, "o-", label=ARM_LABEL[arm])
    ax.set_xlabel("X-specific training samples")
    ax.set_ylabel("frozen-test mAP50 (layout, 11 classes)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    out = ROOT / args.results
    png = out.parent / "scaling_curve.png"
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    print("\n→", png)


if __name__ == "__main__":
    main()
