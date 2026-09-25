"""Export synthesized layout labels to YOLO format with the SAME frozen split
as the atomic detector (data/export_meta_v03_frozen.json groups), plus
sample-scaling train subsets (100/250/500/full) for the GUI-prior study.

Usage:
  .venv/bin/python layout/export_layout_yolo.py \
      --labels data/layout_labels --images data/raw \
      --freeze-from data/export_meta_v03_frozen.json --out data/yolo_layout
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LAYOUT_CLASSES = ["action_row", "card", "compose_bar", "left_nav", "media_region",
                  "overlay", "post", "post_header", "quoted_post", "right_sidebar",
                  "text_region"]  # sorted; class id = index


def group_of(lab: dict) -> str:
    return f"{lab['session']}::{lab['route'].replace('state_', 's_')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/layout_labels")
    ap.add_argument("--images", default="data/raw")
    ap.add_argument("--freeze-from", default="data/export_meta_v03_frozen.json")
    ap.add_argument("--out", default="data/yolo_layout")
    ap.add_argument("--scales", default="100,250,500")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lab_root, img_root = ROOT / args.labels, ROOT / args.images
    out = ROOT / args.out
    frozen = json.loads((ROOT / args.freeze_from).read_text())
    g_train = set(frozen["groups_train"])
    g_val = set(frozen["groups_val"])
    g_test = set(frozen["groups_test"])

    labs = []
    for jf in sorted(lab_root.glob("*/*.json")):
        lab = json.loads(jf.read_text())
        g = group_of(lab)
        if g in g_test:
            split = "test"
        elif g in g_val:
            split = "val"
        elif g in g_train:
            split = "train"
        else:  # groups collected after v0.3 freeze → train (same policy as atomic)
            split = "train"
        labs.append((lab, split))

    # write full dataset
    shutil.rmtree(out, ignore_errors=True)
    counts = Counter()
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for lab, split in labs:
        img = img_root / lab["image"]
        if not img.exists():
            continue
        dst = out / "images" / split / img.name.replace("shot", f"{lab['session']}_{lab['sample_id']}")
        shutil.copy(img, dst)
        lines = []
        for b in lab["boxes"]:
            x1, y1, x2, y2 = b["bbox"]
            cx, cy = (x1 + x2) / 2 / lab["vw"], (y1 + y2) / 2 / lab["vh"]
            w, h = (x2 - x1) / lab["vw"], (y2 - y1) / lab["vh"]
            if w <= 0 or h <= 0 or w > 1 or h > 1:
                continue
            lines.append(f"{LAYOUT_CLASSES.index(b['cls'])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            counts[f"{split}:{b['cls']}"] += 1
        (out / "labels" / split / (dst.stem + ".txt")).write_text("\n".join(lines))

    n = {s: len(list((out / "images" / s).glob("*.png"))) for s in ("train", "val", "test")}
    inst = Counter()
    for b in LAYOUT_CLASSES:
        inst[b] = counts[f"train:{b}"] + counts[f"val:{b}"] + counts[f"test:{b}"]
    meta = {"n_images": n, "classes": LAYOUT_CLASSES, "instances_per_class": dict(inst),
            "split_source": "export_meta_v03_frozen.json (groups identical to atomic)",
            "labels_source": "layout/build_labels.py (DOM-synthesized)"}
    (out / "layout_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"train={n['train']} val={n['val']} test={n['test']} (frozen test == atomic)")
    print("per-class:", {k: v for k, v in inst.most_common()})

    # scaling subsets: TRAIN-only manifests (VM side builds the subset tree; val/test
    # always come from the full export so every arm sees the identical frozen eval)
    random.seed(args.seed)
    order = sorted((out / "images" / "train").glob("*.png"))
    random.shuffle(order)
    for target in [int(s) for s in args.scales.split(",")]:
        picked = order[:target]
        (out / f"scale_{target}.txt").write_text(
            "\n".join(p.name for p in picked))
        print(f"scale_{target}: {len(picked)} train frames → manifest")


if __name__ == "__main__":
    main()
