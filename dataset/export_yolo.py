"""Export fused annotations to YOLO detection format with leakage-safe split.

Split unit: session × route (all screenshots of one route in one session stay together).

Usage:
  .venv/bin/python dataset/export_yolo.py --in data/processed --out data/yolo --val 0.15 --test 0.15
"""
import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    proc_root = ROOT / args.inp
    out = ROOT / args.out
    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    classes = sorted(tax["classes"].keys())
    cid = {c: i for i, c in enumerate(classes)}

    rows = []
    for mf in sorted(proc_root.glob("*/manifest.jsonl")):
        rows += [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no rows")

    # group = (session, route-family) — same route scroll sequence stays together
    def group(r: dict) -> str:
        route = r["route"].replace("state_", "s_")
        return f"{r['session']}::{route}"

    groups = sorted({group(r) for r in rows})
    random.seed(args.seed)
    random.shuffle(groups)
    n_test = max(1, int(len(groups) * args.test)) if len(groups) > 2 else 0
    n_val = max(1, int(len(groups) * args.val)) if len(groups) > 2 else 1
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    train_g = set(groups[n_test + n_val:])

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    stats = Counter()
    split_counts = Counter()
    kept_rows = 0
    for r in rows:
        g = group(r)
        split = "test" if g in test_g else ("val" if g in val_g else "train")
        img_src = proc_root.parent / "raw" / r["image"]
        if not img_src.exists():
            continue
        name = f"{r['session']}_{r['sample_id']}"
        shutil.copy(img_src, out / "images" / split / f"{name}.png")
        lines = []
        for e in r["els"]:
            x, y, w, h = e["bbox"]
            # YOLO xywh normalized
            cx, cy = (x + w / 2) / r["vw"], (y + h / 2) / r["vh"]
            nw, nh = w / r["vw"], h / r["vh"]
            if nw <= 0 or nh <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
                continue
            lines.append(f"{cid[e['role']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[e["role"]] += 1
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
        split_counts[split] += 1
        kept_rows += 1

    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for c, i in cid.items()},
    }
    (out / "data.yaml").write_text(yaml.dump(data_yaml, allow_unicode=True, sort_keys=False))

    meta = {
        "n_images": kept_rows,
        "splits": dict(split_counts),
        "groups_total": len(groups),
        "groups_train": sorted(train_g),
        "groups_val": sorted(val_g),
        "groups_test": sorted(test_g),
        "instances_per_class": dict(stats.most_common()),
    }
    (out / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"images: {kept_rows} | splits: {dict(split_counts)}")
    print("instances/class (top15):")
    for c, n in stats.most_common(15):
        print(f"  {n:6d}  {c}")
    print(f"zero-shot classes:", [c for c in classes if stats[c] == 0] or "none")
    print("→", out / "data.yaml")


if __name__ == "__main__":
    main()
