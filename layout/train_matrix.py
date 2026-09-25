"""VM-side layout training matrix (runs INSIDE Colab).

Builds scaling subsets from manifests, then trains every (prior × scale) arm on
the SAME frozen val/test, appending one metrics row per arm to
experiments/layout_matrix/results.jsonl.

Usage (VM): python layout/train_matrix.py --dataset data/yolo_layout \
    --arms '{"coco_nano":"yolo11n.pt"}' --scales 100,250,500,full --epochs 60
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

VM = Path("/content/x-ocr")


def build_subset(root: Path, scale: int, classes: list[str]) -> Path:
    sub = root / f"sub_{scale}"
    if sub.exists():
        return sub
    for split in ("train", "val", "test"):
        (sub / "images" / split).mkdir(parents=True)
        (sub / "labels" / split).mkdir(parents=True)
    manifest = (root / f"scale_{scale}.txt").read_text().split()
    for name in manifest:
        src = root / "images" / "train" / name
        shutil.copy(src, sub / "images" / "train" / name)
        shutil.copy(root / "labels" / "train" / (Path(name).stem + ".txt"),
                    sub / "labels" / "train" / (Path(name).stem + ".txt"))
    for split in ("val", "test"):
        for p in (root / "images" / split).glob("*"):
            shutil.copy(p, sub / "images" / split / p.name)
            lb = root / "labels" / split / (p.stem + ".txt")
            if lb.exists():
                shutil.copy(lb, sub / "labels" / split / lb.name)
    (sub / "data.yaml").write_text(
        f"path: {sub}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(classes)}\nnames: {classes}\n")
    # train_yolo.py asserts an export_meta.json beside data.yaml (anti-stale guard)
    counts = json.loads((root / "layout_meta.json").read_text())["n_images"]
    n_tr = len(list((sub / "images" / "train").glob("*")))
    (sub / "export_meta.json").write_text(json.dumps(
        {"n_images": {"train": n_tr, "val": counts["val"], "test": counts["test"]},
         "splits": {"train": n_tr, "val": counts["val"], "test": counts["test"]}}))
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/yolo_layout")
    ap.add_argument("--arms", required=True,
                    help='json dict arm_name -> weights path/url')
    ap.add_argument("--scales", default="100,250,500,full")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    root = VM / args.dataset
    meta = json.loads((root / "layout_meta.json").read_text())
    classes = meta["classes"]
    arms = json.loads(args.arms)
    scales = [s if s == "full" else int(s) for s in args.scales.split(",")]

    outdir = VM / "experiments" / "layout_matrix"
    outdir.mkdir(parents=True, exist_ok=True)
    results_f = outdir / "results.jsonl"

    for arm, weights in arms.items():
        for scale in scales:
            exp = f"layout_{arm}_{scale}"
            rows_done = [json.loads(l) for l in results_f.read_text().splitlines() if l.strip()] if results_f.exists() else []
            done = any(r.get("exp") == exp and r.get("ok") for r in rows_done)
            if done:
                print(f"skip {exp} (done)")
                continue
            if scale == "full":
                data_yaml = root / "data.yaml"
                (root / "data.yaml").write_text(
                    f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"nc: {len(classes)}\nnames: {classes}\n")
                (root / "export_meta.json").write_text(json.dumps(
                    {"n_images": meta["n_images"],
                     "splits": meta["n_images"]}))
            else:
                data_yaml = build_subset(root, scale, classes) / "data.yaml"
            print(f"=== TRAIN {exp} data={data_yaml} weights={weights} ===", flush=True)
            r = subprocess.run(
                [sys.executable, str(VM / "training" / "train_yolo.py"),
                 "--data", str(data_yaml), "--model", weights,
                 "--imgsz", str(args.imgsz), "--epochs", str(args.epochs),
                 "--exp", exp, "--out", str(outdir)],
                capture_output=True, text=True)
            log = outdir / f"{exp}.log"
            log.write_text(r.stdout[-20000:] + "\n=== STDERR ===\n" + r.stderr[-20000:])
            row = {"exp": exp, "arm": arm, "scale": scale, "weights": weights,
                   "epochs": args.epochs, "ok": r.returncode == 0}
            mfile = outdir / exp / "metrics.json"
            if mfile.exists():
                m = json.loads(mfile.read_text())
                row.update({k: m.get(k) for k in
                            ("mAP50", "mAP50-95", "precision", "recall", "params",
                             "per_class_ap50")})
                row["dataset"] = m.get("dataset")
            else:
                row["error"] = r.stderr[-2000:]
            with open(results_f, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"=== DONE {exp} mAP50={row.get('mAP50')}", flush=True)

    print("MATRIX COMPLETE →", results_f)


if __name__ == "__main__":
    main()
