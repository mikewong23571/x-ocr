"""Train YOLO11 detector on the X-OCR dataset. Designed to run on Colab GPU.

Run (Colab VM, dataset at /content/x-ocr/data/yolo):
  python training/train_yolo.py --data data/yolo/data.yaml --model yolo11n.pt --imgsz 1280 \
      --epochs 60 --batch auto --exp exp001_baseline_nano --out /content/x-ocr/experiments

Produces: best.pt, last.pt, ONNX (fp32/fp16), metrics json, per-class table.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", default="auto")
    ap.add_argument("--device", default="0")
    ap.add_argument("--exp", default="exp000")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/content/x-ocr/experiments")
    args = ap.parse_args()

    from ultralytics import YOLO

    # data.yaml portability: force `path` to this dataset's real location
    import yaml as _yaml

    dp = Path(args.data)
    _d = _yaml.safe_load(dp.read_text())
    _d["path"] = str(dp.resolve().parent)
    dp.write_text(_yaml.dump(_d, allow_unicode=True, sort_keys=False))

    ds_meta = {}
    _em = dp.parent / "export_meta.json"
    if _em.exists():
        _m = json.loads(_em.read_text())
        ds_meta = {"n_images": _m.get("n_images"), "splits": _m.get("splits")}
    assert ds_meta.get("n_images"), f"dataset meta missing beside {dp}"
    print("DATASET:", ds_meta)

    exp_dir = Path(args.out) / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ultralytics >= 8.4: batch=0 means auto, float = fraction of GPU mem
    batch = 0 if args.batch == "auto" else (float(args.batch) if "." in str(args.batch) else int(args.batch))

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        project=str(exp_dir),
        name="run",
        exist_ok=True,
        plots=True,
        verbose=True,
        cache=False,
    )

    best = exp_dir / "run" / "weights" / "best.pt"
    # validation on the frozen test split
    m = YOLO(str(best))
    val_test = m.val(data=args.data, split="test", imgsz=args.imgsz, device=args.device,
                     name="val_test", exist_ok=True)

    # per-class AP50-95 from ultralytics metrics
    per_class = (
        {name: float(val_test.box.maps[i]) for i, name in m.names.items()}
        if hasattr(val_test.box, "maps") and val_test.box.maps is not None
        else {}
    )

    metrics = {
        "exp": args.exp,
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "dataset": ds_meta,
        "mAP50": float(val_test.box.map50),
        "mAP50-95": float(val_test.box.map),
        "precision": float(val_test.box.mp),
        "recall": float(val_test.box.mr),
        "per_class_ap50": per_class,
        "params": sum(p.numel() for p in m.model.parameters()),
    }
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("METRICS:", json.dumps({k: v for k, v in metrics.items() if k != "per_class_ap50"}))

    # exports: onnx fp32 + fp16
    for fmt, half, tag in [("onnx", False, "onnx_fp32"), ("onnx", True, "onnx_fp16")]:
        try:
            p = m.export(format=fmt, half=half, imgsz=args.imgsz, dynamic=False, simplify=True)
            shutil.copy(p, exp_dir / f"best_{tag}.onnx")
        except Exception as e:  # noqa: BLE001
            print(f"export {tag} failed: {e}")
    print("ARTIFACTS:", sorted(str(p.name) for p in exp_dir.rglob("*") if p.is_file() and p.suffix in {".pt", ".onnx", ".json"}))


if __name__ == "__main__":
    sys.exit(main())
