"""Frozen test set evaluator: mAP + per-class recall + Key Interactive Element Recall.

Runs on Colab (torch) against the frozen test split.
Usage:
  python evaluation/eval_frozen.py --data data/yolo/data.yaml --weights experiments/expXXX/best.pt \
      --key-classes compose_button,post_button,search_input,nav_item,reply_button,repost_button,like_button,bookmark_button,share_button,menu_item,close_button,overflow_button,compose_textbox --imgsz 1280
"""
import argparse
import json
from pathlib import Path

KEY_CLASSES_DEFAULT = (
    "compose_button,post_button,search_input,nav_item,reply_button,repost_button,"
    "like_button,bookmark_button,share_button,menu_item,close_button,overflow_button,compose_textbox"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--key-classes", default=KEY_CLASSES_DEFAULT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    key = set(args.key_classes.split(","))
    m = YOLO(args.weights)
    val = m.val(data=args.data, split="test", imgsz=args.imgsz, conf=args.conf,
                device=args.device, name="eval_frozen", exist_ok=True)

    names = m.names
    per_class = {}
    # per-class recall from confusion: use box.maps (AP50-95) + p/r curves via val API
    try:
        per_class_r = val.box.r
        per_class_p = val.box.p
        for i, name in names.items():
            per_class[name] = {
                "ap5095": float(val.box.maps[i]),
                "recall": float(per_class_r[i]) if per_class_r is not None else None,
                "precision": float(per_class_p[i]) if per_class_p is not None else None,
            }
    except Exception as e:  # noqa: BLE001
        for i, name in names.items():
            per_class[name] = {"ap5095": float(val.box.maps[i])}

    # small-element recall: recompute from predictions vs labels by area on test split
    small_recall = None
    try:
        small_recall = small_element_recall(m, args, names)
    except Exception as e:  # noqa: BLE001
        print("small-element recall skipped:", e)

    key_aps = {c: per_class.get(c, {}).get("ap5095") for c in sorted(key) if c in per_class}
    key_present = [c for c in key if c in per_class]
    report = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "mAP50": float(val.box.map50),
        "mAP50-95": float(val.box.map),
        "precision": float(val.box.mp),
        "recall": float(val.box.mr),
        "key_interactive_classes": sorted(key_present),
        "key_interactive_mean_ap5095": (
            sum(key_aps[c] for c in key_present) / len(key_present) if key_present else None
        ),
        "small_element_recall": small_recall,
        "per_class": per_class,
        "n_params": sum(p.numel() for p in m.model.parameters()),
        "torch": torch.__version__,
    }
    text = json.dumps(report, indent=1)
    print(text[:4000])
    out = Path(args.out or Path(args.weights).parent / "eval_frozen.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("→", out)


def small_element_recall(m, args, names) -> dict | None:
    """Element recall for small GT boxes (<32px and <16px on shortest side) at conf threshold."""
    from ultralytics.data import build_dataloader, build_yolo_dataset
    import torch

    cfg = m.model.args if hasattr(m.model, "args") else None
    ds = build_yolo_dataset(m.model.args if cfg else {}, str(Path(args.data).parent / "images/test"),
                            args.imgsz, stride=int(m.model.stride.max()))
    dl = build_dataloader(ds, batch=8, workers=2, shuffle=False)
    inv = {v: k for k, v in names.items()}
    stats = {k: {"gt": 0, "hit": 0} for k in ("<16px", "<32px")}
    m.model.eval()
    from ultralytics.utils.ops import non_max_suppression
    import torch

    with torch.no_grad():
        for batch in dl:
            imgs = batch["img"].to(m.device)
            preds = m.model(imgs)
            if isinstance(preds, tuple):
                preds = preds[0]
            # batched NMS: rows = [batch_idx, x1, y1, x2, y2, conf, cls]
            nms = non_max_suppression(preds, conf_thres=args.conf)
            imh, imw = imgs.shape[-2:]
            for bi in range(imgs.shape[0]):
                lbl = batch["batch_idx"] == bi
                gboxes = batch["bboxes"][lbl]
                gcls = batch["cls"][lbl]
                dets = nms[bi] if nms and nms[bi] is not None else torch.zeros((0, 6))
                for j in range(gboxes.shape[0]):
                    cx, cy, w, h = gboxes[j].tolist()
                    short = min(w * imw, h * imh)
                    bucket = "<16px" if short < 16 else ("<32px" if short < 32 else None)
                    if not bucket:
                        continue
                    cls_id = int(gcls[j].item())
                    stats[bucket]["gt"] += 1
                    for d in dets:
                        x1, y1, x2, y2, _, dcls = d.tolist()
                        if int(dcls) != cls_id:
                            continue
                        if x1 <= cx * imw <= x2 and y1 <= cy * imh <= y2:
                            stats[bucket]["hit"] += 1
                            break
    out = {}
    for k, v in stats.items():
        out[k] = {"gt": v["gt"], "recall": round(v["hit"] / v["gt"], 4) if v["gt"] else None}
    return out


if __name__ == "__main__":
    main()
