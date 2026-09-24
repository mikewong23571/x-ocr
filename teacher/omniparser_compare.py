"""OmniParser teacher vs DOM ground-truth comparison (runs on Colab GPU).

Downloads OmniParser icon_detect_v3 (YOLOv9-E, MIT-based) from HuggingFace, runs it
on the frozen test split, and quantifies match / DOM-only / OP-only / conflict
against our fused DOM annotations. Disagreements are listed for data/disagreement/.

Usage (VM):
  python teacher/omniparser_compare.py --data-root /content/x-ocr/data/yolo \
      --anno-root /content/x-ocr/data/processed --out /content/x-ocr/experiments/teacher_cmp
"""
import argparse
import json
from pathlib import Path


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="yolo dataset dir (images/test)")
    ap.add_argument("--anno-root", required=True, help="processed dir with */manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--iou-match", type=float, default=0.4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- load DOM ground truth (interactive elements only) ---
    gt = {}
    for mf in sorted(Path(args.anno_root).glob("*/manifest.jsonl")):
        for line in mf.read_text().splitlines():
            r = json.loads(line)
            name = f"{r['session']}_{r['sample_id']}"
            gt[name] = [e for e in r["els"]]  # keep all roles; compare interactives
    print(f"DOM GT samples: {len(gt)}")

    # --- fetch + load OmniParser icon detector ---
    from ultralytics import YOLO

    wdir = Path("/content/weights")
    wdir.mkdir(parents=True, exist_ok=True)
    model_path = wdir / "icon_detect_model.pt"
    if not model_path.exists():
        from huggingface_hub import hf_hub_download

        picked = None
        # prefer icon_detect_v3 (YOLOv9, MIT-based) from PR #37; fall back to v1_5 (AGPL, teacher-eval only)
        for repo, fn, rev in [
            ("microsoft/OmniParser", "icon_detect_v3/model.pt", "refs/pr/37"),
            ("microsoft/OmniParser", "icon_detect_v1_5/model_v1_5.pt", None),
            ("microsoft/OmniParser", "icon_detect/model.safetensors", None),
        ]:
            try:
                p = hf_hub_download(repo, fn, revision=rev)
                picked = (fn, p)
                break
            except Exception as e:  # noqa: BLE001
                print(f"skip {fn}: {type(e).__name__}")
        assert picked, "no OmniParser weights downloadable"
        import shutil

        shutil.copy(picked[1], model_path)
        print("OmniParser weights:", picked[0])
    op = YOLO(str(model_path))
    print("OmniParser icon detector loaded:", type(op.model).__name__)

    # --- run on frozen test split ---
    imgs = sorted((Path(args.data_root) / "images" / "test").glob("*.png"))
    print(f"test images: {len(imgs)}")

    match = dom_only = op_only = 0
    per_sample_disagreement = []
    lat = []
    import time

    for p in imgs:
        name = p.stem
        if name not in gt:
            continue
        t0 = time.time()
        res = op.predict(str(p), conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        lat.append(time.time() - t0)
        preds = [tuple(b.tolist()) for b in res.boxes.xyxy]
        # DOM interactive GT boxes
        gts = [tuple(e["bbox"]) for e in gt[name] if e.get("source") in ("dom-testid", "dom-aria", "heuristic")
               and e["role"] != "post_text"]
        used = set()
        for g in gts:
            gx = (g[0], g[1], g[0] + g[2], g[1] + g[3])
            best, bi = 0.0, -1
            for i, pr in enumerate(preds):
                v = iou(gx, pr)
                if v > best:
                    best, bi = v, i
            if best >= args.iou_match:
                match += 1
                used.add(bi)
            else:
                dom_only += 1
        extra = [i for i in range(len(preds)) if i not in used]
        op_only += len(extra)
        if extra or dom_only:
            per_sample_disagreement.append({
                "sample": name, "n_dom_only": dom_only, "n_op_only": len(extra),
                "op_only_boxes": [tuple(round(v, 1) for v in preds[i]) for i in extra][:20],
            })

    report = {
        "n_images": len(imgs),
        "matched_elements": match,
        "dom_only_elements": dom_only,
        "op_only_elements": op_only,
        "mean_op_latency_ms": round(1000 * sum(lat) / max(1, len(lat)), 1),
        "conf": args.conf, "iou_match": args.iou_match,
        "disagreements": per_sample_disagreement[:100],
    }
    (out / "teacher_comparison.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "disagreements"}, indent=1))
    print("→", out / "teacher_comparison.json")


if __name__ == "__main__":
    main()
