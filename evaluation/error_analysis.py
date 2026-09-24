"""Error analysis on the frozen test split: FP / FN / low-conf montages + confusion.

Runs on Colab (torch). Produces montages under <out>/:
  false_positives.jpg  false_negatives.jpg  low_confidence.jpg
  small_misses.jpg     confusion.csv         summary.json

Usage:
  python evaluation/error_analysis.py --data data/yolo/data.yaml \
      --weights experiments/expXXX/run/weights/best.pt --out experiments/expXXX/errors --imgsz 1280
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-per_montage", type=int, default=24)
    args = ap.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = YOLO(args.weights)
    names = m.names
    inv = {v: k for k, v in names.items()}

    data_root = Path(args.data).parent
    imgs = sorted((data_root / "images" / "test").glob("*.png"))
    results = m.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=args.conf, verbose=False)

    fp_counter, fn_counter, low_conf = Counter(), Counter(), Counter()
    small_miss = Counter()
    confusion = Counter()  # (gt, pred)
    crops = defaultdict(list)

    for img_p, r in zip(imgs, results):
        gt_map = defaultdict(list)
        lbl = img_p.parent.parent / "labels" / "test" / (img_p.stem + ".txt")
        if lbl.exists():
            from PIL import Image

            with Image.open(img_p) as im:
                iw, ih = im.size
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                gt_map[names[c]].append((cx * iw, cy * ih, w * iw, h * ih))
        dets = defaultdict(list)
        for b, cl, cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            dets[names[int(cl)]].append((tuple(b.tolist()), float(cf)))

        for role, gts in gt_map.items():
            for gx, gy, gw, gh in gts:
                matched = any(
                    abs((d[0][0] + d[0][2]) / 2 - gx) < gw and abs((d[0][1] + d[0][3]) / 2 - gy) < gh
                    for d in dets[role]
                )
                if not matched:
                    fn_counter[role] += 1
                    if min(gw, gh) < 24:
                        small_miss[role] += 1
                    if len(crops["fn"]) < args.max_per_montage:
                        crops["fn"].append((img_p, (gx - gw, gy - gh, gx + gw, gy + gh), role))
        for role, ds in dets.items():
            for box, cf in ds:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hit = any(abs(cx - g[0]) < g[2] and abs(cy - g[1]) < g[3] for g in gt_map[role])
                if not hit:
                    fp_counter[role] += 1
                    if len(crops["fp"]) < args.max_per_montage:
                        crops["fp"].append((img_p, box, role))
                elif cf < 0.5:
                    low_conf[role] += 1
                    if len(crops["low"]) < args.max_per_montage:
                        crops["low"].append((img_p, box, role))

    summary = {
        "false_positives": dict(fp_counter.most_common()),
        "false_negatives": dict(fn_counter.most_common()),
        "small_object_misses": dict(small_miss.most_common()),
        "low_confidence_hits": dict(low_conf.most_common()),
        "n_images": len(imgs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    with (out / "confusion.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["n_images", "conf", "imgsz"])
        w.writerow([len(imgs), args.conf, args.imgsz])
        for (g, p), n in confusion.most_common():
            w.writerow([g, p, n])
    print(json.dumps(summary, indent=1)[:3000])

    # montages
    from PIL import Image, ImageDraw

    for kind, items in crops.items():
        if not items:
            continue
        tiles = []
        for img_p, box, role in items[: args.max_per_montage]:
            with Image.open(img_p) as im:
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                s = 160
                crop = im.crop((int(max(0, cx - s)), int(max(0, cy - s)),
                                int(min(im.width, cx + s)), int(min(im.height, cy + s))))
            d = ImageDraw.Draw(crop)
            d.rectangle([x1 - max(0, cx - s), y1 - max(0, cy - s),
                         x2 - max(0, cx - s), y2 - max(0, cy - s)], outline="#ff2222", width=2)
            d.text((4, 4), role, fill="#ffff00")
            tiles.append(crop.resize((320, 320)))
        cols = 6
        rows = (len(tiles) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * 320, rows * 320), "#111")
        for i, t in enumerate(tiles):
            canvas.paste(t, ((i % cols) * 320, (i // cols) * 320))
        canvas.save(out / f"{kind}_montage.jpg", quality=85)
        print("montage:", out / f"{kind}_montage.jpg")


if __name__ == "__main__":
    main()
