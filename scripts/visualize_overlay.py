"""Overlay visualization: draw fused annotations on screenshots + montage grid.

Usage:
  .venv/bin/python scripts/visualize_overlay.py --in data/processed --out data/overlays [--n 200] [--montage]
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080", "#ffffff", "#ff6b6b",
    "#51d0de", "#ff9f45", "#b3de69", "#f7b2e7", "#6a4c93", "#1d976c", "#f4d35e",
    "#00b4d8", "#ef476f",
]


def load_font(size: int = 11):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_sample(row: dict, proc_root: Path, out_dir: Path) -> Path | None:
    img_f = proc_root.parent / "raw" / row["image"]
    if not img_f.exists():
        return None
    im = Image.open(img_f).convert("RGB")
    d = ImageDraw.Draw(im)
    font = load_font()
    classes = sorted({e["role"] for e in row["els"]})
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    for e in row["els"]:
        x, y, w, h = e["bbox"]
        c = cmap[e["role"]]
        d.rectangle([x, y, x + w, y + h], outline=c, width=2)
        label = e["role"]
        tb = d.textbbox((x, y - 12), label, font=font)
        d.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1], fill=c)
        d.text((x, y - 12), label, fill="#000", font=font)
    out_f = out_dir / f"{row['session']}_{row['sample_id']}.jpg"
    im.save(out_f, quality=82)
    return out_f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed")
    ap.add_argument("--out", default="data/overlays")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--montage", action="store_true")
    args = ap.parse_args()

    proc_root = ROOT / args.inp
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for mf in sorted(proc_root.glob("*/manifest.jsonl")):
        rows += [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    random.seed(7)
    random.shuffle(rows)
    rows = rows[: args.n]
    drawn = [p for p in (draw_sample(r, proc_root, out_dir) for r in rows) if p]
    print(f"{len(drawn)} overlays → {out_dir}")

    if args.montage and drawn:
        thumbs = []
        TW = 480
        for p in drawn[:36]:
            im = Image.open(p)
            th = min(320, int(im.height * TW / im.width))
            thumbs.append(im.resize((TW, th)))
        cols = 3
        rowsn = (len(thumbs) + cols - 1) // cols
        H = max(t.height for t in thumbs)
        canvas = Image.new("RGB", (cols * TW, rowsn * H), "#222")
        for i, t in enumerate(thumbs):
            canvas.paste(t, ((i % cols) * TW, (i // cols) * H))
        mg = out_dir / "montage.jpg"
        canvas.save(mg, quality=80)
        print("montage →", mg)


if __name__ == "__main__":
    main()
