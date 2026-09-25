"""E3-supplement — OCR robustness across SCALE (viewport zoom / DPR changes).

Takes GT text-line crops, resamples them by browser-like factors
(0.5x / 0.75x / 1.0x / 1.25x, bilinear), and measures rec-only CER for both
engines per scale bucket. Simulates the same text seen at different window
sizes / device pixel ratios without recapture.

Usage: .venv/bin/python ocr/eval_scale.py [--lines 400] [--out experiments/ocr]
"""
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ocr.eval_ocr import V4, V6, cer  # noqa: E402

SCALES = [0.5, 0.75, 1.0, 1.25]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=400)
    ap.add_argument("--out", default="experiments/ocr")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / "data/ocr_benchmark/benchmark.jsonl").read_text().splitlines() if l.strip()]
    lines = []
    for rec in rows:
        shot = Image.open(ROOT / "data/raw" / rec["shot"])
        for l in rec["lines"]:
            if l["text"].strip() and len(l["text"].strip()) >= 4:
                lines.append((shot, l["rect"], l["text"]))
    lines = lines[: args.lines]
    print(f"{len(lines)} GT lines × {len(SCALES)} scales × 2 engines")

    out = {}
    for eng in (V4(), V6()):
        per = defaultdict(list)
        for shot, (x1, y1, x2, y2), ref in lines:
            crop = shot.crop((max(0, x1 - 4), max(0, y1 - 3), x2 + 4, y2 + 3))
            for s in SCALES:
                im2 = crop.resize((max(4, int(crop.width * s)), max(4, int(crop.height * s))),
                                  Image.BILINEAR)
                buf = io.BytesIO()
                im2.save(buf, format="PNG")
                txt, _ = eng.rec_line(buf.getvalue())
                per[s].append(cer(txt.strip(), ref))
        out[eng.name] = {str(s): round(sum(v) / len(v), 4) for s, v in per.items()}
        print(eng.name, out[eng.name])

    dst = ROOT / args.out / "scale_robustness.json"
    dst.write_text(json.dumps({"n_lines": len(lines), "scales": SCALES,
                               "cer_rec_line": out}, indent=1))
    print("→", dst)


if __name__ == "__main__":
    main()
