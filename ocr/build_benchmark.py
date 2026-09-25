"""Build the fixed Regional-OCR benchmark from existing DOM text_spans.

Region granularity = DOM element (post_text / username_link / menu_item / tab /
news_card / trend_item / user_cell / search_input / notification_item); line
granularity = text_span (Range.getClientRects first-rect, near-exact GT).

Output: data/ocr_benchmark/{benchmark.jsonl, crops/<id>.png}

Usage: .venv/bin/python ocr/build_benchmark.py [--max-per 60] [--target 420]
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

TEXT_ROLES = ("post_text", "username_link", "menu_item", "tab", "news_card",
              "trend_item", "user_cell", "search_input", "notification_item",
              "link_card", "compose_textbox")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/ocr_benchmark")
    ap.add_argument("--max-per", type=int, default=60,
                    help="cap per (role, theme) stratum")
    ap.add_argument("--target", type=int, default=420)
    ap.add_argument("--min-fs", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = ROOT / args.out
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)

    strat: dict[tuple, list[dict]] = {}
    n_sess = 0
    for sd in sorted((ROOT / args.raw).glob("*/")):
        annos = list(sd.glob("*/anno.json"))
        if not annos:
            continue
        a0 = json.loads(annos[0].read_text())
        if "text_spans" not in a0:
            continue
        n_sess += 1
        # theme via body bg captured at fuse time; use stored theme_bg heuristics
        bg = (a0.get("theme_bg") or "").lower()
        light = "255, 255, 255" in bg or "255,255,255" in bg
        for an in annos:
            try:
                a = json.loads(an.read_text())
            except Exception:
                continue
            if not a.get("text_spans"):
                continue
            spans = a["text_spans"]
            els = [e for e in a["els"] if e["role"] in TEXT_ROLES]
            vw, vh = a["vw"], a["vh"]
            for e in els:
                r = e["rect"]
                lines = []
                for s in spans:
                    sr = s["rect"]
                    cx, cy = sr["x"] + sr["w"] / 2, sr["y"] + sr["h"] / 2
                    if r["x"] - 2 <= cx <= r["x"] + r["w"] + 2 and r["y"] - 2 <= cy <= r["y"] + r["h"] + 2:
                        if s.get("fs", 15) >= args.min_fs and s["text"].strip():
                            lines.append(s)
                if not lines:
                    continue
                strat.setdefault((e["role"], "light" if light else "dark"), []).append({
                    "session": sd.name, "anno": an.parent.name,
                    "shot": str((an.parent / "shot.png").relative_to(ROOT / args.raw)),
                    "role": e["role"], "theme": "light" if light else "dark",
                    "region": [r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]],
                    "vw": vw, "vh": vh,
                    "lines": [{"text": s["text"],
                               "rect": [s["rect"]["x"], s["rect"]["y"],
                                        s["rect"]["x"] + s["rect"]["w"],
                                        s["rect"]["y"] + s["rect"]["h"]],
                               "fs": s.get("fs", 15)} for s in lines],
                })
            # fallback: unassigned spans clustered into text_block regions (covers
            # sparse-DOM frames — dark/login-wall pages whose text has no testid el)
            used = set()
            for e in [x for x in a["els"] if x["role"] in TEXT_ROLES]:
                r = e["rect"]
                for k, s in enumerate(spans):
                    sr = s["rect"]
                    cx, cy = sr["x"] + sr["w"] / 2, sr["y"] + sr["h"] / 2
                    if r["x"] - 2 <= cx <= r["x"] + r["w"] + 2 and r["y"] - 2 <= cy <= r["y"] + r["h"] + 2:
                        used.add(k)
            unass = [s for k, s in enumerate(spans)
                     if k not in used and s["text"].strip() and s.get("fs", 15) >= args.min_fs]
            unass.sort(key=lambda s: s["rect"]["y"])
            blocks, cur = [], []
            for s in unass:
                if cur and (s["rect"]["y"] - cur[-1]["rect"]["y"] > 30 or
                            abs(s["rect"]["x"] - cur[-1]["rect"]["x"]) > 260):
                    blocks.append(cur)
                    cur = []
                cur.append(s)
            if cur:
                blocks.append(cur)
            for blk in blocks:
                if len(blk) < 1:
                    continue
                x1 = min(s["rect"]["x"] for s in blk)
                y1 = min(s["rect"]["y"] for s in blk)
                x2 = max(s["rect"]["x"] + s["rect"]["w"] for s in blk)
                y2 = max(s["rect"]["y"] + s["rect"]["h"] for s in blk)
                strat.setdefault(("text_block", "light" if light else "dark"), []).append({
                    "session": sd.name, "anno": an.parent.name,
                    "shot": str((an.parent / "shot.png").relative_to(ROOT / args.raw)),
                    "role": "text_block", "theme": "light" if light else "dark",
                    "region": [x1, y1, x2, y2], "vw": vw, "vh": vh,
                    "lines": [{"text": s["text"],
                               "rect": [s["rect"]["x"], s["rect"]["y"],
                                        s["rect"]["x"] + s["rect"]["w"],
                                        s["rect"]["y"] + s["rect"]["h"]],
                               "fs": s.get("fs", 15)} for s in blk],
                })

    random.seed(args.seed)
    picked = []
    for key, rows in sorted(strat.items()):
        random.shuffle(rows)
        picked += rows[: args.max_per]
    random.shuffle(picked)
    picked = picked[: args.target]

    meta_rows = []
    for i, rec in enumerate(picked):
        rid = hashlib.sha1(json.dumps([rec["session"], rec["anno"], rec["region"]],
                                      ensure_ascii=False).encode()).hexdigest()[:10]
        img = Image.open((ROOT / args.raw) / rec["shot"])
        x1, y1, x2, y2 = rec["region"]
        pad = 6
        crop = img.crop((max(0, int(x1 - pad)), max(0, int(y1 - pad)),
                         min(img.width, int(x2 + pad)), min(img.height, int(y2 + pad))))
        crop.save(crops / f"{rid}.png")
        meta_rows.append({
            "id": rid, "role": rec["role"], "theme": rec["theme"],
            "session": rec["session"], "sample": rec["anno"], "shot": rec["shot"],
            "region": [round(v, 1) for v in rec["region"]],
            "crop": f"crops/{rid}.png",
            "lines": [{"text": l["text"], "rect": [round(v, 1) for v in l["rect"]],
                       "fs": l["fs"]} for l in rec["lines"]],
            "provenance": "dom-text_spans-v1.0",
        })

    with open(out / "benchmark.jsonl", "w") as f:
        for r in meta_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    roles = Counter((r["role"], r["theme"]) for r in meta_rows)
    n_lines = sum(len(r["lines"]) for r in meta_rows)
    print(f"{n_sess} source sessions → {len(meta_rows)} regions / {n_lines} lines → {out}")
    for (role, theme), n in sorted(roles.items()):
        print(f"  {role:18s} {theme:5s} {n}")


if __name__ == "__main__":
    main()
