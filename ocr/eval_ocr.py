"""E3 — Regional OCR benchmark: engines × modes on the fixed DOM-truth set.

Engines:
  v4  = rapidocr_onnxruntime (PP-OCRv4 mobile det+rec) — current runtime baseline
  v6  = rapidocr v3.9.2 PP-OCRv6 rec_small (det_small + rec_small)
Modes:
  detrec_region : det+rec on the element-region crop (layout-guided region OCR)
  rec_line      : rec-only on each GT text-line crop (bbox given, no det)
  detrec_full   : det+rec on the WHOLE screenshot, matched back to GT lines
                  (the forbidden-by-default full-screen baseline, for quantifying)

Metrics per line: CER (miss=1.0), exact-match; latency per region / per line;
breakdown by role/theme/font-size bucket.

Usage: .venv/bin/python ocr/eval_ocr.py [--engines v4,v6] [--out experiments/ocr]
"""
import argparse
import io
import json
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BM = ROOT / "data" / "ocr_benchmark"


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0
    p, r = list(pred), list(ref)
    prev = list(range(len(r) + 1))
    for i, pc in enumerate(p):
        cur = [i + 1]
        for j, rc in enumerate(r):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (pc != rc)))
        prev = cur
    return min(prev[-1] / len(r), 1.0)


class V4:  # rapidocr_onnxruntime — current runtime baseline
    name = "v4"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self.full = RapidOCR()
        self.rec = self.full

    def detrec(self, img) -> tuple[list, float]:
        t = time.time()
        r, _ = self.full(img)
        out = []
        for item in (r or []):
            box, txt = item[0], item[1]
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append(([min(xs), min(ys), max(xs), max(ys)], txt))
        return out, time.time() - t

    def rec_line(self, img) -> tuple[str, float]:
        t = time.time()
        r, _ = self.full(img, use_det=False, use_cls=False, use_rec=True)
        return ((r or [["", 0]])[0][0], time.time() - t)


class V6:  # rapidocr v3.9.2, PP-OCRv6 small
    name = "v6small"

    def __init__(self):
        from rapidocr import RapidOCR
        from rapidocr.utils.parse_parameters import OCRVersion, ModelType
        common = {"Rec.ocr_version": OCRVersion("PP-OCRv6"), "Rec.model_type": ModelType("small")}
        self.full = RapidOCR(params=common)
        self.rec = RapidOCR(params={**common, "Global.use_det": False, "Global.use_cls": False})

    def detrec(self, img) -> tuple[list, float]:
        t = time.time()
        r = self.full(img)
        out = []
        if r.boxes is not None:
            for box, txt, sc in zip(r.boxes, r.txts, r.scores):
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                out.append(([min(xs), min(ys), max(xs), max(ys)], txt))
        return out, time.time() - t

    def rec_line(self, img) -> tuple[str, float]:
        t = time.time()
        r = self.rec(img)
        return ((r.txts[0] if r.txts is not None and len(r.txts) else ""), time.time() - t)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def match_lines(preds, gts, thr=0.35):
    """greedy IoU match → [(pred_txt, ref_txt)] + n_miss"""
    pairs, used = [], set()
    for i in range(len(preds)):
        best, bv = None, thr
        for j, gt in enumerate(gts):
            if j in used:
                continue
            v = iou(preds[i][0], gt["rect"])
            if v > bv:
                best, bv = j, v
        if best is not None:
            used.add(best)
            pairs.append((preds[i][1], gts[best]["text"]))
    return pairs, len(gts) - len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="v4,v6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="experiments/ocr")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (BM / "benchmark.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    # hygiene: frames captured before the stability guard can carry ~100px DOM/pixel
    # misalignment — their crops read neighbouring text. Drop whole frames whose
    # quick rec pass median CER > 0.5 (counted, reported, never silent).
    import statistics
    probe = V6()
    bad_shots = set()
    by_shot = defaultdict(list)
    for r_ in rows:
        by_shot[r_["shot"]].append(r_)
    for sh, rs in by_shot.items():
        cers = []
        shot_img = Image.open(ROOT / "data/raw" / sh)
        for r_ in rs[:6]:
            for l_ in r_["lines"]:
                x1, y1, x2, y2 = l_["rect"]
                buf = io.BytesIO()
                shot_img.crop((max(0, x1 - 4), max(0, y1 - 3), x2 + 4, y2 + 3)).save(buf, format="PNG")
                txt, _ = probe.rec_line(buf.getvalue())
                cers.append(cer(txt.strip(), l_["text"]))
        if cers and statistics.median(cers) > 0.5:
            bad_shots.add(sh)
    n_before = len(rows)
    rows = [r_ for r_ in rows if r_["shot"] not in bad_shots]
    print(f"hygiene: dropped {n_before - len(rows)} regions on {len(bad_shots)} misaligned frames")
    engines = []
    for name in args.engines.split(","):
        engines.append(V4() if name == "v4" else V6())

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    shot_cache = {}

    for eng in engines:
        stats = {"detrec_region": [], "rec_line": [], "detrec_full": []}
        det = {"lat_region": 0.0, "n_region": 0, "lat_line": 0.0, "n_line": 0}
        per = defaultdict(lambda: {"cer": [], "em": [], "miss": 0})
        fs_buckets = defaultdict(lambda: {"cer": [], "em": [], "miss": 0})

        for rec in rows:
            img_path = BM / rec["crop"]
            region = rec["region"]
            crop = Image.open(img_path)
            ox, oy = max(0, region[0] - 6), max(0, region[1] - 6)

            # det+rec on region crop
            preds, lat = eng.detrec(str(img_path))
            det["lat_region"] += lat
            det["n_region"] += 1
            gts = [{"text": l["text"],
                    "rect": [l["rect"][0] - ox, l["rect"][1] - oy,
                             l["rect"][2] - ox, l["rect"][3] - oy]}
                   for l in rec["lines"]]
            pairs, miss = match_lines([(b, t) for b, t in preds], gts)
            for pred, ref in pairs:
                c = cer(pred.strip(), ref)
                stats["detrec_region"].append(c)
                per[(rec["role"], "detrec")]["cer"].append(c)
                per[(rec["role"], "detrec")]["em"].append(float(pred.strip() == ref))
            per[(rec["role"], "detrec")]["miss"] += miss
            for _ in range(miss):
                stats["detrec_region"].append(1.0)

            # rec-only per GT line crop
            shot = shot_cache.get(rec["shot"])
            if shot is None:
                shot = Image.open(ROOT / "data/raw" / rec["shot"])
                shot_cache[rec["shot"]] = shot
            for line in rec["lines"]:
                x1, y1, x2, y2 = line["rect"]
                lc = shot.crop((max(0, x1 - 4), max(0, y1 - 3), x2 + 4, y2 + 3))
                buf = io.BytesIO()
                lc.save(buf, format="PNG")
                txt, lat = eng.rec_line(buf.getvalue())
                det["lat_line"] += lat
                det["n_line"] += 1
                c = cer(txt.strip(), line["text"])
                stats["rec_line"].append(c)
                key = (rec["role"], rec["theme"])
                per[(key, "rec")]["cer"].append(c)
                per[(key, "rec")]["em"].append(float(txt.strip() == line["text"]))
                b = "small" if line["fs"] < 14 else ("mid" if line["fs"] < 18 else "large")
                fs_buckets[b]["cer"].append(c)
                fs_buckets[b]["em"].append(float(txt.strip() == line["text"]))

        # full-screen pass (unique shots)
        full_lat, full_pairs, full_miss = 0.0, [], 0
        shots = sorted({r["shot"] for r in rows})
        gt_by_shot = defaultdict(list)
        meta_by_shot = {}
        for r in rows:
            meta_by_shot[r["shot"]] = r
            for l in r["lines"]:
                gt_by_shot[r["shot"]].append(l)
        for sh in shots:
            big = Image.open(ROOT / "data/raw" / sh)
            preds, lat = eng.detrec(str(ROOT / "data/raw" / sh))
            full_lat += lat
            pairs, miss = match_lines(preds, gt_by_shot[sh])
            full_pairs += pairs
            full_miss += miss
        n_full_gt = sum(len(v) for v in gt_by_shot.values())
        full_cers = [cer(p.strip(), r) for p, r in full_pairs] + [1.0] * full_miss

        def m(v):
            return sum(v) / len(v) if v else None

        row = {
            "engine": eng.name,
            "n_regions": len(rows),
            "n_lines": sum(len(r["lines"]) for r in rows),
            "detrec_region": {"cer": m(stats["detrec_region"]),
                              "em": m([float(c == 0) for c in stats["detrec_region"]]) if stats["detrec_region"] else None,
                              "lat_per_region_ms": 1000 * det["lat_region"] / max(det["n_region"], 1)},
            "rec_line": {"cer": m(stats["rec_line"]),
                         "em": m([float(c == 0) for c in stats["rec_line"]]) if stats["rec_line"] else None,
                         "lat_per_line_ms": 1000 * det["lat_line"] / max(det["n_line"], 1)},
            "detrec_full": {"cer": m(full_cers), "coverage": 1 - full_miss / max(n_full_gt, 1),
                            "lat_per_frame_ms": 1000 * full_lat / max(len(shots), 1),
                            "n_gt_lines": n_full_gt},
            "theme_cer_rec": {
                th: m([c for (k, mode), v in per.items() for c in v["cer"]
                       if mode == "rec" and k[1] == th])
                for th in ("light", "dark")},
            "fs_bucket_cer_rec": {b: m(v["cer"]) for b, v in fs_buckets.items()},
            "role_cer_rec": {f"{k[0]}|{k[1]}": m(v["cer"])
                             for k, v in per.items() if k[1] == "rec"},
        }
        results.append(row)
        print(json.dumps({k: row[k] for k in ("engine", "detrec_region", "rec_line", "detrec_full")},
                         ensure_ascii=False, indent=1))

    (out_dir / "ocr_benchmark_results.json").write_text(
        json.dumps({"benchmark": f"{len(rows)} regions", "results": results},
                   ensure_ascii=False, indent=1))
    print("→", out_dir / "ocr_benchmark_results.json")


if __name__ == "__main__":
    main()
