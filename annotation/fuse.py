"""Label fusion / cleaning: raw DOM annotations → dataset-ready annotations.

Steps per sample: clip boxes to viewport, drop degenerate boxes, same-role
IoU>=0.9 dedup, keep provenance (source), attach sample meta.
Output: data/processed/<session>/manifest.jsonl + per-sample anno files.

Usage: .venv/bin/python annotation/fuse.py --in data/raw --out data/processed
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def iou(a: dict, b: dict) -> float:
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter)


def _theme_is_dark(meta: dict) -> bool:
    """X app theme beats system preference: derive from body bg when available."""
    bg = meta.get("theme_bg") or ""
    if "rgb(" in bg:
        nums = bg.replace("rgb(", "").replace(")", "").replace(" ", "").split(",")
        try:
            r, g, b = (int(x) for x in nums[:3])
            return (r + g + b) / 3 < 128
        except ValueError:
            pass
    return bool(meta.get("dark"))


def fuse_sample(anno: dict, meta: dict) -> dict:
    vw, vh = anno["vw"], anno["vh"]
    els = []
    for e in anno["els"]:
        r = e["rect"]
        # clip to viewport
        x1, y1 = max(0.0, r["x"]), max(0.0, r["y"])
        x2, y2 = min(float(vw), r["x"] + r["w"]), min(float(vh), r["y"] + r["h"])
        w, h = x2 - x1, y2 - y1
        if w < 3 or h < 3:  # below detectable size after clip
            continue
        els.append({**e, "rect": {"x": round(x1, 1), "y": round(y1, 1), "w": round(w, 1), "h": round(h, 1)}})
    # same-role near-dup suppression (nested duplicates of same role)
    els.sort(key=lambda e: e["rect"]["w"] * e["rect"]["h"], reverse=True)
    kept = []
    for e in els:
        if any(e["role"] == k["role"] and iou(e["rect"], k["rect"]) > 0.92 for k in kept):
            continue
        kept.append(e)
    # provenance-based annotation confidence (v1 heuristic; teacher fusion will refine)
    SRC_CONF = {"dom-testid": 1.0, "dom-aria": 0.9, "heuristic": 0.7}
    return {
        "sample_id": meta["sample_id"],
        "session": meta["session"],
        "image": f"{meta['session']}/{meta['sample_id']}/shot.png",
        "vw": vw, "vh": vh,
        "route": meta["route"],
        "viewport": meta["viewport"],
        "dark": _theme_is_dark(meta),
        "els": [
            {
                "role": e["role"], "source": e.get("source"),
                "confidence": SRC_CONF.get(e.get("source"), 0.7),
                "bbox": [e["rect"]["x"], e["rect"]["y"], e["rect"]["w"], e["rect"]["h"]],
                "testid": e.get("testid"), "aria": e.get("aria"), "text": e.get("text"),
                "expanded": e.get("expanded"), "disabled": bool(e.get("disabled")),
            }
            for e in kept
        ],
    }


def phash_hamming(a: str, b: str) -> int:
    n = max(len(a), len(b))
    return sum(x != y for x, y in zip(a.ljust(n), b.ljust(n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--phash-dedup", type=int, default=8,
                    help="drop samples within this phash hamming distance of an earlier sample (same session+route)")
    args = ap.parse_args()

    raw_root = ROOT / args.inp
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    n_samples = 0
    n_els = 0
    for session_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        smanifest = out_root / session_dir.name / "manifest.jsonl"
        smanifest.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        seen_hashes: dict[str, list[str]] = {}
        n_dropped = 0
        for sample_dir in sorted(session_dir.glob("[0-9]*")):
            anno_f, meta_f = sample_dir / "anno.json", sample_dir / "meta.json"
            if not (anno_f.exists() and meta_f.exists()):
                continue
            try:
                anno = json.loads(anno_f.read_text())
                meta = json.loads(meta_f.read_text())
            except Exception:
                continue
            if not (shot := sample_dir / "shot.png").exists():
                continue
            # near-duplicate elimination (per session+route)
            key = f"{meta['session']}::{meta['route']}"
            h = meta.get("phash", "")
            if h and any(phash_hamming(h, p) <= args.phash_dedup for p in seen_hashes.get(key, [])):
                n_dropped += 1
                continue
            if h:
                seen_hashes.setdefault(key, []).append(h)
            rec = fuse_sample(anno, meta)
            rows.append(rec)
            n_samples += 1
            n_els += len(rec["els"])
        with smanifest.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{session_dir.name}: {len(rows)} samples ({n_dropped} near-dups dropped)")
    print(f"TOTAL {n_samples} samples, {n_els} elements → {out_root}")


if __name__ == "__main__":
    main()
