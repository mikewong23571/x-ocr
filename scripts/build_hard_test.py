"""Build data/hard_test/ from processed manifests (task_spec §十一).

Hard criteria (score-based selection, deterministic):
  - menu/dialog overlay states (state_* routes)        +3
  - dense feeds (element count in top quartile)        +2
  - small viewport (1280x720)                          +1
  - threads with media (video/post_media present)      +1
  - notifications / search (rare states)                +1
Copies sample dirs + writes hard_test/manifest.jsonl. Never removes from train.

Usage: .venv/bin/python scripts/build_hard_test.py [--top 60]
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def score(row: dict) -> int:
    s = 0
    route = row.get("route", "")
    if route.startswith("state_"):
        s += 3
    if any(e["role"] in ("menu_item", "dialog") for e in row["els"]):
        s += 2
    if row.get("viewport") == "1280x720":
        s += 1
    roles = {e["role"] for e in row["els"]}
    if "video" in roles or "post_media" in roles:
        s += 1
    if route in ("notifications", "status") or route.startswith("state_repost"):
        s += 1
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed")
    ap.add_argument("--out", default="data/hard_test")
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()

    rows = []
    for mf in sorted((ROOT / args.inp).glob("*/manifest.jsonl")):
        rows += [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no manifests")
    counts = sorted(len(r["els"]) for r in rows)
    q75 = counts[int(len(counts) * 0.75)]

    for r in rows:
        s = score(r) + (2 if len(r["els"]) >= q75 else 0)
        r["_score"] = s
    rows.sort(key=lambda r: (-r["_score"], r["session"], r["sample_id"]))

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kept = []
    for r in rows[: args.top]:
        src_dir = ROOT / "data/raw" / r["session"] / r["sample_id"]
        if not src_dir.exists():
            continue
        dst = out / f"{r['session']}_{r['sample_id']}"
        shutil.copytree(src_dir, dst)
        kept.append({"sample": f"{r['session']}_{r['sample_id']}", "score": r["_score"],
                     "route": r["route"], "n_els": len(r["els"]), "els": r["els"]})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n")
    print(f"hard_test: {len(kept)} samples → {out} (q75_els={q75})")


if __name__ == "__main__":
    main()
