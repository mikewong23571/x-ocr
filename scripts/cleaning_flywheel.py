"""Cleaning flywheel: full-library crosscheck + disagreement triage (P0).

For every processed sample: run detector, associate with DOM truth, split disagreements:
  - model_only  → extractor-rule-gap candidates (data cleaning: fix annotation rules)
  - dom_only    → model weak spots → hard-sample list for next training round
Outputs:
  experiments/flywheel/<ts>/summary.json
  experiments/flywheel/<ts>/rule_gaps.jsonl     (model-only, aggregated patterns)
  experiments/flywheel/<ts>/hard_samples.jsonl  (dom-only rich samples for active learning)

Usage: .venv/bin/python scripts/cleaning_flywheel.py [--conf 0.30]
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.api import ScreenParser  # noqa: E402
from scripts.dom_model_crosscheck import associate, to_xyxy  # noqa: E402

import yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"), names,
                      conf=args.conf, check_purity=False)

    out = ROOT / "experiments/flywheel" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for mf in sorted((ROOT / "data/processed").glob("*/manifest.jsonl")):
        for line in mf.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    agg = Counter()
    rule_gaps = []
    hard = []
    per_role_miss = Counter()
    per_role_total = Counter()
    t0 = time.time()
    for k, r in enumerate(rows):
        img = ROOT / "data/raw" / r["image"]
        if not img.exists():
            continue
        dom = [dict(e) for e in r["els"]]
        for g in dom:
            g["_xyxy"] = [g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]]
        dets = sp(img)
        pairs, used = associate(dets, dom)
        n_gap = n_miss = 0
        for det, g, _ in pairs:
            if g is None:
                n_gap += 1
                rule_gaps.append({"sample": r["image"], "role": det["role"],
                                  "conf": det["confidence"], "bbox": det["bbox"]})
                agg[f"model_only:{det['role']}"] += 1
        for i, g in enumerate(dom):
            per_role_total[g["role"]] += 1
            if i not in used:
                n_miss += 1
                per_role_miss[g["role"]] += 1
        if n_miss:
            hard.append({"sample": r["image"], "route": r["route"], "dark": r.get("dark"),
                         "n_missed": n_miss,
                         "missed_roles": [g["role"] for i, g in enumerate(dom) if i not in used][:12]})
        agg["samples"] += 1
        agg["dom_only_total"] += n_miss
        agg["model_only_total"] += n_gap
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)

    (out / "rule_gaps.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rule_gaps) + "\n")
    (out / "hard_samples.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in hard) + "\n")
    weakest = sorted(((c, per_role_miss[c], per_role_total[c]) for c in per_role_total),
                     key=lambda x: -(x[1] / max(1, x[2])))[:10]
    summary = {
        "n_samples": agg["samples"],
        "model_only_total": agg["model_only_total"],
        "dom_only_total": agg["dom_only_total"],
        "model_only_by_role": {k.split(":", 1)[1]: v for k, v in agg.items() if k.startswith("model_only:")},
        "weakest_recall": [
            {"role": c, "missed": m, "total": t, "recall": round(1 - m / max(1, t), 3)}
            for c, m, t in weakest],
        "n_hard_samples": len(hard),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("→", out)


if __name__ == "__main__":
    main()
