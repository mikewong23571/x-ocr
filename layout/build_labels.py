"""Synthesize layout-region labels from existing DOM annotations (no recapture).

Reads fused manifests (data/processed/<session>/manifest.jsonl, the same sample
set as the atomic detector's dataset) and derives 11-class layout boxes per
frame per schema/layout_taxonomy.yaml. Output: data/layout_labels/<session>/<sample>.json

Usage: .venv/bin/python layout/build_labels.py [--in data/processed] [--out data/layout_labels]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ACTION_ROLES = ("reply_button", "repost_button", "like_button", "bookmark_button", "share_button")
RAIL_ROLES = ("trend_item", "user_cell", "news_card", "follow_button")


def _u(box, x, y, w, h):  # union accumulator: box=(x1,y1,x2,y2)
    return (min(box[0], x), min(box[1], y), max(box[2], x + w), max(box[3], y + h))


def _ok(b, vw, vh):
    return b[2] - b[0] >= 8 and b[3] - b[1] >= 8 and b[0] < vw and b[1] < vh


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _contains(outer, inner, pad=0):
    cx, cy = _center(inner)
    return (outer[0] - pad <= cx <= outer[2] + pad
            and outer[1] - pad <= cy <= outer[3] + pad)


def synth(row: dict) -> list[dict]:
    els = [{"role": e["role"], "bbox": (e["bbox"][0], e["bbox"][1],
                                        e["bbox"][0] + e["bbox"][2],
                                        e["bbox"][1] + e["bbox"][3])}
           for e in row["els"]]
    by = {}
    for e in els:
        by.setdefault(e["role"], []).append(e)
    vw, vh = row["vw"], row["vh"]
    out = []
    parents = []  # (child_idx, parent_class, parent_bbox) resolved at the end

    posts = [(e["bbox"]) for e in by.get("post_container", [])]
    for p in posts:
        out.append({"cls": "post", "bbox": [round(v, 1) for v in p]})

    # feed column from posts (fallback: usernames) — anchors left/right rails
    if posts:
        fx0, fx1 = min(p[0] for p in posts), max(p[2] for p in posts)
    elif by.get("username_link"):
        us = [e["bbox"] for e in by["username_link"]]
        fx0, fx1 = min(u[0] for u in us), max(u[2] for u in us)
    else:
        fx0, fx1 = vw * 0.25, vw * 0.72  # last resort: middle band

    # per-post derived regions
    for p in posts:
        members = [e for e in els if e["role"] not in ("post_container",)
                   and _contains(p, e["bbox"])]
        heads = [e for e in members if e["role"] in ("avatar", "username_link")
                 and _center(e["bbox"])[1] < p[1] + 140]
        if heads:
            b = (vw, vh, 0, 0)
            for e in heads:
                b = _u(b, *e["bbox"])
            if _ok(b, vw, vh):
                idx = len(out)
                out.append({"cls": "post_header", "bbox": [round(v, 1) for v in b]})
                parents.append((idx, "post", p))
        media = [e for e in members if e["role"] in ("post_media", "video")]
        if media:
            b = (vw, vh, 0, 0)
            for e in media:
                b = _u(b, *e["bbox"])
            if _ok(b, vw, vh):
                idx = len(out)
                out.append({"cls": "media_region", "bbox": [round(v, 1) for v in b]})
                parents.append((idx, "post", p))

    # text_region: one per post_text (standalone; parent = containing post if any)
    for e in by.get("post_text", []):
        host = next((p for p in posts if _contains(p, e["bbox"])), None)
        idx = len(out)
        out.append({"cls": "text_region", "bbox": [round(v, 1) for v in e["bbox"]]})
        if host:
            parents.append((idx, "post", host))

    # action_row: cluster same-y action buttons (inside or outside posts)
    acts = sorted((e for e in els if e["role"] in ACTION_ROLES),
                  key=lambda e: e["bbox"][1])
    row_acts, used = [], set()
    for a in acts:
        if id(a) in used:
            continue
        cy = _center(a["bbox"])[1]
        grp = [g for g in acts if id(g) not in used and abs(_center(g["bbox"])[1] - cy) < 24]
        for g in grp:
            used.add(id(g))
        if len(grp) >= 2:
            row_acts.append(grp)
    for grp in row_acts:
        b = (vw, vh, 0, 0)
        for e in grp:
            b = _u(b, *e["bbox"])
        b = (max(0, b[0] - 4), max(0, b[1] - 4), min(vw, b[2] + 4), min(vh, b[3] + 4))
        if not _ok(b, vw, vh):
            continue
        idx = len(out)
        out.append({"cls": "action_row", "bbox": [round(v, 1) for v in b]})
        host = next((p for p in posts if _contains(p, b)), None)
        if host:
            parents.append((idx, "post", host))

    # quoted_post: link_card as-is
    for e in by.get("link_card", []):
        idx = len(out)
        out.append({"cls": "quoted_post", "bbox": [round(v, 1) for v in e["bbox"]]})
        host = next((p for p in posts if _contains(p, e["bbox"])), None)
        if host:
            parents.append((idx, "post", host))

    # compose_bar: textbox ±40px band ∪ toolbar icons ∪ post_button
    tb = by.get("compose_textbox")
    if tb:
        b = (vw, vh, 0, 0)
        cy = _center(tb[0]["bbox"])[1]
        band = [e for e in els if abs(_center(e["bbox"])[1] - cy) < 90
                and e["role"] in ("compose_textbox", "post_button", "icon_button",
                                  "avatar", "overflow_button", "close_button", "tab")]
        # strip off avatars that clearly belong to the first post below the bar
        band = [e for e in band if _center(e["bbox"])[1] > cy - 80]
        for e in band:
            b = _u(b, *e["bbox"])
        if _ok(b, vw, vh):
            out.append({"cls": "compose_bar", "bbox": [round(v, 1) for v in b]})

    # left_nav / right_sidebar: column unions
    left = [e for e in els if e["role"] in ("nav_item", "compose_button", "account_button")
            and _center(e["bbox"])[0] < fx0 - 30]
    if len(left) >= 2:
        b = (vw, vh, 0, 0)
        for e in left:
            b = _u(b, *e["bbox"])
        b = (max(0, b[0] - 8), max(0, b[1] - 8), min(vw, b[2] + 8), min(vh, b[3] + 8))
        out.append({"cls": "left_nav", "bbox": [round(v, 1) for v in b]})
        left_box = out[-1]["bbox"]
    else:
        left_box = None
    right = [e for e in els if e["role"] in RAIL_ROLES and _center(e["bbox"])[0] > fx1 + 30]
    if len(right) >= 2:
        b = (vw, vh, 0, 0)
        for e in right:
            b = _u(b, *e["bbox"])
        b = (max(0, b[0] - 8), max(0, b[1] - 8), min(vw, b[2] + 8), min(vh, b[3] + 8))
        out.append({"cls": "right_sidebar", "bbox": [round(v, 1) for v in b]})
        right_box = out[-1]["bbox"]
    else:
        right_box = None

    # overlay: dialog, or menu-item cluster on state_* frames
    if by.get("dialog"):
        out.append({"cls": "overlay", "bbox": [round(v, 1) for v in by["dialog"][0]["bbox"]]})
    elif len(by.get("menu_item", [])) >= 3 and str(row.get("route", "")).startswith("state_"):
        b = (vw, vh, 0, 0)
        for e in by["menu_item"]:
            b = _u(b, *e["bbox"])
        b = (max(0, b[0] - 10), max(0, b[1] - 24), min(vw, b[2] + 10), min(vh, b[3] + 10))
        out.append({"cls": "overlay", "bbox": [round(v, 1) for v in b]})

    # card: news_card / user_cell individually
    for e in by.get("news_card", []) + by.get("user_cell", []):
        if not _ok(e["bbox"], vw, vh):
            continue
        idx = len(out)
        out.append({"cls": "card", "bbox": [round(v, 1) for v in e["bbox"]]})
        if right_box and _contains(right_box, e["bbox"]):
            out[idx]["parent_hint"] = "right_sidebar"

    # resolve recorded parent links onto boxes
    for idx, pcls, pbox in parents:
        out[idx]["parent_hint"] = pcls
    return [o for o in out if _ok(o["bbox"], vw, vh)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed")
    ap.add_argument("--out", default="data/layout_labels")
    args = ap.parse_args()
    out_root = ROOT / args.out
    stats, class_counts = [], Counter()
    for mf in sorted((ROOT / args.inp).glob("*/manifest.jsonl")):
        session = mf.parent.name
        n = 0
        for l in mf.read_text().splitlines():
            if not l.strip():
                continue
            row = json.loads(l)
            boxes = synth(row)
            if not boxes:
                continue
            d = out_root / session
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{row['sample_id']}.json").write_text(json.dumps({
                "image": row["image"], "session": session, "sample_id": row["sample_id"],
                "route": row["route"], "viewport": row["viewport"], "dark": row["dark"],
                "vw": row["vw"], "vh": row["vh"],
                "source": "synthesized-from-dom-fused-v1.0",
                "boxes": boxes,
            }, ensure_ascii=False))
            n += 1
            class_counts.update(b["cls"] for b in boxes)
        stats.append((session, n))
    total = sum(n for _, n in stats)
    print(f"{total} frames with layout labels → {out_root}")
    for c, k in class_counts.most_common():
        print(f"  {c:14s} {k}")
    print(f"  mean boxes/frame: {sum(class_counts.values())/max(total,1):.1f}")


if __name__ == "__main__":
    main()
