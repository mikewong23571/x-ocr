"""Content-conditioned like targeting: detector groups posts, OCR reads text crops, match → click target.

Runtime composition (task_spec §十二/§十八): ONNX detector (bbox+role) + ONNX OCR (crop-only)
+ trivial matching policy. No DOM, no browser, no VLM. DRY-RUN only: prints the would-click
coordinate, never performs a real like.

Usage:
  .venv/bin/python scripts/like_by_content.py --img data/raw/s19/0003/shot.png --match "关键词"
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.api import parse_screen  # noqa: E402


def group_posts(els: list[dict]) -> list[dict]:
    """Associate each like_button with its post_container and post_text (geometry only)."""
    containers = [e for e in els if e["role"] == "post_container"]
    texts = [e for e in els if e["role"] == "post_text"]
    likes = [e for e in els if e["role"] == "like_button"]
    posts = []
    for c in containers:
        cx1, cy1, cx2, cy2 = c["bbox"]
        span = cy1, cy2
        # text of this post: text boxes whose center sits inside the container
        my_texts = [t for t in texts if span[0] <= (t["bbox"][1] + t["bbox"][3]) / 2 <= span[1]
                    and cx1 <= (t["bbox"][0] + t["bbox"][2]) / 2 <= cx2]
        # like of this post: the like button in the action row right below the text, inside container span
        my_likes = [l for l in likes
                    if span[0] <= (l["bbox"][1] + l["bbox"][3]) / 2 <= span[1]
                    and cx1 <= l["bbox"][0] <= cx2]
        if my_likes:
            posts.append({"container": c, "texts": my_texts, "like": my_likes[0]})
    # posts partially visible above fold still carry action rows: also catch like rows not inside any container
    claimed = {id(p["like"]) for p in posts}
    for l in likes:
        if id(l) in claimed:
            continue
        ly = (l["bbox"][1] + l["bbox"][3]) / 2
        my_texts = [t for t in texts if 0 <= ly - (t["bbox"][1] + t["bbox"][3]) / 2 <= 220]
        posts.append({"container": None, "texts": my_texts[:1], "like": l})
    return posts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--match", required=True, help="内容关键词（子串匹配）")
    args = ap.parse_args()

    from rapidocr_onnxruntime import RapidOCR
    from PIL import Image

    ocr = RapidOCR()
    im = Image.open(args.img)
    els = parse_screen(args.img)
    posts = group_posts(els)

    print(f"{len(posts)} 条帖子（含操作行），开始内容匹配: 「{args.match}」\n")
    hit = None
    for i, p in enumerate(posts):
        if not p["texts"]:
            continue
        x1, y1, x2, y2 = [int(v) for v in p["texts"][0]["bbox"]]
        x1, y1 = max(0, x1 - 4), max(0, y1 - 4)
        x2, y2 = min(im.width, x2 + 4), min(im.height, y2 + 4)
        crop = im.crop((x1, y1, x2, y2))
        res, _ = ocr(str(ROOT / "data/demo_cache/_ocr_tmp.png")) if False else (None, None)
        crop.save(ROOT / "data/demo_cache/_ocr_tmp.png")
        res, _ = ocr(str(ROOT / "data/demo_cache/_ocr_tmp.png"))
        text = " ".join(r[1] for r in res) if res else ""
        mark = ""
        if args.match.lower() in text.lower():
            mark = "  ← ★ 匹配"
            lx = p["like"]["bbox"]
            hit = {
                "click": (int((lx[0] + lx[2]) / 2), int((lx[1] + lx[3]) / 2)),
                "text": text[:60],
            }
        print(f"  post[{i}] text=「{text[:50]}」{mark}")

    if hit:
        print(f"\n目标: click({hit['click'][0]}, {hit['click'][1]})  # 该帖 like_button 中心（dry-run，不真点）")
        print(f"命中内容: 「{hit['text']}」")
    else:
        print("\n本屏无匹配帖子（Agent 可滚动后重试）")


if __name__ == "__main__":
    main()
