"""Render homepage hero images: model detections on curated screenshots.

Bigger labels + conf filter + zoom crop, tuned for README display size
(demo-gallery overlays are too dense/small to read at 900px width).

Usage: .venv/bin/python scripts/release/render_hero.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageColor

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#008080", "#ffbeff", "#9a6324", "#ffd8b1"]

FRAMES = {
    "hero_dark_home": "data/raw/s18/0014/shot.png",
    "hero_light_home": "data/raw/s19/0001/shot.png",
    "hero_dialog": "data/raw/s15/0057/shot.png",
    "hero_loggedout": "data/yolo/images/train/loggedout_s02_0011.png",
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in ("/System/Library/Fonts/Helvetica.ttc",
              "/System/Library/Fonts/SFNSMono.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def detect(path: str, conf: float = 0.5):
    from runtime.api import ScreenParser
    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"),
                      names, conf=conf, check_purity=False)
    return sp(path)


def draw(im: Image.Image, dets: list[dict], font_px: int, box_w: int,
         region: tuple[int, int, int, int] | None = None) -> Image.Image:
    """Draw boxes+labels; optionally only inside region (crop coords)."""
    x0, y0, x1, y1 = region or (0, 0, im.width, im.height)
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    d = ImageDraw.Draw(im)
    font = load_font(font_px)
    roles = sorted({e["role"] for e in dets})
    cmap = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(roles)}
    # draw thin boxes first, labels last (readable), small boxes on top
    for e in sorted(dets, key=lambda e: (e["bbox"][2]-e["bbox"][0])*(e["bbox"][3]-e["bbox"][1]), reverse=True):
        bx1, by1, bx2, by2 = e["bbox"]
        c = cmap[e["role"]]
        rgb = ImageColor.getrgb(c)
        d.rectangle([bx1, by1, bx2, by2], outline=c, width=box_w)
    for e in sorted(dets, key=lambda e: (e["bbox"][2]-e["bbox"][0])*(e["bbox"][3]-e["bbox"][1]), reverse=True):
        bx1, by1, bx2, by2 = e["bbox"]
        label = f"{e['role']} {e['confidence']:.2f}"
        tb = od.textbbox((bx1, by1 - font_px - 4), label, font=font)
        od.rectangle([tb[0]-4, tb[1]-2, tb[2]+4, tb[3]+2], fill=rgb + (230,))
        od.text((bx1, by1 - font_px - 4), label, fill=(0, 0, 0, 255), font=font)
    im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
    return im.crop((x0, y0, x1, y1))


def main() -> None:
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    for name, path in FRAMES.items():
        dets = detect(path)
        im = Image.open(ROOT / path).convert("RGB")
        im = draw(im, dets, font_px=22, box_w=3)
        im = im.resize((1400, int(im.height * 1400 / im.width)), Image.LANCZOS)
        f = out / f"{name}.jpg"
        im.save(f, quality=88)
        print(f"{f.name}: {len(dets)} dets {im.size}")

    # zoom: the post whose action row is most confident (sells per-post semantics)
    dets = detect(FRAMES["hero_dark_home"])

    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / u if u > 0 else 0

    dedup = []
    for e in sorted(dets, key=lambda e: -e["confidence"]):
        if not any(e["role"] == g["role"] and _iou(e["bbox"], g["bbox"]) > 0.55 for g in dedup):
            dedup.append(e)
    dets = dedup
    # zoom only: keep one box per role (a quoted post repeats the action row and
    # the duplicate low-conf labels read as noise in a showcase crop)
    seen_roles: dict[str, dict] = {}
    for e in dets:
        if e["role"] not in seen_roles or e["confidence"] > seen_roles[e["role"]]["confidence"]:
            seen_roles[e["role"]] = e
    dets_zoom = list(seen_roles.values())

    acts = [e for e in dets if e["role"] in ("like_button", "reply_button", "repost_button", "bookmark_button", "share_button")]
    lk = max(acts, key=lambda e: e["confidence"])
    conts = [e for e in dets if e["role"] == "post_container"
             and e["bbox"][0] < lk["bbox"][0] < e["bbox"][2]
             and e["bbox"][1] < lk["bbox"][1] < e["bbox"][3]]
    c = conts[0]["bbox"] if conts else lk["bbox"]
    # crop from the username down through the FULL action row; a post_container
    # box often ends above the action row, so anchor the bottom on the row itself
    row = [e for e in acts if abs(e["bbox"][1] - lk["bbox"][1]) < 30]
    y1 = max(e["bbox"][3] for e in row) + 16 if row else lk["bbox"][3] + 16
    y0 = max(0, min(lk["bbox"][1] - 320, (c[1] - 16) if conts else lk["bbox"][1] - 320))
    x0, x1 = max(0, min(e["bbox"][0] for e in row) - 140), min(1920, max(e["bbox"][2] for e in row) + 140)
    im = Image.open(ROOT / FRAMES["hero_dark_home"]).convert("RGB")
    crop = draw(im, dets_zoom, font_px=26, box_w=4, region=(int(x0), int(y0), int(x1), int(y1)))
    crop = crop.resize((int(crop.width*1.5), int(crop.height*1.5)), Image.LANCZOS)
    crop.save(out / "hero_zoom.jpg", quality=90)
    print(f"hero_zoom.jpg {crop.size}")


if __name__ == "__main__":
    main()
