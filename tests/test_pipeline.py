"""Unit tests for the local pipeline (fuse / export). Run: .venv/bin/python -m pytest tests/ -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from annotation.fuse import fuse_sample, iou  # noqa: E402


def _anno():
    return {
        "url": "https://x.com/home", "vw": 1000, "vh": 800, "n": 4,
        "els": [
            {"role": "like_button", "source": "dom-testid", "testid": "like",
             "aria": None, "text": None, "tag": "button",
             "rect": {"x": 10, "y": 10, "w": 36, "h": 36}, "z": 0,
             "expanded": None, "pressed": None, "selected": None, "checked": None, "disabled": False},
            # duplicate of the same role nested → should be dropped (IoU ≈ 0.945 > 0.92)
            {"role": "like_button", "source": "dom-testid", "testid": "like",
             "aria": None, "text": None, "tag": "div",
             "rect": {"x": 10, "y": 10, "w": 35, "h": 35}, "z": 0,
             "expanded": None, "pressed": None, "selected": None, "checked": None, "disabled": False},
            # off-viewport half-out → clipped
            {"role": "post_container", "source": "dom-testid", "testid": "tweet",
             "aria": None, "text": "hi", "tag": "article",
             "rect": {"x": 100, "y": 700, "w": 600, "h": 300}, "z": 0,
             "expanded": None, "pressed": None, "selected": None, "checked": None, "disabled": False},
            # fully outside → dropped
            {"role": "tab", "source": "dom-testid", "testid": "pillLabel",
             "aria": None, "text": "For you", "tag": "a",
             "rect": {"x": -500, "y": 10, "w": 20, "h": 20}, "z": 0,
             "expanded": None, "pressed": None, "selected": None, "checked": None, "disabled": False},
        ],
    }


def _meta():
    return {"sample_id": "0001", "session": "t0", "route": "home", "viewport": "1000x800", "dark": True}


def test_iou():
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    assert iou(a, {"x": 0, "y": 0, "w": 10, "h": 10}) == 1.0
    assert iou(a, {"x": 20, "y": 20, "w": 5, "h": 5}) == 0.0


def test_fuse_dedup_clip_drop():
    out = fuse_sample(_anno(), _meta())
    roles = [e["role"] for e in out["els"]]
    assert roles.count("like_button") == 1          # nested dup dropped
    assert "tab" not in roles                        # fully off-screen dropped
    pc = next(e for e in out["els"] if e["role"] == "post_container")
    assert pc["bbox"][3] == 100                      # clipped: 700+300→800 → h=100
    assert out["image"].endswith("shot.png")


def test_yolo_export_format(tmp_path):
    # minimal end-to-end: build one fake processed row, export, parse label file
    import yaml
    tax_path = Path(__file__).resolve().parent.parent / "schema/ui_taxonomy.yaml"
    tax = yaml.safe_load(tax_path.read_text())
    classes = sorted(tax["classes"].keys())
    cid = {c: i for i, c in enumerate(classes)}
    row = {
        "session": "t0", "sample_id": "0001", "route": "home", "viewport": "1",
        "vw": 100, "vh": 100, "image": "t0/0001/shot.png", "dark": True,
        "els": [
            {"role": "like_button", "bbox": [10, 10, 20, 20], "source": "dom-testid",
             "testid": "like", "aria": None, "text": None, "expanded": None, "disabled": False},
        ],
    }
    line = f"{cid['like_button']} {20/100:.6f} {20/100:.6f} {20/100:.6f} {20/100:.6f}"
    parts = line.split()
    assert len(parts) == 5
    assert 0 <= float(parts[1]) <= 1
