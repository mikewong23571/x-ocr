"""Runtime inference API: parse_screen(image) -> structured UI elements.

Pure vision: ONNX Runtime + numpy + PIL ONLY. No DOM, no browser, no Playwright,
no OmniParser, no VLM/LLM. The import guard below enforces it at test time.

Usage:
    from runtime.api import parse_screen
    els = parse_screen("shot.png")   # path | PIL.Image | np.ndarray
    # -> [{"role": "like_button", "bbox": [x1,y1,x2,y2], "confidence": 0.98}, ...]
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

# --- runtime purity guard: these modules must NOT be importable in this process ---
FORBIDDEN_SUBSTRINGS = ("playwright", "selenium", "omniparser", "torch", "transformers")


def assert_pure_runtime() -> None:
    loaded = {m for m in sys.modules if any(f in m.lower() for f in FORBIDDEN_SUBSTRINGS)}
    if loaded:
        raise RuntimeError(f"runtime purity violated, forbidden modules loaded: {sorted(loaded)[:8]}")


class ScreenParser:
    def __init__(self, onnx_path: str | Path, names: dict[int, str],
                 imgsz: int = 1280, conf: float = 0.30, iou: float = 0.45):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.iname = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = names

    def _pre(self, img: "Image.Image") -> np.ndarray:
        import PIL.Image as I

        w, h = img.size
        scale = min(self.imgsz / w, self.imgsz / h)
        nw, nh = int(w * scale), int(h * scale)
        im = img.convert("RGB").resize((nw, nh), I.BILINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = np.asarray(im)
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return x, scale

    def __call__(self, image) -> list[dict]:
        import PIL.Image as I

        assert_pure_runtime()
        if isinstance(image, (str, Path)):
            image = I.open(image)
        if not isinstance(image, I.Image):
            image = I.fromarray(image)
        x, scale = self._pre(image)
        out = self.sess.run(None, {self.iname: x})[0]
        pred = out[0] if out.ndim == 3 else out
        if pred.shape[0] < 64 and pred.shape[-1] > pred.shape[0]:
            # raw export layout: (4+nc, N); boxes are cx,cy,w,h in input coords
            pred = pred.T
            cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
            scores = pred[:, 4:]
            confs = scores.max(1)
            cls = scores.argmax(1)
        else:
            # nms'd layout: (N, 6) = x1,y1,x2,y2,conf,cls
            boxes = pred[:, :4].copy()
            confs = pred[:, 4]
            cls = pred[:, 5].astype(int)
        boxes = boxes / scale
        keep = confs > self.conf
        boxes, confs, cls = boxes[keep], confs[keep], cls[keep]
        # NMS per class (simple greedy)
        order = confs.argsort()[::-1]
        picked: list[int] = []
        def iou(a, b):
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            return inter / u if u > 0 else 0
        for i in order:
            if all(cls[i] != cls[j] or iou(boxes[i], boxes[j]) < self.iou for j in picked):
                picked.append(int(i))
        return [
            {
                "role": self.names[int(cls[i])],
                "bbox": [round(float(v), 1) for v in boxes[i]],
                "confidence": round(float(confs[i]), 3),
            }
            for i in sorted(picked, key=lambda i: boxes[i][1])
        ]


_DEFAULT_ONNX = Path(__file__).resolve().parent.parent / "experiments" / "latest" / "best_onnx_fp32.onnx"
_parser: ScreenParser | None = None


def parse_screen(image, onnx_path: str | Path | None = None, names: dict[int, str] | None = None):
    """Screenshot → structured UI elements. Pure vision (see assert_pure_runtime)."""
    global _parser
    if names is None:
        import yaml

        tax = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "schema" / "ui_taxonomy.yaml").read_text()
        )
        names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    if _parser is None or (onnx_path and str(getattr(_parser, "_src", "")) != str(onnx_path)):
        _parser = ScreenParser(onnx_path or _DEFAULT_ONNX, names)
        _parser._src = onnx_path or _DEFAULT_ONNX
    return _parser(image)


if __name__ == "__main__":
    import json
    import sys

    src = sys.argv[1]
    print(json.dumps(parse_screen(src), indent=1)[:2000])
