#!/usr/bin/env python3
"""x-ocr model-v1.0-exp010 standalone inference example.

No repo dependencies: onnxruntime + numpy + pillow + pyyaml only.
Runtime is pure vision — no DOM, no browser, no Playwright, no VLM.

Usage:
    pip install onnxruntime numpy pillow pyyaml
    python example_inference.py best_onnx_fp32.onnx ui_taxonomy.yaml shot.png
"""
import json
import sys

import numpy as np
import yaml


def load(onnx_path: str, taxonomy_path: str):
    import onnxruntime as ort

    tax = yaml.safe_load(open(taxonomy_path))
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess, names


def parse(sess, names, image_path: str, imgsz: int = 1280,
          conf: float = 0.30, iou: float = 0.45) -> list[dict]:
    from PIL import Image

    img = Image.open(image_path)
    w, h = img.size
    scale = min(imgsz / w, imgsz / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = np.asarray(img.convert("RGB").resize((nw, nh), Image.BILINEAR))
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    pred = out[0] if out.ndim == 3 else out
    if pred.shape[0] < 64 and pred.shape[-1] > pred.shape[0]:
        pred = pred.T                                   # (4+nc, N) → (N, 4+nc)
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        scores = pred[:, 4:]
        confs, cls = scores.max(1), scores.argmax(1)
    else:                                               # nms'd export: (N, 6)
        boxes, confs, cls = pred[:, :4].copy(), pred[:, 4], pred[:, 5].astype(int)
    boxes /= scale
    keep = confs > conf
    boxes, confs, cls = boxes[keep], confs[keep], cls[keep]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

    def _iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0

    order = confs.argsort()[::-1]
    picked = []
    for i in order:
        if all(cls[i] != cls[j] or _iou(boxes[i], boxes[j]) < iou for j in picked):
            picked.append(int(i))
    return [
        {"role": names[int(cls[i])],
         "bbox": [round(float(v), 1) for v in boxes[i]],
         "confidence": round(float(confs[i]), 3)}
        for i in sorted(picked, key=lambda i: boxes[i][1])
    ]


if __name__ == "__main__":
    onnx, tax, img = sys.argv[1], sys.argv[2], sys.argv[3]
    sess, names = load(onnx, tax)
    els = parse(sess, names, img)
    print(f"{len(els)} elements (conf>0.30):")
    print(json.dumps(els, indent=1))
