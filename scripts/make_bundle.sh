#!/bin/zsh
# Bundle code + YOLO dataset for Colab (notebook expects /content/x-ocr-bundle.zip)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[ -d data/yolo ] || { echo "run dataset/export_yolo.py first"; exit 1; }
rm -f x-ocr-bundle.zip
zip -qr x-ocr-bundle.zip training evaluation schema data/yolo data/yolo/export_meta.json -x "*.DS_Store"
ls -lh x-ocr-bundle.zip
