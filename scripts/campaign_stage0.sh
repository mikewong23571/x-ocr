#!/bin/zsh
# Stage 0 campaign: sequential collection sessions → fuse → export → release v0.3 → train exp003
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for s in s05 s06 s07 s08; do
  echo "=== session $s ==="
  case $s in
    s05) VPS="1920x1080,1440x900,1280x720";;
    s06) VPS="1440x900,1280x720,1920x1080";;
    s07) VPS="1280x720,1920x1080,1440x900";;
    s08) VPS="1920x1080,1280x720,1440x900";;
  esac
  $PY collector/collect.py --session $s --target 150 --states feed,menus,threads --viewports $VPS 2>&1 | tail -3
done
echo "=== fuse + export ==="
$PY annotation/fuse.py --in data/raw --out data/processed | tail -2
$PY dataset/export_yolo.py --in data/processed --out data/yolo | tail -6
$PY scripts/build_hard_test.py --top 80 | tail -1
echo "=== bundle + release ==="
./scripts/make_bundle.sh | tail -1
gh release create data-v0.3 x-ocr-bundle.zip --repo mikewong23571/x-ocr \
  --title "Dataset v0.3 (Stage 0 campaign)" --notes "Multi-session collection, 3 viewports, menus/dialogs/threads." || echo "release failed"
echo "CAMPAIGN DONE"
