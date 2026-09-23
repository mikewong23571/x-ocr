#!/bin/zsh
# Orchestrate training on Colab via colab CLI (all training happens on Colab).
# Usage: scripts/colab_train.sh <exp_name> <epochs> [yolo_dir] [model] [imgsz]
# e.g.:  scripts/colab_train.sh exp001_smoke 3 data/yolo yolo11n.pt 1280
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${1:?exp name}"
EPOCHS="${2:?epochs}"
YOLO_DIR="${3:-data/yolo}"
MODEL="${4:-yolo11n.pt}"
IMGSZ="${5:-1280}"
SESSION="xocr"
VM="/content/x-ocr"

cd "$ROOT"
echo "[1/6] fuse + export dataset"
.venv/bin/python annotation/fuse.py --in data/raw --out data/processed
if [ -f data/export_meta_v03_frozen.json ]; then
  .venv/bin/python dataset/export_yolo.py --in data/processed --out "$YOLO_DIR" --freeze-from data/export_meta_v03_frozen.json
else
  .venv/bin/python dataset/export_yolo.py --in data/processed --out "$YOLO_DIR"
fi
rm -f /tmp/xocr_dataset.zip
zip -qr /tmp/xocr_dataset.zip "$YOLO_DIR" training evaluation schema -x "*.DS_Store"

echo "[2/6] ensure Colab session (T4)"
colab sessions | grep -q "$SESSION" || colab new -s "$SESSION" --gpu T4
sleep 5

echo "[3/6] deliver dataset to VM"
# Preferred: direct upload of the exact local export (guarantees dataset==local).
# Fallback: pull from GitHub Release (stale-asset risk) then local-zip extract.
colab upload -s "$SESSION" /tmp/xocr_dataset.zip /content/xocr_dataset_local.zip 2>/dev/null && echo "local zip uploaded" || echo "local upload failed → release fetch"
cat > /tmp/xocr_fetch_ds.py <<PYEOF
import os, subprocess, zipfile, urllib.request
asset = "https://github.com/mikewong23571/x-ocr/releases/download/data-v0.4/x-ocr-bundle.zip"
vm = "$VM"
os.makedirs(vm + "/training", exist_ok=True)
src = None
if os.path.exists("/content/xocr_dataset_local.zip"):
    src = "/content/xocr_dataset_local.zip"
else:
    try:
        if not os.path.exists("/content/x-ocr-bundle.zip"):
            urllib.request.urlretrieve(asset, "/content/x-ocr-bundle.zip")
        src = "/content/x-ocr-bundle.zip"
    except Exception as e:
        print("release fetch failed:", e)
assert src, "no dataset source"
with zipfile.ZipFile(src) as z:
    z.extractall(vm)
print("dataset from", src, "ok")
import json as _j
print("VM dataset meta:", _j.load(open(vm + "/$YOLO_DIR/export_meta.json"))["splits"])
import subprocess as sp
print(sp.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
             capture_output=True, text=True).stdout.strip() or "no gpu")
PYEOF
colab exec -s "$SESSION" -f /tmp/xocr_fetch_ds.py --timeout 600

cat > /tmp/xocr_setup.py <<PYEOF
import os, sys, subprocess
vm = "$VM"
assert os.path.exists(vm + "/training/train_yolo.py"), "bundle must contain training code"
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxsim"],
                   capture_output=True, text=True)
print("setup ok:", r.returncode)
PYEOF
colab exec -s "$SESSION" -f /tmp/xocr_setup.py --timeout 600

echo "[4/6] train: $EXP epochs=$EPOCHS model=$MODEL imgsz=$IMGSZ"
cat > /tmp/xocr_train.py <<PYEOF
import subprocess, sys
vm = "$VM"
r = subprocess.run([sys.executable, vm + "/training/train_yolo.py",
    "--data", vm + "/$YOLO_DIR/data.yaml", "--model", "$MODEL",
    "--imgsz", "$IMGSZ", "--epochs", "$EPOCHS", "--exp", "$EXP",
    "--out", vm + "/experiments"], capture_output=True, text=True)
print(r.stdout[-4000:])
print(r.stderr[-1500:])
PYEOF
colab exec -s "$SESSION" -f /tmp/xocr_train.py --timeout 14400

echo "[5/6] fetch metrics"
cat > /tmp/xocr_fetch.py <<PYEOF
import json, os
p = "$VM/experiments/$EXP/metrics.json"
print(open(p).read() if os.path.exists(p) else "NO METRICS")
PYEOF
colab exec -s "$SESSION" -f /tmp/xocr_fetch.py

echo "[6/6] download artifacts"
mkdir -p "$ROOT/experiments/$EXP"
for f in metrics.json best_onnx_fp32.onnx best_onnx_fp16.onnx; do
  colab download -s "$SESSION" "$VM/experiments/$EXP/$f" "$ROOT/experiments/$EXP/$f" 2>/dev/null || echo "  (missing $f)"
done
echo "done → $ROOT/experiments/$EXP (best.pt kept on VM; download on demand)"
