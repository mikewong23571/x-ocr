# x-ocr：X-specific 纯视觉 UI 感知模型实验

> **命题**（[docs/task_spec.md](docs/task_spec.md)）：运行时只看 screenshot、参数几百万的专用检测器，能否替代 X 场景下昂贵的通用 GUI parsing？
> 运行时禁令：无 DOM / 无 Playwright / 无 OmniParser / 无 VLM——`runtime/api.py` 的 `assert_pure_runtime()` 可机器验证。

## 结果速览（随实验更新）

| 实验 | 模型 | 数据 | mAP50 | mAP50-95 | 备注 |
|---|---|---|---|---|---|
| exp001_smoke | yolo11n (2.6M) | 53 图 | 0.0 | 0.0 | 3 epochs 管线冒烟 |
| exp002_baseline_nano | yolo11n (2.6M) | 87 图 | 0.542 | 0.466 | 60ep@1280，test 仅 7 图噪声大；conf=0.25 时 mAP50=0.365 |

Runtime 实测（`experiments/exp002_baseline_nano/best_onnx_fp32.onnx`，CPU/ONNX Runtime）：
单帧 53 检测、nav_item/post_container/news_card 与 DOM 真值完全命中，进程内零禁运模块。

## 从零复现

### 0. 依赖与浏览器
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python playwright pillow pyyaml imagehash pytest onnxruntime
# Chrome 装 "Playwright Extension" 扩展（chrome.google.com/webstore，id mmlmfjhmonkocbjadbfplnigmagldckm）
# 扩展 status 页显示 token → 存入 .env.local:
echo 'PLAYWRIGHT_MCP_EXTENSION_TOKEN=<token>' > .env.local
echo 'PW_SESSION=xocr' >> .env.local
npm i --prefix tools/pwcli   # pin @playwright/cli@0.1.21（协议 v2，须与扩展匹配）
./scripts/pw attach --extension=chrome   # 复用日常 Chrome（含 X 登录态）
```

### 1. 采集（安全：仅导航/滚动/开关菜单弹窗，无公开动作）
```bash
.venv/bin/python collector/collect.py --session s01 --target 100 \
  --states feed,menus,threads --viewports 1920x1080,1440x900,1280x720
```
产出 `data/raw/<session>/<seq>/{shot.png,anno.json,meta.json}`（含 phash、route、viewport、theme）。

### 2. 标注融合 + 数据集导出 + hard_test + 可视化
```bash
.venv/bin/python annotation/fuse.py              # 裁剪/同角色去重/phash 近重复消除/provenance
.venv/bin/python dataset/export_yolo.py          # YOLO 格式 + session×route group split（防泄漏）
.venv/bin/python scripts/build_hard_test.py      # 菜单/弹窗/密集流等困难样本
.venv/bin/python scripts/visualize_overlay.py --montage   # 人工检查
```

### 3. 训练（全部在 Colab，本地零 torch）
```bash
./scripts/make_bundle.sh                                   # 代码+数据集打包
gh release create data-vX <zip>                            # 数据中转（VM 内 1s 拉取）
./scripts/colab_train.sh exp002_baseline_nano 60 data/yolo yolo11n.pt 1280
# 或用 notebooks/train_colab.ipynb 手动跑
```

### 4. 评测 + 错误分析（frozen test split）
```bash
# VM 上：python evaluation/eval_frozen.py ...（KIER/per-class）
#        python evaluation/error_analysis.py ...（FP/FN/低置信 montage）
```

### 5. Runtime 推理（纯视觉证明）
```bash
.venv/bin/python -c "
from runtime.api import parse_screen
print(parse_screen('shot.png'))   # [{role, bbox(xyxy), confidence}, ...]
"
```

## 工程地图

```
collector/   采集（playwright-cli 扩展会话驱动真实 Chrome）
annotation/  DOM 标注融合（fuse.py）
teacher/     OmniParser 对比（omniparser_compare.py）
dataset/     YOLO 导出 + group split
training/    Colab 训练脚本（train_yolo.py）
evaluation/  frozen 评测（eval_frozen.py）+ 错误分析（error_analysis.py）
runtime/     parse_screen API（ONNX Runtime，纯度可验证）
schema/      ui_taxonomy.yaml（30 类 X 专属 agent-oriented 角色）
docs/        task_spec / research / system_boundary / goal / final_report
scripts/     pw wrapper / colab_train.sh / campaign / overlays / hard_test
experiments/ expXXX（config+metrics+checkpoint 引用）
```

## 已知边界（[docs/system_boundary.md](docs/system_boundary.md)）

- **扩展 Input 通道不可用**（实证：CDP 鼠标/键盘事件 0 到达）→ 一切交互走页面内 eval；
  X 主题色块无法合成点击 → **light 主题需人工在 Display 设置切换一次**（当前仅 dark/lightsout）。
- Colab VM 临时 → 数据集走 GitHub Release 中转（本机零重复流量）。
- 训练一律 Colab（用户策略）；本地仅采集/数据集/ONNX 推理。

## License 提示
主实验 ultralytics YOLO11 为 **AGPL-3.0**（商用闭源需注意，报告已记录）；OmniParser icon_detect_v3
基于 MIT 的 YOLOv9；ScreenParse 生态 MIT/Apache-2.0。数据集为真实 X 截图，仅研究用途。
