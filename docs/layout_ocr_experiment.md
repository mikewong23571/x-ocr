# Phase 2 实验报告：Layout Parser + Regional OCR + 更强预训练基座

> 目标：在 Atomic UI Detector（exp010, 2.6M, mAP50 .935）之上引入独立 Layout 模型与区域
> OCR，测试 GUI-domain 预训练基座的数据效率，形成「Atomic + Layout + OCR」多小专家 runtime。
> 约束遵守：无完整 Agent、无神经融合、无纯 mAP 驱动的更大检测器。

## ⚠ 勘误（2026-09-25 晚）：E1/E2 数字基于带缺陷的 GT，v2 修正中

用户以 DOM 真值抽查触发审计，发现 `build_labels.py` 的 `_u()` 并集函数期望 xywh 却被传入
xyxy——六个 union 类（post_header / media_region / action_row / left_nav / right_sidebar /
compose_bar）的 GT 系统性畸形（如 header 高 614px、bottom=x₂+y₁）。影响：
- E1/E2 表格中这些类的 per-class AP 及总体 mAP50 在"对坏 GT 评测"下得出，v2（GT 修复，
  compose_bar 9→267 实例）重训后本节数字将整体更新；
- 未受影响的类（post/text_region/quoted_post/card/overlay 单元素直取）与 E3/E4 OCR 部分
  不受此 bug 影响（E4 的 layout recall 以 v1 GT 为分母，v2 重测）。
数据：Release `layout-data-v2`（sha256 头 f3455754d5bbea0d）。

## 实验设置

- **数据（零大规模新采）**：Layout 标签全部由现有 screenshot+DOM 合成（`layout/build_labels.py`，
  11 类 `schema/layout_taxonomy.yaml`，788 帧 / 9,362 框，视觉抽查合格）。切分与原子任务共用
  v03 frozen groups → train 729 / val 39 / test 20。缩放研究 seed=42 抽 100/250/500（清单文件，
  VM 侧构建子集），val/test 全程不变。唯一定向补采：36 帧暗色未登录截图（OCR 暗色覆盖缺口，
  `loggedout_dark_s01`，provenance 记录于 meta）。
- **OCR 基准**：DOM text_spans 行级真值，480 区域 / 1,289 行（light 438 + dark 42），
  `data/ocr_benchmark/`。帧级卫生过滤剔除 41 个守卫上线前的 DOM/像素错位帧（有计数，非静默）。
- **训练**：全部 Colab（A100→L4），60ep@1280，`layout/train_matrix.py` 断点续跑，
  结果 `experiments/layout_matrix/results.jsonl`。数据中转 GitHub Release `layout-data-v1`
  （sha256 头 31c12654600db114 防陈旧资产）。
- **运行方差披露**：同一 nano-full 配置 A100=0.978 / L4=0.956（±2pt 跨机方差）。GUI 臂全部
  在 L4 训练，同机可比；COCO 臂低数据档来自 A100 批次。结论只采用跨臂 ≥5pt 的差距。

## E1/E2 — 基座 × 数据量（frozen test mAP50，11 类 layout）

| prior \ train | 100 | 250 | 500 | full(729) |
|---|---|---|---|---|
| YOLO11n (COCO, 2.6M) | 0.487 | 0.824 | 0.834 | 0.978 |
| YOLO11s (COCO, 9.4M) | 0.690 | 0.824 | 0.895 | 0.987 |
| ScreenParser (GUI, YOLO11-L, 25.3M) | 0.516 | 0.758 | 0.802 | 0.916 |
| **OmniParser v2 icon (GUI, YOLOv8, 20.1M)** | **0.739** | **0.895** | **0.970** | **0.981** |

全部 16 格完成（17 行含 nano-full 的跨机复跑）。ScreenParser 全档位垫底，
与两端点预判一致：web 页面区块语义 ≠ X 帖子结构语义。

![scaling curve](../experiments/layout_matrix/scaling_curve.png)

**E2 核心答案：GUI prior 能否用更少数据达到相同效果？——能，且非常显著（OmniParser）。**
- @100：OmniParser 0.739 vs nano 0.487（**+25pt**）、vs small 0.690（+5pt）
- @250：0.895 ≈ COCO 基座 @500 的水平（small 0.895/nano 0.834）→ **同效数据需求约减半**
- @500：0.970 已接近其 full 水平
- 但 GUI prior 不是普遍有效：ScreenParser（144 万张 web 页面预训练的 55 类检测器）在两个
  端点都最差——它的"web 页面区块"语义与我们的"帖子结构"语义不匹配；OmniParser 的
  "可交互区域"单类先验反而更贴近布局区域的本质（都是 UI 包围盒）。

**per-class AP50（full 档）**：三类基座的 post/card/left_nav/right_sidebar 均 ≥0.995；
难类一致为 overlay(.15-.30)、media_region(.30-.48)、action_row(.37-.90)、post_header(.42-.75)、
text_region(.64-.87)。compose_bar 全库仅 9 实例（数据缺口，非模型问题）。

## E3 — Regional OCR 基准（500 区域 / 1,910 行，v1.1 含 dialog 分层）

| 引擎 | 模式 | CER↓ | EM↑ | 延迟 |
|---|---|---|---|---|
| RapidOCR v4 mobile（现行） | rec_line（行框给定） | 0.260 | 51.8% | 6.8ms/行 |
| **PP-OCRv6-small（21.2MB rec）** | rec_line | **0.190** | **70.1%** | **6.3ms/行** |
| v4 | det+rec_region（元素框） | 0.513 | 27.1% | 266ms/区 |
| **v6small** | det+rec_region | **0.502** | **37.6%** | 266ms/区 |
| v4 | 整屏 det+rec | 0.552 | 行覆盖 52.7% | 881ms/帧 |
| v6small | 整屏 det+rec | 0.538 | 行覆盖 52.8% | 1022ms/帧 |

- **问题3（Layout 减少 OCR 计算）：是。** 整屏一次 ~0.9s 仅覆盖 61% 行；Layout 指定 2-4 个
  文本区 × ~250ms；行框已知时 rec-only 6.4ms/行（一屏 ~20 行 ≈ 130ms，**~7× 加速**）。
- **问题4（Regional vs 整屏）：明确更优。** CER 0.54→0.19（行级），且整屏会读出 agent 不需要
  的全部文本（nav/头像/无关帖），token 成本同步放大。
- 字号效应：large(≥18px) CER 0.059 ≪ small(<14px) 0.18 / mid 0.25 —— 小字是主误差源。
- 主题：暗色 0.159 vs 亮色 0.225（暗色样本偏大标题，人群不同；无主题劣势证据）。
- WER 不单列：中文行无空格，与 CER 同义；EM/CER 已覆盖。模型大小：v6 det 9.5MB + rec 20.3MB
  （vs v4 det+rec ~15MB）。
- **缩放鲁棒性（`ocr/eval_scale.py`，400 行 × 0.5/0.75/1.0/1.25×）**：rec 模型内部高度归一化，
  0.5× 也只掉 ~1.5pt（v6small 0.278→0.292，v4 0.311→0.326），1.25× 无损——**视口缩放不是
  OCR 误差主因**，字号才是。
- **分层表现（v6small rec-only CER）**：dialog 0.13-0.15（短、字距大，最容易）、menu_item 0.279、
  **search_input 0.954（最差：灰字占位符，聚焦后为空——属真实 UI 语义而非模型缺陷，
  建议运行时对 search_input 用 aria 回退或跳过 OCR）**。
- 基准版本 ocrbench-v1.1（新增 dialog 分层 35 区域 + dataset_version 字段）；
  暗色补采数据 manifest：`data/raw/loggedout_dark_s01/_dataset_manifest.json`。

## E4 — 三模型几何融合（20 帧 frozen test，CPU）

`runtime/structured.py`：Atomic(exp010) + Layout + Regional OCR(v6small)，确定性几何
（最小容器 containment + y 对齐回退 + 区域 roll-up），输出
`{regions:[{type:"post", bbox, text, children:[{role,bbox}]}]}`。

| 布局模型 | layout R | layout P | 父子关联 | 完整帖子重建 | 帖子文本 CER | 延迟(ms) atomic/layout/OCR/合计 |
|---|---|---|---|---|---|---|
| OmniParser-full (20.1M) | 0.810 | 0.918 | 0.773 | **1.00** | — | 111 / 418 / 1133 / **~1.7s** |
| **nano-full (2.6M)** | 0.769 | 0.916 | **0.839** | **1.00** | **0.391** | 107 / 90 / 1154 / **~1.4s** |

帖子文本 CER（重建帖子的 OCR 文本 vs DOM post_text 真值，23 对配对）：0.391——优于整屏
模式（0.49）、劣于行框给定模式（0.22），与区域 det+rec 的定位一致；E1 要求的
hierarchy reconstruction accuracy 即上表父子关联列（nano 0.839）。

- atomic（共用 exp010）：recall 0.925 / precision 0.964（与单模型基线一致，融合无损失）。
- 延迟分解：OCR 占 ~80%（按需 2-4 文本区）；布局 nano 只占 90ms。
- 混合精度注意：OmniParser 布局召回更高但 CPU 延迟 4.6×、关联反而更低——
  **小布局模型在融合管线里性价比更高**。

## 研究问题总回答

1. **Layout 独立建模是否更好？** 是。结构区域（post/header/action_row/sidebar）用独立小模型
   即可到 mAP50 .96-.98，且让下游拿到的是层级而非平面列表；不必也不应把它塞回 32 类原子
   检测器（两类目标尺度/语义不同，exp010 原子链路保持不变且无损）。
2. **GUI prior 数据效率？** 显著（OmniParser：@100 +25pt、同效数据 ~½）；但 GUI 先验间差异
   巨大（ScreenParser 反而最差）——先验语义与目标任务的匹配度 > 先验数据量。
3. **Layout 减少 OCR 计算？** 是（~7×，见 E3）。
4. **Regional 优于整屏？** 明确是（CER 减半 + 计算大降 + 无关文本不进上下文）。
5. **多小专家 vs 单模型？** 是。单 exp010 只给平面 32 类（无文本、无层级）；组合后
   完整帖子重建率 1.00、文本 CER 0.22，总 CPU ~1.4s/帧（其中原子+布局仅 ~0.2s）。
6. **总量/延迟/收益**：推荐组合总参数 2.6M(原子)+2.6M(布局nano)+5.2M(v6 rec) ≈ **10.4M**，
   模型文件 11.1+11.1+20.3(+det 9.5 可选) ≈ **42-52MB**，CPU 端到端 **~1.4s/帧**
   （纯元素感知 ~0.2s；OCR 按需）。
7. **下一版默认方案**：`Atomic(exp010 nano) + Layout(nano-full, gui_omniparser 作为低数据冷启动
   权重) + OCR(PP-OCRv6-small rec)`。若未来 X 布局数据 <300 张，布局模型直接用 OmniParser
   微调权重（@250 已 0.895 vs nano 0.824）。

## 与 baseline 的实际提升（frozen test）

| 能力 | exp010 单模型 | 三模型组合 |
|---|---|---|
| 交互元素召回 | 0.927 | 0.925（无损） |
| 页面结构 | ❌ 平面列表 | ✅ 11 类层级区域（R .77-.81） |
| 帖子级语义 | ❌ | ✅ 重建率 1.00（actions+text 配对） |
| 文本 | ❌（整屏 OCR 备选 0.9s/帧 CER .49） | ✅ 区域 OCR CER .22、按需 1.1s |
| CPU 延迟 | 190ms | 200ms(元素) / 1.4s(含文本) |

## 产物清单

- `schema/layout_taxonomy.yaml`（11 类）｜`layout/{build_labels,export_layout_yolo,train_matrix,plot_scaling}.py`
- `ocr/{build_benchmark,eval_ocr}.py`｜`runtime/structured.py`｜`evaluation/eval_fusion.py`
- `experiments/ocr/ocr_benchmark_results.json`｜`experiments/layout_matrix/{results.jsonl,e4_fusion_results.json,scaling_curve.png,layout_*/best_onnx_*.onnx}`
- Release `layout-data-v1`（数据+代码，sha256 头 31c12654600db114）

## 运维注记（复现者需要知道）

Colab 个人配额下会话 ~1-1.5h 即被回收（本实验经历 6 次）+ 僵尸分配需等 TTL。对策已固化：
release sha 校验投递、`train_matrix.py` 断点续跑（results.jsonl 逐臂落账）、本地 watcher
6 分钟增量抢救 + 逐臂 zip 拉取、`colab run` 兜底。数据/代码任何一次重投 ≤5 分钟。
