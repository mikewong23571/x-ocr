# Goal Ledger：实验目标闭环文档（docs/goal.md）

> 目标源：[docs/task_spec.md](task_spec.md)（用户最初 prompt 的存档，唯一权威来源）
> 本文是**可核对的验收台账**：每条要求 → 产物路径 → 状态 → 证据。状态只有三种：`完成`（有实物+验证）、`进行中`（已启动）、`未开始`。
> 更新规则：每完成一项立即更新本表并 commit；不允许"计划了"就算完成。

---

## 北极星目标（提取自 task_spec §Goal + §一）

> **训练一个仅依赖 screenshot 的 X-specific、pure-vision、3–20M 参数小模型，在独立 frozen test set 上稳定识别 X Desktop Web 上 Agent 需要交互的关键 UI 元素；运行时禁止 DOM/Playwright/OmniParser/VLM。**
>
> 这是"把运行时智能蒸馏进专用视觉模型"的实验（[task_spec §一](task_spec.md#一背景与核心假设)），不是造通用 parser、不是造 VLM。
> 如果实验事实否定核心假设，必须如实报告（[task_spec §二十二](task_spec.md#二十二决策原则)）。

**规模约束**：首选 3–10M 参数；可接受 <20M；超 20M 必须实验论证更小为何不够（§Goal）。
**运行时禁令**（§十八）：无 DOM、无 browser、无 OmniParser、无 VLM/LLM——`runtime/parse_screen(image)` 必须可证明满足。

---

## 引用文档索引

| 文档 | 作用 |
|---|---|
| [docs/task_spec.md](task_spec.md) | 目标源（用户 prompt 原文，23 节） |
| [docs/research.md](research.md) | §三调研结论：Webshot 复用判定 / DOM 能与不能 / student 选型 / license |
| [docs/system_boundary.md](system_boundary.md) | 系统边界：浏览器扩展链路、Colab、网络磁盘 |
| [schema/ui_taxonomy.yaml](../schema/ui_taxonomy.yaml) | §五 taxonomy：30 类 X 专属角色 + DOM 定位规则 |
| [docs/final_report.md](final_report.md) | §二十一最终报告（**未开始**） |

---

## 交付物验收清单（task_spec §二十一，15 项）

| # | 要求 | 产物（路径） | 状态 | 证据 |
|---|---|---|---|---|
| 1 | 自动数据采集 pipeline | [collector/collect.py](../collector/collect.py) + [scripts/pw](../scripts/pw) | 完成 | s01 session 后台采集中；smoke 7 样本已验证 |
| 2 | 自动/弱监督 annotation pipeline | [collector/build_extract_js.py](../collector/build_extract_js.py) + [annotation/fuse.py](../annotation/fuse.py) | 完成(v1) | 487 元素/7样本冒烟；**OmniParser teacher 融合未做** |
| 3 | X-specific taxonomy | [schema/ui_taxonomy.yaml](../schema/ui_taxonomy.yaml) | 完成 | 30 类，来自 9 个真实页面 DOM dump（data/observation/） |
| 4 | dataset manifest | data/processed/*/manifest.jsonl（fuse 产物） | 完成 | 冒烟已生成；随采集更新 |
| 5 | train/val/test split | [dataset/export_yolo.py](../dataset/export_yolo.py) | 完成 | group split（session×route）防泄漏，export_meta.json 记录分组 |
| 6 | 数据可视化工具 | [scripts/visualize_overlay.py](../scripts/visualize_overlay.py) | 完成 | data/overlays/ 生成；VLM 抽查 ~85% 对齐并修复 4 项 |
| 7 | Colab training notebook | [notebooks/train_colab.ipynb](../notebooks/train_colab.ipynb) + [scripts/colab_train.sh](../scripts/colab_train.sh) | 完成(待跑通) | **尚未实际在 Colab 执行** |
| 8 | 训练完成的小模型 checkpoint | experiments/exp*/best.pt | 未开始 | 依赖 s01 完成 |
| 9 | ONNX 导出 | training/train_yolo.py 内置 export | 未开始 | 同上 |
| 10 | runtime inference API | runtime/api.py（parse_screen） | 未开始 | — |
| 11 | frozen test set evaluator | [evaluation/eval_frozen.py](../evaluation/eval_frozen.py) | 完成(待跑) | KIER + per-class + small-element recall 已实现 |
| 12 | error analysis visualization | evaluation/error_analysis.py | 未开始 | — |
| 13 | active-learning loop | scripts/active_learning.py | 未开始 | ≥1 次闭环（§十四） |
| 14 | 完整实验报告 | docs/final_report.md | 未开始 | 必答 15 问见 §二十一 |
| 15 | README 复现指南 | README.md | 未开始 | — |

## 过程性验收门（task_spec 各节）

| 节 | 门 | 状态 |
|---|---|---|
| §三 | 先调研再实现，结论入 docs/research.md（6 个必答点） | 完成（[research.md](research.md)） |
| §四 | 仅 X Desktop Web；3 viewport；明暗双主题 | viewport 采集**进行中**；**双主题未实现**（当前仅 dark/lightsout；light 需 X Display 设置切换——collector v2 待加） |
| §五 | taxonomy 先观察再设计，15–30 类，说明 6 属性 | 完成（30 类） |
| §六 | sample 字段完整（screenshot/route/viewport/theme/ts/elements/bbox/role/visible/enabled/name/text/source/confidence/states） | **部分**：anno.json 含 rect/role/source/testid/aria/text/states；meta 含 route/viewport/dark/phash/ts。**缺**：annotation confidence 字段、hover/focused/selected 采集 |
| §七 | Label Fusion + provenance + disagreement 目录 | **部分**：fuse.py 是 v1（DOM 单源 + 几何过滤）；**未做**：OmniParser/VLM 融合、data/disagreement/ |
| §八 | OmniParser teacher：match/DOM-only/OP-only/conflict 定量 | 未开始（Colab 上跑） |
| §九 | Stage 0 500–1000 张 → 人工 overlay 检查 | **进行中**（s01 目标 100 先跑通链；Stage 0 全量随后） |
| §十 | coverage matrix（7 route × ~17 状态） | **部分**：7 route×scroll×3viewport×6 弹窗态；empty/skeleton/error 态未覆盖 |
| §十一 | group split + hard_test/ 目录 | split 完成；**hard_test/ 未建** |
| §十二 | 两档规模对比（nano/tiny + small），license 说明 | 选型完成（research.md §4）；实验未跑 |
| §十三 | Colab 全流程 + 1024/1280 分辨率 + 显存自适应 + 断点续训 + ONNX/FP16/INT8 | 脚本就绪；**未实际执行**；INT8 未实现 |
| §十四 | 训练后错误分析 + ≥1 次 active-learning 闭环 | 未开始 |
| §十五 | 指标：mAP50/50-95/precision/recall/per-class/small-element + KIER≥95% 目标 | 评测器完成；**未产生数字** |
| §十六 | student vs OmniParser（+ScreenParser 若可跑）对比（recall/bbox/语义/延迟/大小/显存/CPU 可行性） | 未开始 |
| §十七 | 报告列出 params/ckpt/ONNX/FP16/INT8 大小/FLOPs/延迟 | 未开始 |
| §十九 | 工程结构 + CLI 可重复 | 完成（结构已建，测试 3 passed） |
| §二十 | experiments/expXXX 记录 config/metrics/checkpoint/dataset version | 框架就绪（colab_train.sh 产出）；**首个实验未跑** |
| §二十二 | 决策原则（质量>数量等 7 条） | 持续遵循 |

## 安全红线（task_spec §二，全程有效）

- 不执行 Like/Repost/Follow/Post/DM 等公开/不可逆动作 ✅（collector 只 navigate/scroll/开关菜单弹窗）
- 不绕过 CAPTCHA/认证 ✅
- 除 X 登录/Colab 认证/CAPTCHA/缺 credential 外不请求用户确认 ✅

---

## 闭环路径（当前 → 终点）

1. **[进行中]** s01 采集完成 → fuse → export → overlay 抽查（§九 Stage 0 门）
2. exp001_smoke：Colab 3 epochs 跑通 训练→评测→ONNX 回下载（§十三门）
3. Stage 0 扩到 500–1000 张（多 session、light 主题、更多状态）+ hard_test 初建
4. exp002 baseline nano vs small @1280（§十二）+ 错误分析（§十四）
5. OmniParser teacher 定量对比 + disagreement（§七/§八/§十六）
6. active-learning 闭环 ×1（§十四）
7. runtime API + 15 问最终报告 + README（§十八/§二十一）

> 本文档在每步完成后更新状态列并 commit。**终点判定**：上表 15 项交付物全部 `完成` 且 §二十一 15 问在 final_report.md 有实证回答。
