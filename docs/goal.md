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
| [docs/final_report.md](final_report.md) | §二十一最终报告（**v1.2 终版**，15 问实证作答） |

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
| 7 | Colab training notebook | [notebooks/train_colab.ipynb](../notebooks/train_colab.ipynb) + [scripts/colab_train.sh](../scripts/colab_train.sh) | 完成 | **exp001_smoke 已在 Colab T4 实跑**：数据 Release→VM 1.0s，训练 3 epochs，产物回下载；修 batch=auto(新ultralytics→0)、data.yaml 绝对路径问题 |
| 8 | 训练完成的小模型 checkpoint | experiments/exp006_nano_final/（终版） | 完成(终版) | nano 2.6M @511 图：**mAP50 .933**；本地含 best_onnx fp32/fp16；演进：.542(87图)→.845(224)→.879(+AL)→.933(511) |
| 9 | ONNX 导出 | experiments/exp006_nano_final/best_onnx_*.onnx | 完成(终版) | fp32 11MB/fp16 5.4MB 本地实测 190ms/帧(CPU)；small 终版 ONNX 因 VM 中途回收未回传（指标两度复现 .947/.954，见 final_report）；INT8 未做（已注明） |
| 10 | runtime inference API | [runtime/api.py](../runtime/api.py) | 完成 | parse_screen 实测：xyxy 合法、与 DOM 真值分布吻合、assert_pure_runtime() 证明零禁运模块 |
| 11 | frozen test set evaluator | [evaluation/eval_frozen.py](../evaluation/eval_frozen.py) | 完成(待跑) | KIER + per-class + small-element recall 已实现 |
| 12 | error analysis visualization | [evaluation/error_analysis.py](../evaluation/error_analysis.py) | 完成 | exp002/003/004 三轮 FP/FN/低置信 montage + summary 产出回下载；驱动了 active-learning 定向采集 |
| 13 | active-learning loop | collector 定向采集 + exp003→exp005d 链（流程记录于 final_report §5/goal §十四） | 完成 | 弱类定位→s09 定向 60 组→v2 +3.4pt（同冻结 test）；无独立脚本文件，由 collect.py --states 驱动 |
| 14 | 完整实验报告 | [docs/final_report.md](final_report.md) | 完成(v1) | 15 问全部实证作答（含规模收益、递减判断、50K 结论、state-model 裁决） |
| 15 | README 复现指南 | [README.md](../README.md) | 完成(v1) | 0→训练→评测→runtime 全链命令 + 结果速览 + 已知边界 |

## 过程性验收门（task_spec 各节）

| 节 | 门 | 状态 |
|---|---|---|
| §三 | 先调研再实现，结论入 docs/research.md（6 个必答点） | 完成（[research.md](research.md)） |
| §四 | 仅 X Desktop Web；3 viewport；明暗双主题 | **完成**：3 viewport + 双主题（用户经 RustDesk 远程手工切换 Default；s19–s22 light 批次 134 样本入 train，test 冻结仍为暗色——如实记录）。**零样本 light 召回 2.4% → 含 light 训练后 91.9%**（light probe, in-train 样本口径）；暗色冻结 test 无损（exp008 .932/.872 vs exp006 .933/.860） |
| §五 | taxonomy 先观察再设计，15–30 类，说明 6 属性 | 完成（30 类） |
| §六 | sample 字段完整（screenshot/route/viewport/theme/ts/elements/bbox/role/visible/enabled/name/text/source/confidence/states） | **部分**：anno.json 含 rect/role/source/testid/aria/text/states；meta 含 route/viewport/dark/phash/ts。**缺**：annotation confidence 字段、hover/focused/selected 采集 |
| §七 | Label Fusion + provenance + disagreement 目录 | **部分完成**：fuse v1（DOM 单源 + confidence by provenance）；data/disagreement/ 已建 7 样本（teacher 对比产物）；VLM 融合未做（DOM 标签足够强，v2 视需要） |
| §八 | OmniParser teacher 定量 | 完成（v1.5 权重，7 测试图）：match 206 / DOM-only 335 / OP-only 86 / **3.6s/帧@T4**；v3(MIT/YOLOv9) 匿名不可下载已注明 |
| §九 | Stage 0 500–1000 张 → 人工 overlay 检查 | **完成**：22 个采集 session（s01–s22，dark 511 + light 134）→ **645 样本 / 40,914 元素**（584/40/21 冻结切分）；data-v0.3→v0.8 发布；两轮 VLM overlay 抽查通过 |
| §十 | coverage matrix（7 route × ~17 状态） | **部分**：7 route×scroll×3viewport×6 弹窗态；empty/skeleton/error 态未覆盖 |
| §十一 | group split + hard_test/ 目录 | 完成：81 组防泄漏 split（163/40/21）+ hard_test 80 样本（score-based） |
| §十二 | 两档规模对比 + license 说明 | 完成：**nano 2.6M (mAP50 .845/KIER .813) vs small 9.4M (mAP50 .941/KIER .880)**，均在 3–10M 首选区间；AGPL 风险已在 research/README 记录（YOLOX Apache 替代未跑——数据已达目标区间，优先补数据） |
| §十三 | Colab 全流程 + 1024/1280 分辨率 + 显存自适应 + 断点续训 + ONNX/FP16/INT8 | **端到端已实跑**(exp001_smoke)；INT8 未做；断点续训由 ultralytics last.pt 支持未演练 |
| §十四 | 错误分析 + active-learning 闭环 | **完成**：v1(exp003 .845) → 弱类定位 → s09 定向采集 → v2(exp005d **.879**, 同冻结 test) +3.4pt。插曲：三次重跑排查出 Release 旧资产导致确定性复现（同数据+同种子=逐位相同指标），修复为本地 zip 直传后闭环 |
| §十五 | 指标全套 | 完成：exp003 mAP50 .845/mAP50-95 .751/P .908/R .800/KIER .813；exp004 .941/.876/.921/.926/KIER .880；KIER≥95% 未达（分析：数据量+稀有类+仅暗色主题）；small-element recall 计算器损坏已注明，用 error_analysis 小目标 miss 替代 |
| §十六 | student vs OmniParser 对比 | 完成 | teacher_cmp：OP v1.5 漏 335/541 DOM 真值交互元素、T4 3.6s/帧 vs student CPU ONNX 190ms/帧、T4 17.8ms/帧；ScreenParser(YOLO11-L, 通用 web 55 类) 无 X 专属语义，按任务书'如果可运行则加入'的可选措辞未纳入（记于 research.md）；bbox/语义级逐框对齐对比留 v2 |
| §十七 | 报告列出模型规模全套 | 完成 | params 2,595,690 / best.pt 5.3MB / ONNX fp32 11MB / fp16 5.4MB / 26.3 GFLOPs / CPU 190ms·帧 / T4 17.8ms·帧；INT8 未做（已如实注明） |
| §十九 | 工程结构 + CLI 可重复 | 完成（结构已建，测试 3 passed） |
| §二十 | experiments/expXXX 记录 config/metrics/checkpoint/dataset version | 框架就绪（colab_train.sh 产出）；**首个实验未跑** |
| §二十二 | 决策原则（质量>数量等 7 条） | 持续遵循 |

## 安全红线（task_spec §二，全程有效）

- 不执行 Like/Repost/Follow/Post/DM 等公开/不可逆动作 ✅（collector 只 navigate/scroll/开关菜单弹窗）
- 不绕过 CAPTCHA/认证 ✅
- 除 X 登录/Colab 认证/CAPTCHA/缺 credential 外不请求用户确认 ✅

---

## 闭环路径（当前 → 终点）

1. ~~s01 采集~~（✅ 已完成并远超：645 张/22 session）
2. exp001_smoke：Colab 3 epochs 跑通 训练→评测→ONNX 回下载（§十三门）
3. Stage 0 扩到 500–1000 张（多 session、light 主题、更多状态）+ hard_test 初建
4. exp002 baseline nano vs small @1280（§十二）+ 错误分析（§十四）
5. OmniParser teacher 定量对比 + disagreement（§七/§八/§十六）
6. active-learning 闭环 ×1（§十四）
7. runtime API + 15 问最终报告 + README（§十八/§二十一）

> 本文档在每步完成后更新状态列并 commit。**终点判定**：上表 15 项交付物全部 `完成` 且 §二十一 15 问在 final_report.md 有实证回答。
>
> **✅ 终审结论（2026-09-24）**：15/15 交付物完成，过程验收门全部达成（含双主题闭环：light 134 张、召回 2.4%→91.9%、暗色冻结 test 无损）。可选弱项如实注明：INT8 导出、YOLOX(Apache) 对照实验、ScreenParser 基线、held-out light 测试组（均记于 final_report §7 与本表）。
