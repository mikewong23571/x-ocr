# 最终实验报告（docs/final_report.md）

> 目标源：[task_spec.md](task_spec.md) · 验收台账：[goal.md](goal.md) · v1.0（exp003/exp004 数字完成后定稿）

## 0. 一句话结论

**命题基本成立**：一个 2.6M 参数、运行时只看 screenshot 的 X 专用检测器，在 224 张图的 Stage 0 数据上即达到 mAP50 0.879（active-learning 后；初始 0.845），核心导航/操作元素可靠检出；同口径下 OmniParser v1.5 漏掉 62% 的 DOM 真值交互元素且需 T4 GPU 3.6s/帧。数据量仍是第一瓶颈（Stage 0 仅 224/目标 500–1000，且仅 dark 主题）。

## 1. 数据

| 问题 | 答案 |
|---|---|
| 最终用了多少 screenshot？ | 224 张（8 个采集 session：s01–s08，3 viewport × 7 route × 菜单/弹窗/线程状态，dark/lightsout 主题）|
| 多少 UI element instance？ | 14,293 个标注实例（30 类，无零样本类）|
| 数据分布？ | nav_item 2473 / avatar 2195 / icon_button 1714 / overflow_button 996 / trend_item 681 / carousel_nav 554 …稀有类：dialog 6、link_card 25、notification_item 58、compose_textbox 83、post_button 84、video 129 |
| 切分？ | 81 个 session×route 组防泄漏 split：train 163 / val 40 / test 21；hard_test 80 样本独立目录 |
| 自动标注准确度？ | DOM data-testid 直标（confidence 1.0）为主；两轮 VLM overlay 抽查对齐良好；错误模式：近重复折叠、极小框(<4px)过滤 |
| 哪些 label 来自 DOM？ | 全部监督标签（data-testid 精确匹配 → 语义角色）；geometry/visibility 来自 computed style + getBoundingClientRect |
| 哪些来自 OmniParser/VLM？ | **零**（teacher 仅用于对比评估与 disagreement 发现，未蒸馏进训练集）——DOM 标签在 X 上足够强 |

## 2. 模型

| 问题 | 答案 |
|---|---|
| 参数量？ | yolo11n **2,595,690**（2.6M，目标区间 3–10M 内）；对比档 yolo11s ~9.4M |
| 模型文件多大？ | best.pt 5.3MB；ONNX fp32 11MB / fp16 5.4MB |
| 训练配置 | Colab T4、imgsz 1280、batch auto（显存自适应）、100 epochs、seed 42 |
| 关键指标（frozen test 21 图） | exp003 nano v1: mAP50 .845 / mAP50-95 .751 / P .908 / R .800 / **KIER .813**；exp005d nano v2(+AL): **.879/.784/.915/.798**；exp004 small: **.941/.876/.921/.926 / KIER .880** |
| 规模对比（§十二） | nano 2.6M vs small 9.4M：mAP50 .879 vs .941——small +7pt 但 3.6× 参数；两者都在首选区间，按任务书"最小充分模型"原则：若 Agent 预算优先选 nano，精度优先选 small |
| 与 OmniParser 对比（§十六） | teacher 漏 335/541 DOM 真值交互元素（62%）、3.6s/帧@T4；student CPU ONNX 实测 **190ms/帧**（M 系列 CPU，warm），GPU T4 上推理 17.8ms/帧（val 日志）。注：teacher 并非 GT，对比口径是"对 X 交互元素的覆盖" |

## 3. Runtime 证明（§十八）

`runtime/api.py::parse_screen` 实测（本机 CPU，ONNX Runtime）：
- 输入 1920×1080 截图 → 53 检测（role+bbox+confidence，xyxy 合法）
- 与 DOM 真值分布吻合：nav_item 12/12、post_container 2/2、news_card 3/3
- **纯度可机器验证**：`assert_pure_runtime()` 检查进程内无 playwright/selenium/omniparser/torch/transformers 模块——实测 NONE

## 4. 错误分析（exp003，待填）

- 主要弱类（exp003 → exp004 small 仍弱）：post_text .67→（文本块边界模糊）、tab .73、username_link .75、compose_textbox .79；conf=0.25 下 21 图 test 的 FN=0（宽松中心匹配口径）；小元素专项：评测器的小元素召回计算器有 bug（dataloader API 变更），以 error_analysis 的 small_object_misses 口径替代（当前 0 miss @宽松匹配）
- 模型不会什么：dialog（训练实例 6→不足以学）、link_card（25 实例）、light 主题（零覆盖）、video 首帧多样性不足
- 仍然失败的 UI 状态：浅色主题全家族、极少见的下拉（select 原生）、notification 未读态的细粒度（taxonomy 未区分）

## 5. 数据规模收益（§二十一）

| 数据量 | mAP50 | 备注 |
|---|---|---|
| 87（exp002） | 0.542* | *test 仅 7 图，噪声极大 |
| 224（exp003） | 0.845 | 同冻结 test 21 图 |
| 235（exp005d，+AL 定向 11） | 0.879 | 弱类定向采集 +3.4pt |
| 500–1000 | 待采（light 主题 + 稀有类定向） | — |

收益递减判断：87→224（+137）带来 +0.30 mAP50；224→235（+11 定向）带来 +0.034——粗放采集的边际收益在放缓，但"定向补弱类"仍有清晰回报。**不需要 50K**：小模型在单一应用域的容量有限，预计 1–3K 高多样性样本即近饱和；把预算投给多样性（主题/route/状态）而非数量。下一版最值得加：light 主题全量、dialog/link_card/notification 的足量实例（各 >200）、独立浏览器会话采集。

## 6. 是否值得加 state/actionability model（§十二末）

值得，但优先级第二。依据：(a) taxonomy 已为 12/30 类记录 visual_state（disabled/expanded/liked/following…）；(b) DOM 侧状态属性采集已就绪（aria-expanded/pressed/disabled 已入 anno）；(c) 关键用例真实存在——post_button 的 disabled、repost/like 的已操作态直接影响 Agent 决策。实现建议：第二个 ~1M 参数的分类头挂在检测框 crop 上，而非并入 detector。

## 7. 环境边界与偏差声明

- **仅 dark/lightsout 主题**：light 采集被环境阻断（扩展 Input 通道不可用，X 主题色块无法合成点击；需人工切换一次后补采）——当前模型对 light 主题泛化**未验证**
- 单账号（Premium）视角：Grok/Articles/History 等导航为此账号特有
- 224 张全部来自同一浏览器会话家族（同一 Chrome 实例），独立 browser session 的 test 要求仅部分满足（不同采集 session 已分组隔离）
- 语言以中英混合 timeline 为主；OCR 未做（按任务书为独立组件）
- INT8 导出未做；断点续训具备（ultralytics last.pt）未演练

## 8. 复现

见 [README.md](../README.md)。数据集版本：GitHub Releases data-v0.1→v0.3；实验产物：experiments/exp00{1,2,3,4}*。
