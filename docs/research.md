# Research：现有实现调研（docs/research.md）

日期：2026-09-24。所有结论基于官方仓库/论文/HF 页面 + 本地 clone 源码阅读（`third_party/screenparse`）。

## 0. 项目版图（五个名字是一家人 + 一个对手）

| 名字 | 是什么 | 规模 | License |
|---|---|---|---|
| **Webshot** | 自动化数据管线：URL 采样 → Playwright 渲染 → DOM 密集标注 → VLM 重标注/过滤 → 去重 → 导出 | 产出 771K(v1)/1.45M(v2) 截图、21M+/25.5M+ 元素标注 | 代码 MIT（repo 根 LICENSE） |
| **ScreenParse** | Webshot 产出的数据集 + benchmark（ICML 2026, IBM–ETH/docling-project） | 55 类 | 论文 CC-BY-4.0；数据集 HF `docling-project/screenparse` |
| **ScreenParser** | 在 ScreenParse v2 上微调的 **YOLO11-L** 检测器（HF `docling-project/ScreenParser`） | YOLO11-L @1280px，无文本 | **Apache-2.0** |
| **ScreenVLM** | SigLIP-2 视觉编码器 + Granite-165M 解码器（SmolDocling 系初始化） | 316M | HF docling-project |
| **OmniParser** (Microsoft) | 通用屏幕解析：icon 检测 + Florence-2 图标描述 + OCR | 最新 icon_detect_v3 = YOLOv9-E | repo CC-BY-4.0；**icon_detect_v3 权重基于 MIT 的 YOLOv9**；caption 模型 MIT；早期 ultralytics 版 icon 检测器 **AGPL** |

论文：*Moving Beyond Sparse Grounding with Complete Screen Parsing*（arXiv 2602.14276）。

## 1. Webshot 管线拆解（可复用性逐组件判定）

已 clone 源码逐文件阅读。管线：`crawl.py`(URL) → `worker.py`(渲染+DOM提取) → `labels.py`(粗分类) → `vlm_refine.py`(Qwen-3-VL-8B 重标注) → `vlm_quality_score.py`(VLM-as-judge 页面级过滤,阈值0.70) → `filtering.py`(叶子选择/包装器剪枝/去重) → `dedupe.py`(感知哈希近重复消除) → `yolo_export.py`(YOLO/COCO 导出+group split)。

### 直接复用（改造量小）
1. **`worker.py` 的 DOM 提取 JS**（约 160 行注入脚本）：`isVisible()` 综合检查（display/visibility/opacity/filter/transform scale(0)/max-height 塌陷/offset<-1000/clip-path/祖先 overflow-hidden 裁剪/视口边界）+ `getBoundingClientRect` 几何 + 全属性 + z-index + 父子层级索引 + shadow DOM 遍历 + text span（Range.getClientRects）。**这是本实验 DOM 侧标注的地基**，X 页面同样适用。
2. **`filtering.py` 的过滤规则**（993 行）：min box 4px²、视口覆盖、纯布局容器判定 `is_pure_layout_container`、交互指示 `has_interactive_indicators`、wrapper 剪枝 `_prune_wrapper_containers`、叶子选择 `get_leaf_elements`、IoU 0.95 重复删除、IoU 0.65+containment 类内清理。规则可直接搬。
3. **`dedupe.py`**：感知哈希（Hamming 半径 8）近重复消除——任务书要求同款。
4. **`yolo_export.py`**：类别映射、YOLO txt 格式、group-aware split `_choose_split`。
5. **`visualize.py`**：overlay 可视化。

### 必须为 X 重写
1. **导航模型**：Webshot 是 URL 爬虫（随机网站、每页一次采集）；X 是登录态 SPA，数据多样性来自 route×scroll×UI state 矩阵，需要专门的 navigation policy + 状态触发（开菜单/开弹窗/hover/主题切换/viewport 切换）。
2. **taxonomy**：Webshot 55 类是跨站通用设计系统分类（HIG/Material/Fluent）；X 有稳定 `data-testid` 属性（`tweet`/`like`/`retweet`/`reply`/`bookmark`/`SideNav_NewTweet_Button`/`cellInnerDiv` 等），可以做 **agent-oriented 的语义角色标注**，比 Webshot 的 tag/role 启发式强得多。不继承 55 类。
3. **登录态与浏览器控制**：走用户的真实 Chrome（Playwright Extension attach），非无头新 profile（X 拒绝无头 UA）。
4. **VLM 重标注**：第一版不需要——X 的 DOM 标签足够强（data-testid + aria）。VLM 只在 disagreement 复核时用（可选）。
5. **广告过滤**：Webshot 的大篇幅 ad 检测对 X 无关（X 有自己的 ad cell，作为 `post` 的一个变体状态即可）。

## 2. Browser DOM 能提供 / 不能提供

**能（strong labels）**：
- 精确 bbox（getBoundingClientRect，含 iframe 坐标映射）
- 可见性（上述 isVisible 全套）+ 遮挡祖先裁剪
- 语义：data-testid（X 最强信号）、ARIA role/name、tag、可点击性（button/a/role/onclick/tabindex）
- 文本内容 + 文本 span bbox（OCR 真值）
- 状态：disabled、aria-expanded、aria-selected、aria-checked、aria-pressed、focus（document.activeElement）、hover（注入 mousemove 记录 :hover 目标——可选）
- 层级/嵌套关系（父容器 vs 叶子）

**不能（DOM 盲区，需要 OmniParser/启发式/视觉侧补充）**：
- 纯视觉遮挡（z-index 高的 overlay 挡住内容，DOM 依然"可见"）→ 需要 z-index/遮挡交叉验证或视觉复核
- 图标的视觉语义（SVG 图标长什么样、视觉上像不像按钮）
- 图像/视频内容本身（media 的语义类型：动图/视频/多图布局）
- canvas 绘制内容（X 少见，Grok 相关页面可能有）
- skeleton/loading 态的视觉形态（DOM 能看到结构但"看起来像什么"要视觉确认）
- 虚拟列表视口外内容（X timeline 有虚拟化——只标注 DOM 内现存元素即可，但滚动会销毁/重建节点，需要按帧采集）
- 系统级弹层（浏览器 context menu、通知）

## 3. OmniParser 的角色（teacher，不是 ground truth）

用途：
1. **icon/可点击区域探测器**：发现 DOM 语义弱的地方（X 的 icon button 有 data-testid，其实覆盖较好——预期 OmniParser 的增量主要在视觉侧 sanity check 与 media/未知元素）。
2. **对比 baseline**：frozen test set 上比较 X-specific student vs 通用 teacher（任务书第十六节）。

定量分析计划：DOM labels × OmniParser labels 的 match / DOM-only / OP-only / conflict 矩阵；conflict 进 `data/disagreement/`。

版本选择：**icon_detect_v3（YOLOv9-E）**——基于 MIT 的 YOLOv9 实现，避开早期 ultralytics AGPL 权重；caption（Florence）MIT。OCR 用 PaddleOCR（Apache-2.0）独立组件。teacher 全部在 Colab 跑。

## 4. Student architecture 选型

约束：3–10M 参数（可接受 <20M）、Colab 可训、可导 ONNX/FP16/INT8、维护良好。

| 候选 | 参数量 | License | 判定 |
|---|---|---|---|
| **YOLO11n** (ultralytics) | ~2.6M | **AGPL-3.0** | 主实验 nano 档 ✅（工具链最好） |
| **YOLO11s** (ultralytics) | ~9.4M | AGPL-3.0 | 主实验 small 档 ✅ |
| **YOLOX-tiny/s** (Megvii 原版) | ~5M / ~9M | **Apache-2.0** | 宽松许可替代实验 ✅（任务书要求） |
| RT-DETR-r18 (原版 repo) | ~20M | Apache-2.0 | 边界规模，备选 |
| YOLO-NAS | - | 自定义限制许可 | ❌ |

结论：主实验 **YOLO11n/s @ 1024/1280**（ultralytics 生态：Colab 一键训、ONNX/FP16/INT8 导出、训练日志完善）；**license 风险**（AGPL-3.0 对闭源商用不友好）在最终报告明确说明，并补一组 **YOLOX-tiny/s（Apache-2.0）** 对照实验。检测器不负责读字——OCR 作为 runtime 独立组件。

## 5. 数据策略借鉴（来自 Webshot 的已验证经验）

- 渲染 settle（加载后等 ~800ms）再提取；禁用动画
- 视口级采集（不整页长截图）——与我们的 screenshot-per-state 一致
- 近重复消除：感知哈希 + IoU 类内清理（对连续 scroll 尤其重要）
- group-aware split（页面/域为组，防泄漏）——对应我们的 session/route-state group split
- VLM-as-judge 页面级质量分 → 我们的替代：DOM 标注完整度启发式（关键元素覆盖率）+ 人工抽查 overlay

## 6. 与本实验的差异声明

Webshot/ScreenParse 证明"DOM→密集标注→小模型"在**跨站通用**设定下成立（YOLO11-L@1280）。本实验问的是**单应用(X)、更小模型(3-10M)、更强语义角色(agent-oriented taxonomy)** 能否达到接近/超过通用 parser 的关键元素覆盖。不做通用泛化声明。

## 引用
- 论文: https://arxiv.org/abs/2602.14276 · 项目页: https://saidgurbuz.github.io/screenparse · 代码: https://github.com/Saidgurbuz/screenparse (MIT)
- ScreenParser: https://huggingface.co/docling-project/ScreenParser (Apache-2.0)
- OmniParser: https://github.com/microsoft/OmniParser
