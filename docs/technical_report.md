# X-OCR 技术报告：把运行时智能蒸馏进 2.6M 参数的纯视觉 UI 感知模型

> 版本 v1.1 · 2026-09-24（+清洗飞轮/提取器v2/DOM校准/屏幕文档/exp010） · 仓库 [mikewong23571/x-ocr](https://github.com/mikewong23571/x-ocr)
> 相关文档：[任务书](task_spec.md) · [验收台账](goal.md) · [调研](research.md) · [系统边界](system_boundary.md) · [实验报告](final_report.md)

## 摘要

我们验证了一个命题：**对于 X（Twitter）这样的单一应用，可以在训练阶段利用 DOM/浏览器等特权信息自动生产监督数据，把"这个像素块是什么、能怎么操作"的知识压缩进一个 2.6M–9.4M 参数的纯视觉检测器；运行时只吃 screenshot（190ms/帧 @CPU），不碰 DOM、不开浏览器、不调 VLM**。

**结果**：在 681 张自采集截图（暗/亮双主题 × 登录/未登录 × 3–4 种 viewport × 7 路由 × 菜单/弹窗/线程/错误态）上训练 YOLO11n（2.6M，32 类语义角色），冻结测试集 mAP50 **0.930**，关键交互元素召回 **95.9%**；换 9.4M 的 YOLO11s 达 **0.947–0.954**。对照组 OmniParser v1.5 漏掉 62% 的 DOM 真值交互元素且需 T4 GPU 3.6s/帧。模型对未训练过的第三方 App 截图表现出非受控的语义迁移（头像/操作行等检出正确）。全链路（含按需 OCR 的内容选靶、视觉坐标点击、像素级动作验证）已打通并开源。

---

## 1 背景与动机

通用 GUI Agent 的感知层通常是 `Screenshot → OmniParser/VLM → UI elements`，对固定产品存在大量重复计算，且每帧都要付出通用模型的算力与延迟。X 是强约束环境：组件集合有限、设计系统固定、同一交互原语反复出现——**适合把感知知识"预蒸馏"进专用小模型**。

训练阶段允许作弊（DOM/几何/无障碍树都是特权信息），运行时只留 screenshot：

```
训练期（作弊）: screenshot + DOM + geometry ──► 数据集 ──► 2.6M 检测器
运行期（纯净）: screenshot ──190ms──► [{role, bbox, confidence}, ...]
```

## 2 系统架构

```
┌─ 数据引擎 ──────────────────────────────────────────────┐
│ 用户真实 Chrome（Playwright 扩展中继，含登录态）            │
│ + 本地 playwright 浏览器（未登录态，独立 session）          │
│ → DOM 语义标注（data-testid/aria/启发式）                 │
│ → 融合（裁剪/同角色去重/感知哈希近重复消除/provenance）      │
│ → group-split 防泄漏 + frozen test                        │
├─ 训练（全部 Colab A100）────────────────────────────────┤
│ YOLO11n(2.6M) / YOLO11s(9.4M) @1280 → ONNX fp32/fp16    │
├─ 运行时（纯 ONNX）──────────────────────────────────────┤
│ parse_screen(image) → 元素列表（纯度可机器证明）            │
│ verify_parse / verify_state_change（结构自检+像素验证）    │
├─ 应用层 ────────────────────────────────────────────────┤
│ 内容选靶(检测分组+crop OCR) → 视觉坐标点击 → 像素验证闭环   │
└──────────────────────────────────────────────────────────┘
```

## 3 数据引擎

### 3.1 采集
- **登录态**：通过 Playwright Extension（协议 v2）attach 用户日常 Chrome，复用真实登录态与 Premium 功能；23 个采集 session（s01–s22 + loggedout_s01）
- **未登录态**：独立 playwright 浏览器 + 专用 profile（不污染登录态），覆盖登录墙/空态/骨架屏/错误页
- **状态矩阵**：home/explore/notifications/profile/grok/search/status × 滚动序列 × 菜单（More/账号/帖内⋯/转发）× 弹窗（发帖框）× 1280/1440/1920(/1680) viewport × 暗色/亮色主题
- **安全约束**：只做导航/滚动/开关菜单弹窗；不执行 like/repost/follow/post 等公开动作；主题切换由用户经 RustDesk 远程手工完成（自动化通道见 §8）
- **去重**：采集时 + 融合时双层感知哈希（phash，Hamming≤8，按 session×route 分桶）

### 3.2 标注（Label Fusion v1）
单一真相源 = DOM 语义 + 浏览器几何，每个标注带 provenance 与置信度：

| 来源 | 置信度 | 覆盖 |
|---|---|---|
| `data-testid` 精确匹配 | 1.0 | 27/32 类（reply/retweet/like/bookmark/tweet/nav…） |
| aria/role | 0.9 | share 按钮、menu_item、dialog |
| 结构/文本启发式 | 0.7 | icon_button 兜底、登录墙输入框与继续按钮 |

可见性判定继承 Webshot 管线的全套 computed-style 检查（display/opacity/transform/clip/祖先 overflow 裁剪/视口边界）。

### 3.3 Taxonomy（32 类，agent-oriented）
**设计原则：只区分影响 Agent action selection 的类别，纯视觉差异不建类。** 每类含语义定义、DOM 定位规则、interactive/ocr/visual_state 三属性。分组：导航(3) · 输入(3) · 帖子结构(8) · 帖子操作(6) · 通用控件(4) · 列表内容(6) · 结构层(1) · 登录态(2)。完整定义见 [schema/ui_taxonomy.yaml](../schema/ui_taxonomy.yaml)。

### 3.4 数据集
| 版本 | 样本 | 元素 | 说明 |
|---|---|---|---|
| v0.1–0.2 | 27–87 | 3.8K–5.8K | 冒烟 + 修复菜单原子采集 |
| v0.3 | 224 | 14,293 | 4-session 战役 |
| v0.4 | 235 | 14,870 | active-learning 定向 +11 |
| v0.5–0.7 | 376→**511** | 32,619 | Stage 0 数量门跨过 |
| v0.8-light | 645 | 40,914 | +134 亮色（用户手工切主题后） |
| **v0.9-loggedout** | **681** | **41,190** | +36 未登录态，taxonomy 32 类 |
| **v1.0-textspans** | **826** | **47,778** | +145 提取器v2样本（行级text_spans+层级索引） |

**切分**：227 个 session×route 组防泄漏；**train 620 / val 40 / test 21，test 自 v0.3 冻结**（所有模型严格同 test 可比，`--freeze-from` 机制保证）；hard_test 121 样本独立目录；disagreement 7 样本（teacher 对比产物）。

## 4 训练（全部 Colab，A100 优先）

- 框架 Ultralytics YOLO11 @ imgsz 1280，batch 显存自适应，seed 42，100 epochs
- 基建：数据集经 GitHub Release 中继（VM 内 1.0s 拉取，本机零重复流量）；后期改**本地 zip 直传 + 数据集版本写入 metrics.json**（根治了"Release 旧资产 + 确定性训练 = 逐位相同指标"的复现陷阱）；A100 训练用 VM 侧 nohup + 服务器侧 watch 抗会话回收

| 实验 | 模型 | 数据 | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|---|
| exp002 | nano | 87 图 | 0.542* | 0.466 | 0.77 | 0.50 |
| exp003 | nano | 224 | 0.845 | 0.751 | 0.91 | 0.80 |
| exp005d | nano | +11 定向 | 0.879 | 0.784 | 0.92 | 0.80 |
| exp006/007 | nano/small | 511 | 0.933 / 0.947–0.954† | 0.860 / 0.876–0.913 | — | — |
| exp008 | nano | 645(+light) | 0.932 | 0.872 | 0.94 | — |
| **exp009** | **nano** | **681, 32 类** | **0.930** | **0.866** | 0.93 | 0.90 |
| **exp010** | **nano** | **826 (v1.0)** | **0.935** | **0.876** | 0.91 | **0.93** |

\* test 仅 7 图，噪声大 † small 两次独立训练复现
规模爬坡曲线：**87→0.542 · 224→0.845 · 235→0.879 · 511→0.933**（+137 图≈+0.30；其后每翻倍约 +0.03–0.09，收益递减明确，**无需扩到 50K**）。

## 5 评测

- **冻结 test**（21 图，暗色，从未参与训练）：见上表
- **关键交互元素**（15 类，实例级召回）：**95.9%**（813/848）；屏级存在召回 96.0%；KIER(mean AP50-95 口径) small@224 = 0.880
- **主题泛化**（40 张亮色图前后对照）：暗色模型零样本 **2.4%** → 含亮色训练后 **91.9%**（+89.5pt，in-train 口径）——同时暗色冻结 test 无损，证明模型原本学的是主题相关外观而非纯布局
- **Teacher 对比**（7 张 test）：OmniParser v1.5（icon_detect）漏 **335/541（62%）** DOM 真值交互元素、**3578ms/帧@T4**；本模型 190ms/帧@CPU、17.8ms/帧@T4。ScreenParser（YOLO11-L，通用 web 55 类）无 X 专属语义未纳入
- **错误分析**：弱类集中在 dialog（6 实例）/link_card（25）/notification_item（58）；conf=0.25 下冻结 test FN=0（宽松中心匹配）
- **域外抽样**（非受控）：第三方 App 竖屏截图上 avatar×3、reply/repost/bookmark/share 操作行、post_text 检出正确（VLM 逐项核对）——语义视觉特征发生了迁移

## 6 运行时（纯度可证明）

```python
from runtime.api import parse_screen
parse_screen("shot.png")
# → [{"role": "like_button", "bbox": [x1,y1,x2,y2], "confidence": 0.98}, ...]
```

- **190ms/帧（CPU, M 系, warm）**；ONNX fp32 11MB / fp16 5.4MB；2,595,690 参数；26.3 GFLOPs
- `assert_pure_runtime()` 机器验证进程内**零禁运模块**（playwright/selenium/omniparser/torch/transformers）
- `verify_parse` 结构不变式（曾当场抓到 6/8 图框越界真 bug → 修复为输出钳位后 8/8 通过）
- `verify_state_change` 像素差分动作验证（菜单区域 0.38 vs 静态区域 0.12 可分离）

## 6.5 清洗飞轮与运行时增强（v1.1 新增）

- **清洗飞轮**（`scripts/cleaning_flywheel.py`）：全库 681 张 54 秒跑完，产出 715 个 model-only（提取器规则盲区候选：icon_button 171 / video 77 / post_media 70）与 2,199 个 dom-only（硬样本清单，最弱类 menu_item 召回 0.898）——disagreement 自动分流，替代人工全库审查
- **提取器 v2**：行级文本框（Range.getClientRects，~60 span/屏）+ 元素层级索引，为新类（text_line）与 OCR 对齐数据铺路
- **DOM 校准运行时**（`runtime/calibrate.py`）：混合模式下框吸附（实测 IoU 0.96 基础）+ FP 抑制 + 类别仲裁；作者名对比：纯视觉 OCR 读成 "Gazi √ aziozmn1"（认证徽章幻觉为 √），DOM 校准给 "颜探长 @laoyan86 · 6h"（精确）
- **结构化屏幕文档**（`runtime/screen_doc.py` + demo `/screendoc` 端点）：检测→几何分组→按需 OCR→~500 token 屏幕文档，双模式（纯视觉/混合），直接作为 LLM 决策上下文

## 7 应用层：内容条件化操作闭环

**"给内容提到 X 的帖子点赞"** 的完整链路（全部已实测，dry-run）：

```
① parse_screen → post_container/post_text/like_button 坐标
② 纯几何分组：like 的 y 中心落在容器 span 内 ⇒ 归属该帖（零成本）
③ 仅对 post_text 框 crop → RapidOCR(ONNX) → 帖子文字
④ 关键词/LLM 匹配 ⇒ click(该帖 like 中心)；全不中 ⇒ 滚动重试
⑤ 点击后 verify_state_change(同框 before/after) 确认生效
```

实测例：OCR 正确读出「感觉AI没有护城河…」并给出 `click(910,622)`。视觉坐标点击（`page.mouse.click`，零选择器）在本地 playwright 浏览器实测通过：视觉定位登录输入框→真点击→输入→像素验证 19.2% 变化确认。

## 8 工程边界与踩坑记录（复现者必读）

1. **扩展中继的 Input 通道不可用**：CDP 鼠标/键盘事件 0 到达（监听器实证）；X 主题色块对合成 click/KeyboardEvent 免疫；主题为账号级服务端设置（无 cookie/localStorage）。⇒ 一切交互走页面内 eval + 单次 async eval 原子化菜单采集；真实输入用本地 playwright 浏览器
2. **协议版本匹配**：CLI 的 playwright 协议 v1 vs 扩展 v2 → 连接超时（公开 issue 同因）；pin `@playwright/cli@0.1.21`
3. **Release 旧资产陷阱**：VM 从 Release 拉到旧数据集 + 确定性训练 = 与旧实验逐位相同的"假结果"；解法=本地 zip 直传 + metrics.json 内嵌数据集版本
4. **A100 会话回收**：改 VM 侧 nohup + 服务器侧 watch
5. **主题标志 bug**：`prefers-color-scheme` 反映系统而非 X 应用主题 → 改按 body 背景亮度判定

## 9 局限与路线图

| 项 | 现状 | 计划 |
|---|---|---|
| state head（enabled/liked/expanded） | 未做（DOM 侧状态属性已随数据采集） | **v2 首选**：~1M 分类头挂检测框 crop，对 Agent 决策价值 > 再+1% 召回 |
| 稀有类 | dialog 6 / link_card 25 / notification 58 实例 | 定向补采各 200+ |
| held-out 亮色测试组 | 亮色 probe 为 in-train 口径（已注明） | 独立亮色 test 组 |
| INT8 / YOLOX(Apache) 对照 / ScreenParser 基线 | 未做（任务书"如果可行"级） | 按商用需求决定 |
| 小元素召回计算器 | dataloader API 变更致损坏，以错误分析口径替代 | 修复 |
| 99.9% | 不可达（标注噪声天花板 ~98% + 视口切割物理不可见） | Agent 层滚动重试即 ~100% 操作成功率 |

## 10 复现

```bash
git clone https://github.com/mikewong23571/x-ocr && cd x-ocr
# 完整流程见 README.md：采集→融合→导出→Colab 训练→评测→runtime
./scripts/colab_train.sh exp009_nano_32class_a100 100 data/yolo yolo11n.pt 1280
.venv/bin/python -c "from runtime.api import parse_screen; print(parse_screen('shot.png'))"
```

数据集版本链：GitHub Releases `data-v0.1 → v0.9-loggedout`（含 YOLO 格式与导出元数据）。

## 附录：关键文件索引

| 组件 | 路径 |
|---|---|
| 采集器（扩展会话/未登录） | `collector/collect.py` · `collector/collect_loggedout.py` |
| 标注生成与融合 | `collector/build_extract_js.py` · `annotation/fuse.py` |
| Taxonomy | `schema/ui_taxonomy.yaml`（32 类） |
| 数据集导出（frozen split） | `dataset/export_yolo.py` |
| 训练/评测/错误分析 | `training/train_yolo.py` · `evaluation/eval_frozen.py` · `evaluation/error_analysis.py` |
| Teacher 对比 | `teacher/omniparser_compare.py` |
| 运行时 API + 验证 | `runtime/api.py` · `runtime/verify.py` |
| 内容选靶/视觉点击/演示 | `scripts/like_by_content.py` · `scripts/vision_automation_demo.py` · `scripts/demo_server.py` |
| 实验记录 | `experiments/exp00{1..9}*`（metrics 内嵌数据集版本） |
