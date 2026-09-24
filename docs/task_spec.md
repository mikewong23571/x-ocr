# Goal

完成一个端到端实验：

**利用 X（Twitter）Web 页面自动产生视觉训练数据，训练一个仅依赖 screenshot、无需 DOM / Playwright / OmniParser / VLM 的小型 UI perception 模型，并通过独立测试集证明它在 X Desktop Web 上能够稳定识别 Agent 需要交互的关键 UI 元素。**

最终目标不是构造一个通用 OmniParser，也不是训练一个新的 VLM。

最终产物必须是：

**一个 X-specific、pure-vision、small model。**

期望模型规模优先控制在：

- 首选：3M–10M parameters
- 可接受：<20M parameters
- 如果超过 20M，必须通过实验说明更小模型为什么明显不够

最终 inference：

Screenshot → Small Model → structured UI elements

运行时禁止依赖：

DOM、HTML、CSS、Accessibility Tree、Playwright、Browser DevTools、OmniParser、大语言模型、VLM。

这些工具仅允许在 training/data-generation 阶段作为 privileged information / teacher 使用。

---

# 一、背景与核心假设

传统通用 GUI Agent 通常使用：

Screenshot → OmniParser / VLM → UI elements → LLM → action

这种方式具有较强泛化能力，但对于固定产品 X 而言可能存在大量重复计算。

X 是一个强约束环境：

- 页面组件集合有限；
- UI design system 相对固定；
- interaction primitive 有限；
- navigation/layout 有明显规律；
- 同一类 button/icon/component 会持续重复出现。

因此核心假设是：

> 对于 X 这样一个固定应用，完全可以在训练阶段利用 browser/DOM/OmniParser 等昂贵信息自动产生大量监督数据，再把这些知识压缩进一个数百万参数的视觉模型。

训练阶段允许作弊：

Screenshot + DOM + browser geometry + accessibility + OmniParser + heuristic + action trace

↓

Dataset

↓

Small Model

运行阶段：

Screenshot

↓

Small Model

↓

UI state representation

这是一项“把运行时智能蒸馏进专用视觉模型”的实验。

---

# 二、你需要自主完成整个实验

请把当前任务视为 Long Horizon Task。

不要只输出方案。

需要实际：

研究现有实现 → 创建工程 → 编写采集器 → 运行采集 → 生成数据集 → 可视化检查 → 清洗 → 划分数据集 → 创建 Colab 训练入口 → 训练模型 → 评测 → 分析错误 → 自动补充 hard cases → 再训练 → 输出最终报告。

除以下真正需要用户权限的情况外，不要中途要求确认：

1. X 登录需要人工认证；
2. Google/Colab/Drive 登录需要人工认证；
3. CAPTCHA 或网站明确要求人工交互；
4. 缺少必须由用户提供的 credential。

遇到工程问题、依赖问题、模型选择问题、目录设计问题、训练参数问题，请自行调查、实验、修复并继续。

不要绕过 CAPTCHA、认证、安全机制或网站访问限制。

不要执行 Like、Repost、Follow、Post、Send DM 等会影响真实账号/其他用户的不可逆或公开动作。

允许执行无副作用操作，例如：

- navigation；
- scroll；
- open/close menu；
- open/close dialog；
- switch tabs；
- hover；
- focus input；
- 输入临时文本但不提交，然后清除。

---

# 三、首先研究而不是盲目实现

开始前检查并理解当前最新版：

- ScreenParse
- Webshot
- ScreenParser
- ScreenVLM
- Microsoft OmniParser

重点研究 Webshot 的：

browser screenshot → DOM-derived annotation → filtering → YOLO export

不要机械 fork 整个项目。

判断哪些组件可以直接复用，哪些需要针对 X 重写。

将调查结论记录到：

docs/research.md

其中明确：

- 可复用的 Webshot 能力；
- OmniParser 适合承担什么 teacher 角色；
- browser DOM 能提供哪些 strong labels；
- 哪些视觉状态无法从 DOM 直接获得；
- 最终选择什么 student architecture；
- 许可/license 对实验和未来发布有什么影响。

---

# 四、实验范围

第一版严格限制为：

X Desktop Web

不要一开始支持：

- Android
- iOS
- Native App

推荐覆盖多个 Desktop viewport，例如：

1280×720  
1440×900  
1920×1080

同时覆盖：

Light Theme  
Dark Theme

不要因为支持过多平台导致问题空间失控。

---

# 五、设计 X-specific taxonomy

不要直接继承 ScreenParse 的 55 classes。

先实际观察 X UI，然后设计一个尽可能小、Agent-oriented 的 taxonomy。

目标约：

15–30 个 semantic roles。

应该重点覆盖类似：

navigation item  
search input  
compose textbox  
primary button  
secondary button  
post container  
avatar  
profile link  
overflow/menu button  
reply action  
repost action  
like action  
bookmark action  
share action  
media image  
video  
link card  
tab  
dialog  
close button  
menu item  
dropdown  
checkbox/toggle  
notification item  
conversation item

这只是候选集合。

必须根据实际 X 页面分析后确定最终 taxonomy。

原则：

**只区分会影响 Agent action selection 的类别。**

不要仅因为视觉上不同就建立新 class。

需要输出：

schema/ui_taxonomy.yaml

并说明每个类别：

- semantic definition
- positive examples
- negative examples
- 是否 interactive
- 是否需要 OCR
- 是否存在 visual state

---

# 六、数据生产

构造自动化浏览器 collector。

每个 sample 至少保存：

screenshot  
URL route category  
viewport  
theme  
timestamp  
DOM-derived elements  
element bounding boxes  
semantic role  
visibility  
enabled/disabled  
accessible name（若存在）  
text（若存在）  
annotation source  
annotation confidence

如果能够安全获取，还记录：

hovered  
focused  
selected  
expanded/collapsed  
pressed/unpressed

原始 DOM/HTML 可以用于训练数据构造，但不得成为最终模型输入。

---

# 七、不要简单把 DOM 当真值

DOM 是 privileged information，但它也有缺陷：

- bounding box 可能包含不可见区域；
- 元素可能被遮挡；
- clickable parent/child 可能重复；
- accessibility role 可能错误或缺失；
- SVG/icon 语义不一定清楚；
- virtualized UI 可能产生特殊问题。

因此建立 Label Fusion Pipeline。

优先级大致：

browser geometry / visibility
+
DOM / accessibility semantics
+
OmniParser
+
必要时 VLM semantic relabel
+
heuristics
+
action result

产生最终 fused annotation。

每个 annotation 必须记录 source provenance。

不要默默覆盖冲突。

把冲突样本保存为：

data/disagreement/

后续重点评测。

---

# 八、OmniParser 的角色

OmniParser 是 teacher / weak labeler，而不是 ground truth。

使用它发现：

- DOM 没覆盖到的视觉元素；
- icon；
- visually clickable region；
- browser metadata 无法明确表达的 element。

需要定量分析：

DOM labels 和 OmniParser labels 的：

match  
DOM-only  
OmniParser-only  
conflict

不要把 OmniParser 的错误原样蒸馏进 student。

---

# 九、数据规模

采用渐进式策略。

Stage 0：

500–1,000 screenshots

只验证：

采集 → annotation → visualization → dataset export

必须人工可视化生成至少一批 overlay 图。

Stage 1：

5,000–8,000 screenshots

训练第一个 baseline。

只有当 baseline 明显能够学习 X UI 后，再继续扩展。

Stage 2：

目标 20,000–30,000 个高多样性 screenshots。

重点是 state coverage，而不是 frame count。

避免连续 scroll 产生大量近重复 screenshot。

使用：

perceptual hash  
image similarity  
DOM/layout signature

进行 near-duplicate elimination。

如果 20K 后 validation improvement 已明显饱和，不要为了达到数字继续采集。

如果 hard-case recall 仍明显提升，可以扩展到 50K。

---

# 十、必须特别设计状态覆盖

建立 coverage matrix。

至少覆盖主要 route，例如：

Home  
Explore/Search  
Notifications  
Profile  
Post detail/thread  
Lists/Bookmarks（若账号可访问）  
Settings 中安全可浏览页面

对于每个 route，尽可能覆盖：

empty state  
normal state  
long content  
short content  
text-only post  
image post  
multiple-image post  
video post  
link card  
quoted post  
thread  
menu opened  
dialog opened  
disabled/enabled controls  
hover/focus  
loading/skeleton  
error/empty state（如果自然出现）

不要为了制造数据执行真实公开动作。

---

# 十一、数据集切分非常重要

禁止随机逐截图 split。

否则连续截图会造成严重 leakage。

按：

session  
route-state group  
content thread  
collection run

进行 group split。

推荐：

70% train  
15% validation  
15% test

Test set 必须尽可能包含：

未出现在 train 中的帖子内容；
独立 browser session；
独立 collection run；
不同 UI state combination。

另外创建：

hard_test/

专门保存：

popup  
occlusion  
dense feed  
unusual media  
long text  
edge viewport  
loading state  
rare menu

最终所有模型都必须在同一个 frozen test set 上比较。

---

# 十二、Small Model

首选训练一个 object detector / lightweight screen parser。

不要第一版就训练 300M VLM。

优先比较至少两个轻量级规模，例如：

nano/tiny  
small

选择当前维护良好、Colab 容易训练、可导出 ONNX 的 architecture。

如果模型 license 对未来闭源/商业使用存在限制，要在报告里明确说明，并尽量再提供一个许可更宽松的替代实验。

主要模型输入：

RGB screenshot

主要输出：

bbox  
semantic role  
confidence

OCR 暂时允许作为独立 runtime component。

不要要求 detector 自己识别所有文字。

如果数据充分，可以增加第二个极小模型预测：

enabled  
selected  
expanded  
interactive/actionable

但必须先把 element detection 做好。

---

# 十三、Colab

生成：

notebooks/train_colab.ipynb

要求 notebook 能够从零运行：

install dependencies  
load dataset  
validate dataset  
train  
evaluate  
export model  
produce plots  
save checkpoint

适配常见 Colab GPU。

训练分辨率优先从：

1024 或 1280

开始。

不要默认 640，因为 X 页面上存在大量小 icon。

根据显存自动调整 batch size。

支持断点续训。

保存：

best checkpoint  
ONNX  
FP16 export  
如果可行再提供 INT8 export

---

# 十四、第一版训练后必须主动找错误

不能训练一次就结束。

完成 baseline 后自动生成：

false positives  
false negatives  
low-confidence positives  
class confusion  
small-object failures  
occlusion failures

每类至少生成可视化 montage。

分析：

模型究竟不会什么？

然后进行一次 active-learning / hard-case loop：

Model v1
→ 在未标注 X screenshots 上推理
→ 找 low confidence / disagreement / rare states
→ collector 定向收集
→ teacher pipeline 标注
→ 加入训练数据
→ Model v2

至少完成一次这样的闭环。

---

# 十五、核心指标

不要只报告 overall mAP。

至少报告：

mAP50  
mAP50-95  
precision  
recall  
per-class AP  
per-class recall  
small-element recall

另外定义：

Key Interactive Element Recall

对于 Agent 关键组件，例如：

compose  
send/post control  
menu  
reply  
repost  
like  
search  
navigation  
dialog controls

重点关注 recall。

实验目标：

Key Interactive Element Recall 尽量达到 ≥95%。

这个值是实验目标，不允许为了达到目标修改 test set。

若达不到，分析为什么。

---

# 十六、和 Teacher 比较

在 frozen test set 上比较：

Small Model  
OmniParser  
如果可运行则加入 ScreenParser

比较：

element recall  
bbox accuracy  
semantic class accuracy  
latency  
model size  
GPU memory  
CPU inference feasibility

Teacher 不是 ground truth。

Ground truth 来源于 fused annotation + 独立检查。

目标不是证明 student 全面优于 OmniParser。

要回答：

> 在 X 这个受限 domain 内，小模型能否以显著更低计算成本获得接近或超过通用 parser 的关键 UI coverage？

---

# 十七、模型规模目标

最终报告必须列出：

parameter count  
checkpoint size  
ONNX size  
FP16 size  
INT8 size（若成功）  
FLOPs/MACs（可计算时）  
inference latency

优先选择满足需求的最小模型。

如果：

5M model = 95% key recall  
15M model = 96%

优先讨论 5M model 的系统价值，而不是机械选择最大模型。

---

# 十八、运行时最终 API

提供一个简单 API：

parse_screen(image) -> elements

返回结构类似：

[
  {
    "role": "like_button",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.98
  }
]

如果有 state head：

{
  "role": "menu_button",
  "bbox": [...],
  "confidence": 0.99,
  "state": {
    "interactive": true,
    "enabled": true,
    "expanded": false
  }
}

运行该 API 时必须能够证明：

没有访问 DOM；
没有启动 browser；
没有调用 OmniParser；
没有调用 VLM/LLM。

---

# 十九、工程结构

建立清晰项目结构，大致包括：

collector/  
annotation/  
teacher/  
dataset/  
training/  
evaluation/  
runtime/  
notebooks/  
configs/  
scripts/  
tests/  
docs/

不要把所有逻辑写进一个 notebook。

Notebook 主要承担 Colab training entry point。

数据生产和评测必须可以通过 CLI 重复执行。

---

# 二十、实验可复现性

固定：

random seeds  
dataset manifest  
train/val/test split  
model config  
training config

所有实验写入：

experiments/

例如：

exp001_baseline_nano  
exp002_small  
exp003_nano_1280  
exp004_active_learning

每个 experiment 保存：

config  
metrics  
checkpoint reference  
git commit  
dataset version

---

# 二十一、最终交付

任务结束时必须真正留下可以运行的产物，而不是只有文档。

至少包括：

1. 自动数据采集 pipeline
2. 自动/弱监督 annotation pipeline
3. X-specific taxonomy
4. dataset manifest
5. train/val/test split
6. 数据可视化工具
7. Colab training notebook
8. 至少一个训练完成的小模型 checkpoint
9. ONNX 导出
10. runtime inference API
11. frozen test set evaluator
12. error analysis visualization
13. active-learning loop
14. 完整实验报告
15. README：从零复现实验的方法

最终报告：

docs/final_report.md

必须明确回答：

- 最终用了多少 screenshot？
- 有多少 UI element instance？
- 数据分布如何？
- 自动标注准确度如何？
- 哪些 label 来自 DOM？
- 哪些来自 OmniParser/VLM？
- 最终模型参数量多少？
- 最终模型文件多少 MB？
- X-specific model 与 OmniParser 相比如何？
- 哪些 UI state 仍然失败？
- 从 5K → 10K → 20K 数据时性能增长如何？
- 数据是否已经出现收益递减？
- 是否真的有必要继续扩到 50K？
- 下一版最值得增加什么数据？
- 是否值得增加 state/actionability model？

---

# 二十二、决策原则

始终遵循：

数据质量 > 数据数量

state coverage > 连续帧数量

真实 held-out evaluation > training metric

smallest sufficient model > biggest available model

strong browser labels > weak teacher labels

domain specialization > generality

最终目标不是论文指标。

最终目标是验证这个命题：

> 一个运行时只看 screenshot、参数规模约几百万到一两千万的专用视觉模型，是否足以替代 X 场景中大量昂贵的通用 GUI parsing。

如果实验事实否定这个假设，也必须如实报告。

不要为了证明假设成立而修改数据或指标。

---

# 二十三、执行方式

现在直接开始执行。

先完成 repository inspection 和 research，然后建立最小 pipeline，并尽快跑通：

100 screenshots
→ annotation
→ visualization
→ YOLO-format dataset
→ tiny model smoke training
→ inference

只有这条链跑通之后再扩大数据。

每完成一个阶段：

运行测试；
检查生成物；
记录实验；
commit 当前稳定状态；
然后继续下一阶段。

不要停留在计划阶段。
