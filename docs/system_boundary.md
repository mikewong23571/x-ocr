# 系统边界检查报告（2026-09-24）

实验全链路的外部依赖逐项验证结果。结论：**全部就绪**。

## 1. 浏览器自动化（数据采集）— ✅ 闭环

**方案**：复用用户日常 Chrome（含 X 登录态），通过 **Playwright Extension**（Chrome 应用商店版 v0.4.0，id `mmlmfjhmonkocbjadbfplnigmagldckm`，装于 Default profile）+ **playwright-cli `attach --extension=chrome`**。

**为什么不是别的**：
- ZCode browser-use 插件：本机只有 in-app browser backend，拿不到用户 Chrome 的登录态。
- Chrome 136+ 禁止对默认 profile 开 `--remote-debugging-port`，CDP 直连日常浏览器不可行。
- 无头 Playwright + 新 profile：x.com 直接拒绝无头 UA（实测 `ERR_HTTP_RESPONSE_CODE_FAILURE`），且无登录态。

**踩过的坑（记录避免复发）**：
- 用户 fork 的 playwright-cli（playwright 1.59-alpha）协议 v1，与扩展 0.4.0（只接受 v2）不匹配 → `Extension connection timeout`（GitHub 已知问题：CLI/MCP 版本与扩展版本不匹配）。
- 正确工具链：`@playwright/cli@0.1.21`（playwright 1.64-alpha，协议 v2），已 pin 在 `tools/pwcli/package.json`。
- 扩展 token 存于 connect.html 的 localStorage（`auth-token`），与 `PLAYWRIGHT_MCP_EXTENSION_TOKEN` 环境变量比对；token 已存 `.env.local`（不入库）。
- 官方命令是 `attach --extension=chrome`（不是 `open --extension`）。

**用法**：
```bash
./scripts/pw attach --extension=chrome   # 建立会话 xocr（会留一个 connect 标签页）
./scripts/pw goto https://x.com/home
./scripts/pw snapshot / eval / screenshot <path> / resize <w> <h> / tab-new / tab-list
```

**实测**：attach ✓、goto x.com/home ✓、snapshot 显示已登录侧边栏（Home/Notifications/DM/Grok/Profile/Post）✓。

**边界注意**：
- viewport 用 `resize` 直接改用户 Chrome 窗口尺寸（采集期间会可见地 resize，需告知用户）。
- 明暗主题：优先用 X 自带 Display 设置（Default/Dim/Lights out，可还原）；`run-code` 可能支持 `page.emulateMedia({colorScheme})`，采集器开发时验证。
- 不执行任何公开/不可逆动作（like/repost/follow/post/DM）。

## 2. Colab（全部训练任务）— ✅ 闭环

- `colab` CLI（google-colab-cli）已认证：oauth2，scopes 含 `colaboratory` + `drive.file`（email mikewong23571@gmail.com）。
- 用法要点（来自 `colab skill`）：`colab new -s <name> [--gpu T4]`、`colab exec -s <name> -f script.py`（kernel 状态跨调用持久）、`colab run`（一次性跑完自毁）、`colab upload/download`、**用完必须 `colab stop`**。
- GPU 配额未实测（分配时如 400/无配额则降级 T4 或 CPU，如实汇报）。
- **策略（用户指定）**：所有训练任务一律 Colab，本地不装 torch、不做训练。本地仅：采集、数据集构建、可视化、打包上传。

## 3. 网络 — ✅
x.com / huggingface.co / pypi / raw.githubusercontent.com 全部可达（curl 200）。

## 4. 磁盘 — ✅
521 GiB 可用。

## 5. Python 环境 — ✅（仅轻量依赖）
`.venv`（Python 3.12，uv 管理）：playwright（已弃用）、pillow、pyyaml、imagehash。训练依赖全部在 Colab 侧安装。

## 6. X 账号 — ✅
用户 Chrome Default profile 已登录 X（含 Premium 功能：Grok/Articles/History 导航项）。

## 7. 遗留待办（均可自助，无需用户）
- OmniParser teacher：HF 公开权重下载（首次运行时）+ 在 Colab 上跑 teacher 标注。
- Colab GPU 首次分配验证。
- 采集会占用用户 Chrome（导航/resize/开标签页）——长采集任务前告知用户运行时间窗口。
