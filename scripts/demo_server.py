"""Demo server: gallery of screenshots with model detections overlaid.

Serves on 0.0.0.0:<port> (default 8800). Uses the current 32-class ONNX model
(experiments/latest) on curated dataset screenshots; inference is cached.

Usage: .venv/bin/python scripts/demo_server.py [--port 8800] [--n 24]
"""
import argparse
import json
import shutil
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
import yaml  # noqa: E402

from runtime.api import ScreenParser  # noqa: E402

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6",
    "#bcf60c", "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#8f8f8f", "#ff6b6b", "#51d0de", "#ff9f45",
    "#b3de69", "#f7b2e7", "#6a4c93", "#1d976c", "#f4d35e", "#00b4d8", "#ef476f",
    "#c9ada7", "#84a59d", "#f28482", "#9d4edd",
]


def load_font(size: int = 15):
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNSMono.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def pick_samples(n: int) -> list[dict]:
    """Curated spread: (route-family, dark) → richest sample; plus logged-out."""
    import glob

    rows = []
    for mf in sorted((ROOT / "data/processed").glob("*/manifest.jsonl")):
        if "loggedout" in mf.parent.name:
            continue
        rows += [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]

    def family(r: dict) -> str:
        route = r["route"].replace("state_", "s_")
        return f"{route}|{'dark' if r['dark'] else 'light'}"

    best: dict[str, dict] = {}
    for r in rows:
        key = family(r)
        if key not in best or len(r["els"]) > len(best[key]["els"]):
            best[key] = r

    picks = []
    for key in sorted(best):
        r = best[key]
        img = ROOT / "data/raw" / r["session"] / r["sample_id"] / "shot.png"
        if img.exists():
            picks.append({"key": key, "route": r["route"], "theme": "dark" if r["dark"] else "light",
                          "img": str(img.relative_to(ROOT)), "n_dom": len(r["els"])})
    # logged-out picks
    for sid, key in [("0028", "loggedout_login"), ("0024", "loggedout_error"), ("0030", "loggedout_explore")]:
        img = ROOT / f"data/raw/loggedout_s01/{sid}/shot.png"
        if img.exists():
            picks.append({"key": key, "route": key, "theme": "loggedout",
                          "img": str(img.relative_to(ROOT)), "n_dom": 0})

    # balance: keep variety of route+theme, cap n
    seen_route_theme = set()
    out = []
    for p in picks:
        rt = (p["route"].split("_")[0], p["theme"])
        if rt in seen_route_theme and len(out) >= n:
            continue
        seen_route_theme.add(rt)
        out.append(p)
        if len(out) >= n:
            break
    return out


def render_overlay(src: Path, dets: list[dict], out_overlay: Path, out_thumb: Path) -> None:
    im = Image.open(src).convert("RGB")
    ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    font = load_font(16)
    classes = sorted({e["role"] for e in dets})
    cmap = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(classes)}
    for e in dets:
        x1, y1, x2, y2 = e["bbox"]
        c = cmap[e["role"]]
        hex_rgb = tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
        d.rectangle([x1, y1, x2, y2], fill=hex_rgb + (36,), outline=hex_rgb + (255,), width=3)
        label = f"{e['role']} {e['confidence']:.2f}"
        tb = d.textbbox((0, 0), label, font=font)
        ly = y1 - (tb[3] - tb[1]) - 8
        if ly < 0:
            ly = y2 + 2
        d.rectangle([x1, ly, x1 + (tb[2] - tb[0]) + 10, ly + (tb[3] - tb[1]) + 8], fill=hex_rgb + (230,))
        d.text((x1 + 5, ly + 4), label, fill="#000", font=font)
    base = im.convert("RGBA")
    base.alpha_composite(ov)
    base.convert("RGB").save(out_overlay, quality=88)
    th = base.convert("RGB")
    th.thumbnail((560, 400))
    th.save(out_thumb, quality=80)


INDEX_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>X-OCR 纯视觉识别 Demo</title>
<style>
 :root{--bg:#0d1117;--card:#161b22;--line:#30363d;--txt:#e6edf3;--mut:#8b949e;--acc:#58a6ff}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 -apple-system,"PingFang SC",sans-serif}
 header{padding:18px 24px;border-bottom:1px solid var(--line);position:sticky;top:0;background:rgba(13,17,23,.92);backdrop-filter:blur(6px);z-index:5}
 h1{margin:0;font-size:18px} h1 span{color:var(--acc)}
 .meta{color:var(--mut);font-size:12px;margin-top:4px}
 .bar{display:flex;gap:16px;align-items:center;padding:12px 24px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
 .bar label{font-size:12px;color:var(--mut)}
 input[type=range]{width:220px;accent-color:var(--acc)}
 #confv{color:var(--acc);font-weight:600;min-width:38px}
 .chips{display:flex;gap:6px;flex-wrap:wrap;max-width:70%}
 .chip{border:1px solid var(--line);border-radius:99px;padding:2px 10px;font-size:12px;cursor:pointer;color:var(--mut)}
 .chip.on{color:#fff;border-color:var(--acc);background:rgba(88,166,255,.15)}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:18px 24px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer;transition:.15s}
 .card:hover{transform:translateY(-2px);border-color:var(--acc)}
 .card img{width:100%;display:block}
 .cap{padding:8px 12px;display:flex;justify-content:space-between;font-size:12px;color:var(--mut)}
 .cap b{color:var(--txt)}
 #modal{position:fixed;inset:0;background:rgba(0,0,0,.82);display:none;z-index:10;padding:24px}
 #modal.on{display:flex;gap:18px;justify-content:center;align-items:flex-start}
 #mimg{max-width:78vw;max-height:88vh;border-radius:8px;box-shadow:0 8px 40px rgba(0,0,0,.6)}
 #side{width:300px;max-height:84vh;overflow:auto;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
 #side h3{margin:0 0 8px;font-size:14px} .close{position:absolute;top:18px;right:26px;font-size:26px;cursor:pointer;color:var(--mut)}
 .el{display:flex;justify-content:space-between;padding:4px 6px;border-radius:6px;font-size:12px;margin:2px 0;cursor:pointer}
 .el:hover{background:rgba(255,255,255,.06)}
 .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
</style></head><body>
<header><h1>X-OCR · 纯视觉 UI 感知 Demo <span>32 类 · ONNX · 无 DOM/浏览器</span></h1>
<div class="meta">模型 experiments/latest（exp009, 2.6M 参数）· 输入=screenshot · 输出=role+bbox+confidence</div></header>
<div class="bar">
 <label>置信度阈值 <input id="conf" type="range" min="0" max="0.95" step="0.05" value="0.30"> <span id="confv">0.30</span></label>
 <div class="chips" id="chips"></div>
 <span class="meta" id="stat"></span>
</div>
<div class="grid" id="grid"></div>
<div id="modal"><span class="close" onclick="modal.classList.remove('on')">×</span>
 <img id="mimg"><div id="side"></div></div>
<script>
let DATA=[], OFF=[];
fetch('/api/manifest').then(r=>r.json()).then(m=>{DATA=m.items;
 const classes=[...new Set(m.items.flatMap(i=>i.els.map(e=>e.role)))].sort();
 const chips=document.getElementById('chips');
 chips.innerHTML=classes.map(c=>`<span class="chip on" data-c="${c}">${c}</span>`).join('');
 chips.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>{ch.classList.toggle('on');render()});
 document.getElementById('conf').oninput=e=>{document.getElementById('confv').textContent=e.target.value;render()};
 document.getElementById('stat').textContent=m.items.length+' 张精选截图（暗/亮主题 × 各路由 × 菜单弹窗 × 登录墙）';
 render();});
function active(){return [...document.querySelectorAll('.chip.on')].map(c=>c.dataset.c)}
function render(){const th=+document.getElementById('conf').value;const act=active();const g=document.getElementById('grid');
 g.innerHTML=DATA.map((it,idx)=>{const n=it.els.filter(e=>e.confidence>=th&&act.includes(e.role)).length;
  return `<div class="card" onclick="big(${idx})"><img loading="lazy" src="/img/${it.thumb}">
  <div class="cap"><b>${it.key}</b><span>${n} 元素 · ${it.theme}</span></div></div>`}).join('')}
function big(idx){const th=+document.getElementById('conf').value;const act=active();const it=DATA[idx];
 const els=it.els.filter(e=>e.confidence>=th&&act.includes(e.role));
 document.getElementById('mimg').src='/img/'+it.overlay;
 const side=document.getElementById('side');
 side.innerHTML=`<h3>${it.key} · ${els.length} 个检测</h3>`+els
  .map(e=>`<div class="el"><span><span class="dot" style="background:${e.color}"></span>${e.role}</span><span>${e.confidence.toFixed(2)}</span></div>`).join('');
 document.getElementById('modal').classList.add('on')}
document.getElementById('modal').onclick=e=>{if(e.target.id==='modal')e.currentTarget.classList.remove('on')};
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--n", type=int, default=27)
    args = ap.parse_args()

    cache = ROOT / "data/demo_cache"
    if cache.exists():
        shutil.rmtree(cache)
    (cache / "img").mkdir(parents=True)

    tax = yaml.safe_load((ROOT / "schema/ui_taxonomy.yaml").read_text())
    names = {i: c for i, c in enumerate(sorted(tax["classes"].keys()))}
    sp = ScreenParser(str(ROOT / "experiments/latest/best_onnx_fp32.onnx"), names, conf=0.10)

    items = []
    palette_by_class = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(names.values()))}
    for k, p in enumerate(pick_samples(args.n)):
        src = ROOT / p["img"]
        dets = sp(src)
        stem = f"{k:02d}_{p['key']}".replace("/", "_").replace("|", "_")
        overlay = f"{stem}.jpg"
        thumb = f"{stem}_t.jpg"
        render_overlay(src, dets, cache / "img" / overlay, cache / "img" / thumb)
        items.append({
            "key": p["key"], "theme": p["theme"], "overlay": overlay, "thumb": thumb,
            "els": [{**e, "color": palette_by_class[e["role"]]} for e in dets],
        })
        print(f"  [{k+1}/{args.n}] {p['key']}: {len(dets)} dets")

    (cache / "manifest.json").write_text(json.dumps({"items": items}, ensure_ascii=False))
    print(f"cached {len(items)} samples → {cache}")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: N802
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            import base64
            import io
            import urllib.parse as _up
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/parse":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))
                    data = body["image"].split(",", 1)[1]
                    raw = base64.b64decode(data)
                    import tempfile
                    from runtime.api import ScreenParser as _SP
                    from runtime.screen_doc import screen_document as _sd
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(raw)
                        tmp = f.name
                    els = sp(tmp)
                    try:
                        doc = _sd(tmp, conf=0.35)
                        doc.pop("_hybrid", None)
                    except Exception:
                        doc = None
                    import os
                    os.unlink(tmp)
                    self._send(200, json.dumps({"elements": els, "doc": doc},
                                               ensure_ascii=False).encode(), "application/json")
                except Exception as e:  # noqa: BLE001
                    self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_GET(self):  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            if u.path == "/" or u.path == "/index.html":
                self._send(200, INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif u.path == "/report":
                import markdown
                md = (ROOT / "docs/technical_report.md").read_text()
                html = ("<doctype html><html><head><meta charset='utf-8'><title>X-OCR 技术报告</title>"
                        "<style>body{max-width:860px;margin:24px auto;padding:0 16px;font:15px/1.7 -apple-system,'PingFang SC',sans-serif;color:#1a1a1a}"
                        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px 10px;font-size:13px}"
                        "pre{background:#f5f5f5;padding:12px;border-radius:8px;overflow:auto}code{background:#f0f0f0;padding:2px 5px;border-radius:4px}"
                        "h1,h2{border-bottom:1px solid #ddd;padding-bottom:6px}blockquote{border-left:4px solid #58a6ff;margin:0;padding:2px 14px;color:#555;background:#f6f9ff}</style>"
                        "</head><body>" + markdown.markdown(md, extensions=["tables"]) + "</body></html>")
                self._send(200, html.encode(), "text/html; charset=utf-8")
            elif u.path == "/screendoc":
                import urllib.parse as _up
                q = _up.parse_qs(u.query)
                src = (q.get("src") or [""])[0]
                allowed_root = str(ROOT / "data")
                full = (ROOT / src).resolve()
                if not src or not str(full).startswith(allowed_root) or not full.exists():
                    self._send(400, b"bad src", "text/plain"); return
                from runtime.screen_doc import screen_document
                doc = screen_document(str(full))
                body = json.dumps(doc, ensure_ascii=False, indent=1)
                html = ("<doctype html><html><head><meta charset='utf-8'><title>屏幕文档</title>"
                        "<style>body{background:#0d1117;color:#e6edf3;font:13px/1.6 Menlo,monospace;padding:20px}"
                        "pre{white-space:pre-wrap}a{color:#58a6ff}</style></head><body>"
                        f"<a href='/'>← demo</a> | <b>{src}</b><pre>" +
                        body.replace("&", "&amp;").replace("<", "&lt;") + "</pre></body></html>")
                self._send(200, html.encode(), "text/html; charset=utf-8")
            elif u.path == "/api/manifest":
                self._send(200, (cache / "manifest.json").read_bytes(), "application/json")
            elif u.path.startswith("/img/"):
                f = cache / "img" / u.path[len("/img/"):]
                if f.exists():
                    self._send(200, f.read_bytes(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), H)
    print(f"DEMO: http://0.0.0.0:{args.port}  (LAN: http://<本机IP>:{args.port})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
