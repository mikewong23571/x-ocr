"""Minimal reference agent: consumes the perception stack, acts, verifies.

The agent's ONLY eyes are runtime.hybrid.perceive (screen document + element list).
The brain is pluggable: any callable(elements, doc) -> [toolcall, ...].
An LLM policy = send build_prompt(doc) to your model, parse tool calls from JSON.

Tools: click(x,y) · type(text) · scroll(dy) · wait(s) · verify(bbox) · done(result)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runtime.hybrid import perceive  # noqa: E402
from runtime.verify import verify_state_change  # noqa: E402

TOOL_SPEC = """可用工具（JSON 数组，每步可多个）:
[{"tool":"click","x":int,"y":int}, {"tool":"type","text":"..."}, {"tool":"scroll","dy":int},
 {"tool":"wait","s":1}, {"tool":"verify","bbox":[x1,y1,x2,y2]}, {"tool":"done","result":"..."}]"""


def build_prompt(doc: dict, elements: list[dict], task: str) -> str:
    """The exact context an LLM policy receives. ~500-900 tokens."""
    import json
    els_compact = [
        {"role": e["role"], "xy": [int((e["bbox"][0] + e["bbox"][2]) / 2), int((e["bbox"][1] + e["bbox"][3]) / 2)],
         "text": (e.get("text") or "")[:40] or None, "conf": round(e["confidence"], 2)}
        for e in elements if e["confidence"] > 0.5
    ]
    return (f"任务: {task}\n\n屏幕文档:\n{json.dumps(doc, ensure_ascii=False)[:2200]}\n\n"
            f"元素索引(role/中心坐标/文本):\n{json.dumps(els_compact, ensure_ascii=False)[:1500]}\n\n{TOOL_SPEC}")


class Executor:
    """Runs tool calls on a playwright page. Vision coordinates, pixel verify."""

    def __init__(self, page, workdir: Path = None):
        self.page = page
        self.workdir = workdir or Path("/tmp")
        self.n = 0

    def _shot(self, tag="") -> Path:
        self.n += 1
        p = self.workdir / f"agent_{self.n:02d}{tag}.png"
        self.page.screenshot(path=str(p))
        return p

    def __call__(self, call: dict, last_bbox=None) -> dict:
        t = call.get("tool")
        if t == "click":
            before = self._shot("_pre")
            self.page.mouse.click(int(call["x"]), int(call["y"]))
            time.sleep(1.2)
            after = self._shot("_post")
            # auto-verify a 120px region around the click
            x, y = int(call["x"]), int(call["y"])
            r = verify_state_change(str(before), str(after), (x - 60, y - 60, x + 60, y + 60),
                                    min_changed_ratio=0.01)
            return {"tool": "click", "at": [x, y], "region_changed": r["changed"],
                    "changed_ratio": r["changed_ratio"]}
        if t == "type":
            self.page.keyboard.type(call["text"], delay=25)
            return {"tool": "type", "chars": len(call["text"])}
        if t == "scroll":
            self.page.mouse.wheel(0, int(call.get("dy", 600)))
            time.sleep(1.5)
            return {"tool": "scroll", "dy": call.get("dy", 600)}
        if t == "wait":
            time.sleep(call.get("s", 1))
            return {"tool": "wait"}
        if t == "verify" and last_bbox:
            before = self._shot("_vpre")
            time.sleep(0.6)
            return verify_state_change(str(before), str(self._shot("_vpost")), call["bbox"])
        if t == "done":
            return {"tool": "done", "result": call.get("result")}
        return {"tool": t, "error": "unknown"}


def run(page, task: str, policy, max_steps: int = 6, verbose=bool) -> list[dict]:
    """Agent loop: perceive → decide → act (+verify) → repeat until done."""
    trace = []
    for step in range(max_steps):
        els, dom = perceive(page, raw=True)
        doc = perceive(page)
        doc.pop("_hybrid", None)
        calls = policy(els, doc, task, trace)
        if not calls:
            trace.append({"step": step, "error": "policy returned no calls"})
            break
        for c in calls:
            out = Executor(page)(c)
            trace.append({"step": step, "call": c, "result": out})
            if verbose:
                print(f"  step{step}: {c.get('tool')} → {out}")
            if c.get("tool") == "done":
                return trace
    return trace
