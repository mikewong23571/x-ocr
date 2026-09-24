const PAL = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6","#bcf60c","#008080","#e6beff","#9a6324","#fffac8","#800000","#aaffc3","#808000","#ffd8b1","#000075","#8f8f8f","#ff6b6b","#51d0de","#ff9f45","#b3de69","#f7b2e7","#6a4c93","#1d976c","#f4d35e","#00b4d8","#ef476f","#c9ada7","#84a59d","#f28482","#9d4edd"];
let DATA = null;

document.getElementById("go").onclick = async () => {
  const stat = document.getElementById("stat");
  stat.textContent = "截图中 → 推理中（~1s）…";
  const resp = await chrome.runtime.sendMessage({ type: "parse", dpr: window.devicePixelRatio || 1 });
  if (resp.error) { stat.textContent = ""; document.getElementById("hint").innerHTML = `<span class="err">${resp.error}</span>`; return; }
  DATA = resp;
  const els = resp.elements || [];
  stat.textContent = `${els.length} 个元素 · conf≥0.35`;
  document.getElementById("hint").style.display = "none";
  drawPreview(resp.image, els);
  renderList(els);
  renderDoc(resp.doc);
};

document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
  document.querySelectorAll(".tab").forEach(x => x.classList.toggle("on", x === t));
  document.getElementById("list").classList.toggle("on", t.dataset.t === "list");
  document.getElementById("doc").classList.toggle("on", t.dataset.t === "doc");
});

function drawPreview(dataUrl, els) {
  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  const img = new Image();
  img.onload = () => {
    const W = cv.clientWidth || 340;
    const s = W / img.width;
    cv.width = W; cv.height = img.height * s;
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    const classes = [...new Set(els.map(e => e.role))].sort();
    els.forEach(e => {
      const c = PAL[classes.indexOf(e.role) % PAL.length];
      const [x1, y1, x2, y2] = e.bbox;
      ctx.strokeStyle = c; ctx.lineWidth = 1.5;
      ctx.strokeRect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    });
  };
  img.src = dataUrl;
}

function renderList(els) {
  const by = {};
  els.forEach(e => (by[e.role] = by[e.role] || []).push(e));
  const html = Object.entries(by).sort((a, b) => b[1].length - a[1].length).map(([role, es]) => {
    const c = PAL[Object.keys(by).sort().indexOf(role) % PAL.length];
    const rows = es.map(e =>
      `<div class="el"><span><span style="color:${c}">■</span> ${role}</span>
       <span class="c">${e.confidence.toFixed(2)} · ${Math.round(e.bbox[0])},${Math.round(e.bbox[1])}</span></div>`).join("");
    return `<div class="grp"><h3>${role} × ${es.length}</h3>${rows}</div>`;
  }).join("");
  document.getElementById("list").innerHTML = html;
  document.getElementById("list").classList.add("on");
}

function renderDoc(doc) {
  if (!doc) { document.getElementById("doc").innerHTML = "<span class='meta'>无文档</span>"; return; }
  document.getElementById("doc").innerHTML = "<pre>" +
    JSON.stringify(doc, null, 1).replace(/&/g, "&amp;").replace(/</g, "&lt;") + "</pre>";
}
