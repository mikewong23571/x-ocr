chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type !== "parse") return;
  (async () => {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
      const r = await fetch("http://127.0.0.1:8800/api/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      const j = await r.json();
      j.image = dataUrl;
      j.dpr = msg.dpr || 1;
      sendResponse(j);
    } catch (e) {
      sendResponse({ error: String(e) });
    }
  })();
  return true; // async response
});
