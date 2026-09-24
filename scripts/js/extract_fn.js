() => {
  function isVisible(el) {
    if (!el || el.nodeType !== Node.ELEMENT_NODE) return false;
    if (el.hidden) return false;
    if (el.getAttribute('aria-hidden') === 'true') return false;
    const s = getComputedStyle(el);
    if (s.display === 'none' || s.visibility === 'hidden' || s.visibility === 'collapse') return false;
    if (parseFloat(s.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return false;
    if (r.right < 0 || r.bottom < 0) return false;
    if (r.left > innerWidth || r.top > innerHeight) return false;
    return true;
  }
  const out = [];
  for (const n of document.querySelectorAll('*')) {
    if (!isVisible(n)) continue;
    const r = n.getBoundingClientRect();
    const s = getComputedStyle(n);
    const interactive = !!(n.tagName === 'BUTTON' || n.tagName === 'A' || n.getAttribute('role') === 'button'
      || n.getAttribute('role') === 'link' || n.getAttribute('role') === 'textbox' || n.tagName === 'INPUT'
      || n.tagName === 'SELECT' || n.hasAttribute('contenteditable')
      || (s.cursor === 'pointer' && n.getElementsByTagName('*').length === 0));
    out.push({
      tag: n.tagName.toLowerCase(),
      testid: n.getAttribute('data-testid'),
      role: n.getAttribute('role'),
      aria: (n.getAttribute('aria-label') || '').slice(0, 80) || null,
      text: (n.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 80) || null,
      rect: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
      inter: interactive,
      disabled: n.disabled === true || n.getAttribute('aria-disabled') === 'true' || null,
    });
  }
  return { url: location.href, vw: innerWidth, vh: innerHeight, n: out.length, els: out };
}
