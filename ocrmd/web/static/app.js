/* ocrmd web UI — vanilla JS single-page app. No build step, no external dependencies. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
const STATUSES = ["verified", "reviewed", "flagged", "suspect", "illegible", "blank", "unverified", "error"];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleTimeString() : "");
const fmtNum = (v, d = 2) => (v == null ? "" : Number(v).toFixed(d));
let timers = [];

function toast(msg, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`; el.textContent = msg;
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

async function api(path, opts = {}) {
  const init = { method: opts.method || "GET", headers: {} };
  if (opts.body !== undefined) {
    if (opts.body instanceof FormData) init.body = opts.body;
    else { init.headers["Content-Type"] = "application/json"; init.body = JSON.stringify(opts.body); }
  }
  const r = await fetch(path, init);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

function every(ms, fn) { const id = setInterval(fn, ms); timers.push(id); return id; }
function clearTimers() { timers.forEach(clearInterval); timers = []; }

/* ---------- tiny markdown renderer for the ocrmd dialect ---------- */
function inline(s) {
  s = esc(s);
  s = s.replace(/==\[(.+?)\]==\[\^ocrmd-(\d+)\]/g, '<mark>$1</mark><sup>[$2]</sup>');
  s = s.replace(/\[\^([^\]]+)\]/g, '<sup>[$1]</sup>');
  s = s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/(^|[^*])\*([^*]+)\*/g, "$1<i>$2</i>");
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  return s;
}
function renderMd(md) {
  const lines = md.split("\n"); const out = []; let para = [], list = null, table = null, quote = [];
  const flushP = () => { if (para.length) { out.push(`<p>${inline(para.join(" "))}</p>`); para = []; } };
  const flushL = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; } };
  const flushT = () => { if (table) { out.push(`<table>${table.map((r, i) => `<tr>${r.map((c) => `<${i ? "td" : "th"}>${inline(c)}</${i ? "td" : "th"}>`).join("")}</tr>`).join("")}</table>`); table = null; } };
  const flushQ = () => { if (quote.length) { out.push(`<blockquote>${inline(quote.join(" "))}</blockquote>`); quote = []; } };
  const flushAll = () => { flushP(); flushL(); flushT(); flushQ(); };
  for (const raw of lines) {
    const l = raw.replace(/\s+$/, "");
    let m;
    if (!l.trim()) { flushAll(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)/))) { flushAll(); out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); continue; }
    if ((m = l.match(/^\[\^([^\]]+)\]:\s*(.*)/))) { flushAll(); out.push(`<div class="fn"><sup>[${esc(m[1])}]</sup> ${inline(m[2])}</div>`); continue; }
    if ((m = l.match(/^>\s?(.*)/))) { flushP(); flushL(); flushT(); quote.push(m[1]); continue; }
    if (/^\|.*\|$/.test(l)) { flushP(); flushL(); flushQ(); if (/^\|[\s:|-]+\|$/.test(l)) continue; table = table || []; table.push(l.slice(1, -1).split("|").map((c) => c.trim())); continue; }
    if ((m = l.match(/^\s*[-*+]\s+(.*)/))) { flushP(); flushT(); flushQ(); if (!list || list.tag !== "ul") { flushL(); list = { tag: "ul", items: [] }; } list.items.push(m[1]); continue; }
    if ((m = l.match(/^\s*\d+[.)]\s+(.*)/))) { flushP(); flushT(); flushQ(); if (!list || list.tag !== "ol") { flushL(); list = { tag: "ol", items: [] }; } list.items.push(m[1]); continue; }
    flushL(); flushT(); flushQ(); para.push(l);
  }
  flushAll();
  return out.join("\n");
}

/* ---------- shared widgets ---------- */
function statusBar(counts) {
  const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
  return `<div class="statusbar">${STATUSES.filter((s) => counts[s]).map((s) => `<span class="bg-${s}" style="width:${(100 * counts[s]) / total}%" title="${counts[s]} ${s}"></span>`).join("")}</div>`;
}
function countChips(counts, active, onclickAttr) {
  return `<div class="chips">${STATUSES.filter((s) => counts[s]).map((s) => `<span class="chip ${active === s ? "on" : ""} s-${s}" ${onclickAttr ? `onclick="${onclickAttr}('${s}')"` : ""}>${counts[s]} ${s}</span>`).join("")}</div>`;
}
const badge = (s) => `<span class="badge s-${s}">${esc(s)}</span>`;
function logHtml(entries) {
  return entries.map((e) => `<div class="${esc(e.level)}"><span class="t">${fmtTime(e.at)}</span>${esc(e.line)}</div>`).join("");
}

/* ---------- top status (scan / jobs) ---------- */
async function refreshTop() {
  try {
    const [scan, jobs] = await Promise.all([api("/api/scan"), api("/api/jobs")]);
    const running = jobs.filter((j) => j.status === "running" || j.status === "queued").length;
    $("#top-status").innerHTML = (scan.running ? `<span class="live">● scanning ${esc(scan.doc)}</span>` : "") +
      (running ? ` &nbsp; <a href="#/jobs">${running} job(s) running</a>` : "");
  } catch (_) { /* server down */ }
}

/* ---------- views ---------- */
const exportLinks = (slug) => ["md", "txt", "json", "jsonl"].map((f) =>
  `<a href="/api/documents/${encodeURIComponent(slug)}/export.${f}" download title="download the document as ${f}">${f}</a>`).join(" ");
/* An FTS snippet marks the match as [word]; show it as a highlight. */
const snippetHtml = (s) => esc(s).replace(/\[([^\]]+)\]/g, "<mark>$1</mark>");

function docCard(d) {
  const who = d.author && d.author.length ? d.author.join(", ") : "";
  const what = [d.kind, d.year, who].filter(Boolean).map(esc).join(" · ");
  const reshoot = (d.counts.illegible || 0) + (d.counts.suspect || 0);
  return `
    <div class="card">
      <div class="row"><h3><a href="#/doc/${esc(d.slug)}">${esc(d.title || d.slug)}</a></h3>
        ${d.scanning ? '<span class="right ok small">● scanning</span>' : ""}</div>
      <div class="muted small">${esc(d.slug)} · ${d.pages} page(s)${d.page_labels ? " · printed " + esc(d.page_labels) : ""}${what ? "<br>" + what : ""}</div>
      ${statusBar(d.counts)}
      ${countChips(d.counts)}
      <div class="row small" style="margin-top:8px">
        ${reshoot ? `<span class="err">${reshoot} to re-shoot</span>` : ""}
        ${d.counts.flagged ? `<a href="#/review/${esc(d.slug)}">${d.counts.flagged} to review</a>` : ""}
        ${d.counts.unverified ? `<span class="muted">${d.counts.unverified} without LLM pass</span>` : ""}
        <span class="right"></span>
        <span class="muted">export</span> ${exportLinks(d.slug)}
      </div>
    </div>`;
}

/* The library: every document scanned into this root, its catalogue entry, its progress — and one
   search box over every page of all of them. */
async function dashboard() {
  const [lib, doctor] = await Promise.all([api("/api/library"), api("/api/doctor?cameras=false"), sinksStatus()]);
  const docs = lib.documents.map((d) => ({ ...d, scanning: d.slug === lib.scanning }));
  const pages = docs.reduce((a, d) => a + d.pages, 0);
  view.innerHTML = `
  <div class="grid side">
    <div>
      <div class="row"><h1>Library</h1><span class="muted small">${docs.length} document(s) · ${pages} page(s)</span><span class="right"></span>
        <a class="btn" href="#/scan">New scan</a></div>
      <form class="row" id="searchform" style="margin-bottom:12px"><input id="q" placeholder="search every page ever scanned… (quote a phrase, word* for a prefix)" style="flex:1"><button class="primary">Search</button></form>
      <div class="cards">${docs.map(docCard).join("") || '<div class="muted">Nothing scanned yet. Start a scan, or create a document and drop page images on it.</div>'}</div>
    </div>
    <div>
      <h2>System</h2>
      <div class="panel">
        <dl class="kv">
          <dt>OCR</dt><dd>${esc(doctor.ocr_engine || "none")}</dd>
          <dt>LLM</dt><dd>${esc(doctor.config.provider)} / ${esc(doctor.config.model)} ${doctor.credentials.anthropic || doctor.credentials.xai ? '<span class="ok">credentials found</span>' : '<span class="warn">no credentials — OCR only</span>'}</dd>
          <dt>Prompt</dt><dd>${esc(doctor.config.prompt_version)}</dd>
          <dt>Index</dt><dd>${doctor.library.indexed} page(s) ${doctor.library.fts ? "(FTS5)" : '<span class="warn">(substring scan — no FTS5)</span>'}</dd>
          <dt>Database</dt><dd>${doctor.db_integrity === "ok" ? '<span class="ok">ok</span>' : `<span class="err">${esc(doctor.db_integrity)}</span>`}</dd>
          <dt>Root</dt><dd class="mono">${esc(doctor.root)}</dd>
          ${doctor.orphan_artifacts ? `<dt>Orphans</dt><dd>${doctor.orphan_artifacts} file(s)</dd>` : ""}
          ${doctor.open_batches ? `<dt>Batches</dt><dd>${doctor.open_batches} open</dd>` : ""}
        </dl>
        <div style="margin-top:8px"><a href="#/settings">full doctor report →</a></div>
      </div>
      <h2>Destinations</h2>
      <div class="panel small">${(sinks || []).length
        ? (sinks || []).map((s) => `<div><b>${esc(s.descriptor?.name || s.name)}</b> <span class="muted">${esc(s.url)}</span>${s.descriptor && s.descriptor.reachable === false ? ` <span class="err">${esc(s.descriptor.error || "unreachable")}</span>` : ""}</div>`).join("")
        : '<span class="muted">none configured — a <code>[[sinks]]</code> table in ocrmd.toml adds a “Send” button to every page (see ocrmd.example.toml)</span>'}</div>
      <h2>Spend</h2>
      <div class="panel">
        ${doctor.costs.length ? `<table><tr><th>model</th><th class="num">calls</th><th class="num">in</th><th class="num">out</th><th class="num">avg ms</th></tr>
        ${doctor.costs.map((c) => `<tr><td class="mono">${esc(c.model)}</td><td class="num">${c.n}</td><td class="num">${c.tokens_in.toLocaleString()}</td><td class="num">${c.tokens_out.toLocaleString()}</td><td class="num">${c.avg_latency_ms}</td></tr>`).join("")}</table>` : '<span class="muted">no LLM calls yet</span>'}
      </div>
    </div>
  </div>`;
  $("#searchform").onsubmit = (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (q) location.hash = `#/search?q=${encodeURIComponent(q)}`; };
}

/* ---------- search ---------- */
async function searchView(q) {
  const r = q ? await api(`/api/search?q=${encodeURIComponent(q)}&limit=100`) : { hits: [], fts: true };
  view.innerHTML = `
  <div class="row"><h1>Search</h1><span class="muted small">every page of every document</span></div>
  <form class="row" id="searchform" style="margin-bottom:12px"><input id="q" value="${esc(q || "")}" placeholder="words to find; quote a phrase; word* for a prefix" style="flex:1"><button class="primary">Search</button></form>
  ${q ? `<div class="muted small" style="margin-bottom:8px">${r.hits.length} hit(s) for <b>${esc(q)}</b>${r.fts ? "" : " · substring scan (this SQLite has no FTS5)"}</div>` : ""}
  ${r.hits.map((h) => `
    <div class="readout" style="margin-top:6px">
      <div class="row small"><a href="#/doc/${esc(h.document)}/page/${h.sequence}"><b>${esc(h.title || h.document)}</b> · page ${String(h.sequence).padStart(4, "0")}${h.page_label ? ` (p. ${esc(h.page_label)})` : ""}</a> ${badge(h.status || "unverified")}</div>
      <div class="small">${snippetHtml(h.snippet)}</div>
    </div>`).join("")}`;
  $("#searchform").onsubmit = (e) => { e.preventDefault(); location.hash = `#/search?q=${encodeURIComponent($("#q").value.trim())}`; };
  $("#q").focus();
}

/* ---------- scan ---------- */
let scanState = null;
async function scanView() {
  const [docs, cams, snap, settings] = await Promise.all([api("/api/documents"), api("/api/cameras").catch(() => []), api("/api/scan"), api("/api/settings"), sinksStatus()]);
  scanState = snap;
  view.innerHTML = `
  <div class="grid side">
    <div>
      <div class="video" id="video">
        <img id="stream" alt="" style="${snap.running ? "" : "display:none"}">
        <div class="overlay"><span class="gate" id="gate">${snap.running ? "…" : "idle"}</span><span id="pxpage" class="pxpage"></span><span id="occ" class="mono small" style="color:#ccc;display:none"></span></div>
        <div id="video-placeholder" class="loading" style="${snap.running ? "display:none" : ""}">No scan running. Configure and start on the right.</div>
      </div>
      <div class="meters" id="meters"></div>
      <div class="controls" id="controls" style="${snap.running ? "" : "display:none"}">
        <button class="primary" data-cmd="capture" title="space">Capture now</button>
        <button data-cmd="focus_lock" title="f">Lock focus</button>
        <button data-cmd="focus_auto" title="F">Autofocus</button>
        <button data-cmd="rotate" title="r">Rotate ⟳</button>
        <button data-cmd="rotate_ccw" title="R">Rotate ⟲</button>
        <button data-cmd="zoom_in" title="+">Zoom +</button>
        <button data-cmd="zoom_out" title="-">Zoom −</button>
        <button data-cmd="pan_up">↑</button><button data-cmd="pan_down">↓</button><button data-cmd="pan_left">←</button><button data-cmd="pan_right">→</button>
        <button data-cmd="reset" title="0">Reset view</button>
        <span class="right"></span>
        <button class="danger" id="stop">Stop scan</button>
      </div>
      <h2>Events</h2>
      <div class="log" id="scanlog"></div>
    </div>
    <div>
      <div class="panel" id="scan-form" style="${snap.running ? "display:none" : ""}">
        <h3 style="margin-top:0">Start a scan</h3>
        <label>Document</label>
        <div class="row"><select id="f-doc" class="grow">${docs.map((d) => `<option value="${esc(d.slug)}">${esc(d.slug)}${d.title ? " — " + esc(d.title) : ""}</option>`).join("")}<option value="__new__">new document…</option></select></div>
        <input id="f-newdoc" placeholder="new-slug (letters, digits, - _ .)" style="display:${docs.length ? "none" : "block"};width:100%;margin-top:6px">
        <label>Camera</label>
        <select id="f-cam" style="width:100%">${cams.map((c) => `<option value="${c.index}">${c.index}: ${esc(c.name)}${c.format ? ` (${c.format[0]}×${c.format[1]})` : ""}</option>`).join("") || '<option value="0">0 (no cameras probed)</option>'}</select>
        <div class="grid cols-2">
          <div><label>Focus</label><select id="f-focus" style="width:100%">${["settle", "continuous", "locked", "off"].map((f) => `<option ${f === settings.focus || (f === "settle" && settings.focus === "continuous") ? "selected" : ""}>${f}</option>`).join("")}</select></div>
          <div><label>Resolution</label><input id="f-res" placeholder="largest (e.g. 1280x960)" style="width:100%"></div>
          <div><label>Stop after (s)</label><input id="f-dur" type="number" min="10" placeholder="unlimited" style="width:100%"></div>
          <div><label>Max pages</label><input id="f-max" type="number" min="1" placeholder="unlimited" style="width:100%"></div>
        </div>
        <label>Title (new documents)</label><input id="f-title" style="width:100%">
        <label><input type="checkbox" id="f-nollm"> OCR only (no LLM pass)</label>
        <div class="row" style="margin-top:12px"><button class="primary" id="start">Start scan</button>
          <span class="muted small">LLM: ${esc(settings.provider)}/${esc(settings.model)} × ${settings.llm_workers}</span></div>
      </div>
      <div class="panel" id="scan-status" style="${snap.running ? "" : "display:none"}">
        <h3 style="margin-top:0">Session</h3>
        <dl class="kv" id="scan-kv"></dl>
        <h3>Recent pages</h3>
        <div class="recent" id="recent"></div>
        <div id="advice" class="warn small" style="margin-top:8px"></div>
      </div>
      <div class="panel small" style="margin-top:12px">
        <b>Keys</b> (when the video has focus): <kbd>space</kbd> capture · <kbd>f</kbd>/<kbd>F</kbd> focus · <kbd>r</kbd> rotate · <kbd>+</kbd>/<kbd>-</kbd> zoom · <kbd>w a s d</kbd> pan · <kbd>0</kbd> reset.<br>
        Start with your hands out of the frame; the hand gate learns what paper looks like from the first frames.
      </div>
    </div>
  </div>`;
  $("#f-doc").onchange = () => { $("#f-newdoc").style.display = $("#f-doc").value === "__new__" ? "block" : "none"; };
  $("#start").onclick = startScan;
  $("#stop").onclick = () => api("/api/scan/stop", { method: "POST" }).then(() => toast("stopping — finishing queued pages")).catch((e) => toast(e.message, "err"));
  view.querySelectorAll("[data-cmd]").forEach((b) => (b.onclick = () => sendCmd(b.dataset.cmd)));
  document.onkeydown = (ev) => {
    if (!scanState?.running || ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    const map = { " ": "capture", f: "focus_lock", F: "focus_auto", r: "rotate", R: "rotate_ccw", "+": "zoom_in", "=": "zoom_in", "-": "zoom_out", w: "pan_up", s: "pan_down", a: "pan_left", d: "pan_right", "0": "reset" };
    if (map[ev.key]) { ev.preventDefault(); sendCmd(map[ev.key]); }
  };
  if (snap.running) attachStream();
  renderScan(snap);
  every(500, pollScan);
}
function sendCmd(cmd) { api("/api/scan/command", { method: "POST", body: { command: cmd } }).catch((e) => toast(e.message, "err")); }
function attachStream() { const img = $("#stream"); if (img && !img.src) { img.src = `/api/scan/stream?t=${Date.now()}`; img.style.display = "block"; $("#video-placeholder").style.display = "none"; } }
async function startScan() {
  const docSel = $("#f-doc").value;
  const doc = docSel === "__new__" ? $("#f-newdoc").value.trim() : docSel;
  if (!doc) return toast("choose or name a document", "err");
  const body = { doc, camera: $("#f-cam").value, focus: $("#f-focus").value, no_llm: $("#f-nollm").checked,
    resolution: $("#f-res").value || null, duration: Number($("#f-dur").value) || null, max_pages: Number($("#f-max").value) || null,
    title: $("#f-title").value || null };
  try {
    const snap = await api("/api/scan/start", { method: "POST", body });
    scanState = snap;
    if (snap.error) return toast(snap.error, "err");
    $("#scan-form").style.display = "none"; $("#scan-status").style.display = ""; $("#controls").style.display = "";
    attachStream(); renderScan(snap); toast(`scanning into ${doc}`, "ok");
  } catch (e) { toast(e.message, "err"); }
}
async function pollScan() {
  try { scanState = await api("/api/scan"); } catch (_) { return; }
  renderScan(scanState);
}
function renderScan(s) {
  if (!$("#gate")) return;
  const g = s.running ? (s.gate || {}) : {};
  const gateEl = $("#gate");
  const label = { hand: "HAND IN FRAME", "same page": "SAME PAGE", "page moved": "PAGE MOVING" }[g.gate]
    || (g.gate || (s.running ? "…" : "idle")).toUpperCase();
  gateEl.textContent = label;
  gateEl.className = `gate ${(g.gate || "").replace(" page", "").replace(" moved", "moved")}`;
  const px = $("#pxpage");
  if (g.page_px) { px.textContent = `${g.page_px}px/page`; px.className = `pxpage ${g.page_px >= (g.min_page_px || 900) ? "ok" : "low"}`; } else px.textContent = "";
  $("#occ").textContent = g.occlusion ? `${g.occlusion}${g.clipped?.length ? "  CUT OFF: " + g.clipped.join("+") : ""}` : "";
  $("#meters").innerHTML = s.running ? [
    // the settle target grows after a page turn, so the bar is drawn against what this scene needs
    [g.settling_from ? `settling after ${g.settling_from}` : "still",
     Math.min(1, (g.still || 0) / (g.stable_frames || 8)),
     g.settling_from ? `${g.still || 0}/${g.stable_frames || 8}` : `motion ${fmtNum(g.motion, 1)}`],
    ["new page", Math.min(1, (g.novelty || 0) / (g.novelty_threshold || 12)), `${g.novelty ?? ""}`],
    ["sharpness", Math.min(1, (g.sharp || 0) / ((g.floor || 60) * 3)), `${fmtNum(g.sharp, 0)} / floor ${fmtNum(g.floor, 0)}`],
  ].map(([n, v, d]) => `<div class="meter"><div class="row"><span>${n}</span><span class="right muted">${d}</span></div><div class="bar"><i style="width:${v * 100}%"></i></div></div>`).join("") : "";
  const kv = $("#scan-kv");
  if (kv) kv.innerHTML = `
    <dt>Document</dt><dd><a href="#/doc/${esc(s.doc)}">${esc(s.doc)}</a></dd>
    <dt>Camera</dt><dd>${esc(s.camera?.name || "")} ${g.resolution ? `${g.resolution[0]}×${g.resolution[1]}` : ""}</dd>
    <dt>Focus</dt><dd>${esc(g.focus || "")}${g.adjusting ? " (moving)" : ""}${
      g.focus_note && !g.focus_note.startsWith("focus ok")
        ? ` <span class="${g.focus_note.includes("soft") ? "err" : "warn"}">${esc(g.focus_note)}</span>`
        : g.rig_best ? ` <span class="muted small">${Math.round(100 * (g.focus_ratio ?? 1))}% of this rig's best (${Math.round(g.rig_best)})</span>` : ""}</dd>
    <dt>View</dt><dd class="mono small">${esc(g.view || "")}</dd>
    <dt>Captured</dt><dd>${s.captured}</dd>
    <dt>Queue</dt><dd>${s.queued} ${s.degraded ? '<span class="err">LLM OFF (degraded)</span>' : ""}</dd>
    <dt>Working</dt><dd>${esc(s.working || "idle")}</dd>
    <dt>LLM</dt><dd>${s.llm ? esc(s.llm.provider + "/" + s.llm.model) : "OCR only"}</dd>`;
  const rec = $("#recent");
  if (rec) rec.innerHTML = (s.recent || []).map((r) => {
    const seq = /^\d+$/.test(r.label) ? Number(r.label) : null;
    const doc = s.doc || (s.opts && s.opts.doc) || "";
    const page = `<span class="s-${r.status}">${seq != null && doc ? `<a href="#/doc/${esc(doc)}/page/${seq}">${esc(r.label)}</a>` : esc(r.label)} ${esc(r.status)}</span>`;
    return page + (seq != null && doc && sinks && sinks.length
      ? ` <a class="chip" href="#/doc/${esc(doc)}/page/${seq}?send=1" title="send this page to a destination">send</a>` : "");
  }).join(" ");
  const adv = $("#advice"); if (adv) adv.textContent = s.last_error || s.advice || "";
  const log = $("#scanlog");
  if (log && s.events) { const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 20; log.innerHTML = logHtml(s.events); if (atBottom) log.scrollTop = log.scrollHeight; }
  if (!s.running && $("#controls")?.style.display !== "none") {
    $("#controls").style.display = "none"; $("#scan-status").style.display = "none"; $("#scan-form").style.display = "";
    const img = $("#stream"); if (img) { img.src = ""; img.style.display = "none"; } $("#video-placeholder").style.display = "";
    if (s.error) toast(s.error, "err");
  }
}

/* ---------- document ---------- */
let docFilter = null;
async function docView(slug) {
  const d = await api(`/api/documents/${encodeURIComponent(slug)}`);
  const rows = d.page_list.filter((p) => !docFilter || p.status === docFilter);
  view.innerHTML = `
  <div class="row"><h1>${esc(d.title || slug)}</h1><span class="muted">${esc(slug)} · ${d.pages} pages</span><span class="right"></span>
    ${d.flagged ? `<a class="btn" href="#/review/${esc(slug)}">Review ${d.flagged} flagged</a>` : ""}
    <button id="j-verify">Verify</button><button id="j-build">Build</button><button id="j-rebuild" title="Rebuild every page from its Markdown and score it against the photograph">Rebuild</button><button id="j-reprocess" class="danger">Reprocess</button>
    <span class="small"><span class="muted">export</span> ${exportLinks(slug)}</span>
    <a class="btn" href="#/settings/${esc(slug)}">Metadata</a>
    <button id="j-delete-doc" class="danger" title="delete the whole document">Delete…</button></div>
  ${statusBar(d.counts)}
  <div class="row">${countChips(d.counts, docFilter, "setDocFilter")} ${docFilter ? '<span class="chip on" onclick="setDocFilter(null)">× all</span>' : ""}
    <span class="right muted small">${d.duplicates ? d.duplicates + " duplicate capture(s) skipped · " : ""}${d.refused ? `<span class="warn">${d.refused} capture(s) refused as unreadable</span> · ` : ""}${d.needs_reshoot.length ? `<span class="err">${d.needs_reshoot.length} to re-shoot</span>` : ""}</span></div>
  <div class="grid side" style="margin-top:12px">
    <div>
    <div class="row small" id="bulkbar" style="margin-bottom:6px">
      <label style="margin:0"><input type="checkbox" id="sel-all"> select all shown</label>
      <span id="sel-count" class="muted"></span>
      <span class="right"></span>
      <button class="danger small" id="del-sel" disabled>Delete selected…</button>
    </div>
    <table><tr><th></th><th class="num">seq</th><th>label</th><th>status</th><th class="num">px</th><th class="num">agree</th><th class="num">overlap</th><th>note</th></tr>
    ${rows.map((p) => `<tr class="click" onclick="location.hash='#/doc/${esc(slug)}/page/${p.sequence}'">
      <td onclick="event.stopPropagation()"><input type="checkbox" class="sel" value="${p.sequence}"></td>
      <td class="num mono">${String(p.sequence).padStart(4, "0")}</td><td>${esc(p.page_label || "")}${p.part ? ` <span class="muted small">${p.part}</span>` : ""}</td>
      <td>${badge(p.status)}</td><td class="num ${p.width_px && p.width_px < 900 ? "warn" : ""}">${p.width_px ?? ""}</td>
      <td class="num">${fmtNum(p.agreement)}</td><td class="num">${fmtNum(p.overlap)}</td>
      <td class="small">${p.llm_error ? `<span class="err">${esc(p.llm_error)}</span>` : ""}${p.continuity ? `<span class="warn">${esc(p.continuity)}</span> ` : ""}<span class="muted">${esc(p.advice || "")}</span></td></tr>`).join("")}
    </table>
    </div>
    <div>
      <div class="dropzone" id="drop">Drop page images here (or click) to ingest into <b>${esc(slug)}</b><input type="file" id="files" multiple accept="image/*" style="display:none"></div>
      <div class="panel" id="etl-panel" style="margin-top:12px"><h3 style="margin-top:0">Extract</h3><div id="etl" class="small muted">…</div></div>
      <div id="job-out" style="margin-top:12px"></div>
    </div>
  </div>`;
  renderPipelines(slug, d);
  const run = (kind, options = {}) => runJob(kind, slug, options, $("#job-out"));
  $("#j-verify").onclick = () => run("verify");
  $("#j-build").onclick = () => run("build");
  $("#j-rebuild").onclick = () => run("reconstruct");
  $("#j-reprocess").onclick = () => confirm(`Reprocess '${slug}': every page is rebuilt from raw and cached OCR/LLM results are discarded (the LLM pass re-runs and costs tokens). Continue?`) && run("reprocess");
  const selected = () => [...view.querySelectorAll(".sel:checked")].map((c) => Number(c.value));
  const syncSel = () => { const n = selected().length; $("#del-sel").disabled = !n; $("#sel-count").textContent = n ? `${n} selected` : ""; };
  view.querySelectorAll(".sel").forEach((c) => (c.onchange = syncSel));
  $("#sel-all").onchange = (e) => { view.querySelectorAll(".sel").forEach((c) => (c.checked = e.target.checked)); syncSel(); };
  $("#del-sel").onclick = () => deletePages(slug, selected());
  $("#j-delete-doc").onclick = () => deleteDocument(slug, d.pages);
  const drop = $("#drop"), files = $("#files");
  drop.onclick = () => files.click();
  files.onchange = () => upload(slug, files.files, $("#job-out"));
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); upload(slug, e.dataTransfer.files, $("#job-out")); };
}
window.setDocFilter = (s) => { docFilter = s; route(); };
/* ---------- extraction pipelines (see ocrmd/etl.py) ---------- */
async function renderPipelines(slug, d) {
  const box = $("#etl"); if (!box) return;
  let list = [];
  try { list = await api(`/api/documents/${encodeURIComponent(slug)}/pipelines`); } catch { /* offline */ }
  const files = (d.extracted || []).map((f) =>
    `<a href="/api/documents/${encodeURIComponent(slug)}/extracted/${encodeURIComponent(f)}" download>${esc(f)}</a>`).join(" · ");
  if (!list.length) {
    box.innerHTML = `<div class="muted small">No extraction pipelines. Put one in
      <span class="mono">documents/${esc(slug)}/etl/*.yaml</span> (see <span class="mono">examples/cases.pipeline.yaml</span>).</div>
      ${files ? `<div class="small" style="margin-top:6px">Extracted: ${files}</div>` : ""}`;
    return;
  }
  box.innerHTML = `
    <div class="row small">
      <select id="etl-pick">${list.map((p) => `<option value="${esc(p.file)}" ${p.error ? "disabled" : ""}>${esc(p.name)}${p.error ? " (broken)" : ""}</option>`).join("")}</select>
      <button class="small" id="etl-dry" title="show what it would ask, spend nothing">Dry run</button>
      <button class="small primary" id="etl-run">Extract</button>
      <label class="muted" style="margin:0">first <input id="etl-limit" type="number" min="1" style="width:70px" placeholder="all"> page(s)</label>
    </div>
    <div class="muted small" id="etl-desc"></div>
    ${files ? `<div class="small" style="margin-top:6px">Extracted: ${files}</div>` : ""}`;
  const describe = () => {
    const p = list.find((x) => x.file === $("#etl-pick").value) || {};
    $("#etl-desc").innerHTML = p.error
      ? `<span class="err">${esc(p.error)}</span>`
      : `${(p.steps || []).map(esc).join(" → ")} · <span class="${(p.llm_steps || []).length ? "warn" : "muted"}">${(p.llm_steps || []).length} operator(s) cost tokens</span>`;
  };
  $("#etl-pick").onchange = describe; describe();
  const go = (dry) => runJob("etl", slug, {
    pipeline: $("#etl-pick").value, dry_run: dry,
    limit: Number($("#etl-limit").value) || null,
  }, $("#job-out"));
  $("#etl-dry").onclick = () => go(true);
  $("#etl-run").onclick = () => go(false);
}

async function upload(slug, fileList, out) {
  if (!fileList.length) return;
  const fd = new FormData();
  for (const f of fileList) fd.append("files", f, f.name);
  try {
    const job = await api(`/api/documents/${encodeURIComponent(slug)}/ingest`, { method: "POST", body: fd });
    toast(`ingesting ${fileList.length} file(s)`, "ok"); followJob(job.id, out, () => route());
  } catch (e) { toast(e.message, "err"); }
}
async function runJob(kind, doc, options, out) {
  try {
    const job = await api("/api/jobs", { method: "POST", body: { kind, doc, options } });
    toast(`${kind} started`, "ok"); followJob(job.id, out, () => route());
  } catch (e) { toast(e.message, "err"); }
}
function followJob(id, out, onDone) {
  if (!out) return;
  out.innerHTML = `<div class="panel"><div class="row"><b id="jt"></b><span class="right muted small" id="jp"></span></div><div class="progress" style="margin:6px 0"><i id="jb" style="width:0"></i></div><div class="log" id="jl" style="height:200px"></div></div>`;
  let offset = 0; const lines = [];
  const id2 = every(700, async () => {
    try {
      const j = await api(`/api/jobs/${id}?since=${offset}`);
      lines.push(...j.log); offset += j.log.length;
      $("#jt").textContent = `${j.kind} ${j.doc || ""} — ${j.status}`;
      $("#jp").textContent = j.total ? `${j.done}/${j.total}` : "";
      $("#jb").style.width = j.total ? `${(100 * j.done) / j.total}%` : (j.status === "done" ? "100%" : "0");
      const l = $("#jl"); if (l) { l.innerHTML = logHtml(lines); l.scrollTop = l.scrollHeight; }
      if (["done", "failed", "cancelled"].includes(j.status)) { clearInterval(id2); if (j.status === "failed") toast(j.error, "err"); if (onDone && j.status === "done") setTimeout(onDone, 800); }
    } catch (_) { clearInterval(id2); }
  });
}

/* ---------- deletion ---------- */
async function deletePages(slug, sequences, opts = {}) {
  if (!sequences.length) return;
  const label = sequences.length === 1 ? `page ${String(sequences[0]).padStart(4, "0")}` : `${sequences.length} pages`;
  const whole = opts.askWhole && confirm(`Also delete the other half of the same capture(s) (the spread partner)?\n\nOK = both halves, Cancel = just ${label}.`);
  if (!confirm(`Delete ${label} from '${slug}'?\n\nRemoves the page file(s), OCR/LLM artifacts and — when nothing else refers to it — the raw capture. This cannot be undone.`)) return;
  try {
    const r = await api(`/api/documents/${encodeURIComponent(slug)}/pages/delete`, { method: "POST", body: { sequences, purge_raw: true, whole_capture: !!whole } });
    toast(r.summary, "ok");
    if (opts.after) opts.after(r); else route();
  } catch (e) { toast(e.message, "err"); }
}
async function deleteDocument(slug, pages) {
  const typed = prompt(`Delete document '${slug}' with ${pages} page(s), its files and raw captures?\nThis cannot be undone. Type the slug to confirm:`);
  if (typed == null) return;
  if (typed !== slug) return toast("slug did not match; nothing deleted", "err");
  try {
    const r = await api(`/api/documents/${encodeURIComponent(slug)}?confirm=${encodeURIComponent(slug)}`, { method: "DELETE" });
    toast(`deleted '${slug}': ${r.summary}`, "ok");
    location.hash = "#/";
  } catch (e) { toast(e.message, "err"); }
}

/* ---------- page ---------- */
let pageTab = "edit", imgKind = "flat", imgZoom = 1, reviewMode = false;
let sel = null, selPage = null, readout = null;   // the dragged rectangle, whose page, and what it read
async function pageView(slug, seq) {
  const [p] = await Promise.all([api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}`), sinksStatus()]);
  const fm = p.frontmatter || {}, v = fm.verification || {}, ctx = fm.context || {}, proc = fm.processing || {};
  const kinds = Object.keys(p.images);
  if (!kinds.includes(imgKind)) imgKind = kinds[0];
  view.innerHTML = `
  <div class="row" style="margin-bottom:10px">
    <a href="#/doc/${esc(slug)}">${esc(slug)}</a> <span class="muted">/</span> <b>page ${String(seq).padStart(4, "0")}</b>
    ${p.page_label ? `<span class="muted">printed ${esc(p.page_label)}</span>` : ""} ${p.part ? `<span class="muted small">${p.part}</span>` : ""}
    ${badge(p.status)} <span class="muted small">agree ${fmtNum(v.agreement)} · overlap ${fmtNum(v.overlap)} · ${p.width_px || "?"}px</span>
    <span class="right"></span>
    ${reviewMode ? '<span class="chip on">review mode</span>' : ""}
    <button ${p.prev == null ? "disabled" : ""} onclick="go('${esc(slug)}',${p.prev})" title="p">◀ prev</button>
    <button ${p.next == null ? "disabled" : ""} onclick="go('${esc(slug)}',${p.next})" title="n">next ▶</button>
    <button ${p.next_flagged == null ? "disabled" : ""} onclick="go('${esc(slug)}',${p.next_flagged})" title="shift+n">next flagged ⏭</button>
    ${p.status === "reviewed" ? `<button id="reopen">Reopen</button>` : `<button class="primary" id="signoff" ${p.unresolved ? "disabled" : ""} title="${p.unresolved ? p.unresolved + " unresolved" : "sign off as reviewed"}">Sign off</button>`}
    ${sendButton("send-open", "Send…", "hand this page's reading to a destination — a study platform, a knowledge base")}
    <button class="danger" id="delete-page" title="delete this page">Delete…</button>
  </div>
  <div class="grid page">
    <div>
      <div class="row small" style="margin-bottom:6px">
        ${kinds.map((k) => `<span class="chip ${k === imgKind ? "on" : ""}" onclick="setImg('${k}')" title="${IMG_HELP[k] || ""}">${k}</span>`).join("")}
        <span class="right"></span><button class="small" onclick="zoomImg(-0.25)">−</button><span class="mono" id="zoomlbl">${Math.round(imgZoom * 100)}%</span><button class="small" onclick="zoomImg(0.25)">+</button><button class="small" onclick="zoomImg(0)">fit</button>
      </div>
      <div class="imgbox" id="imgbox"><div class="imgwrap" id="imgwrap" style="width:${imgZoom * 100}%"><img id="pimg" src="${p.images[imgKind] || ""}"><div class="selrect" id="selrect" hidden></div></div></div>
      <div id="selbar" class="selbar small"></div>
      <div id="send-panel"></div>
      ${p.advice ? `<div class="warn small" style="margin-top:6px">${esc(p.advice)}</div>` : ""}
      ${proc.continuity ? `<div class="warn small">${esc(proc.continuity)}</div>` : ""}
      ${p.llm_error ? `<div class="err small">LLM: ${esc(p.llm_error)}</div>` : ""}
      ${v.illegible_reason ? `<div class="muted small">model: ${esc(v.illegible_reason)}</div>` : ""}
    </div>
    <div>
      <div class="tabs">${["edit", "preview", "meta"].map((t) => `<button class="${t === pageTab ? "on" : ""}" onclick="setTab('${t}')">${t}${t === "edit" && p.unresolved ? ` (${p.unresolved})` : ""}</button>`).join("")}</div>
      <div id="tab"></div>
    </div>
  </div>`;
  attachSelect(slug, seq);
  renderSend(slug, seq, null, p.unresolved);
  const btn = $("#send-open");
  if (btn) btn.onclick = () => { sendOpen = !sendOpen; sendOut = null; renderSend(slug, seq, null, p.unresolved); };
  const tab = $("#tab");
  if (pageTab === "edit") {
    tab.innerHTML = `
      ${p.flags.length ? `<h3 style="margin-top:0">Disagreements</h3>` + p.flags.map((f) => `
        <div class="flag" data-n="${f.n}">
          <div class="ctx">${esc(f.context).replace(/\[(.+?)\]/, "<b>[$1]</b>")}</div>
          <div class="opts">
            <button class="small" onclick="resolveFlag('${esc(slug)}',${seq},${f.n},${JSON.stringify(f.ocr).replace(/"/g, "&quot;")})">OCR: <span class="val">${esc(f.ocr)}</span></button>
            <button class="small" onclick="resolveFlag('${esc(slug)}',${seq},${f.n},${JSON.stringify(f.llm).replace(/"/g, "&quot;")})">LLM: <span class="val">${esc(f.llm)}</span></button>
            <button class="small" onclick="resolveCustom('${esc(slug)}',${seq},${f.n})">type…</button>
          </div>
        </div>`).join("") : '<div class="muted small" style="margin-bottom:8px">No open disagreements.</div>'}
      <label>Markdown body</label>
      <textarea id="body" rows="22">${esc(p.body)}</textarea>
      <div class="row" style="margin-top:8px"><button class="primary" id="save">Save body</button><span class="muted small">Saving marks the page as hand-edited; a later verify keeps your text only once signed off.</span></div>`;
    $("#save").onclick = async () => {
      try { await api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}`, { method: "PUT", body: { body: $("#body").value } }); toast("saved", "ok"); route(); } catch (e) { toast(e.message, "err"); }
    };
  } else if (pageTab === "preview") {
    tab.innerHTML = `<div class="md">${renderMd(p.body) || '<span class="muted">empty page</span>'}</div>`;
  } else {
    const field = (k, val, ph = "") => `<label>${k}</label><input data-ctx="${k}" value="${esc(Array.isArray(val) ? val.join(", ") : val ?? "")}" placeholder="${ph}" style="width:100%">`;
    tab.innerHTML = `
      <div class="grid cols-2">
        <div><label>printed page label</label><input id="m-label" value="${esc(p.page_label || "")}" style="width:100%"></div>
        <div>${field("chapter", ctx.chapter)}</div><div>${field("section", ctx.section)}</div><div>${field("running_head", ctx.running_head)}</div>
        <div>${field("title", ctx.title)}</div><div>${field("tags", ctx.tags, "comma separated")}</div>
      </div>
      <div class="row" style="margin:10px 0"><button class="primary" id="save-meta">Save metadata</button></div>
      <h3>Verification</h3>
      <dl class="kv">
        <dt>status</dt><dd>${badge(v.status || p.status)}</dd>
        <dt>agreement / overlap / cer</dt><dd>${fmtNum(v.agreement)} / ${fmtNum(v.overlap)} / ${fmtNum(v.cer)}</dd>
        <dt>flagged spans</dt><dd>${v.flagged_spans ?? 0}</dd>
        ${v.reviewed_by ? `<dt>reviewed by</dt><dd>${esc(v.reviewed_by.who || v.reviewed_by)} ${esc(v.reviewed_by.at || "")}</dd>` : ""}
        ${v.llm_uncertain?.length ? `<dt>model unsure of</dt><dd class="mono small">${v.llm_uncertain.map(esc).join(", ")}</dd>` : ""}
        ${v.llm_notes ? `<dt>model notes</dt><dd class="small">${esc(v.llm_notes)}</dd>` : ""}
      </dl>
      <h3>Provenance</h3>
      <dl class="kv small">
        <dt>captured</dt><dd>${esc(fm.captured_at || "")} via ${esc(fm.source?.type || "")} ${esc(fm.source?.device || "")}</dd>
        <dt>processing</dt><dd>${esc(proc.pipeline || "")} · orientation ${proc.orientation ?? ""}° · deskew ${proc.deskew ?? ""}° · blur ${proc.blur ?? ""} · ${proc.page_width_px ?? ""}px${proc.clipped_edges?.length ? " · clipped " + proc.clipped_edges.join("+") : ""}</dd>
        <dt>ocr</dt><dd>${esc(fm.ocr?.engine || "")} conf ${fmtNum(fm.ocr?.confidence)}${fm.ocr?.agreement != null ? ` · engines agree ${fmtNum(fm.ocr.agreement)}` : ""}</dd>
        ${fm.ocr?.focus ? `<dt>focus</dt><dd>${fm.ocr.focus.variants} bracketed exposure(s) · ${fm.ocr.focus.improved} line(s) improved · ${fm.ocr.focus.filled} filled · ${fm.ocr.focus.unresolved} unresolved</dd>` : ""}
        <dt>llm</dt><dd>${fm.llm ? `${esc(fm.llm.provider)}/${esc(fm.llm.model)} ${esc(fm.llm.prompt)} · tokens ${fm.llm.tokens?.in ?? "?"}→${fm.llm.tokens?.out ?? "?"} · <span class="mono">${esc(fm.llm.request_id || "")}</span>` : "none"}</dd>
        <dt>id</dt><dd class="mono">${esc(p.id)}</dd>
        <dt>file</dt><dd class="mono">documents/${esc(slug)}/pages/${String(seq).padStart(4, "0")}.md</dd>
      </dl>`;
    $("#save-meta").onclick = async () => {
      const context = {};
      tab.querySelectorAll("[data-ctx]").forEach((i) => { const k = i.dataset.ctx; context[k] = k === "tags" ? i.value.split(",").map((x) => x.trim()).filter(Boolean) : (i.value || null); });
      try { await api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}`, { method: "PUT", body: { page_label: $("#m-label").value, context } }); toast("saved", "ok"); route(); } catch (e) { toast(e.message, "err"); }
    };
  }
  const so = $("#signoff");
  if (so) so.onclick = async () => {
    const who = localStorage.getItem("ocrmd.who") || prompt("Your name (recorded as reviewer)", "reviewer");
    if (!who) return; localStorage.setItem("ocrmd.who", who);
    try {
      const r = await api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}/signoff`, { method: "POST", body: { who } });
      toast(`page ${seq} signed off`, "ok");
      if (reviewMode && r.next_flagged != null) go(slug, r.next_flagged); else route();
    } catch (e) { toast(e.message, "err"); }
  };
  $("#delete-page").onclick = () => deletePages(slug, [seq], {
    askWhole: !!p.part,
    after: (r) => {   // the spread partner may have gone too: pick the nearest page that still exists
      const to = [p.next, p.prev].find((x) => x != null && !(r.deleted || []).includes(x));
      location.hash = to != null ? `#/doc/${slug}/page/${to}${reviewMode ? "?review=1" : ""}` : `#/doc/${slug}`;
    },
  });
  const ro = $("#reopen");
  if (ro) ro.onclick = () => api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}/reopen`, { method: "POST" }).then(route).catch((e) => toast(e.message, "err"));
  document.onkeydown = (ev) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (ev.key === "n" && p.next != null) go(slug, p.next);
    if (ev.key === "p" && p.prev != null) go(slug, p.prev);
    if (ev.key === "N" && p.next_flagged != null) go(slug, p.next_flagged);
  };
}
window.go = (slug, seq) => { location.hash = `#/doc/${slug}/page/${seq}${reviewMode ? "?review=1" : ""}`; };
window.setTab = (t) => { pageTab = t; route(); };
window.setImg = (k) => { imgKind = k; route(); };
window.zoomImg = (d) => {
  imgZoom = d === 0 ? 1 : Math.max(0.25, Math.min(4, imgZoom + d));
  const wrap = $("#imgwrap"); if (wrap) wrap.style.width = `${imgZoom * 100}%`;
  const lbl = $("#zoomlbl"); if (lbl) lbl.textContent = `${Math.round(imgZoom * 100)}%`;
};

/* ---------- sending a page to a destination ----------
   ocrmd's one outbound path (ocrmd/sinks.py): the reconciled reading of a page (or of one box on it)
   is handed to a receiver that speaks the sink contract — a study platform, a knowledge base, a
   notebook. The text travels, the photograph does not; the page id goes with it so a reviewer over
   there can come back here. Each receiver declares its own options in a descriptor and the panel
   draws them. The whole UI is absent unless a destination is configured. */
let sinks = null;                                 // the /api/sinks probe, fetched once per load
let sendOpen = false, sendOut = null, sendSink = null;

async function sinksStatus() {
  if (sinks === null) sinks = await api("/api/sinks").catch(() => []);
  return sinks;
}

function sendButton(id, label, title) {
  return sinks && sinks.length ? `<button class="small" id="${id}" title="${esc(title)}">${label}</button>` : "";
}

/* One input per declared option, typed as the receiver said: a select for an enum, a checkbox for
   a boolean, a number box, a text box. A blank means "let the receiver decide". */
function optionField(o, val) {
  const cur = val ?? o.default ?? "";
  const key = esc(o.key), label = esc(o.label || o.key), help = esc(o.help || "");
  if (o.type === "enum") return `<select class="small" data-opt="${key}" title="${help}"><option value="">${label}: auto</option>${(o.choices || []).map((c) => `<option ${String(c) === String(cur) ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
  if (o.type === "boolean") return `<label style="margin:0" title="${help}"><input type="checkbox" data-opt="${key}" ${cur === true || String(cur) === "true" ? "checked" : ""}> ${label}</label>`;
  const num = o.type === "integer" || o.type === "number";
  return `<input class="small" type="${num ? "number" : "text"}" data-opt="${key}" value="${esc(cur)}" placeholder="${label}${o.help ? " — " + help : ""}" style="${num ? "width:90px" : "flex:1;min-width:160px"}" title="${help}">`;
}

/* The page-level panel: pick a destination, fill its options, send. `region` is set when the send
   comes from a region readout, and then the readout's text is what travels rather than the page body. */
function renderSend(slug, seq, region, unresolved = 0) {
  const box = $("#send-panel"); if (!box) return;
  if (!sendOpen || !sinks || !sinks.length) { box.innerHTML = ""; return; }
  const cur = sinks.find((s) => s.name === sendSink) || sinks[0];
  sendSink = cur.name;
  const d = cur.descriptor || {};
  const defaults = d.defaults || {};
  box.innerHTML = `
    <div class="panel" style="margin-top:8px">
      <div class="row">
        <b>Send</b>
        ${sinks.length > 1 ? `<select id="send-sink" class="small">${sinks.map((s) => `<option ${s.name === cur.name ? "selected" : ""}>${esc(s.name)}</option>`).join("")}</select>` : `<b>→ ${esc(d.name || cur.name)}</b>`}
        <span class="muted small">${region ? "the region reading" : "this page"} — the text and its provenance travel; the photograph does not</span>
        <span class="right"></span><button class="small" id="send-close">close</button>
      </div>
      ${d.description ? `<div class="muted small" style="margin-top:4px">${esc(d.description)}</div>` : ""}
      ${d.reachable === false ? `<div class="err small" style="margin-top:4px">${esc(d.error || "unreachable")}</div>` : ""}
      <div class="row" style="margin-top:6px">
        ${(d.options || []).map((o) => optionField(o, defaults[o.key])).join("")}
        <button class="small primary" id="send-go">Send to ${esc(d.name || cur.name)}</button>
      </div>
      <div class="muted small" style="margin-top:4px">${esc(cur.url)}</div>
      ${!region && unresolved ? `<div class="warn small">${unresolved} unresolved disagreement${unresolved === 1 ? "" : "s"} in this page — the marks travel with it, and so does the count</div>` : ""}
      <div id="send-out"></div>`;
  const pick = $("#send-sink");
  if (pick) pick.onchange = () => { sendSink = pick.value; sendOut = null; renderSend(slug, seq, region, unresolved); };
  $("#send-close").onclick = () => { sendOpen = false; sendOut = null; renderSend(slug, seq, region, unresolved); };
  $("#send-go").onclick = () => sendPage(slug, seq, region);
  box.querySelectorAll("input[type=text]").forEach((i) => (i.onkeydown = (e) => { if (e.key === "Enter") sendPage(slug, seq, region); }));
  const first = box.querySelector("[data-opt]"); if (first) first.focus();
  renderSendOut();
}

async function sendPage(slug, seq, region) {
  const out = $("#send-out");
  out.innerHTML = '<div class="muted small" style="margin-top:8px">sending…</div>';
  const options = {};
  $("#send-panel").querySelectorAll("[data-opt]").forEach((el) => { options[el.dataset.opt] = el.type === "checkbox" ? el.checked : el.value; });
  const body = { sink: sendSink, options };
  if (region) { body.text = region.text; body.bbox = region.bbox; }
  try {
    sendOut = await api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}/send`, { method: "POST", body });
    renderSendOut();
    toast(sendOut.summary, sendOut.items.length ? "ok" : "");
  } catch (e) { out.innerHTML = ""; toast(e.message, "err"); }
}

/* What came back: the receiver's one-line summary, one row per item it made, and its warnings. */
function renderSendOut() {
  const out = $("#send-out"); if (!out || !sendOut) return;
  const r = sendOut;
  out.innerHTML = `
    <div class="${r.items.length ? "ok" : "warn"} small" style="margin-top:8px">${esc(r.summary)}</div>
    ${r.items.map((it) => `
      <div class="readout" style="margin-top:6px"><div class="row small">
        ${it.status ? `<span class="chip on">${esc(it.status)}</span>` : ""}<b>${esc(it.title || "")}</b>
        <span class="muted">${esc(it.detail || "")}</span><span class="right"></span><span class="mono muted">${esc(it.id || "")}</span>
      </div></div>`).join("")}
    ${r.warnings.map((w) => `<div class="warn small">${esc(w)}</div>`).join("")}`;
}

/* ---------- selecting a region of the page ---------- */
const IMG_HELP = {
  flat: "flattened and enhanced for reading — illumination divided out, ink and paper stretched apart",
  processed: "the cropped, rotated page as the LLM sees it",
  bin: "adaptive threshold; what older OCR engines were given",
  raw: "the untouched camera frame",
};

function pct(x) { return `${(x * 100).toFixed(1)}%`; }

function drawSel() {
  const box = $("#selrect"); if (!box) return;
  if (!sel) { box.hidden = true; return; }
  box.hidden = false;
  Object.assign(box.style, { left: pct(sel.x), top: pct(sel.y), width: pct(sel.w), height: pct(sel.h) });
}

/* The rectangle is kept in page fractions, not pixels: it has to survive zooming, switching between
   the flat and processed views, and the re-render that follows every read. */
function attachSelect(slug, seq) {
  const wrap = $("#imgwrap"); if (!wrap) return;
  if (selPage !== `${slug}/${seq}`) { sel = null; readout = null; selPage = `${slug}/${seq}`; }
  drawSel(); renderSelbar(slug, seq);
  if (imgKind === "raw") return;                 // the raw frame is not in page coordinates
  wrap.classList.add("selectable");
  let start = null;
  wrap.onpointerdown = (e) => {
    if (e.button !== 0) return;
    const r = wrap.getBoundingClientRect();
    start = { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
    wrap.setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  wrap.onpointermove = (e) => {
    if (!start) return;
    const r = wrap.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
    sel = { x: Math.min(start.x, x), y: Math.min(start.y, y), w: Math.abs(x - start.x), h: Math.abs(y - start.y) };
    drawSel();
  };
  wrap.onpointerup = () => {
    start = null;
    if (sel && sel.w * sel.h < 0.0004) sel = null;    // a stray click, not a request
    drawSel(); renderSelbar(slug, seq);
  };
}

function renderSelbar(slug, seq) {
  const bar = $("#selbar"); if (!bar) return;
  if (!sel) {
    bar.innerHTML = imgKind === "raw"
      ? '<span class="muted">Switch to the flat or processed view to select a region.</span>'
      : '<span class="muted">Drag a box on the page to read that region on its own.</span>';
    return;
  }
  bar.innerHTML = `
    <div class="row">
      <span class="mono">${pct(sel.w)}×${pct(sel.h)} at ${pct(sel.x)},${pct(sel.y)}</span>
      <button class="small primary" id="r-ocr">Read region</button>
      <button class="small" id="r-llm" title="sends this one crop to the model; costs tokens">Ask the model</button>
      <input id="r-ask" class="small" placeholder="ask something about it…" style="flex:1;min-width:140px">
      <button class="small" id="r-clear">clear</button>
    </div>
    <div id="readout"></div>`;
  $("#r-clear").onclick = () => { sel = null; readout = null; drawSel(); renderSelbar(slug, seq); };
  $("#r-ocr").onclick = () => readRegion(slug, seq, false);
  $("#r-llm").onclick = () => readRegion(slug, seq, true);
  $("#r-ask").onkeydown = (e) => { if (e.key === "Enter") readRegion(slug, seq, true); };
  renderReadout(slug, seq);
}

async function readRegion(slug, seq, withLlm) {
  if (!sel) return;
  const question = ($("#r-ask")?.value || "").trim() || null;
  const out = $("#readout");
  out.innerHTML = `<div class="muted small" style="margin-top:8px">reading${withLlm ? " and asking the model" : ""}…</div>`;
  try {
    readout = await api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}/region`, {
      method: "POST", body: { ...sel, llm: withLlm, question },
    });
    renderReadout(slug, seq);
  } catch (e) { out.innerHTML = ""; toast(e.message, "err"); }
}

function renderReadout(slug, seq) {
  const out = $("#readout"); if (!out) return;
  if (!readout) { out.innerHTML = ""; return; }
  const r = readout;
  out.innerHTML = `
    <div class="readout">
      <img src="${esc(r.image)}" alt="the crop as it was read">
      <div class="row small muted">${r.size[0]}×${r.size[1]}px enlarged ${r.scale}× · ${esc(r.engine || "no OCR engine")}${r.text ? ` · conf ${fmtNum(r.confidence)}` : ""}</div>
      <label>local OCR</label>
      <pre class="mono" id="r-text">${esc(r.text) || '<span class="muted">nothing read</span>'}</pre>
      ${r.llm ? `<label>model</label><pre class="mono" id="r-llmtext">${esc(r.llm)}</pre>` : ""}
      ${r.llm_error ? `<div class="err small">${esc(r.llm_error)}</div>` : ""}
      <div class="row">
        <button class="small" data-copy="text">Copy OCR</button>
        ${r.llm ? '<button class="small" data-copy="llm">Copy model</button>' : ""}
        <button class="small" data-insert="${r.llm ? "llm" : "text"}">Insert into body</button>
        ${sendButton("r-send", "Send…", "send this region's reading to a destination")}
      </div>
    </div>`;
  out.querySelectorAll("[data-copy]").forEach((b) => (b.onclick = () => {
    navigator.clipboard.writeText(b.dataset.copy === "llm" ? r.llm : r.text).then(() => toast("copied", "ok"));
  }));
  const ins = out.querySelector("[data-insert]");
  if (ins) ins.onclick = () => insertIntoBody(ins.dataset.insert === "llm" ? r.llm : r.text);
  const toSend = out.querySelector("#r-send");
  if (toSend) toSend.onclick = () => {
    // The model's reading when there is one: a region is re-read at several times the page's glyph
    // height, and that reading is the better evidence about what the box says.
    sendOpen = true; sendOut = null;
    renderSend(slug, seq, { text: r.llm || r.text, bbox: [r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]] });
    $("#send-panel")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
}

/* Inserted at the cursor, and only into the editor: a region read is one unreconciled reading, so it
   is a suggestion for the person looking at the crop, never something the pipeline writes by itself. */
function insertIntoBody(text) {
  const ta = $("#body");
  if (!ta) { toast("open the edit tab to insert", "err"); return; }
  const at = ta.selectionStart ?? ta.value.length;
  ta.value = ta.value.slice(0, at) + text + ta.value.slice(ta.selectionEnd ?? at);
  ta.focus(); ta.selectionStart = ta.selectionEnd = at + text.length;
  toast("inserted — remember to save", "ok");
}
window.resolveFlag = async (slug, seq, n, chosen) => {
  try { await api(`/api/documents/${encodeURIComponent(slug)}/pages/${seq}/resolve`, { method: "POST", body: { n, chosen } }); route(); } catch (e) { toast(e.message, "err"); }
};
window.resolveCustom = (slug, seq, n) => { const t = prompt("Replacement text"); if (t != null) resolveFlag(slug, seq, n, t); };

async function reviewView(slug) {
  const d = await api(`/api/documents/${encodeURIComponent(slug)}`);
  const first = d.page_list.find((p) => p.status === "flagged");
  reviewMode = true;
  if (!first) { view.innerHTML = `<div class="panel">No flagged pages in <b>${esc(slug)}</b>. <a href="#/doc/${esc(slug)}">Back to the document</a>.</div>`; return; }
  location.replace(`#/doc/${slug}/page/${first.sequence}?review=1`);
}

/* ---------- jobs ---------- */
async function jobsView() {
  const [jobs, docs] = await Promise.all([api("/api/jobs"), api("/api/documents")]);
  view.innerHTML = `
  <div class="grid side">
    <div>
      <h1>Jobs</h1>
      <table><tr><th>when</th><th>kind</th><th>document</th><th>status</th><th class="num">progress</th><th></th></tr>
      ${jobs.map((j) => `<tr class="click" onclick="showJob('${j.id}')"><td class="small">${fmtTime(j.created_at)}</td><td>${esc(j.kind)}</td><td>${esc(j.doc || "all")}</td>
        <td class="${j.status === "failed" ? "err" : j.status === "done" ? "ok" : ""}">${esc(j.status)}${j.error ? ` — ${esc(j.error)}` : ""}</td>
        <td class="num">${j.total ? `${j.done}/${j.total}` : ""}</td>
        <td>${["queued", "running"].includes(j.status) ? `<button class="small" onclick="event.stopPropagation();cancelJob('${j.id}')">cancel</button>` : ""}</td></tr>`).join("") || '<tr><td colspan="6" class="muted">no jobs yet</td></tr>'}
      </table>
      <div id="job-out" style="margin-top:12px"></div>
    </div>
    <div class="panel">
      <h3 style="margin-top:0">Run</h3>
      <label>Kind</label><select id="k" style="width:100%"><option>verify</option><option>build</option><option>reconstruct</option><option>reprocess</option></select>
      <label>Document</label><select id="d" style="width:100%"><option value="">all</option>${docs.map((x) => `<option>${esc(x.slug)}</option>`).join("")}</select>
      <label><input type="checkbox" id="o-flag"> flagged only (verify)</label>
      <label><input type="checkbox" id="o-refresh"> refresh: re-run pages made by an older prompt/model — costs tokens</label>
      <label><input type="checkbox" id="o-nollm"> OCR only (no LLM)</label>
      <div style="margin-top:10px"><button class="primary" id="run">Run</button></div>
    </div>
  </div>`;
  $("#run").onclick = () => {
    const kind = $("#k").value, doc = $("#d").value || null;
    if (kind === "reprocess" && !doc) return toast("reprocess needs a document", "err");
    if (kind === "reprocess" && !confirm("Reprocess discards cached OCR/LLM results and re-runs the LLM pass (costs tokens). Continue?")) return;
    runJob(kind, doc, { flagged_only: $("#o-flag").checked, refresh: $("#o-refresh").checked, no_llm: $("#o-nollm").checked }, $("#job-out"));
  };
  every(2000, async () => { if (location.hash.startsWith("#/jobs") && !$("#jl")) jobsView(); });
}
window.showJob = (id) => followJob(id, $("#job-out"));
window.cancelJob = (id) => api(`/api/jobs/${id}/cancel`, { method: "POST" }).then(() => route()).catch((e) => toast(e.message, "err"));

/* ---------- settings ---------- */
async function settingsView(slug) {
  const [settings, doctor, docs] = await Promise.all([api("/api/settings"), api("/api/doctor"), api("/api/documents")]);
  slug = slug || docs[0]?.slug;
  const meta = slug ? await api(`/api/documents/${encodeURIComponent(slug)}/meta`) : null;
  const F = ["title", "author", "edition", "publisher", "year", "isbn", "citation", "citation_key", "kind", "tags", "project"];
  view.innerHTML = `
  <div class="grid cols-2">
    <div>
      <h1>Document metadata</h1>
      <div class="panel">
        <label>Document</label><select id="m-doc" style="width:100%">${docs.map((d) => `<option ${d.slug === slug ? "selected" : ""}>${esc(d.slug)}</option>`).join("")}</select>
        ${meta ? F.map((k) => `<label>${k}</label><input data-k="${k}" value="${esc(Array.isArray(meta[k]) ? meta[k].join(", ") : meta[k] ?? "")}" style="width:100%">`).join("") : ""}
        <div class="row" style="margin-top:12px"><button class="primary" id="save-doc" ${meta ? "" : "disabled"}>Save document.yaml</button><span class="muted small">Inherited into every page on build; a page's own value wins.</span></div>
      </div>
      <h2>Effective settings</h2>
      <div class="panel"><dl class="kv small">${Object.entries(settings).map(([k, v]) => `<dt>${esc(k)}</dt><dd class="mono">${esc(JSON.stringify(v))}</dd>`).join("")}</dl>
        <div class="muted small" style="margin-top:8px">Set in <code>ocrmd.toml</code> or <code>OCRMD_*</code> environment variables; restart the server to apply.</div></div>
    </div>
    <div>
      <h1>Doctor</h1>
      <div class="panel"><dl class="kv">
        <dt>root</dt><dd class="mono">${esc(doctor.root)}</dd>
        <dt>OCR</dt><dd>${esc(doctor.ocr_engine || "NONE — macOS: pyobjc-framework-Vision; elsewhere pip install 'ocrmd[rapidocr]'")}
          <span class="muted">installed: ${esc((doctor.ocr_engines_installed || []).join(", ") || "none")}</span></dd>
        <dt>anthropic</dt><dd>${doctor.credentials.anthropic ? '<span class="ok">credentials found</span>' : '<span class="warn">no credentials (ANTHROPIC_API_KEY or CLAUDE_CODE_OAUTH_TOKEN in .env)</span>'}</dd>
        <dt>xai</dt><dd>${doctor.credentials.xai ? '<span class="ok">XAI_API_KEY set</span>' : '<span class="muted">no credentials</span>'}</dd>
        <dt>dictionary</dt><dd>${doctor.dictionary ? "present" : '<span class="warn">/usr/share/dict/words missing</span>'}</dd>
        <dt>camera access</dt><dd>${doctor.camera_authorized === false ? '<span class="err">NOT authorized — System Settings › Privacy & Security › Camera</span>' : "ok"}</dd>
        <dt>database</dt><dd><span class="mono">${esc(doctor.db_path)}</span> ${doctor.db_integrity === "ok" ? '<span class="ok">ok</span>' : `<span class="err">${esc(doctor.db_integrity)}</span>`}</dd>
        <dt>artifacts</dt><dd>${doctor.missing_artifacts.raw.length} without raw, ${doctor.missing_artifacts.processed.length} without processed, ${doctor.orphan_artifacts} orphan(s)</dd>
        <dt>batches</dt><dd>${doctor.open_batches} open</dd>
      </dl></div>
      <h2>Cameras</h2>
      <div class="panel">${doctor.cameras.map((c) => `<div><b>${c.index}: ${esc(c.name)}</b> ${c.format ? `${c.format[0]}×${c.format[1]}` : ""} <span class="muted small">focus ${esc(c.focus || "?")} (${(c.focus_modes || []).join(", ") || "fixed"}) · exposure ${esc(c.exposure || "?")}</span>${c.warning ? `<div class="warn small">${esc(c.warning)}</div>` : ""}</div>`).join("") || '<span class="muted">none found</span>'}</div>
      <h2>Documents</h2>
      <div class="panel">${doctor.documents.map((d) => `<div><b>${esc(d.slug)}</b> ${d.pages} page(s): ${Object.entries(d.counts).map(([k, n]) => `${n} ${k}`).join(", ")}</div>`).join("") || '<span class="muted">none</span>'}</div>
    </div>
  </div>`;
  $("#m-doc").onchange = () => { location.hash = `#/settings/${$("#m-doc").value}`; };
  const sd = $("#save-doc");
  if (sd) sd.onclick = async () => {
    const body = {};
    view.querySelectorAll("[data-k]").forEach((i) => { const k = i.dataset.k; body[k] = ["author", "tags"].includes(k) ? i.value.split(",").map((x) => x.trim()).filter(Boolean) : k === "year" ? (i.value ? Number(i.value) : null) : (i.value || null); });
    try { await api(`/api/documents/${encodeURIComponent(slug)}/meta`, { method: "PUT", body }); toast("document.yaml saved — run Build to inherit into pages", "ok"); } catch (e) { toast(e.message, "err"); }
  };
}

/* ---------- router ---------- */
async function route() {
  clearTimers();
  document.onkeydown = null;
  const hash = location.hash || "#/";
  const [path, query] = hash.slice(1).split("?");
  const parts = path.split("/").filter(Boolean);
  const q = new URLSearchParams(query || "");
  reviewMode = q.get("review") === "1";
  if (q.get("send") === "1") { sendOpen = true; sendOut = null; }
  const nav = parts[0] || "dash";
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav || (nav === "doc" && a.dataset.nav === "dash")));
  try {
    if (!parts.length) await dashboard();
    else if (parts[0] === "scan") await scanView();
    else if (parts[0] === "search") await searchView(q.get("q") || "");
    else if (parts[0] === "jobs") await jobsView();
    else if (parts[0] === "settings") await settingsView(parts[1] && decodeURIComponent(parts[1]));
    else if (parts[0] === "review") await reviewView(decodeURIComponent(parts[1]));
    else if (parts[0] === "doc" && parts[2] === "page") await pageView(decodeURIComponent(parts[1]), Number(parts[3]));
    else if (parts[0] === "doc") await docView(decodeURIComponent(parts[1]));
    else view.innerHTML = '<div class="panel">not found</div>';
  } catch (e) {
    view.innerHTML = `<div class="panel err">${esc(e.message)}</div>`;
  }
  refreshTop();
}
window.addEventListener("hashchange", route);
every(3000, refreshTop);
route();
