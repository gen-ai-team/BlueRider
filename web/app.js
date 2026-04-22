/* -------------------------------------------------------------------- *
 * Exhibition viewer — client
 *
 * Polls /api/state, diffs against the last snapshot, and re-renders only
 * the pieces that changed. Images are preloaded before being swapped in
 * so the hero never flashes half-loaded.
 *
 * Fullscreen viewer:
 *   - Click the hero or any thumbnail -> single-work mode for that iteration.
 *   - Click a series title/card -> series-overview mode.
 *   - ← / →   navigate within the current series (single-work mode).
 *   - S       jump from a work to its series overview.
 *   - Esc     close.
 * -------------------------------------------------------------------- */

const POLL_MS = 4000;
const IMG_BASE = "/output/";
const FOLDERS_POLL_MS = 30000;   // re-scan available folders periodically
const FOLDER_STORAGE_KEY = "imggen.folder";

const el = (id) => document.getElementById(id);
const pad3 = (n) => String(n).padStart(3, "0");

/** Currently-selected folder name (server-side directory name). `null` means
 *  use the server's default — we don't pass a ?folder= argument. */
let CURRENT_FOLDER = null;
try {
  const saved = localStorage.getItem(FOLDER_STORAGE_KEY);
  if (saved) CURRENT_FOLDER = saved;
} catch (_) { /* private mode etc. */ }

function folderQuery(prefix = "?") {
  if (!CURRENT_FOLDER) return "";
  return prefix + "folder=" + encodeURIComponent(CURRENT_FOLDER);
}

const imgUrl = (n) =>
  `${IMG_BASE}art_${pad3(n)}.png${folderQuery("?")}`;

// ---------------------------------------------------------------------- *
// cached state
// ---------------------------------------------------------------------- */

/** Latest snapshot from /api/state. */
let STATE = null;
/** Map iteration number -> iteration JSON, rebuilt on every fetch. */
let ITERS_BY_NUM = new Map();
/** Map series id -> series object, rebuilt on every fetch. */
let SERIES_BY_ID = new Map();

// Signatures for diff-based render skipping.
let lastSig = {
  current: null,
  seriesSig: null,
  pathSig: null,
};

// Viewer own state.
const viewer = {
  open: false,
  mode: null,       // "work" | "series"
  iterNum: null,
  seriesId: null,
};

// ---------------------------------------------------------------------- *
// tiny helpers
// ---------------------------------------------------------------------- */

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

function preloadImage(src) {
  return new Promise((resolve) => {
    const im = new Image();
    im.onload = () => resolve(src);
    im.onerror = () => resolve(null);
    im.src = src;
  });
}

function plural(count, one, few, many) {
  const n = Math.abs(count) % 100;
  const n10 = n % 10;
  if (n > 10 && n < 20) return many;
  if (n10 > 1 && n10 < 5) return few;
  if (n10 === 1) return one;
  return many;
}

function countWorksLabel(n) {
  return `${n} ${plural(n, "работа", "работы", "работ")}`;
}

// ---------------------------------------------------------------------- *
// main rendering (unchanged panels)
// ---------------------------------------------------------------------- */

async function renderHero(current) {
  if (!current) {
    setText(el("hero-title"), "Ожидаем первую работу…");
    setText(el("hero-series"), "—");
    setText(el("hero-iter"), "—");
    el("hero-img").removeAttribute("src");
    return;
  }
  setText(el("hero-title"), current.title || "—");
  setText(el("hero-series"),
          `Серия ${current.series_id} · ${current.series_name || ""}`.trim());
  setText(el("hero-iter"), `итерация ${pad3(current.iteration)}`);

  const src = IMG_BASE + (current.image_path || `art_${pad3(current.iteration)}.png`) + folderQuery("?");
  const img = el("hero-img");
  if (img.getAttribute("src") !== src) {
    const loaded = await preloadImage(src);
    if (loaded) {
      img.style.opacity = "0";
      img.src = loaded;
      requestAnimationFrame(() => { img.style.opacity = "1"; });
    }
  }
}

function renderThoughts(current) {
  const map = [
    ["t-see",       "what_i_see"],
    ["t-resonates", "what_resonates"],
    ["t-silent",    "what_is_silent"],
    ["t-next",      "where_next"],
  ];
  for (const [id, key] of map) {
    const v = (current && current[key]) || "—";
    setText(el(id), v);
  }
}

function renderPath(path) {
  const body = el("path-body");
  if (!path) {
    body.innerHTML = '<p class="muted">путь ещё не сформирован…</p>';
    return;
  }
  const order = [
    ["opening",           "Начальная мысль"],
    ["what_attracts_me",  "Что меня притягивает"],
    ["what_i_dislike",    "Что мне не нравится"],
    ["what_works",        "Что у меня получается"],
    ["where_i_stumble",   "Где я спотыкаюсь"],
    ["emerging_language", "Формирующийся язык"],
    ["current_series",    "Текущая серия"],
    ["next_hypothesis",   "Следующая гипотеза"],
  ];
  const frag = document.createDocumentFragment();
  for (const [key, label] of order) {
    const v = path[key];
    if (!v) continue;
    const wrap = document.createElement("div");
    wrap.className = "path-item";
    const h = document.createElement("h4");
    h.textContent = label;
    const p = document.createElement("p");
    p.textContent = v;
    wrap.appendChild(h);
    wrap.appendChild(p);
    frag.appendChild(wrap);
  }
  body.innerHTML = "";
  body.appendChild(frag);
}

function renderSeries(seriesList) {
  const host = el("series-list");
  if (!seriesList || seriesList.length === 0) {
    host.innerHTML = '<p class="muted">серий пока нет…</p>';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const s of seriesList) {
    const open = s.closed_at == null;
    const card = document.createElement("article");
    card.className = "series-card" + (open ? " open series-open" : "");
    card.dataset.seriesId = s.id;

    const head = document.createElement("div");
    head.className = "series-head";

    const title = document.createElement("h3");
    title.className = "series-name";
    title.textContent = `#${s.id} · ${s.name}`;
    title.title = "Открыть серию";
    title.addEventListener("click", () => openViewerSeries(s.id));

    const metaWrap = document.createElement("div");
    const count = (s.iterations || []).length;
    const chip = document.createElement("span");
    chip.className = "series-status-chip " + (open ? "open" : "closed");
    chip.textContent = open ? "идёт" : "закрыта";
    const meta = document.createElement("div");
    meta.className = "series-meta";
    meta.textContent = countWorksLabel(count);
    metaWrap.appendChild(chip);
    metaWrap.appendChild(document.createElement("br"));
    metaWrap.appendChild(meta);

    head.appendChild(title);
    head.appendChild(metaWrap);
    card.appendChild(head);

    if (s.thesis) {
      const p = document.createElement("p");
      p.className = "series-thesis";
      p.textContent = s.thesis;
      card.appendChild(p);
    }

    if (count > 0) {
      const thumbs = document.createElement("div");
      thumbs.className = "series-thumbs";
      for (const n of s.iterations) {
        const im = document.createElement("img");
        im.loading = "lazy";
        im.alt = `iter ${n}`;
        im.title = `Итерация ${pad3(n)}`;
        im.src = imgUrl(n);
        im.addEventListener("click", () => openViewerWork(n));
        thumbs.appendChild(im);
      }
      card.appendChild(thumbs);
    }

    if (s.retrospective) {
      const r = s.retrospective;
      const retro = document.createElement("div");
      retro.className = "series-retro";
      if (r.title) {
        const t = document.createElement("p");
        t.className = "series-retro-title";
        t.textContent = `«${r.title}»`;
        retro.appendChild(t);
      }
      if (r.arc_summary) {
        const a = document.createElement("p");
        a.className = "series-retro-arc";
        a.textContent = r.arc_summary;
        retro.appendChild(a);
      }
      card.appendChild(retro);
    }

    frag.appendChild(card);
  }
  host.innerHTML = "";
  host.appendChild(frag);
}

function renderStatus(data) {
  const lines = data.log_tail || [];
  const last = lines[lines.length - 1] || "";
  const m = last.match(/^\S+\s+\S+\s+(\w+)\s+(.*)$/);
  const level = m ? m[1] : "INFO";
  const msg = m ? m[2] : last;

  const bar = el("statusbar");
  const line = el("status-line");
  const dot = el("status-dot");
  const progress = el("status-progress-bar");
  setText(line, msg || "ожидание…");

  bar.classList.remove("is-stalled", "is-error");
  dot.classList.remove("stalled", "error");
  progress.classList.remove("stalled", "error", "determinate");
  progress.style.width = "38%";
  progress.style.transform = "";

  if (level === "ERROR" || level === "CRITICAL") {
    bar.classList.add("is-error");
    dot.classList.add("error");
    progress.classList.add("error");
  } else if (!last) {
    bar.classList.add("is-stalled");
    dot.classList.add("stalled");
    progress.classList.add("stalled");
  } else if (data.current) {
    const iter = data.current.iteration;
    const current = ITERS_BY_NUM.get(iter);
    const series = current ? SERIES_BY_ID.get(current.series_id) : null;
    const list = series ? seriesWorksSorted(series.id) : [];
    const index = list.indexOf(iter);
    if (index >= 0 && list.length > 0) {
      const pct = Math.max(((index + 1) / list.length) * 100, 10);
      progress.classList.add("determinate");
      progress.style.width = `${pct}%`;
      progress.style.transform = "none";
    }
  }

  const c = data.counts || {};
  const stats =
    `${c.iterations ?? 0} работ · ${c.series ?? 0} серий · ` +
    `активных: ${c.open_series ?? 0}`;
  setText(el("status-stats"), stats);
}

// ---------------------------------------------------------------------- *
// Viewer — single-work mode
// ---------------------------------------------------------------------- */

function seriesWorksSorted(seriesId) {
  const s = SERIES_BY_ID.get(seriesId);
  if (!s) return [];
  return [...(s.iterations || [])].sort((a, b) => a - b);
}

function neighboursInSeries(iterNum) {
  const it = ITERS_BY_NUM.get(iterNum);
  if (!it) return { prev: null, next: null, list: [], index: -1 };
  const list = seriesWorksSorted(it.series_id);
  const idx = list.indexOf(iterNum);
  return {
    list,
    index: idx,
    prev: idx > 0 ? list[idx - 1] : null,
    next: (idx >= 0 && idx < list.length - 1) ? list[idx + 1] : null,
  };
}

function openViewerWork(iterNum) {
  if (!ITERS_BY_NUM.has(iterNum)) return;
  viewer.mode = "work";
  viewer.iterNum = iterNum;
  const it = ITERS_BY_NUM.get(iterNum);
  viewer.seriesId = it.series_id;
  showViewer();
  renderViewerWork();
}

function openViewerSeries(seriesId) {
  if (!SERIES_BY_ID.has(seriesId)) return;
  viewer.mode = "series";
  viewer.seriesId = seriesId;
  viewer.iterNum = null;
  showViewer();
  renderViewerSeries();
}

function showViewer() {
  const v = el("viewer");
  v.hidden = false;
  v.setAttribute("aria-hidden", "false");
  viewer.open = true;
  document.body.classList.add("viewer-open");
}

function closeViewer() {
  const v = el("viewer");
  v.hidden = true;
  v.setAttribute("aria-hidden", "true");
  viewer.open = false;
  viewer.mode = null;
  viewer.iterNum = null;
  viewer.seriesId = null;
  document.body.classList.remove("viewer-open");
}

function setViewerMode(mode) {
  el("viewer-work").hidden = mode !== "work";
  el("viewer-series").hidden = mode !== "series";
  el("v-series-btn").hidden = mode !== "work";
}

async function renderViewerWork() {
  setViewerMode("work");
  const it = ITERS_BY_NUM.get(viewer.iterNum);
  if (!it) return;

  // Crumb
  const crumbSeries = el("v-crumb-series");
  crumbSeries.textContent = `Серия ${it.series_id} · ${it.series_name || ""}`.trim();
  crumbSeries.onclick = () => openViewerSeries(it.series_id);
  setText(el("v-crumb-iter"), `итерация ${pad3(it.iteration)}`);

  // Side panel
  setText(el("v-title"), it.title || "—");
  const { list, index } = neighboursInSeries(it.iteration);
  const pos = index >= 0 ? `${index + 1} / ${list.length}` : "—";
  setText(el("v-meta"),
          `Серия ${it.series_id} «${it.series_name}» · ${pos}`);
  setText(el("v-see"),       it.what_i_see      || "—");
  setText(el("v-resonates"), it.what_resonates  || "—");
  setText(el("v-silent"),    it.what_is_silent  || "—");
  setText(el("v-next-text"), it.where_next      || "—");

  // Image (preload + fade)
  const src = IMG_BASE + (it.image_path || `art_${pad3(it.iteration)}.png`) + folderQuery("?");
  const img = el("v-img");
  if (img.getAttribute("src") !== src) {
    img.style.opacity = "0";
    const loaded = await preloadImage(src);
    if (loaded) {
      img.src = loaded;
      requestAnimationFrame(() => { img.style.opacity = "1"; });
    }
  }

  // Nav buttons
  const nb = neighboursInSeries(it.iteration);
  el("v-prev").disabled = nb.prev == null;
  el("v-next").disabled = nb.next == null;

  // Bottom strip
  renderWorkStrip(it.series_id, it.iteration);
}

function renderWorkStrip(seriesId, activeIter) {
  const strip = el("v-strip");
  strip.innerHTML = "";
  const list = seriesWorksSorted(seriesId);
  for (const n of list) {
    const b = document.createElement("div");
    b.className = "vw-strip-item" + (n === activeIter ? " active" : "");
    b.title = `Итерация ${pad3(n)}`;
    b.addEventListener("click", () => openViewerWork(n));

    const im = document.createElement("img");
    im.loading = "lazy";
    im.alt = `iter ${n}`;
    im.src = imgUrl(n);
    b.appendChild(im);

    const num = document.createElement("span");
    num.className = "strip-num";
    num.textContent = pad3(n);
    b.appendChild(num);

    strip.appendChild(b);
  }
  // Scroll active item into view
  const active = strip.querySelector(".vw-strip-item.active");
  if (active) {
    active.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
  }
}

// ---------------------------------------------------------------------- *
// Viewer — series-overview mode
// ---------------------------------------------------------------------- */

function renderViewerSeries() {
  setViewerMode("series");
  const s = SERIES_BY_ID.get(viewer.seriesId);
  if (!s) return;
  const open = s.closed_at == null;

  // Crumb
  const crumbSeries = el("v-crumb-series");
  crumbSeries.textContent = `Серия ${s.id} · ${s.name}`;
  crumbSeries.onclick = null;
  setText(el("v-crumb-iter"),
          countWorksLabel((s.iterations || []).length));

  // Head
  const chip = el("vs-status");
  chip.textContent = open ? "идёт" : "закрыта";
  chip.className = "series-status-chip " + (open ? "open" : "closed");
  setText(el("vs-count"),
          `#${s.id} · ${countWorksLabel((s.iterations || []).length)}` +
          (s.closed_at ? ` · закрыта на итерации ${pad3(s.closed_at)}` : ""));
  setText(el("vs-name"), s.name);
  setText(el("vs-thesis"), s.thesis || "");

  // Retrospective
  const retroWrap = el("vs-retro");
  if (s.retrospective) {
    const r = s.retrospective;
    retroWrap.hidden = false;
    setText(el("vs-retro-title"), r.title ? `«${r.title}»` : "Взгляд назад");
    setText(el("vs-retro-arc"), r.arc_summary || "");
    const list = el("vs-retro-list");
    list.innerHTML = "";
    const fields = [
      ["what_emerged",     "Что проявилось"],
      ["what_disappointed", "Что разочаровало"],
      ["verdict",          "Вердикт"],
      ["bridge_to_next",   "Мост к следующему"],
    ];
    for (const [k, label] of fields) {
      const v = r[k];
      if (!v) continue;
      const item = document.createElement("div");
      item.className = "vs-retro-item";
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = v;
      item.appendChild(dt);
      item.appendChild(dd);
      list.appendChild(item);
    }
  } else {
    retroWrap.hidden = true;
  }

  // Grid
  const grid = el("vs-grid");
  grid.innerHTML = "";
  const iters = seriesWorksSorted(s.id);
  if (iters.length === 0) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "в серии пока нет работ";
    grid.appendChild(p);
    return;
  }
  for (const n of iters) {
    const it = ITERS_BY_NUM.get(n);
    const card = document.createElement("article");
    card.className = "vs-card";
    card.addEventListener("click", () => openViewerWork(n));

    const im = document.createElement("img");
    im.loading = "lazy";
    im.alt = it ? it.title : `iter ${n}`;
    im.src = imgUrl(n);
    card.appendChild(im);

    const body = document.createElement("div");
    body.className = "vs-card-body";
    const num = document.createElement("div");
    num.className = "vs-card-num";
    num.textContent = `Итерация ${pad3(n)}`;
    const title = document.createElement("h4");
    title.className = "vs-card-title";
    title.textContent = it ? (it.title || "—") : "—";
    body.appendChild(num);
    body.appendChild(title);
    card.appendChild(body);

    grid.appendChild(card);
  }
}

// ---------------------------------------------------------------------- *
// Viewer — event wiring
// ---------------------------------------------------------------------- */

function wireViewer() {
  // Hero -> open current work
  const frame = el("hero-frame");
  const openFromHero = () => {
    if (STATE && STATE.current) openViewerWork(STATE.current.iteration);
  };
  frame.addEventListener("click", openFromHero);
  frame.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openFromHero();
    }
  });

  // Top-bar buttons
  el("v-close").addEventListener("click", closeViewer);
  el("v-series-btn").addEventListener("click", () => {
    if (viewer.mode === "work" && viewer.seriesId != null) {
      openViewerSeries(viewer.seriesId);
    }
  });

  // Nav buttons
  el("v-prev").addEventListener("click", () => {
    if (viewer.iterNum == null) return;
    const { prev } = neighboursInSeries(viewer.iterNum);
    if (prev != null) openViewerWork(prev);
  });
  el("v-next").addEventListener("click", () => {
    if (viewer.iterNum == null) return;
    const { next } = neighboursInSeries(viewer.iterNum);
    if (next != null) openViewerWork(next);
  });

  // Click on dimmed backdrop closes (but clicks inside content don't)
  el("viewer").addEventListener("click", (e) => {
    if (e.target.id === "viewer") closeViewer();
  });

  // Keyboard
  window.addEventListener("keydown", (e) => {
    if (!viewer.open) return;
    switch (e.key) {
      case "Escape":
        closeViewer();
        break;
      case "ArrowLeft":
        if (viewer.mode === "work" && viewer.iterNum != null) {
          const { prev } = neighboursInSeries(viewer.iterNum);
          if (prev != null) openViewerWork(prev);
        }
        break;
      case "ArrowRight":
        if (viewer.mode === "work" && viewer.iterNum != null) {
          const { next } = neighboursInSeries(viewer.iterNum);
          if (next != null) openViewerWork(next);
        }
        break;
      case "s":
      case "S":
        if (viewer.mode === "work" && viewer.seriesId != null) {
          openViewerSeries(viewer.seriesId);
        }
        break;
    }
  });
}

// ---------------------------------------------------------------------- *
// Folder picker
// ---------------------------------------------------------------------- */

/** Cache of the last folder list rendered, so we don't rebuild <option>s
 *  on every poll. */
let lastFoldersSig = null;

function foldersSignature(list, selected) {
  return JSON.stringify({
    s: selected || "",
    f: (list || []).map((f) => [f.name, f.is_default, f.iterations]),
  });
}

function formatFolderOption(f) {
  const suffix = f.iterations
    ? ` · ${f.iterations} ${plural(f.iterations, "работа", "работы", "работ")}`
    : "";
  const def = f.is_default ? " ★" : "";
  return `${f.name}${def}${suffix}`;
}

async function refreshFolders() {
  let data;
  try {
    data = await fetchFolders();
  } catch (e) {
    return;  // silent — next poll will retry
  }
  const folders = data.folders || [];
  const select = el("folder-select");
  if (!select) return;

  // If the persisted CURRENT_FOLDER no longer exists on disk, fall back to
  // the server's default (null == send no folder param).
  if (CURRENT_FOLDER && !folders.some((f) => f.name === CURRENT_FOLDER)) {
    CURRENT_FOLDER = null;
    try { localStorage.removeItem(FOLDER_STORAGE_KEY); } catch (_) {}
  }

  const effective = CURRENT_FOLDER || (data.default || "");
  const sig = foldersSignature(folders, effective);
  if (sig === lastFoldersSig) return;
  lastFoldersSig = sig;

  // Rebuild options (preserve ordering from server — newest first).
  select.innerHTML = "";
  if (folders.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "нет данных";
    select.appendChild(opt);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const f of folders) {
    const opt = document.createElement("option");
    opt.value = f.name;
    opt.textContent = formatFolderOption(f);
    if (f.name === effective) opt.selected = true;
    select.appendChild(opt);
  }
}

function resetLocalState() {
  STATE = null;
  ITERS_BY_NUM = new Map();
  SERIES_BY_ID = new Map();
  lastSig.current = null;
  lastSig.seriesSig = null;
  lastSig.pathSig = null;

  // Clear rendered views so the user sees an immediate transition.
  const img = el("hero-img");
  if (img) { img.removeAttribute("src"); img.style.opacity = "0"; }
  setText(el("hero-title"), "Загружаем…");
  setText(el("hero-series"), "—");
  setText(el("hero-iter"), "—");
  for (const id of ["t-see", "t-resonates", "t-silent", "t-next"]) {
    setText(el(id), "—");
  }
  el("series-list").innerHTML = '<p class="muted">загружаем…</p>';
  el("path-body").innerHTML   = '<p class="muted">загружаем…</p>';

  // Close the fullscreen viewer if it's open — its contents now reference
  // the previous folder.
  if (viewer.open) closeViewer();
}

function wireFolderPicker() {
  const select = el("folder-select");
  if (!select) return;
  select.addEventListener("change", async () => {
    const val = select.value || "";
    CURRENT_FOLDER = val || null;
    try {
      if (CURRENT_FOLDER) localStorage.setItem(FOLDER_STORAGE_KEY, CURRENT_FOLDER);
      else                localStorage.removeItem(FOLDER_STORAGE_KEY);
    } catch (_) { /* ignore */ }
    resetLocalState();
    await tick();          // fetch immediately, don't wait for next poll
  });
}

// ---------------------------------------------------------------------- *
// Poll loop
// ---------------------------------------------------------------------- */

async function fetchState() {
  const res = await fetch("/api/state" + folderQuery("?"), { cache: "no-store" });
  if (!res.ok) throw new Error("state fetch failed: " + res.status);
  return res.json();
}

async function fetchFolders() {
  const res = await fetch("/api/folders", { cache: "no-store" });
  if (!res.ok) throw new Error("folders fetch failed: " + res.status);
  return res.json();
}

function seriesSignature(list) {
  return JSON.stringify((list || []).map((s) => [
    s.id, (s.iterations || []).length, s.closed_at,
    s.retrospective ? 1 : 0,
  ]));
}

function rebuildLookups(data) {
  const iterMap = new Map();
  for (const it of (data.iterations || [])) {
    if (typeof it.iteration === "number") iterMap.set(it.iteration, it);
  }
  const seriesMap = new Map();
  for (const s of (data.series || [])) {
    if (typeof s.id === "number") seriesMap.set(s.id, s);
  }
  ITERS_BY_NUM = iterMap;
  SERIES_BY_ID = seriesMap;
}

async function tick() {
  let data;
  try {
    data = await fetchState();
  } catch (e) {
    el("status-dot").classList.add("error");
    setText(el("status-line"), String(e));
    return;
  }
  STATE = data;
  rebuildLookups(data);

  const cur = data.current;
  const curN = cur ? cur.iteration : null;
  if (curN !== lastSig.current) {
    await renderHero(cur);
    renderThoughts(cur);
    lastSig.current = curN;
    el("hero").classList.remove("fade-enter");
    void el("hero").offsetWidth;
    el("hero").classList.add("fade-enter");
  }

  const sSig = seriesSignature(data.series);
  const seriesChanged = sSig !== lastSig.seriesSig;
  if (seriesChanged) {
    renderSeries(data.series);
    lastSig.seriesSig = sSig;
  }

  const pSig = JSON.stringify(data.path || null);
  if (pSig !== lastSig.pathSig) {
    renderPath(data.path);
    lastSig.pathSig = pSig;
  }

  renderStatus(data);

  // If the viewer is open and underlying data changed, refresh its view
  // in-place (e.g. a new work was appended to the open series).
  if (viewer.open && seriesChanged) {
    if (viewer.mode === "series" && viewer.seriesId != null &&
        SERIES_BY_ID.has(viewer.seriesId)) {
      renderViewerSeries();
    } else if (viewer.mode === "work" && viewer.iterNum != null &&
               ITERS_BY_NUM.has(viewer.iterNum)) {
      renderViewerWork();
    }
  }
}

async function loop() {
  await tick();
  setTimeout(loop, POLL_MS);
}

wireViewer();
wireFolderPicker();
refreshFolders();                                // initial folder list
setInterval(refreshFolders, FOLDERS_POLL_MS);    // keep it fresh
loop();
