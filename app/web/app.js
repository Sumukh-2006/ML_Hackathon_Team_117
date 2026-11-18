/* PathSnap chart sheet.
   Renderer notes: mercator world coords are precomputed once per trip; every frame
   rebuilds screen-space paths. At the payload's 250 m corridor that is a few
   thousand segments, so a rebuild per frame is cheaper than caching and invalidating. */

const $ = (sel, root = document) => root.querySelector(sel);
const nf = new Intl.NumberFormat("en-US");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = getComputedStyle(document.documentElement);
const C = {
  paper: css.getPropertyValue("--paper").trim(),
  paperLift: css.getPropertyValue("--paper-lift").trim(),
  ink: css.getPropertyValue("--ink").trim(),
  soft: css.getPropertyValue("--ink-soft").trim(),
  faint: css.getPropertyValue("--ink-faint").trim(),
  rule: css.getPropertyValue("--rule").trim(),
  ruleSoft: css.getPropertyValue("--rule-soft").trim(),
  road: css.getPropertyValue("--road").trim(),
  casing: css.getPropertyValue("--road-casing").trim(),
  truth: css.getPropertyValue("--truth").trim(),
  pred: css.getPropertyValue("--pred").trim(),
  warn: css.getPropertyValue("--warn").trim(),
};

const el = {
  select: $("#trip-select"),
  tripId: $("#trip-id"),
  partition: $("#trip-partition"),
  place: $("#trip-place"),
  recall: $("#m-recall"),
  recallNote: $("#m-recall-note"),
  precision: $("#m-precision"),
  f1: $("#m-f1"),
  top1: $("#m-top1"),
  dir: $("#m-dir"),
  breaks: $("#m-breaks"),
  missed: $("#m-missed"),
  foot: $("#m-foot"),
  bar: $("#bar-fill"),
  readout: $("#chart-readout"),
  tapeStatus: $("#tape-status"),
  tapeFix: $("#tape-fix"),
  scrub: $("#scrub"),
  cursor: $("#tape-cursor"),
  toast: $("#toast"),
  chart: $("#chart"),
  tape: $("#tape-canvas"),
  apiDialog: $("#api-dialog"),
  apiBody: $("#api-body"),
  apiNote: $("#api-note"),
  aboutDialog: $("#about-dialog"),
  aboutBody: $("#about-body"),
};

const state = {
  trips: [],
  trip: null,
  d: null,
  i: 0,
  layers: { road: true, route: true, pred: true, fix: true },
  view: { cx: 0.5, cy: 0.5, scale: 1e6 },
  fitScale: 1e6,
  focus: null,
  missed: [],
  missIdx: -1,
  drag: null,
  anim: 0,
  cache: new Map(),
};

/* ------------------------------------------------------------------ geometry */

const mercX = (lon) => (lon + 180) / 360;
const mercY = (lat) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
};
const invLon = (x) => x * 360 - 180;
const invLat = (y) => {
  const n = Math.PI * (1 - 2 * y);
  return (Math.atan(Math.sinh(n)) * 180) / Math.PI;
};

const gossip = 40075200; // metres around the equator, for scale-bar only

function fmtDuration(sec) {
  const s = Math.max(0, Math.round(sec));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}
const pct = (v) => (v * 100).toFixed(1);

/* --------------------------------------------------------------------- derive */

function derive(payload) {
  const net = payload.network;
  const arcs = new Float64Array(net.n * 4);
  const geom = new Map();
  for (let a = 0; a < net.n; a++) {
    const o = a * 4;
    const x1 = mercX(net.xy[o]), y1 = mercY(net.xy[o + 1]);
    const x2 = mercX(net.xy[o + 2]), y2 = mercY(net.xy[o + 3]);
    arcs[o] = x1; arcs[o + 1] = y1; arcs[o + 2] = x2; arcs[o + 3] = y2;
    geom.set(net.ids[a], [x1, y1, x2, y2]);
  }

  const tr = payload.track;
  const wpts = new Float64Array(tr.n * 2);
  for (let k = 0; k < tr.n; k++) {
    wpts[k * 2] = mercX(tr.xy[k * 2]);
    wpts[k * 2 + 1] = mercY(tr.xy[k * 2 + 1]);
  }

  // distance already driven at each fix, for the position readout
  const cum = new Float64Array(tr.n);
  const kx = 111320 * Math.cos((payload.bbox[1] * Math.PI) / 180);
  for (let k = 1; k < tr.n; k++) {
    const dx = (tr.xy[k * 2] - tr.xy[k * 2 - 2]) * kx;
    const dy = (tr.xy[k * 2 + 1] - tr.xy[k * 2 - 1]) * 111320;
    cum[k] = cum[k - 1] + Math.hypot(dx, dy);
  }

  const predSet = new Set(payload.pred.arc);
  const seen = new Set();
  const missed = [];
  for (const a of payload.route) {
    if (!predSet.has(a) && !seen.has(a) && geom.has(a)) {
      seen.add(a);
      missed.push(a);
    }
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let k = 0; k < tr.n; k++) {
    const x = wpts[k * 2], y = wpts[k * 2 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  return { payload, arcs, arcs4: net.n, geom, wpts, n: tr.n, missed, predSet, cum, bbox: { minX, minY, maxX, maxY } };
}

function fitView() {
  const d = state.d;
  if (!d) return;
  const { w, h } = chart;
  const bw = Math.max(d.bbox.maxX - d.bbox.minX, 1e-7);
  const bh = Math.max(d.bbox.maxY - d.bbox.minY, 1e-7);
  const scale = Math.min(w / bw, h / bh) * 0.84;
  state.fitScale = scale;
  state.view = {
    cx: (d.bbox.minX + d.bbox.maxX) / 2,
    cy: (d.bbox.minY + d.bbox.maxY) / 2,
    scale,
  };
}

const sx = (wx) => (wx - frame.cx) * frame.scale + frame.w / 2;
const sy = (wy) => (wy - frame.cy) * frame.scale + frame.h / 2;

const frame = { cx: 0, cy: 0, scale: 1, w: 1, h: 1 };
function setFrame(cx, cy, scale, w, h) {
  frame.cx = cx;
  frame.cy = cy;
  frame.scale = scale;
  frame.w = w;
  frame.h = h;
}

/* --------------------------------------------------------------------- canvases */

const chart = { ctx: el.chart.getContext("2d"), w: 0, h: 0 };
const tape = { ctx: el.tape.getContext("2d"), w: 0, h: 0 };

function sizeCanvas(cv, ctx) {
  const rect = cv.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

function resize() {
  const a = sizeCanvas(el.chart, chart.ctx);
  chart.w = a.w; chart.h = a.h;
  const b = sizeCanvas(el.tape, tape.ctx);
  tape.w = b.w; tape.h = b.h;
  if (state.d && !state.userMoved) fitView();
  draw();
}

new ResizeObserver(resize).observe($(".chart-wrap"));
new ResizeObserver(resize).observe(el.tape.parentElement);

/* --------------------------------------------------------------------- drawing */

let raf = 0;
function draw() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    drawChart();
    drawTape();
  });
}

function drawChart() {
  const { ctx, w, h } = chart;
  ctx.clearRect(0, 0, w, h);
  if (!state.d) return;

  setFrame(state.view.cx, state.view.cy, state.view.scale, w, h);
  drawGraticule(ctx, w, h);

  const T = {};
  let t0 = performance.now();
  if (state.layers.road) paintRoads(ctx);
  T.road = performance.now() - t0;
  t0 = performance.now();
  if (state.layers.route) paintRoute(ctx, 9, 0.3);
  if (state.layers.pred) paintPred(ctx, 1.9);
  T.paths = performance.now() - t0;
  t0 = performance.now();
  if (state.focus !== null) paintFocus(ctx);
  if (state.layers.fix) paintFixes(ctx, 1.3, 0.34);
  T.fixes = performance.now() - t0;

  drawEnds(ctx);
  drawCurrent(ctx);
  drawScaleBar(ctx, w, h);
  t0 = performance.now();
  drawInset(ctx, w, h);
  T.inset = performance.now() - t0;
  T.total = T.road + T.paths + T.fixes + T.inset;
  state.timing = T;
  // the inset draws in its own projection; put the main one back for hit tests
  setFrame(state.view.cx, state.view.cy, state.view.scale, w, h);
}

function paintRoads(ctx, casing = 3.6, fill = 1.15) {
  const d = state.d;
  // skip arcs shorter than a pixel: at whole-drive scale most segments are
  // sub-pixel and drawing them costs raster time without adding information
  const min2 = 0.81 / (frame.scale * frame.scale);
  ctx.beginPath();
  const g = d.arcs;
  for (let a = 0; a < d.arcs4; a++) {
    const o = a * 4;
    const dx = g[o + 2] - g[o];
    const dy = g[o + 3] - g[o + 1];
    if (dx * dx + dy * dy < min2) continue;
    ctx.moveTo(sx(g[o]), sy(g[o + 1]));
    ctx.lineTo(sx(g[o + 2]), sy(g[o + 3]));
  }
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = C.casing;
  ctx.lineWidth = casing;
  ctx.stroke();
  ctx.strokeStyle = C.road;
  ctx.lineWidth = fill;
  ctx.stroke();
}

function paintRoute(ctx, width, alpha) {
  const d = state.d;
  ctx.beginPath();
  for (const a of d.payload.route) {
    const g = d.geom.get(a);
    if (!g) continue;
    ctx.moveTo(sx(g[0]), sy(g[1]));
    ctx.lineTo(sx(g[2]), sy(g[3]));
  }
  ctx.strokeStyle = `rgba(32,51,110,${alpha})`;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke();
}

function paintPred(ctx, width) {
  const d = state.d;
  ctx.beginPath();
  for (const a of d.predSet) {
    const g = d.geom.get(a);
    if (!g) continue;
    ctx.moveTo(sx(g[0]), sy(g[1]));
    ctx.lineTo(sx(g[2]), sy(g[3]));
  }
  ctx.strokeStyle = C.pred;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.stroke();
}

function paintFixes(ctx, size, alpha) {
  const d = state.d;
  // above ~6k fixes the extra dots are sub-pixel noise at these scales
  const stride = Math.max(1, Math.ceil(d.n / 6000));
  const w = frame.w;
  const h = frame.h;
  ctx.fillStyle = `rgba(20,24,26,${alpha})`;
  for (let k = 0; k < d.n; k += stride) {
    const x = sx(d.wpts[k * 2]);
    if (x < -2 || x > w + 2) continue;
    const y = sy(d.wpts[k * 2 + 1]);
    if (y < -2 || y > h + 2) continue;
    ctx.fillRect(x - size / 2, y - size / 2, size, size);
  }
}

function paintFocus(ctx) {
  const d = state.d;
  const g = d.geom.get(state.focus);
  if (!g) return;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(sx(g[0]), sy(g[1]));
  ctx.lineTo(sx(g[2]), sy(g[3]));
  ctx.setLineDash([7, 5]);
  ctx.lineCap = "butt";
  ctx.strokeStyle = C.paper;
  ctx.lineWidth = 7;
  ctx.stroke();
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 2.4;
  ctx.stroke();
  ctx.restore();

  const mx = (sx(g[0]) + sx(g[2])) / 2;
  const my = (sy(g[1]) + sy(g[3])) / 2;
  label(ctx, "no fix reaches this segment", mx + 8, my - 9, C.ink);
}

/* One inset that always shows what the main map cannot: at whole-drive scale a
   500 m magnifier around the current fix, once zoomed in a locator for the drive. */
function drawInset(ctx, w, h) {
  const midLat = invLat(state.view.cy);
  const mPerWorld = gossip * Math.cos((midLat * Math.PI) / 180);
  const metresWide = (w / state.view.scale) * mPerWorld;
  if (metresWide > 2600) drawDetail(ctx, w, h, mPerWorld);
  else drawLocator(ctx, w, h);
}

function drawDetail(ctx, w, h, mPerWorld) {
  const d = state.d;
  const i = state.i;
  const iw = Math.round(Math.max(150, Math.min(300, w * 0.24)));
  const ih = Math.round(iw * 0.73);
  const pad = 16;
  const x0 = w - iw - pad;
  const y0 = h - ih - pad;

  ctx.save();
  ctx.fillStyle = C.paperLift;
  ctx.fillRect(x0, y0, iw, ih);
  ctx.beginPath();
  ctx.rect(x0, y0, iw, ih);
  ctx.clip();
  ctx.translate(x0, y0);
  setFrame(d.wpts[i * 2], d.wpts[i * 2 + 1], (iw * mPerWorld) / 500, iw, ih);
  if (state.layers.road) paintRoads(ctx, 2.6, 1);
  if (state.layers.route) paintRoute(ctx, 7, 0.34);
  if (state.layers.pred) paintPred(ctx, 1.6);
  if (state.layers.fix) paintFixes(ctx, 1.2, 0.4);
  drawCurrent(ctx);
  ctx.restore();

  ctx.strokeStyle = C.rule;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, iw - 1, ih - 1);

  const cap = `500 m around fix ${nf.format(i + 1)}`;
  ctx.font = '10.5px "Bahnschrift", "DIN Alternate", sans-serif';
  const tw = ctx.measureText(cap).width;
  ctx.fillStyle = C.paper;
  ctx.fillRect(x0 - 2, y0 - 15, tw + 4, 14);
  ctx.fillStyle = C.soft;
  ctx.fillText(cap, x0, y0 - 4);
}

function drawLocator(ctx, w, h) {
  const d = state.d;
  const iw = Math.round(Math.max(120, Math.min(178, w * 0.18)));
  const ih = Math.round(iw * 0.75);
  const pad = 16;
  const x0 = w - iw - pad;
  const y0 = h - ih - pad;
  const bw = Math.max(d.bbox.maxX - d.bbox.minX, 1e-9);
  const bh = Math.max(d.bbox.maxY - d.bbox.minY, 1e-9);
  const s = Math.min((iw - 18) / bw, (ih - 18) / bh);
  const cx = (d.bbox.minX + d.bbox.maxX) / 2;
  const cy = (d.bbox.minY + d.bbox.maxY) / 2;
  const px = (wx) => x0 + iw / 2 + (wx - cx) * s;
  const py = (wy) => y0 + ih / 2 + (wy - cy) * s;

  ctx.save();
  ctx.fillStyle = "rgba(246,244,234,0.94)";
  ctx.fillRect(x0, y0, iw, ih);
  ctx.strokeStyle = C.rule;
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 + 0.5, y0 + 0.5, iw - 1, ih - 1);

  ctx.beginPath();
  ctx.rect(x0 + 1, y0 + 1, iw - 2, ih - 2);
  ctx.clip();

  ctx.beginPath();
  for (let k = 0; k < d.n; k++) {
    const x = px(d.wpts[k * 2]), y = py(d.wpts[k * 2 + 1]);
    k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.strokeStyle = "rgba(20,24,26,0.42)";
  ctx.lineWidth = 1;
  ctx.stroke();

  const v = state.view;
  const vx0 = v.cx - chart.w / 2 / v.scale;
  const vx1 = v.cx + chart.w / 2 / v.scale;
  const vy0 = v.cy - chart.h / 2 / v.scale;
  const vy1 = v.cy + chart.h / 2 / v.scale;
  ctx.beginPath();
  ctx.rect(px(vx0), py(vy0), (vx1 - vx0) * s, (vy1 - vy0) * s);
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 1.2;
  ctx.stroke();

  ctx.fillStyle = C.soft;
  ctx.font = '9.5px "Bahnschrift", "DIN Alternate", sans-serif';
  ctx.fillText("whole drive", x0 + 7, y0 + 13);
  ctx.restore();
}

function drawEnds(ctx) {
  const d = state.d;
  const ax = sx(d.wpts[0]), ay = sy(d.wpts[1]);
  const bx = sx(d.wpts[(d.n - 1) * 2]), by = sy(d.wpts[(d.n - 1) * 2 + 1]);

  ctx.fillStyle = C.paper;
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 1.3;
  ctx.beginPath();
  ctx.arc(ax, ay, 3.6, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.beginPath();
  ctx.rect(bx - 3.2, by - 3.2, 6.4, 6.4);
  ctx.fill();
  ctx.stroke();

  ctx.font = '10px "Bahnschrift", "DIN Alternate", sans-serif';
  ctx.fillStyle = C.soft;
  ctx.fillText("start", ax + 8, ay + 13);
  ctx.fillText("finish", bx + 8, by - 8);
}

function label(ctx, text, x, y, colour) {
  ctx.font = '11px "Bahnschrift", "DIN Alternate", sans-serif';
  const pad = 4;
  const tw = ctx.measureText(text).width;
  ctx.fillStyle = "rgba(239,237,227,0.92)";
  ctx.fillRect(x - pad, y - 11, tw + pad * 2, 15);
  ctx.fillStyle = colour;
  ctx.fillText(text, x, y);
}

function drawCurrent(ctx) {
  const d = state.d;
  const i = state.i;
  const px = sx(d.wpts[i * 2]);
  const py = sy(d.wpts[i * 2 + 1]);
  const snap = d.payload.pred.snap;
  const qx = sx(mercX(snap[i * 2]));
  const qy = sy(mercY(snap[i * 2 + 1]));
  const off = d.payload.pred.dist[i];

  if (off > 6) {
    ctx.strokeStyle = C.warn;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(qx, qy);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.fillStyle = C.pred;
  ctx.fillRect(qx - 2.2, qy - 2.2, 4.4, 4.4);

  ctx.beginPath();
  ctx.arc(px, py, 6, 0, Math.PI * 2);
  ctx.fillStyle = C.paper;
  ctx.fill();
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 1.6;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(px, py, 1.8, 0, Math.PI * 2);
  ctx.fillStyle = C.ink;
  ctx.fill();
}

function drawGraticule(ctx, w, h) {
  const v = state.view;
  const x0 = v.cx - w / 2 / v.scale, x1 = v.cx + w / 2 / v.scale;
  const y0 = v.cy - h / 2 / v.scale, y1 = v.cy + h / 2 / v.scale;
  const lonSpan = invLon(x1) - invLon(x0);
  const step = pickStep(lonSpan, 6);
  const dp = Math.max(0, Math.ceil(-Math.log10(step)));

  ctx.font = '10px "Bahnschrift", "DIN Alternate", sans-serif';
  ctx.strokeStyle = C.ruleSoft;
  ctx.lineWidth = 1;
  ctx.fillStyle = C.faint;

  const lon0 = Math.floor(invLon(x0) / step) * step;
  for (let lon = lon0; lon <= invLon(x1); lon += step) {
    const x = sx(mercX(lon));
    if (x < 0 || x > w) continue;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(`${lon.toFixed(dp)}°`, x + 4, h - 8);
  }

  const latA = invLat(y0), latB = invLat(y1);
  const latStep = pickStep(Math.abs(latA - latB), 5);
  const latDp = Math.max(0, Math.ceil(-Math.log10(latStep)));
  const start = Math.floor(Math.min(latA, latB) / latStep) * latStep;
  for (let lat = start; lat <= Math.max(latA, latB); lat += latStep) {
    const y = sy(mercY(lat));
    if (y < 0 || y > h) continue;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.fillText(`${lat.toFixed(latDp)}°`, 8, y - 5);
  }
}

function pickStep(span, target) {
  const raw = span / target;
  const ladder = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
    0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 45];
  return ladder.find((s) => s >= raw) ?? 45;
}

function drawScaleBar(ctx, w, h) {
  const midLat = invLat(state.view.cy);
  const mPerPx = (gossip * Math.cos((midLat * Math.PI) / 180)) / state.view.scale;
  const raw = 100 * mPerPx;
  const pow = Math.pow(10, Math.floor(Math.log10(raw)));
  const nice = [1, 2, 5, 10].map((m) => m * pow).find((m) => m >= raw) ?? raw;
  const px = Math.min(nice / mPerPx, w * 0.4);
  const x = 16, y = 22;
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x + px, y);
  ctx.moveTo(x, y - 4); ctx.lineTo(x, y + 4);
  ctx.moveTo(x + px, y - 4); ctx.lineTo(x + px, y + 4);
  ctx.stroke();
  ctx.fillStyle = C.soft;
  ctx.font = '10.5px "Bahnschrift", "DIN Alternate", sans-serif';
  ctx.fillText(nice >= 1000 ? `${(nice / 1000).toFixed(nice % 1000 ? 1 : 0)} km` : `${Math.round(nice)} m`, x, y - 7);
}

/* ------------------------------------------------------------------------ tape */

function drawTape() {
  const { ctx, w, h } = tape;
  ctx.clearRect(0, 0, w, h);
  const d = state.d;
  if (!d) return;
  const { t, n } = state.d.payload.track;
  const on = state.d.payload.pred.on;
  const join = state.d.payload.pred.join;
  const tmax = t[n - 1] || 1;
  const base = h - 15;

  // The drive as a hairline; marks hang above it only where the snap is wrong.
  ctx.fillStyle = "rgba(20,24,26,0.12)";
  ctx.fillRect(0, base - 1, w, 2);

  const cols = new Uint8Array(Math.ceil(w));
  for (let k = 0; k < n; k++) {
    const x = Math.min(cols.length - 1, Math.round((t[k] / tmax) * (w - 1)));
    const sev = on[k] === 1 ? (join[k] ? 0 : 2) : on[k] === 2 ? 3 : 4;
    if (sev > cols[x]) cols[x] = sev;
  }

  const colour = ["", "", "rgba(20,24,26,0.62)", "rgba(166,18,75,0.85)", "rgba(142,58,16,0.88)"];
  const height = [0, 0, 5, 7, 10];
  for (let x = 0; x < cols.length; x++) {
    if (!cols[x]) continue;
    ctx.fillStyle = colour[cols[x]];
    ctx.fillRect(x, base - height[cols[x]], 1, height[cols[x]]);
  }

  const step = timeStep(tmax, w);
  ctx.font = '10px "Bahnschrift", "DIN Alternate", sans-serif';
  for (let s = 0; s <= tmax + 1; s += step) {
    const x = (s / tmax) * (w - 1);
    ctx.fillStyle = C.rule;
    ctx.fillRect(x, base + 1, 1, 3);
    ctx.fillStyle = C.faint;
    ctx.fillText(fmtDuration(s), Math.min(w - 30, x + 3), h - 2);
  }

  const cxp = (t[state.i] / tmax) * (w - 1);
  el.cursor.style.left = `${cxp}px`;
}

function timeStep(tmax, w) {
  const target = tmax / Math.max(2, Math.floor(w / 110));
  return [30, 60, 120, 300, 600, 900, 1800, 3600, 7200].find((s) => s >= target) ?? 7200;
}

/* ------------------------------------------------------------------ readouts */

function updateSheet() {
  const d = state.d;
  const p = d.payload;
  const m = p.metrics;
  el.tripId.textContent = p.id;
  el.partition.textContent = p.partition === "test" ? "student test" : "train";
  el.place.textContent = [p.meta.city, p.meta.country].filter(Boolean).join(", ");

  el.recall.textContent = pct(m.arcs.recall_strict);
  el.precision.textContent = `${pct(m.arcs.precision_strict)}%`;
  el.f1.textContent = `${pct(m.arcs.f1_strict)}%`;
  el.top1.textContent = `${pct(m.point.top1_dir)}%`;
  el.dir.textContent = `${pct(m.arcs.recall_dir)}%`;
  el.breaks.textContent = nf.format(m.continuity.breaks);
  el.missed.textContent = nf.format(m.arcs.missed_arcs_dir);
  el.bar.style.width = `${Math.min(100, m.arcs.recall_strict * 100)}%`;

  const wrongWay = m.arcs.wrong_way_only;
  const never = m.arcs.missed_arcs_dir;
  el.foot.textContent = wrongWay
    ? `${nf.format(never)} of ${nf.format(m.counts.route_arcs)} route segments are never reached; ` +
      `${nf.format(wrongWay)} only the wrong way round.`
    : `${nf.format(never)} of ${nf.format(m.counts.route_arcs)} route segments are never reached by any fix.`;
  el.recallNote.textContent =
    "of the route's segments are touched, direction included";

  const cta = $("#first-miss");
  cta.hidden = m.arcs.missed_arcs === 0;
}

function nearArcText(i) {
  const d = state.d;
  const arc = d.payload.pred.arc[i];
  const on = d.payload.pred.on[i];
  if (on === 1) return `<span class="on">On the reference route</span>`;
  if (on === 2) return `<span class="off">Right road, opposite direction</span>`;
  return `<span class="off">Not on the reference route</span>`;
}

function updatePosition() {
  const d = state.d;
  if (!d) return;
  const p = d.payload;
  const i = state.i;
  const n = p.track.n;
  const arc = p.pred.arc[i];
  const off = p.pred.dist[i].toFixed(1);
  const t = p.track.t[i];
  const join = p.pred.join[i];

  const total = p.meta.drive_m || 0;
  const done = d.cum[i] || 0;

  el.tapeStatus.textContent =
    `Fix ${nf.format(i + 1)} of ${nf.format(n)} at ${fmtDuration(t)} · ${(done / 1000).toFixed(1)} km of ${(total / 1000).toFixed(1)} km`;

  el.readout.innerHTML =
    `<div><strong>${off} m</strong> from the nearest road · arc ${arc}</div>` +
    `<div>${nearArcText(i)}</div>` +
    (join ? "" : `<div class="off">Disconnected from the previous fix</div>`);

  el.tapeFix.textContent = join
    ? `${off} m off the road · arc ${arc}`
    : `${off} m off the road · break before this fix`;

  el.scrub.value = String(i);
  el.cursor.style.left = `${(p.track.t[i] / (p.track.t[n - 1] || 1)) * (tape.w - 1)}px`;
}

function setIndex(i) {
  const d = state.d;
  if (!d) return;
  state.i = Math.max(0, Math.min(d.n - 1, i));
  updatePosition();
  draw();
}

/* ---------------------------------------------------------------- misses */

function goMiss(dir) {
  const d = state.d;
  if (!d || !d.missed.length) {
    toast("Every reference segment on this drive receives at least one fix.");
    return;
  }
  state.missIdx = (state.missIdx + dir + d.missed.length) % d.missed.length;
  const arc = d.missed[state.missIdx];
  const g = d.geom.get(arc);
  if (!g) return;
  state.focus = arc;
  state.userMoved = true;

  // move the tape to the moment the car passed closest to that segment, so the
  // readout describes the same place the map is showing
  const cx = (g[0] + g[2]) / 2;
  const cy = (g[1] + g[3]) / 2;
  let idx = 0;
  let best = Infinity;
  for (let k = 0; k < d.n; k++) {
    const dx = d.wpts[k * 2] - cx;
    const dy = d.wpts[k * 2 + 1] - cy;
    const dd = dx * dx + dy * dy;
    if (dd < best) { best = dd; idx = k; }
  }
  state.i = idx;
  el.scrub.value = String(idx);
  updatePosition();

  flyTo(cx, cy, missScale(g));
  toast(`Miss ${state.missIdx + 1} of ${nf.format(d.missed.length)}: no fix snaps to this segment.`);
}

function missScale(g) {
  const midLat = invLat((g[1] + g[3]) / 2);
  const mPerWorld = gossip * Math.cos((midLat * Math.PI) / 180);
  const arcM = Math.hypot(g[2] - g[0], g[3] - g[1]) * mPerWorld;
  const span = Math.max(700, arcM * 2.5);
  return Math.min(chart.w, chart.h) / (span / mPerWorld);
}

function flyTo(cx, cy, scale) {
  cancelAnimationFrame(state.anim);
  if (reduceMotion) {
    state.view = { cx, cy, scale };
    state.focus = state.focus;
    draw();
    return;
  }
  const from = { ...state.view };
  const t0 = performance.now();
  const dur = 420;
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - k, 3);
    state.view = {
      cx: from.cx + (cx - from.cx) * e,
      cy: from.cy + (cy - from.cy) * e,
      scale: from.scale + (scale - from.scale) * e,
    };
    draw();
    if (k < 1) state.anim = requestAnimationFrame(step);
  };
  state.anim = requestAnimationFrame(step);
}

/* -------------------------------------------------------------------- events */

function bind() {
  el.select.addEventListener("change", () => loadTrip(el.select.value));

  for (const btn of document.querySelectorAll(".layer")) {
    btn.addEventListener("click", () => {
      const key = btn.dataset.layer;
      state.layers[key] = !state.layers[key];
      btn.setAttribute("aria-pressed", String(state.layers[key]));
      draw();
    });
  }

  $("#zoom-in").addEventListener("click", () => zoomBy(1.5, chart.w / 2, chart.h / 2));
  $("#zoom-out").addEventListener("click", () => zoomBy(1 / 1.5, chart.w / 2, chart.h / 2));
  $("#fit").addEventListener("click", () => {
    state.focus = null;
    state.missIdx = -1;
    state.userMoved = false;
    fitView();
    draw();
  });

  el.scrub.addEventListener("input", () => {
    state.focus = null;
    setIndex(Number(el.scrub.value));
  });

  $("#next-miss").addEventListener("click", () => goMiss(1));
  $("#prev-miss").addEventListener("click", () => goMiss(-1));
  $("#first-miss").addEventListener("click", () => goMiss(1));

  const cv = el.chart;
  cv.addEventListener("pointerdown", (ev) => {
    cancelAnimationFrame(state.anim);
    cv.setPointerCapture(ev.pointerId);
    state.drag = { x: ev.clientX, y: ev.clientY, cx: state.view.cx, cy: state.view.cy, moved: 0 };
    cv.classList.add("dragging");
  });

  cv.addEventListener("pointermove", (ev) => {
    if (!state.drag) return;
    const dx = ev.clientX - state.drag.x;
    const dy = ev.clientY - state.drag.y;
    state.drag.moved = Math.max(state.drag.moved, Math.abs(dx) + Math.abs(dy));
    if (state.drag.moved > 4) state.userMoved = true;
    state.view.cx = state.drag.cx - dx / state.view.scale;
    state.view.cy = state.drag.cy - dy / state.view.scale;
    draw();
  });

  const endDrag = (ev) => {
    if (!state.drag) return;
    const moved = state.drag.moved;
    state.drag = null;
    cv.classList.remove("dragging");
    if (moved < 4) pickFixAt(ev);
  };
  cv.addEventListener("pointerup", endDrag);
  cv.addEventListener("pointercancel", () => {
    state.drag = null;
    cv.classList.remove("dragging");
  });

  cv.addEventListener(
    "wheel",
    (ev) => {
      ev.preventDefault();
      const rect = cv.getBoundingClientRect();
      zoomBy(Math.exp(-ev.deltaY * 0.0016), ev.clientX - rect.left, ev.clientY - rect.top);
    },
    { passive: false }
  );

  cv.addEventListener("dblclick", () => zoomBy(1.8, chart.w / 2, chart.h / 2));

  cv.addEventListener("keydown", (ev) => {
    const step = ev.shiftKey ? 10 : 1;
    if (ev.key === "ArrowRight") { ev.preventDefault(); setIndex(state.i + step); }
    else if (ev.key === "ArrowLeft") { ev.preventDefault(); setIndex(state.i - step); }
    else if (ev.key === "f") { fitView(); state.focus = null; draw(); }
    else if (ev.key === "m") { goMiss(ev.shiftKey ? -1 : 1); }
  });

  $("#api-open").addEventListener("click", openApi);
  $("#about-open").addEventListener("click", openMethod);
  document.querySelector(".hint").addEventListener("click", openMethod);

  for (const dlg of document.querySelectorAll("dialog")) {
    dlg.addEventListener("click", (ev) => {
      if (ev.target === dlg) dlg.close();
      if (ev.target.closest("[data-close]")) dlg.close();
    });
  }
}

function pickFixAt(ev) {
  const d = state.d;
  if (!d) return;
  const rect = el.chart.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const my = ev.clientY - rect.top;
  let best = -1;
  let bestDist = 14 * 14;
  for (let k = 0; k < d.n; k++) {
    const dx = sx(d.wpts[k * 2]) - mx;
    const dy = sy(d.wpts[k * 2 + 1]) - my;
    const dd = dx * dx + dy * dy;
    if (dd < bestDist) { bestDist = dd; best = k; }
  }
  if (best >= 0) {
    state.focus = null;
    state.missIdx = -1;
    setIndex(best);
  }
}

function zoomBy(f, mx, my) {
  const v = state.view;
  const wx = (mx - chart.w / 2) / v.scale + v.cx;
  const wy = (my - chart.h / 2) / v.scale + v.cy;
  const min = state.fitScale * 0.6;
  const max = state.fitScale * 260;
  v.scale = Math.max(min, Math.min(max, v.scale * f));
  v.cx = wx - (mx - chart.w / 2) / v.scale;
  v.cy = wy - (my - chart.h / 2) / v.scale;
  state.userMoved = true;
  draw();
}

/* -------------------------------------------------------------------- loading */

async function loadTrip(id) {
  const cached = state.cache.get(id);
  if (cached) {
    setTrip(cached);
    return;
  }
  el.tapeStatus.textContent = `Loading drive ${id}…`;
  try {
    const res = await fetch(`/api/trips/${id}`);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const payload = await res.json();
    state.cache.set(id, payload);
    setTrip(payload);
  } catch (err) {
    el.tapeStatus.textContent = `Could not load ${id}: ${err.message}. Check that the payloads exist.`;
  }
}

function setTrip(payload) {
  state.trip = payload;
  state.d = derive(payload);
  state.i = 0;
  state.focus = null;
  state.missIdx = -1;
  state.userMoved = false;
  fitView();
  updateSheet();
  el.scrub.max = String(payload.track.n - 1);
  setIndex(0);
  draw();
}

async function boot() {
  resize();
  bind();
  try {
    const res = await fetch("/api/trips");
    const data = await res.json();
    state.trips = data.trips || [];
    if (!state.trips.length) {
      el.tapeStatus.textContent = "No payloads yet. Run python -m app.build_payloads.";
      return;
    }
    fillPicker();
    const first = state.trips.find((t) => t.partition === "train") ?? state.trips[0];
    el.select.value = first.id;
    await loadTrip(first.id);
  } catch (err) {
    el.tapeStatus.textContent = `Index unavailable: ${err.message}`;
  }
}

function fillPicker() {
  el.select.innerHTML = "";
  const groups = [
    ["train", "Training drives"],
    ["test", "Student-test drives"],
  ];
  for (const [part, title] of groups) {
    const trips = state.trips.filter((t) => t.partition === part);
    if (!trips.length) continue;
    const og = document.createElement("optgroup");
    og.label = title;
    for (const t of trips) {
      const o = document.createElement("option");
      o.value = t.id;
      o.textContent = t.city ? `${t.id} (${t.city})` : t.id;
      og.append(o);
    }
    el.select.append(og);
  }
}

/* -------------------------------------------------------------------- panels */

async function openApi() {
  if (!state.trip) return;
  el.apiNote.textContent = "Running the matcher…";
  el.apiBody.textContent = "";
  el.apiDialog.showModal();
  try {
    const res = await fetch("/api/match", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ trip_id: state.trip.id }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(`${res.status} ${JSON.stringify(data)}`);
    const shown = {
      trip_id: data.trip_id,
      method: data.method,
      fixes: data.fixes,
      elapsed_ms: data.elapsed_ms,
      metrics: data.metrics,
      arcs: data.arcs.slice(0, 12),
      offset_m: data.offset_m.slice(0, 12),
    };
    el.apiNote.textContent =
      `POST /api/match ran the matcher on ${nf.format(data.fixes)} fixes in ${data.elapsed_ms} ms ` +
      `(this includes loading the drive's network on the first call). ` +
      `The full response is ${(JSON.stringify(data).length / 1024).toFixed(0)} KB; the arc and offset lists are trimmed here.`;
    el.apiBody.textContent = JSON.stringify(shown, null, 2);
  } catch (err) {
    el.apiNote.textContent = `Request failed: ${err.message}`;
  }
}

async function openMethod() {
  el.aboutBody.innerHTML = "<p>Loading…</p>";
  el.aboutDialog.showModal();
  try {
    const m = await fetch("/api/methodology").then((r) => r.json());
    el.aboutBody.innerHTML = `
      <h3>What is drawn</h3>
      <p>${m.summary}</p>
      <p>The map carries four layers: the road network within 650 m of the drive, the
      reference route supplied with the data, the path produced by snapping every fix,
      and the GPS fixes themselves. Where the magenta line leaves the blue band, the
      matcher chose a road the car did not drive.</p>

      <h3>Why a baseline is the point</h3>
      <p>${m.why} It is not a trained model: there is no classifier, no hidden Markov
      model and no tuning here. Those are the next step, and the numbers on this sheet
      are the floor they have to beat.</p>

      <h3>How the numbers are counted</h3>
      <ul>
        <li><b>Route recovered</b> (recall, direction included): share of reference-route
        segments that receive at least one fix. This is the literal task, because the
        output is a sequence of segments.</li>
        <li><b>Roads found</b>: the same count treating a segment and its reverse twin
        as one piece of road. 92% of these segments have a twin at identical geometry, so
        the strict number depends on how ties are broken; this view isolates "found the
        road" from "faced the right way".</li>
        <li><b>Precision and F1</b> use the strict rule on the same segment sets.</li>
        <li><b>Continuity breaks</b>: consecutive fixes whose snapped segments share no
        node, so the output cannot be driven without teleporting.</li>
        <li><b>N/A</b>: macro F1 and log loss are classifier metrics. This matcher emits
        no per-class probabilities, so quoting them would be invented.</li>
      </ul>

      <h3>Assumptions worth challenging</h3>
      <ul>${m.assumptions.map((a) => `<li>${a}</li>`).join("")}</ul>

      <h3>One number to quote</h3>
      <p><code>Arc recall, strict = ${pct(state.trip.metrics.arcs.recall_strict)}%</code>
      on this drive, against a 90-drive median of 57.0%. Point accuracy is deliberately
      not the headline: it averages over fixes, which is exactly the unit the baseline is
      allowed to use.</p>
    `;
  } catch (err) {
    el.aboutBody.innerHTML = `<p>Methodology endpoint failed: ${err.message}</p>`;
  }
}

let toastTimer = 0;
function toast(text) {
  el.toast.textContent = text;
  el.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.toast.classList.remove("show"), 3200);
}

boot();

/* Small console handle for demos and the screenshot tests: pathsnap.state,
   pathsnap.setIndex(400), pathsnap.screenOf(400). */
window.pathsnap = {
  state,
  setIndex,
  goMiss,
  screenOf(i) {
    if (!state.d) return null;
    return [sx(state.d.wpts[i * 2]), sy(state.d.wpts[i * 2 + 1])];
  },
};
