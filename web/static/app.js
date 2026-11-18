// The replay page. All model work happens on the model server; this file
// sends fixes in order, applies updates, and draws. API: docs/api.md.
"use strict";

const COLORS = { hmm: "#2563eb", mcm: "#d97706", segment: "#059669", mdp: "#dc2626", learned: "#7c3aed" };
const $ = (id) => document.getElementById(id);
const map = L.map("map", { preferCanvas: true }).setView([49.85, 8.6], 12);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  { maxZoom: 19, attribution: "© OpenStreetMap contributors" }).addTo(map);

let models = [], corpus = [];
let track = null;     // { map_id, fixes, truth }
let run = null;       // { sid, seq, models, playing, simStart, wallStart, layers, rows }
const trackLayer = L.layerGroup().addTo(map);
const here = L.circleMarker([0, 0], { radius: 6, color: "#111", fillOpacity: 1 });

// GeoJSON is [lon, lat]; Leaflet wants [lat, lon].
const latlngs = (multi) => multi.coordinates.map((part) => part.map(([lon, lat]) => [lat, lon]));

async function api(method, path, body) {
  const options = { method, headers: {} };
  if (body !== undefined) { options.body = JSON.stringify(body); options.headers["Content-Type"] = "application/json"; }
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) { const e = new Error(data.error ? `${data.error.code}: ${data.error.message}` : response.statusText);
                      e.status = response.status; e.body = data; throw e; }
  return data;
}
const say = (text) => { $("message").textContent = text || ""; };

// ---- setup ---------------------------------------------------------------

async function init() {
  [models, corpus] = await Promise.all([api("GET", "/api/models"), api("GET", "/api/corpus")]);
  $("models").innerHTML = models.map((m) => `
    <label class="model" title="${m.reason || m.name}">
      <input type="checkbox" value="${m.id}" ${m.available ? (m.mode === "online" ? "checked" : "") : "disabled"}>
      <span class="swatch" style="background:${COLORS[m.id] || "#444"}"></span>
      ${m.id} <small>(${m.available ? m.mode.replace("_", " ") : "unavailable"})</small></label>`).join("");
  $("corpus").innerHTML = corpus.map((c, i) =>
    `<option value="${i}">${c.split}/${c.id} · ${c.fixes} fixes${c.tags.length ? " · " + c.tags.join(", ") : ""}</option>`).join("");
}

function showTrack(t) {
  track = t; reset();
  trackLayer.clearLayers();
  const points = t.fixes.map(([lon, lat]) => [lat, lon]);
  points.forEach((p) => L.circleMarker(p, { radius: 1.5, color: "#555", weight: 0, fillOpacity: 0.5 }).addTo(trackLayer));
  map.fitBounds(L.latLngBounds(points), { padding: [20, 20] });
  $("play").disabled = false;
  $("progress").textContent = `${t.fixes.length} fixes loaded`;
}

$("load-corpus").onclick = async () => {
  try {
    const c = corpus[$("corpus").value];
    const { fixes } = await api("GET", `/api/corpus/${c.split}/${c.id}/track`);
    showTrack({ map_id: c.map_id, fixes, truth: c.has_route ? { corpus: { split: c.split, id: c.id } } : null });
  } catch (e) { say(e.message); }
};

$("upload").onsubmit = async (event) => {
  event.preventDefault();
  say("uploading…");
  const response = await fetch("/api/upload", { method: "POST", body: new FormData(event.target) });
  const data = await response.json();
  if (!response.ok) return say(`${data.error.code}: ${data.error.message}`);
  say(""); showTrack({ map_id: data.map_id, fixes: data.fixes, truth: null });
};

// ---- the replay loop ------------------------------------------------------

function reset() {
  if (run) run.playing = false;
  if (run) Object.values(run.layers).forEach((l) => { map.removeLayer(l.committed); map.removeLayer(l.tentative); });
  run = null; $("status").innerHTML = ""; here.remove();
  $("pause").disabled = $("stop").disabled = true;
}

$("play").onclick = async () => {
  if (!track) return;
  say("");
  if (!run) {
    const chosen = [...document.querySelectorAll("#models input:checked")].map((i) => i.value);
    if (!chosen.length) return say("pick at least one model");
    try {
      const s = await api("POST", "/api/sessions",
        { map_id: track.map_id, models: chosen, options: { geometry: true, ...(track.truth ? { truth: track.truth } : {}) } });
      run = { sid: s.session_id, seq: 0, models: chosen, layers: {}, rows: {} };
    } catch (e) { return say(e.message); }
    for (const m of run.models) {
      run.layers[m] = { committed: L.layerGroup().addTo(map), tentative: L.layerGroup().addTo(map) };
      run.rows[m] = { arcs: 0, pending: "", cf: "" };
    }
    here.addTo(map);
  }
  run.playing = true;
  run.simStart = track.fixes[run.seq][2];
  run.wallStart = performance.now();
  $("play").disabled = true; $("pause").disabled = $("stop").disabled = false;
  loop(run);
};

$("pause").onclick = () => { if (run) run.playing = false; $("play").disabled = false; $("pause").disabled = true; };

async function loop(r) {
  const speed = Number($("speed").value);
  while (r.playing && r.seq < track.fixes.length) {
    const simNow = r.simStart + speed * (performance.now() - r.wallStart) / 1000;
    let end = r.seq;   // every fix due by now, at most 1000 per request
    while (end < track.fixes.length && track.fixes[end][2] <= simNow && end - r.seq < 1000) end++;
    if (end === r.seq) { await sleep(Math.max(10, (track.fixes[r.seq][2] - simNow) * 1000 / speed)); continue; }
    const batch = track.fixes.slice(r.seq, end);
    let reply;
    try {
      reply = await api("POST", `/api/sessions/${r.sid}/fixes`, { seq: r.seq, fixes: batch });
    } catch (e) {
      if (e.status === 409 && e.body.error && e.body.error.expected !== undefined) { r.seq = e.body.error.expected; continue; }
      if (e.status >= 500 || !e.status) { say(`${e.message}; retrying`); await sleep(1000); continue; }  // same seq: safe
      say(e.message); r.playing = false; break;
    }
    say("");
    r.seq = reply.seq_next;
    apply(r, reply.updates);
    const [lon, lat] = batch[batch.length - 1];
    here.setLatLng([lat, lon]);
    $("progress").textContent = `fix ${r.seq} / ${track.fixes.length}`;
  }
  if (r.seq >= track.fixes.length && r.playing) await stop();
}

function apply(r, updates) {
  for (const [m, u] of Object.entries(updates)) {
    if (u.status === "failed") { r.rows[m].pending = "failed"; say(`${m}: ${u.error.message}`); continue; }
    if (u.kept === null) { r.rows[m].pending = "at finish"; continue; }
    const style = { color: COLORS[m] || "#444", weight: 4, opacity: 0.85 };
    if (u.committed.arcs.length) L.polyline(latlngs(u.geometry.committed), style).addTo(r.layers[m].committed);
    r.layers[m].tentative.clearLayers();
    if (u.tentative.length) L.polyline(latlngs(u.geometry.tentative), { ...style, dashArray: "6 6", opacity: 0.6 })
      .addTo(r.layers[m].tentative);
    r.rows[m].arcs = u.committed.at + u.committed.arcs.length;
    r.rows[m].pending = u.pending_fixes;
  }
  table(r);
}

function table(r) {
  $("status").innerHTML = r.models.map((m) => `<tr>
    <td><span class="swatch" style="background:${COLORS[m]}"></span> ${m}</td>
    <td>${r.rows[m].arcs}</td><td>${r.rows[m].pending}</td><td>${r.rows[m].cf}</td>
    <td>${r.rows[m].done ? `<a href="/api/sessions/${r.sid}/route?model=${m}">.route</a>` : ""}</td></tr>`).join("");
}

// ---- finish: flush online models, run the at-finish ones, draw finals -----

$("stop").onclick = () => stop();

async function stop() {
  const r = run; if (!r) return;
  r.playing = false;
  $("play").disabled = $("pause").disabled = $("stop").disabled = true;
  $("progress").textContent = "finishing…";
  try {
    let state = await api("POST", `/api/sessions/${r.sid}/finish?wait=5`);
    draw(r, state.results);
    while (state.status !== "done") {
      state = await api("GET", `/api/sessions/${r.sid}?wait=25`);
      draw(r, state.results);
    }
    $("progress").textContent = `finished: ${r.seq} fixes`;
  } catch (e) { say(e.message); }
}

function draw(r, results) {
  for (const [m, res] of Object.entries(results)) {
    if (res.status === "failed") { r.rows[m].pending = "failed"; say(`${m}: ${res.error.message}`); continue; }
    if (res.status !== "done" || r.rows[m].done) { if (res.status === "running") r.rows[m].pending = "running…"; continue; }
    r.layers[m].committed.clearLayers(); r.layers[m].tentative.clearLayers();
    L.polyline(latlngs(res.geometry), { color: COLORS[m] || "#444", weight: 4, opacity: 0.85 }).addTo(r.layers[m].committed);
    Object.assign(r.rows[m], { done: true, arcs: res.route.length, pending: 0,
      cf: res.score ? (100 * res.score.correct_fraction).toFixed(2) + "%" : "" });
  }
  table(r);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
init().catch((e) => say(`cannot reach the website API: ${e.message}`));
