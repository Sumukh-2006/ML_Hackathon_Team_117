"""Minimal demo UI (FastAPI): replay a corpus track one GPS fix at a time
through the model server and draw the route as it is matched.

Run: uv run python -m serve            (model server, port 8100)
     uv run python web/minimal.py      (this page, port 5001, all interfaces)

The model server stays on 127.0.0.1; only this page is reachable from other
machines, and it passes on only the calls the page makes (no map uploads).
HOST=127.0.0.1 keeps the page local.
"""

import os

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse

API = os.environ.get("MODEL_API", "http://127.0.0.1:8100/v1")
app, http = FastAPI(), requests.Session()


ALLOWED = ("corpus", "models", "sessions")  # what the page uses; not /maps


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request) -> Response:
    if not path.startswith(ALLOWED) or (request.method == "POST" and not path.startswith("sessions")):
        return Response(status_code=403)
    r = http.request(request.method, f"{API}/{path}", params=request.query_params,
                     data=await request.body(), headers={"Content-Type": "application/json"})
    return Response(r.content, r.status_code, media_type=r.headers.get("Content-Type"))


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Route Matching</title><link rel=stylesheet href=https://unpkg.com/leaflet@1.9.4/dist/leaflet.css>
<style>
  body { margin: 0; font: 14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1f2933; }
  header { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; padding: 10px 16px;
           background: #fff; border-bottom: 1px solid #e4e7eb; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
  h1 { font-size: 16px; font-weight: 600; margin: 0 12px 0 0; }
  select, button { font: inherit; padding: 6px 10px; border: 1px solid #cbd2d9; border-radius: 6px; background: #fff; }
  button { background: #2563eb; border-color: #2563eb; color: #fff; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #out { margin-left: auto; color: #52606d; font-variant-numeric: tabular-nums; }
  #m { height: calc(100vh - 56px); }
  .leaflet-tile-pane { filter: grayscale(.7) contrast(.9) brightness(1.05); }
  .legend { background: #fff; padding: 8px 10px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.2); line-height: 1.7; }
  .legend i { display: inline-block; width: 22px; height: 0; margin-right: 6px; vertical-align: middle; border-top: 3px solid; }
</style>
<header><h1>Route Matching</h1><select id=folder></select><select id=model></select>
<button id=run onclick=go()>Match</button><span id=out>Pick a track and a model</span></header>
<div id=m></div><script src=https://unpkg.com/leaflet@1.9.4/dist/leaflet.js></script><script>
const map = L.map('m').setView([50, 10], 5), T = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
L.tileLayer(T, {attribution: '&copy; OpenStreetMap contributors'}).addTo(map);
let C, drawn = L.layerGroup().addTo(map), legend = L.control({position: 'bottomleft'});
legend.onAdd = () => Object.assign(L.DomUtil.create('div', 'legend'), {innerHTML:
  '<i style="border-color:#9aa5b1"></i>GPS track<br><i style="border-color:#2563eb"></i>Matched (final)<br>' +
  '<i style="border-color:#2563eb;border-top-style:dashed"></i>Matched (tentative)<br><i style="border-color:#e11d48"></i>Final route'});
legend.addTo(map);
const j = (u, b) => fetch('/api/' + u, b && {method: 'POST', body: JSON.stringify(b)}).then(r => r.json());
const line = (g, o) => L.polyline(g.coordinates.map(p => p.map(([x, y]) => [y, x])), o).addTo(drawn);
j('corpus').then(c => { C = c; folder.innerHTML = c.map((r, i) => `<option value=${i}>${r.split}/${r.id}`).join(''); });
j('models').then(ms => model.innerHTML = ms.filter(m => m.available).map(m => `<option>${m.id}`).join(''));
async function go() {
  const c = C[folder.value], m = model.value, {fixes} = await j(`corpus/${c.split}/${c.id}/track`);
  run.disabled = true; drawn.clearLayers();
  L.polyline(fixes.map(([x, y]) => [y, x]), {color: '#9aa5b1', weight: 2}).addTo(drawn);
  const dot = L.circleMarker([fixes[0][1], fixes[0][0]],
    {radius: 7, color: '#fff', weight: 2, fillColor: '#1f2933', fillOpacity: 1}).addTo(drawn);
  map.setView(dot.getLatLng(), 15);
  const s = await j('sessions', {map_id: c.map_id, models: [m], options: {truth: {corpus: {split: c.split, id: c.id}}}});
  let tail = L.layerGroup().addTo(drawn);
  for (let i = 0; i < fixes.length; i++) {           // one GPS fix per request
    const u = (await j(`sessions/${s.session_id}/fixes`, {seq: i, fixes: [fixes[i]]})).updates[m];
    out.textContent = `Fix ${i + 1} / ${fixes.length}`;
    dot.setLatLng([fixes[i][1], fixes[i][0]]); map.panTo(dot.getLatLng(), {animate: false});  // camera follows the fix
    if (u.kept === null) continue;                    // at-finish model: nothing until the end
    line(u.geometry.committed, {color: '#2563eb', weight: 4}); tail.remove();
    tail = line(u.geometry.tentative, {color: '#2563eb', weight: 4, dashArray: '6 6', opacity: .7});
  }
  out.textContent = 'Finishing the match…';
  let r = await j(`sessions/${s.session_id}/finish?wait=30`, {});
  while (r.status !== 'done') r = await j(`sessions/${s.session_id}?wait=30`);
  r = r.results[m]; run.disabled = false;
  if (r.error) return out.textContent = r.error.message;
  line(r.geometry, {color: '#e11d48', weight: 5, opacity: .85});
  out.textContent = `${m}: ${r.route.length} arcs · ${(r.length_m / 1000).toFixed(1)} km · ` +
    `correct fraction ${(100 * r.score.correct_fraction).toFixed(2)}%`;
}
</script>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=5001)
