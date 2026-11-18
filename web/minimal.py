"""Minimal demo UI (FastAPI): replay a corpus track one GPS fix at a time
through the model server and draw the route as it is matched.

Run: uv run python -m serve            (model server, port 8100)
     uv run python web/minimal.py      (this page, http://127.0.0.1:5001)
"""

import os

import requests
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse

API = os.environ.get("MODEL_API", "http://127.0.0.1:8100/v1")
app, http = FastAPI(), requests.Session()


@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request) -> Response:
    r = http.request(request.method, f"{API}/{path}", params=request.query_params,
                     data=await request.body(), headers={"Content-Type": "application/json"})
    return Response(r.content, r.status_code, media_type=r.headers.get("Content-Type"))


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return """<!doctype html><meta charset=utf-8><title>Route matching demo</title>
<link rel=stylesheet href=https://unpkg.com/leaflet@1.9.4/dist/leaflet.css>
<select id=folder></select> <select id=model></select> <button onclick=go()>Match</button> <span id=out></span>
<div id=m style="height:90vh"></div><script src=https://unpkg.com/leaflet@1.9.4/dist/leaflet.js></script><script>
const map = L.map('m').setView([50, 10], 5), T = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
L.tileLayer(T).addTo(map); let C, drawn = L.layerGroup().addTo(map);
const j = (u, b) => fetch('/api/' + u, b && {method: 'POST', body: JSON.stringify(b)}).then(r => r.json());
const line = (g, o) => L.polyline(g.coordinates.map(p => p.map(([x, y]) => [y, x])), o).addTo(drawn);
j('corpus').then(c => { C = c; folder.innerHTML = c.map((r, i) => `<option value=${i}>${r.split}/${r.id}`).join(''); });
j('models').then(ms => model.innerHTML = ms.filter(m => m.available).map(m => `<option>${m.id}`).join(''));
async function go() {
  const c = C[folder.value], m = model.value, {fixes} = await j(`corpus/${c.split}/${c.id}/track`);
  drawn.clearLayers(); L.polyline(fixes.map(([x, y]) => [y, x]), {color: '#999'}).addTo(drawn);
  const dot = L.circleMarker([fixes[0][1], fixes[0][0]], {radius: 6, color: 'black'}).addTo(drawn);
  map.setView(dot.getLatLng(), 16);
  const s = await j('sessions', {map_id: c.map_id, models: [m], options: {truth: {corpus: {split: c.split, id: c.id}}}});
  let tail = L.layerGroup().addTo(drawn);
  for (let i = 0; i < fixes.length; i++) {           // one GPS fix per request
    const u = (await j(`sessions/${s.session_id}/fixes`, {seq: i, fixes: [fixes[i]]})).updates[m];
    out.textContent = `fix ${i + 1}/${fixes.length}`;
    dot.setLatLng([fixes[i][1], fixes[i][0]]); map.panTo(dot.getLatLng(), {animate: false});  // camera follows the fix
    if (u.kept === null) continue;                    // at-finish model: nothing until the end
    line(u.geometry.committed, {color: 'blue'}); tail.remove();
    tail = line(u.geometry.tentative, {color: 'blue', dashArray: '4'});
  }
  let r = await j(`sessions/${s.session_id}/finish?wait=30`);
  while (r.status !== 'done') r = await j(`sessions/${s.session_id}?wait=30`);
  r = r.results[m]; line(r.geometry, {color: 'red', weight: 4});
  out.textContent = r.error ? r.error.message : `${r.route.length} arcs, correct fraction ${r.score.correct_fraction}`;
}
</script>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5001)
