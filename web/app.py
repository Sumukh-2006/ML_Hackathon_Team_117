"""The website: pages, uploads, and a thin proxy to the model server.

Run: uv run python web/app.py   (then open http://127.0.0.1:5000)
The model server (`python -m serve`) must be running.
"""

from flask import Flask, Response, jsonify, render_template, request

import model_client as client

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.errorhandler(client.ModelServerDown)
def server_down(error):
    return jsonify({"error": {"code": "model_server_down",
                              "message": "the model server is not reachable"}}), 502


def relay(response) -> Response:
    """The model server's answer, status and body unchanged."""
    return Response(response.content, status=response.status_code,
                    content_type=response.headers.get("Content-Type"))


@app.get("/")
def index():
    return render_template("index.html")


# ---- proxy: the browser only ever talks to the website -----------------

@app.get("/api/models")
def models():
    return relay(client.call("GET", "/models"))


@app.get("/api/corpus")
def corpus():
    return relay(client.call("GET", "/corpus"))


@app.get("/api/corpus/<split>/<name>/track")
def corpus_track(split, name):
    return relay(client.call("GET", f"/corpus/{split}/{name}/track"))


@app.post("/api/sessions")
def open_session():
    return relay(client.call("POST", "/sessions", json=request.get_json()))


@app.post("/api/sessions/<sid>/fixes")
def push(sid):
    return relay(client.call("POST", f"/sessions/{sid}/fixes", json=request.get_json()))


@app.post("/api/sessions/<sid>/finish")
def finish(sid):
    wait = float(request.args.get("wait", 25))
    return relay(client.call("POST", f"/sessions/{sid}/finish", wait=wait,
                             params={"wait": wait}))


@app.get("/api/sessions/<sid>")
def state(sid):
    wait = float(request.args.get("wait", 0))
    return relay(client.call("GET", f"/sessions/{sid}", wait=wait,
                             params={"wait": wait}))


@app.get("/api/sessions/<sid>/route")
def route(sid):
    model = request.args.get("model", "")
    response = client.call("GET", f"/sessions/{sid}/route", params={"model": model})
    out = relay(response)
    if response.ok:
        out.headers["Content-Disposition"] = f"attachment; filename={sid}_{model}.route"
    return out


# ---- upload: map once, track parsed here into fixes ---------------------

@app.post("/api/upload")
def upload():
    files = request.files
    if not all(k in files for k in ("nodes", "arcs", "track")):
        return jsonify({"error": {"code": "bad_request",
                                  "message": "nodes, arcs and track files are required"}}), 400
    mid = client.ensure_map(files["nodes"].read(), files["arcs"].read())
    if not isinstance(mid, str):
        return relay(mid)
    fixes = []
    for number, line in enumerate(files["track"].read().decode().splitlines(), 1):
        if not line.strip():
            continue
        try:
            lon, lat, t = (float(v) for v in line.split())
        except ValueError:
            return jsonify({"error": {"code": "invalid_track",
                                      "message": f"track:{number}: expected lon lat t, found {line!r}"}}), 400
        fixes.append([lon, lat, t])
    return jsonify({"map_id": mid, "fixes": fixes})


if __name__ == "__main__":
    # threaded: each long poll holds a thread while it waits.
    app.run(host="127.0.0.1", port=5000, threaded=True)
