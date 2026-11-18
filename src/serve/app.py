"""The HTTP API of docs/api.md. One process, many threads: sessions, maps
and results live in this process's memory, so run exactly one of it."""

import hashlib
import math
import secrets
import shutil
import threading
import time
from concurrent.futures import Future, wait as wait_for
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, request

from dataset import Corpus, DatasetError, RoadGraph
from dataset.table import read_table
from serve import models
from serve.workers import Worker, Workers

MAX_UPLOAD: int = 64 * 1024 * 1024    # OWN CHOICE (largest corpus map 6.8 MB)
MAX_FIXES_PER_REQUEST: int = 1000     # OWN CHOICE
MAX_FIXES_PER_SESSION: int = 100_000  # OWN CHOICE (longest corpus track 19 313)
MAX_SESSIONS: int = 64                # OWN CHOICE
IDLE_SECONDS: float = 15 * 60         # OWN CHOICE
MAX_WAIT: float = 30.0
KEPT_RESPONSES: int = 64              # replies kept per session for retries


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **extra) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra


def map_id_of(nodes: bytes, arcs: bytes) -> str:
    inner: bytes = hashlib.sha256(nodes).digest() + hashlib.sha256(arcs).digest()
    return hashlib.sha256(inner).hexdigest()[:16]


@dataclass
class Session:
    sid: str
    map_id: str
    map_dir: str
    models: list[str]
    truth: list[int] | None
    fixes: list[list[float]] = field(default_factory=list)
    committed: dict[str, list[int]] = field(default_factory=dict)
    tentative: dict[str, list[int]] = field(default_factory=dict)
    replies: dict[int, tuple[list, dict]] = field(default_factory=dict)
    results: dict[str, Future | dict] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    finished: bool = False
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


def create_app(corpus_root: Path, results: Path, source: Path,
               streaming_workers: int, batch_workers: int) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD
    store: Path = results / "serve" / "maps"
    store.mkdir(parents=True, exist_ok=True)
    workers = Workers(results, streaming_workers, batch_workers)

    maps: dict[str, Path] = {}          # map_id -> folder in corpus layout
    corpus: list[dict] = []             # GET /corpus rows
    folders: dict[tuple[str, str], Path] = {}
    for split, entries in Corpus.from_root(corpus_root).splits.items():
        for folder in entries:
            name: str = folder.name()
            mid = map_id_of((folder.path / f"{name}.nodes").read_bytes(),
                            (folder.path / f"{name}.arcs").read_bytes())
            maps[mid] = folder.path
            folders[(split, name)] = folder.path
            tags_file = (corpus_root.parent / "corpus_fixed" / split / name
                         / f"{name}.tags")
            corpus.append({
                "split": split, "id": name, "map_id": mid,
                "fixes": sum(1 for _ in open(folder.path / f"{name}.track")),
                "has_route": folder.has_route(),
                "tags": tags_file.read_text().split() if tags_file.exists() else []})
    for path in store.iterdir():
        maps[path.name] = path

    unavailable: dict[str, str | None] = {m: models.availability(m, results)
                                          for m in models.MODELS}
    versions: dict[str, str] = {m: models.version(m, results, source)
                                for m in models.MODELS}
    sessions: dict[str, Session] = {}
    sessions_lock = threading.Lock()

    # ---- helpers ---------------------------------------------------------

    @app.errorhandler(ApiError)
    def api_error(error: ApiError):
        body = {"error": {"code": error.code, "message": error.message, **error.extra}}
        response = jsonify(body)
        response.status_code = error.status
        if error.status == 503:
            response.headers["Retry-After"] = "5"
        return response

    @app.errorhandler(413)
    def too_large(_):
        return api_error(ApiError(413, "too_large", f"upload over {MAX_UPLOAD} bytes"))

    def body() -> dict:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError(400, "bad_request", "expected a JSON object")
        return data

    def session_of(sid: str) -> Session:
        with sessions_lock:
            if sid not in sessions:
                raise ApiError(404, "unknown_session", sid)
            session = sessions[sid]
        session.touched = time.monotonic()
        return session

    def truth_of(options: dict) -> list[int] | None:
        truth = options.get("truth")
        if truth is None:
            return None
        if isinstance(truth, list):
            return [int(a) for a in truth]
        key = (truth.get("corpus", {}).get("split"), truth.get("corpus", {}).get("id"))
        if key not in folders:
            raise ApiError(400, "bad_request", f"no corpus folder {key}")
        path = folders[key] / f"{key[1]}.route"
        if not path.exists():
            raise ApiError(400, "bad_request", f"corpus folder {key} has no route")
        return [int(a) for a in path.read_text().split()]

    def checked_fixes(raw, last_t: float | None) -> list[list[float]]:
        if not isinstance(raw, list) or len(raw) == 0:
            raise ApiError(400, "bad_request", "fixes must be a non-empty list")
        if len(raw) > MAX_FIXES_PER_REQUEST:
            raise ApiError(413, "too_large", f"over {MAX_FIXES_PER_REQUEST} fixes")
        fixes: list[list[float]] = []
        for index, fix in enumerate(raw):
            if (not isinstance(fix, list) or len(fix) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in fix)):
                raise ApiError(400, "invalid_fix", f"fix {index}: expected [lon, lat, t]")
            lon, lat, t = (float(v) for v in fix)
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ApiError(400, "invalid_fix", f"fix {index}: lon/lat out of range")
            if last_t is not None and t <= last_t:
                raise ApiError(400, "invalid_fix",
                               f"fix {index}: time {t} does not come after {last_t}")
            fixes.append([lon, lat, t])
            last_t = t
        return fixes

    def close(session: Session) -> None:
        for model in session.models:
            key = f"{session.sid}/{model}"
            worker: Worker | None = workers.placed.get(key)
            if worker is not None:
                try:
                    worker.call("close", key)
                except RuntimeError:
                    pass
                workers.release(key)

    def sweep() -> None:
        """Drop sessions idle past IDLE_SECONDS."""
        now = time.monotonic()
        with sessions_lock:
            stale = [s for s in sessions.values() if now - s.touched > IDLE_SECONDS]
            for s in stale:
                del sessions[s.sid]
        for s in stale:
            close(s)

    def model_results(session: Session, wait: float) -> tuple[str, dict]:
        """Every model's result, waiting up to `wait` s for running ones."""
        pending = [f for f in session.results.values() if isinstance(f, Future)]
        if pending and wait > 0:
            wait_for(pending, timeout=wait)
        out: dict = {}
        for model in session.models:
            if model in session.failed:
                out[model] = {"status": "failed", "error": {
                    "code": "match_failed", "message": session.failed[model]}}
                continue
            value = session.results.get(model)
            if value is None:
                out[model] = {"status": "streaming" if models.mode(model) == "online"
                              else "waiting_for_finish"}
            elif isinstance(value, Future):
                if not value.done():
                    out[model] = {"status": "running"}
                elif value.exception() is not None:
                    session.failed[model] = str(value.exception())
                    session.results.pop(model)
                    out[model] = {"status": "failed", "error": {
                        "code": "match_failed", "message": session.failed[model]}}
                else:
                    session.results[model] = value.result()
                    out[model] = session.results[model]
            else:
                out[model] = value
        done = all(r["status"] in ("done", "failed") for r in out.values())
        return ("done" if done else "running"), out

    def wait_seconds() -> float:
        try:
            return max(0.0, min(MAX_WAIT, float(request.args.get("wait", 0))))
        except ValueError:
            raise ApiError(400, "bad_request", "wait must be a number")

    # ---- service ---------------------------------------------------------

    @app.get("/v1/health")
    def health():
        return jsonify({
            "ready": workers.alive(),
            "models": {m: {"state": "unavailable" if unavailable[m] else "ready",
                           "mode": models.mode(m)} for m in models.MODELS},
            "workers": {"streaming": len(workers.streaming),
                        "batch": batch_workers},
            "sessions": len(sessions)})

    @app.get("/v1/models")
    def list_models():
        return jsonify([{"id": m, "name": models.NAMES[m], "mode": models.mode(m),
                         "version": versions[m], "available": unavailable[m] is None,
                         "reason": unavailable[m]} for m in models.MODELS])

    # ---- maps and corpus -------------------------------------------------

    @app.route("/v1/maps/<map_id>", methods=["HEAD", "GET"])
    def get_map(map_id: str):
        if map_id not in maps:
            if request.method == "HEAD":
                return Response(status=404)
            raise ApiError(404, "unknown_map", map_id)
        if request.method == "HEAD":
            return Response(status=200)
        g: RoadGraph = models.graph(str(maps[map_id]))
        return jsonify({"map_id": map_id, "nodes": g.node_count(), "arcs": g.arc_count(),
                        "bbox": [float(g.node_lon.min()), float(g.node_lat.min()),
                                 float(g.node_lon.max()), float(g.node_lat.max())]})

    @app.post("/v1/maps")
    def post_map():
        if "nodes" not in request.files or "arcs" not in request.files:
            raise ApiError(400, "bad_request", "multipart fields nodes and arcs required")
        nodes, arcs = request.files["nodes"].read(), request.files["arcs"].read()
        mid = map_id_of(nodes, arcs)
        status = 200
        if mid not in maps:
            folder = store / mid
            folder.mkdir(exist_ok=True)
            (folder / f"{mid}.nodes").write_bytes(nodes)
            (folder / f"{mid}.arcs").write_bytes(arcs)
            try:
                RoadGraph.from_folder(folder)
            except (DatasetError, ValueError) as error:
                shutil.rmtree(folder)
                raise ApiError(400, "invalid_map", str(error))
            maps[mid] = folder
            status = 201
        response = get_map(mid)
        response.status_code = status
        return response

    @app.get("/v1/corpus")
    def list_corpus():
        return jsonify(corpus)

    @app.get("/v1/corpus/<split>/<name>/track")
    def corpus_track(split: str, name: str):
        if (split, name) not in folders:
            raise ApiError(404, "unknown_map", f"{split}/{name}")
        lon, lat, t = read_table(folders[(split, name)] / f"{name}.track",
                                 [float, float, float])
        return jsonify({"fixes": [[a, b, c] for a, b, c in
                                  zip(lon.tolist(), lat.tolist(), t.tolist())]})

    # ---- sessions --------------------------------------------------------

    @app.post("/v1/sessions")
    def open_session():
        sweep()
        data = body()
        mid = data.get("map_id")
        if mid not in maps:
            raise ApiError(404, "unknown_map", str(mid))
        wanted = data.get("models")
        if not isinstance(wanted, list) or len(wanted) == 0:
            raise ApiError(400, "bad_request", "models must be a non-empty list")
        for m in wanted:
            if m not in models.MODELS:
                raise ApiError(404, "unknown_model", str(m))
            if unavailable[m]:
                raise ApiError(503, "model_unavailable", f"{m}: {unavailable[m]}")
        if len(sessions) >= MAX_SESSIONS:
            raise ApiError(503, "too_many_sessions", f"{MAX_SESSIONS} open")
        options = data.get("options", {}) or {}
        session = Session(secrets.token_urlsafe(6), mid, str(maps[mid]),
                          list(dict.fromkeys(wanted)), truth_of(options))
        for m in session.models:
            if models.mode(m) == "online":
                key = f"{session.sid}/{m}"
                workers.place(key, mid).call("open", key, m, session.map_dir)
                session.committed[m], session.tentative[m] = [], []
        with sessions_lock:
            sessions[session.sid] = session
        response = jsonify({
            "session_id": session.sid,
            "models": {m: {"mode": models.mode(m), "version": versions[m]}
                       for m in session.models},
            "seq_next": 0})
        response.status_code = 201
        return response

    @app.post("/v1/sessions/<sid>/fixes")
    def push_fixes(sid: str):
        session = session_of(sid)
        data = body()
        seq = data.get("seq")
        if not isinstance(seq, int) or seq < 0:
            raise ApiError(400, "bad_request", "seq must be a non-negative integer")
        with session.lock:
            if session.finished:
                raise ApiError(409, "session_finished", sid)
            seq_next = len(session.fixes)
            if seq < seq_next:  # a retry of a request already applied
                stored = session.replies.get(seq)
                if stored is not None and stored[0] == data.get("fixes"):
                    return jsonify(stored[1])
                raise ApiError(409, "seq_gap", f"seq {seq} already applied",
                               expected=seq_next)
            if seq > seq_next:
                raise ApiError(409, "seq_gap", f"expected seq {seq_next}",
                               expected=seq_next)
            last_t = session.fixes[-1][2] if session.fixes else None
            fixes = checked_fixes(data.get("fixes"), last_t)
            if seq_next + len(fixes) > MAX_FIXES_PER_SESSION:
                raise ApiError(413, "too_large", f"over {MAX_FIXES_PER_SESSION} fixes")
            session.fixes.extend(fixes)

            updates: dict = {}
            for m in session.models:
                if models.mode(m) != "online":
                    updates[m] = {"kept": None}
                    continue
                if m in session.failed:
                    updates[m] = {"status": "failed", "error": {
                        "code": "match_failed", "message": session.failed[m]}}
                    continue
                try:
                    update = workers.placed[f"{sid}/{m}"].call("push", f"{sid}/{m}", fixes)
                except RuntimeError as error:
                    session.failed[m] = str(error)
                    updates[m] = {"status": "failed", "error": {
                        "code": "match_failed", "message": str(error)}}
                    continue
                at = len(session.committed[m])
                session.committed[m].extend(update["committed"])
                session.tentative[m] = update["tentative"]
                update["committed"] = {"at": at, "arcs": update["committed"]}
                updates[m] = update
            reply = {"seq_next": len(session.fixes), "updates": updates}
            session.replies[seq] = (data.get("fixes"), reply)
            if len(session.replies) > KEPT_RESPONSES:
                session.replies.pop(min(session.replies))
            return jsonify(reply)

    @app.post("/v1/sessions/<sid>/finish")
    def finish(sid: str):
        session = session_of(sid)
        with session.lock:
            if not session.finished:
                if not session.fixes:
                    raise ApiError(400, "bad_request", "no fixes to match")
                session.finished = True
                for m in session.models:
                    if m in session.failed:
                        continue
                    if models.mode(m) == "online":
                        key = f"{sid}/{m}"
                        try:
                            session.results[m] = workers.placed[key].call(
                                "finish", key, session.truth)
                        except RuntimeError as error:
                            session.failed[m] = str(error)
                        workers.release(key)
                    else:
                        session.results[m] = workers.batch.submit(
                            models.run_batch, m, session.map_dir, str(results),
                            session.fixes, session.truth)
        status, out = model_results(session, wait_seconds())
        return jsonify({"status": status, "seq_next": len(session.fixes),
                        "results": out})

    @app.get("/v1/sessions/<sid>")
    def state(sid: str):
        session = session_of(sid)
        status, out = model_results(session, wait_seconds() if session.finished else 0)
        live = {m: {"committed": session.committed[m],
                    "tentative": session.tentative[m]} for m in session.committed}
        return jsonify({"session_id": sid, "map_id": session.map_id,
                        "finished": session.finished, "status": status,
                        "seq_next": len(session.fixes), "live": live,
                        "results": out})

    @app.get("/v1/sessions/<sid>/route")
    def route(sid: str):
        session = session_of(sid)
        model = request.args.get("model", "")
        if model not in session.models:
            raise ApiError(404, "unknown_model", model)
        _, out = model_results(session, 0)
        if out[model].get("status") == "done":
            arcs = out[model]["route"]
        elif model in session.committed:
            arcs = session.committed[model] + session.tentative[model]
        else:
            raise ApiError(409, "bad_request", f"{model} has no route until finish")
        return Response("".join(f"{a}\n" for a in arcs), mimetype="text/plain")

    @app.delete("/v1/sessions/<sid>")
    def delete(sid: str):
        session = session_of(sid)
        with sessions_lock:
            sessions.pop(sid, None)
        close(session)
        return Response(status=204)

    return app
