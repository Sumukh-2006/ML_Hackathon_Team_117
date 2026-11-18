"""FastAPI backend for the PathSnap demo.

Rationale: two kinds of endpoint, on purpose. `/api/trips/*` serves precomputed
payloads so the demo is instant on a projector; `/api/match` runs the matcher
live on a posted track so the evaluator can see the pipeline actually execute
and can screenshot a real JSON response. Same code path, same numbers.
"""
from __future__ import annotations

import json
import os
import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, match

app = FastAPI(title="PathSnap", version="1.0", docs_url="/api/docs")
app.add_middleware(GZipMiddleware, minimum_size=1024)

TRIPS_FILE = os.path.join(config.DATA_DIR, "index.json")


def load_index() -> dict:
    if not os.path.exists(TRIPS_FILE):
        return {"trips": [], "method": config.METHOD}
    with open(TRIPS_FILE, encoding="utf-8") as f:
        return json.load(f)


def trip_path(trip_id: str) -> str:
    p = os.path.join(config.DATA_DIR, f"trip_{os.path.basename(trip_id)}.json")
    if not os.path.exists(p):
        raise HTTPException(404, f"unknown trip {trip_id}")
    return p


@lru_cache(maxsize=4)
def cached_trip(trip_id: str) -> "match.Trip":
    """Keep a few full graphs in memory for live matching.

    ⚠️ assumption: the network for a trip never changes during a session.
    """
    ids = {t for t, p in match.trip_dirs() if p == "test"}
    base = config.TEST_DIR if trip_id in ids else config.TRAIN_DIR
    inst = os.path.join(base, trip_id)
    if not os.path.isdir(inst):
        raise HTTPException(
            503,
            f"raw trip data for {trip_id} is not in this checkout. The provided "
            "train/ and student_test/ folders are not committed; the precomputed "
            "payloads under /api/trips still work.",
        )
    return match.build_trip(inst)


# ------------------------------------------------------------------ endpoints
@app.get("/api/health")
def health():
    idx = load_index()
    return {"status": "ok", "trips": len(idx["trips"]), "method": idx.get("method"),
            "payloads": os.path.isdir(config.DATA_DIR)}


@app.get("/api/trips")
def trips():
    idx = load_index()
    return {"count": len(idx["trips"]), "method": idx.get("method"), "trips": idx["trips"]}


@app.get("/api/trips/{trip_id}")
def trip(trip_id: str):
    with open(trip_path(trip_id), encoding="utf-8") as f:
        return json.load(f)


class MatchRequest(BaseModel):
    trip_id: str = Field(..., description="trip whose road network to match against")
    track: list[list[float]] | None = Field(
        None, description="optional [[lon,lat], ...]; omit to match the stored track"
    )


@app.post("/api/match")
def match_endpoint(req: MatchRequest):
    """Run nearest-arc matching now and return arcs + scores + wall-clock ms."""
    t0 = time.perf_counter()
    trip = cached_trip(req.trip_id)
    if req.track is None:
        P = None
        ll = trip.track[:, :2]
    else:
        ll = np.asarray(req.track, dtype=np.float64)
        if ll.ndim != 2 or ll.shape[1] != 2 or len(ll) < 2:
            raise HTTPException(422, "track must be a list of [lon, lat] pairs, length >= 2")
        P = np.column_stack(match.to_xy(ll, trip.lon0, trip.lat0))

    pred, dist = match.match(trip, P)
    snap, _ = match.snap_to_arc(P if P is not None else trip.track_xy, trip.A, trip.B, pred)
    elapsed = (time.perf_counter() - t0) * 1000

    k = np.cos(np.radians(trip.lat0))
    snap_ll = np.column_stack([
        trip.lon0 + snap[:, 0] / (111320.0 * k),
        trip.lat0 + snap[:, 1] / 111320.0,
    ])
    return {
        "trip_id": req.trip_id,
        "method": config.METHOD,
        "fixes": int(len(pred)),
        "elapsed_ms": round(elapsed, 1),
        "metrics": match.score(trip, pred, dist),
        "arcs": pred.tolist(),
        "snap": [[round(float(a), 5), round(float(b), 5)] for a, b in snap_ll],
        "offset_m": [round(float(v), 1) for v in dist],
    }


@app.get("/api/methodology")
def methodology():
    return {
        "method": config.METHOD,
        "summary": "Each GPS fix is snapped independently to the nearest road segment. "
                   "No history, no connectivity, no smoothing.",
        "why": "This is the floor a real map matcher has to beat, so the UI can show what "
               "sequence information would buy.",
        "scoring": {
            "recall_strict": "share of reference-route arcs that received at least one fix, "
                             "direction included",
            "recall_dir": "same, counting an arc and its reverse twin once (92% of arcs have "
                          "a geometrically identical twin, so strict depends on the tie rule)",
            "top1_dir": "share of fixes snapped to a road the reference route uses, direction ignored",
            "continuity": "consecutive fixes whose snapped arcs share no node",
            "n/a": "macro F1 and log loss are classifier metrics; this matcher produces no "
                   "per-class posteriors, so they do not apply",
        },
        "assumptions": [
            "distances use a local equirectangular plane per trip (<0.1% error at city scale)",
            "the drawn network is a 650 m corridor around the drive; matching uses 1 km",
            "reference routes are treated as reference, not verified truth",
        ],
    }


# static last, so /api/* wins
app.mount("/", StaticFiles(directory=config.WEB_DIR, html=True), name="web")
