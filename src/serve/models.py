"""The matchers the server offers, and what runs inside worker processes.

A model is `online` (it has a streaming decoder that extends the route one
fix at a time) or `at_finish` (its batch `match` runs once on all the fixes
when the session finishes). To make a model online, write its streaming
class with `push(fixes) -> Update` and `finish() -> (Route, counts)`
(see hmm/online.py) and add it to `open_online`.
"""

import hashlib
import importlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmark.metrics import score_route
from dataset import RoadGraph, Route, Track

MODELS: list[str] = ["hmm", "mcm", "segment", "mdp", "learned"]
ONLINE: set[str] = {"hmm"}
NAMES: dict[str, str] = {
    "hmm": "Newson & Krumm HMM",
    "mcm": "Multiple candidate matching (Li et al. 2023)",
    "segment": "Feature-based (Yin et al. 2018)",
    "mdp": "Woeltche MDP (map_matching_2 port)",
    "learned": "Approach 5: learned costs",
}

# The approach-5 session provides `match(graph, track) -> (Route, counts)`
# that needs no ground truth, in this module. Until it exists, `learned`
# is reported unavailable.
LEARNED_MODULE: str = "learned.serving"


def mode(model: str) -> str:
    return "online" if model in ONLINE else "at_finish"


def availability(model: str, results: Path) -> str | None:
    """None if the model can run, else why not."""
    if model == "hmm" and not (results / "hmm" / "parameters.json").exists():
        return "results/hmm/parameters.json missing: run `python -m hmm`"
    if model == "learned":
        try:
            importlib.import_module(LEARNED_MODULE)
        except Exception as error:
            return f"{LEARNED_MODULE} not importable ({error})"
    return None


def version(model: str, results: Path, source: Path) -> str:
    """Hash of what decides the model's output: its measured parameters
    and its hyperparameters file. A new value invalidates old results."""
    digest = hashlib.sha256()
    for path in [results / model / "parameters.json", results / model / "weights.json",
                 source / model / "hyperparameters.py"]:
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()[:8]


# ---- inside worker processes ---------------------------------------------

@lru_cache(maxsize=8)  # OWN CHOICE: parsed maps kept per worker process
def graph(map_dir: str) -> RoadGraph:
    return RoadGraph.from_folder(Path(map_dir))


def open_online(model: str, map_dir: str, results: str):
    """A fresh streaming decoder for one session."""
    if model == "hmm":
        from hmm.online import OnlineHMM
        p: dict = json.loads((Path(results) / "hmm" / "parameters.json").read_text())
        return OnlineHMM(graph(map_dir), p["sigma"], p["beta"])
    raise ValueError(f"{model} has no streaming decoder")


def run_batch(model: str, map_dir: str, results: str, fixes: list,
              truth: list[int] | None) -> dict:
    """The model's batch match on all fixes (process-pool entry point)."""
    g: RoadGraph = graph(map_dir)
    lon, lat, t = (np.array(column, dtype=np.float64) for column in zip(*fixes))
    x, y = g.projection.to_xy(lon, lat)
    track: Track = Track(lon, lat, x, y, t)
    if model == "hmm":
        from hmm.match import match
        p: dict = json.loads((Path(results) / "hmm" / "parameters.json").read_text())
        route, counts = match(g, track, p["sigma"], p["beta"])
    elif model == "learned":
        route, counts = importlib.import_module(LEARNED_MODULE).match(g, track)
    else:
        route, counts = importlib.import_module(f"{model}.match").match(g, track)
    return result(g, route, counts, truth)


def result(g: RoadGraph, route: Route, counts: dict,
           truth: list[int] | None) -> dict:
    """A finished route as the API returns it."""
    out: dict = {"status": "done",
                 "route": route.arcs.tolist(),
                 "length_m": round(route.length(g), 1),
                 "breaks": route.breaks(g).tolist() if len(route.arcs) else [],
                 "stats": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                           for k, v in counts.items()},
                 "geometry": geometry(g, route.arcs.tolist())}
    if truth is not None:
        s = score_route(g, route, Route(np.array(truth, dtype=np.int64)))
        out["score"] = {"correct_fraction": round(s.correct_fraction(), 4),
                        "overlap_m": round(s.overlap, 1),
                        "predicted_length_m": round(s.predicted_length, 1),
                        "true_length_m": round(s.true_length, 1)}
    return out


def geometry(g: RoadGraph, arcs: list[int]) -> dict:
    """GeoJSON MultiLineString, [lon, lat]; a new part wherever an arc
    does not start where the one before it ends."""
    parts: list[list[list[float]]] = []
    previous_end: int = -1
    for arc in arcs:
        start, end = int(g.arc_from[arc]), int(g.arc_to[arc])
        if start != previous_end:
            parts.append([[round(float(g.node_lon[start]), 6),
                           round(float(g.node_lat[start]), 6)]])
        parts[-1].append([round(float(g.node_lon[end]), 6),
                          round(float(g.node_lat[end]), 6)])
        previous_end = end
    return {"type": "MultiLineString", "coordinates": parts}
