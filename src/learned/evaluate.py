"""Matching with learned weights, and scoring held-out folders.

A lattice holds the raw step features (float32, unscaled), so matching
needs no GPU: the reference decoder in decode.py runs per track.
"""

from pathlib import Path

import numpy as np

from benchmark.metrics import RouteScore, score_route
from dataset import Folder, Route
from learned.decode import Labelling, decode, drive
from learned.lattice import Lattice, load
from learned.network import Network
from learned.routes import Router


def labelling(lattice: Lattice, weights: np.ndarray) -> Labelling:
    """The least-cost labelling under `weights` (step features, then skip,
    first distance, restart distance)."""
    costs: list[np.ndarray] = []
    for j in range(lattice.step_count()):
        step: np.ndarray = lattice.features[j].astype(np.float64) @ weights[:-3]
        costs.append(np.where(lattice.skips[j], weights[-3], step))
    fresh: list[np.ndarray] = [weights[-1] * p.distance for p in lattice.points]
    fresh[0] = weights[-2] * lattice.points[0].distance
    return decode(costs, lattice.skips, fresh, 1.0)


def match(folder: Folder, lattice_path: Path, weights: np.ndarray) -> Route:
    graph = folder.graph()
    lattice: Lattice = load(lattice_path)
    router: Router = Router(Network.from_graph(graph))
    return drive(labelling(lattice, weights), lattice.points, router)


def score(job: tuple) -> dict:
    """(folder, lattice path, weights) -> the benchmark's row values."""
    folder, path, weights = job
    graph = folder.graph()
    route: Route = match(folder, path, weights)
    result: RouteScore = score_route(graph, route, folder.route(graph))
    return {"id": folder.name(), "overlap": result.overlap,
            "predicted_length": result.predicted_length,
            "true_length": result.true_length,
            "correct_fraction": result.correct_fraction()}


def weighted_correct_fraction(rows: list[dict]) -> float:
    overlap: float = sum(r["overlap"] for r in rows)
    longer: float = sum(max(r["predicted_length"], r["true_length"]) for r in rows)
    return overlap / longer
