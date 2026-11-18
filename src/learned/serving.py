"""Approach 5 for the model server (src/serve): match one track with no
ground truth, using the frozen prototype in results/learned (parameters.json
from the measure stage, weights.json from `python -m learned freeze` or
`train`). The same steps as evaluate.match, with the lattice built in
memory instead of loaded from results/learned/lattices.
"""

import json
from pathlib import Path

import numpy as np

from dataset import RoadGraph, Route, Track
from learned.decode import Labelling, drive
from learned.evaluate import labelling
from learned.lattice import Builder, Lattice
from learned.network import Network
from learned.routes import Router

RESULTS: Path = Path(__file__).resolve().parents[2] / "results" / "learned"
PARAMETERS: dict = json.loads((RESULTS / "parameters.json").read_text())
WEIGHTS: np.ndarray = np.array(list(
    json.loads((RESULTS / "weights.json").read_text()).values()))


def match(graph: RoadGraph, track: Track) -> tuple[Route, dict]:
    built: Lattice = Builder(graph).lattice(track, PARAMETERS, None)
    chosen: Labelling = labelling(built, WEIGHTS)
    route: Route = drive(chosen, built.points, Router(Network.from_graph(graph)))
    return route, {"fixes": track.fix_count(), "points": len(built.points),
                   "skips": int(np.sum(chosen.skipped))}
