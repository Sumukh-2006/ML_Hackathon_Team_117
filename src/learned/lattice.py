"""Layers 3 and 4 assembled: the lattice of one track, built once.

A lattice is the decision points' candidates and, for every step, the
feature array, the skip mask and (for a labelled track) the step errors.
Everything training and matching need is in it, so neither ever searches
the road network again except to write out a chosen route's arcs.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset import RoadGraph, Route, Track
from learned.candidates import Candidates, adopt, search
from learned.features import (skip_errors, start_errors, step_features)
from learned.hyperparameters import (COVERAGE_FIXES, GRID_CELL, K_NEAREST, RADIUS,
                                     RADIUS_UPPER, ROUTING_MAX_DISTANCE_FACTOR,
                                     SKIP_ROUTE_FACTOR, MEDIAN_MERGE_TOLERANCE,
                                     MEDIAN_MERGE_ROAD_SHARE, WARP_SPEED)
from learned.kernel import step_lattice
from learned.network import ArcGrid, Network
from learned.routes import Router
from learned.track import DecisionTrack, decision_track
from learned.truth import pieces, placement


@dataclass
class Lattice:
    points: list[Candidates]
    features: list[np.ndarray]     # per step: (K, K', 11) float32
    skips: list[np.ndarray]        # per step: (K, K') bool
    errors: list[np.ndarray]       # per step: (K, K') float32, training only
    skip_errors: list[np.ndarray]  # per step: (K',) float32, training only
    start_errors: np.ndarray       # (K_0,) float32, training only
    true_length: float             # metres, training only

    def step_count(self) -> int:
        return len(self.features)


class Builder:
    """Everything that is built once per road graph."""

    def __init__(self, graph: RoadGraph) -> None:
        self.graph: RoadGraph = graph
        self.network: Network = Network.from_graph(graph)
        self.grid: ArcGrid = ArcGrid(self.network, GRID_CELL)
        self.router: Router = Router(self.network)
        nodes: int = self.network.node_count()
        self.tally: np.ndarray = np.zeros(graph.arc_count(), dtype=np.int64)
        self.path: np.ndarray = np.zeros(nodes + 2, dtype=np.int64)
        self.vx: np.ndarray = np.zeros(nodes + 3)
        self.vy: np.ndarray = np.zeros(nodes + 3)

    def decision_track(self, track: Track, parameters: dict) -> DecisionTrack:
        return decision_track(track, self.grid, parameters["tolerance"],
                              parameters["half_window"], WARP_SPEED,
                              MEDIAN_MERGE_TOLERANCE, MEDIAN_MERGE_ROAD_SHARE,
                              RADIUS_UPPER)

    def candidates(self, decided: DecisionTrack) -> list[Candidates]:
        own: list[Candidates] = []
        for i in range(decided.point_count()):
            found: Candidates = search(self.grid, float(decided.x[i]),
                                       float(decided.y[i]), K_NEAREST, RADIUS,
                                       RADIUS_UPPER)
            if found.count() == 0:
                raise ValueError(f"decision point {i} has no road within "
                                 f"{RADIUS_UPPER} m")
            own.append(found)
        return adopt(own, decided.x, decided.y, GRID_CELL)

    def lattice(self, track: Track, parameters: dict,
                truth: Route | None) -> Lattice:
        decided: DecisionTrack = self.decision_track(track, parameters)
        points: list[Candidates] = self.candidates(decided)
        graph, net = self.graph, self.network
        cut: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * decided.point_count()
        if truth is not None:
            cut = pieces(truth, graph, placement(graph, truth, decided, points))

        lattice = Lattice(points, [], [], [], [], np.zeros(0, np.float32), 0.0)
        if truth is not None:
            lattice.start_errors = start_errors(points[0], cut[0],
                                                graph.canonical_arc,
                                                graph.arc_length).astype(np.float32)
            lattice.true_length = truth.length(graph)
        for j in range(decided.point_count() - 1):
            self.add_step(lattice, decided, j, cut, truth is not None,
                          parameters)
        return lattice

    def add_step(self, lattice: Lattice, decided: DecisionTrack, j: int,
                 cut: list[np.ndarray], with_truth: bool,
                 parameters: dict) -> None:
        a, b = lattice.points[j], lattice.points[j + 1]
        fix_x, fix_y = decided.step_fixes(j)
        if len(fix_x) > COVERAGE_FIXES:
            sample: np.ndarray = np.linspace(0, len(fix_x) - 1,
                                             COVERAGE_FIXES).round().astype(np.int64)
            fix_x, fix_y = fix_x[sample], fix_y[sample]
        piece: np.ndarray = cut[j + 1].astype(np.int64)
        net, router = self.network, self.router
        lengths, changes, sharp, cover_fix, cover_route, delta = step_lattice(
            a.arc, a.along, a.px, a.py, b.arc, b.along, b.px, b.py,
            ROUTING_MAX_DISTANCE_FACTOR, fix_x, fix_y, piece, with_truth,
            self.graph.canonical_arc, net.arc_from, net.arc_to, net.length,
            net.heading, net.moves, net.node_x, net.node_y, net.out_start,
            net.out_arc, router.distance, router.entry, router.turning,
            router.sharpest, router.touched, self.tally, self.path, self.vx,
            self.vy)
        features, skip = step_features(j, decided, a, b, lengths, changes,
                                       sharp, cover_fix, cover_route,
                                       parameters["too_fast_speed"],
                                       SKIP_ROUTE_FACTOR)
        lattice.features.append(features.astype(np.float32))
        lattice.skips.append(skip)
        if with_truth:
            lattice.errors.append(delta.astype(np.float32))
            lattice.skip_errors.append(skip_errors(
                b, piece, self.graph.canonical_arc,
                self.graph.arc_length).astype(np.float32))


def save(lattice: Lattice, path: Path) -> None:
    """One uncompressed file (compression costs more time than the disk
    space it saves); ragged per-point and per-step arrays are stored
    flat with the sizes needed to split them again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes: np.ndarray = np.array([p.count() for p in lattice.points])
    flat_points: dict[str, np.ndarray] = {}
    for name in ["arc", "along", "px", "py", "distance"]:
        flat_points[name] = np.concatenate([getattr(p, name) for p in lattice.points])
    arrays: dict[str, np.ndarray] = {
        "sizes": sizes, "start_errors": lattice.start_errors,
        "true_length": np.array([lattice.true_length])}
    for name, values in flat_points.items():
        arrays["point_" + name] = values
    if lattice.step_count() > 0:
        arrays["features"] = np.concatenate([f.reshape(-1, f.shape[-1]) for f in lattice.features])
        arrays["skips"] = np.concatenate([s.ravel() for s in lattice.skips])
    if len(lattice.errors) > 0:
        arrays["errors"] = np.concatenate([e.ravel() for e in lattice.errors])
        arrays["skip_errors"] = np.concatenate(lattice.skip_errors)
    np.savez(path, **arrays)


def load(path: Path) -> Lattice:
    """The lattice saved by `save`. Every array is read from the file once
    (indexing an NpzFile reads the whole array again each time)."""
    with np.load(path) as data:
        arrays: dict[str, np.ndarray] = {name: data[name] for name in data.files}
    sizes: np.ndarray = arrays["sizes"]
    edges: np.ndarray = np.concatenate([[0], np.cumsum(sizes)])
    points: list[Candidates] = []
    for i in range(len(sizes)):
        span: slice = slice(edges[i], edges[i + 1])
        points.append(Candidates(arrays["point_arc"][span], arrays["point_along"][span],
                                 arrays["point_px"][span], arrays["point_py"][span],
                                 arrays["point_distance"][span],
                                 np.zeros(sizes[i], dtype=bool)))
    lattice = Lattice(points, [], [], [], [], arrays["start_errors"],
                      float(arrays["true_length"][0]))
    offset: int = 0
    for j in range(len(sizes) - 1):
        pairs: int = int(sizes[j] * sizes[j + 1])
        shape: tuple[int, int] = (int(sizes[j]), int(sizes[j + 1]))
        span = slice(offset, offset + pairs)
        lattice.features.append(arrays["features"][span].reshape(shape + (-1,)))
        lattice.skips.append(arrays["skips"][span].reshape(shape))
        if "errors" in arrays:
            lattice.errors.append(arrays["errors"][span].reshape(shape))
        offset += pairs
    if "skip_errors" in arrays:
        lattice.skip_errors = np.split(arrays["skip_errors"], np.cumsum(sizes[1:])[:-1])
    return lattice
