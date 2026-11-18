"""Matching one track: thin the fixes, build candidates, decode, and turn
the decoded candidate path into the arcs driven."""

import numpy as np

from dataset import RoadGraph, Route, Track, great_circle
from hmm.candidates import Candidates, GridIndex, candidates
from hmm.hyperparameters import (GRID_CELL, MAX_BREAK, MAX_DETOUR, MAX_SPEED,
                                 PREPROCESS_SIGMAS, SEARCH_RADIUS)
from hmm.routing import Router
from hmm.segments import arc_segments
from hmm.viterbi import Model, decode


def match(graph: RoadGraph, track: Track, sigma: float,
          beta: float) -> tuple[Route, dict]:
    """The route driven, and counts describing the run."""
    fixes: list[int] = thin(track, PREPROCESS_SIGMAS * sigma)
    segment: np.ndarray = arc_segments(graph)
    index: GridIndex = GridIndex(graph, GRID_CELL)
    router: Router = Router(graph)

    near: list[Candidates] = []
    for fix in fixes:
        near.append(candidates(index, segment, float(track.lon[fix]),
                               float(track.lat[fix]), float(track.x[fix]),
                               float(track.y[fix]), SEARCH_RADIUS))

    model: Model = Model(sigma, beta, MAX_DETOUR, MAX_SPEED, MAX_BREAK)
    trips: list[list[tuple[int, float]]] = decode(fixes, near, track, router,
                                                  model)
    route: Route = assemble(trips, router)
    counts: dict = {"fixes": track.fix_count(), "kept": len(fixes),
                    "trips": len(trips),
                    "matched": sum(len(trip) for trip in trips)}
    return route, counts


def thin(track: Track, distance: float) -> list[int]:
    """Section 4.1: drop every fix within `distance` of the last kept one."""
    kept: list[int] = [0]
    for fix in range(1, track.fix_count()):
        last: int = kept[-1]
        apart: float = float(great_circle(track.lon[last], track.lat[last],
                                          track.lon[fix], track.lat[fix]))
        if apart > distance:
            kept.append(fix)
    return kept


def assemble(trips: list[list[tuple[int, float]]], router: Router) -> Route:
    """The arcs driven along every trip, trips one after another. An arc
    repeated back to back is one arc: consecutive candidates on it."""
    arcs: list[int] = []
    for trip in trips:
        for step in range(len(trip)):
            arc, along = trip[step]
            if step == 0:
                driven: list[int] = [arc]
            else:
                previous_arc, previous_along = trip[step - 1]
                driven = router.path(previous_arc, previous_along, arc, along)
            for each in driven:
                if len(arcs) == 0 or arcs[-1] != each:
                    arcs.append(each)
    return Route(np.array(arcs, dtype=np.int64))
