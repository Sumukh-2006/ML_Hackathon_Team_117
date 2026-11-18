"""Matching one track with the MDP: simplify, search and adopt candidates,
solve, and drive the chosen candidates' routes."""

import numpy as np

from dataset import RoadGraph, Route, Track
from mdp.candidates import Candidates, adopt, search
from mdp.clean import median_merge, simplified
from mdp.geometry import TrackShape
from mdp.hyperparameters import (GRID_CELL, K_NEAREST, MEDIAN_MERGE_ROAD_SHARE,
                                 MEDIAN_MERGE_TOLERANCE, RADIUS, RADIUS_UPPER,
                                 ROUTING_MAX_DISTANCE_FACTOR,
                                 SIMPLIFY_TOLERANCE, WARP_SPEED)
from mdp.network import ArcGrid, road_edges
from mdp.routing import Router
from mdp.value_iteration import Policy, solve


def match(graph: RoadGraph, track: Track) -> tuple[Route, dict]:
    """The route driven, and counts describing the run."""
    grid: ArcGrid = ArcGrid(graph, GRID_CELL)
    shape: TrackShape = cleaned_shape(track, grid)
    edges: np.ndarray = road_edges(graph)

    own: list[Candidates] = []
    for i in range(shape.point_count()):
        own.append(search(grid, edges, float(shape.x[i]), float(shape.y[i]),
                          K_NEAREST, RADIUS, RADIUS_UPPER))
    sets: list[Candidates] = adopt(own, shape.x, shape.y, GRID_CELL)

    router: Router = Router(graph)
    policy: Policy = solve(sets, shape, router)
    route: Route = drive(policy, sets, router)
    counts: dict = {"fixes": track.fix_count(), "points": shape.point_count(),
                    "candidates": sum(s.count() for s in sets),
                    "skips": sum(policy.skipped)}
    return route, counts


def cleaned_shape(track: Track, grid: ArcGrid) -> TrackShape:
    """Trajectory simplification; median merge's radius grows to half the
    distance to the nearest road where that is larger (adaptive).
    OWN CHOICE: a point with no road within RADIUS_UPPER is dropped."""
    x, y = simplified(track, WARP_SPEED, SIMPLIFY_TOLERANCE)
    tolerance: np.ndarray = np.empty(len(x))
    reachable: np.ndarray = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        road: float = grid.nearest_road(float(x[i]), float(y[i]), RADIUS_UPPER)
        reachable[i] = road < RADIUS_UPPER
        tolerance[i] = max(MEDIAN_MERGE_TOLERANCE,
                           MEDIAN_MERGE_ROAD_SHARE * road)
    x, y, tolerance = x[reachable], y[reachable], tolerance[reachable]
    x, y = median_merge(x, y, tolerance)
    return TrackShape.from_points(x, y)


def drive(policy: Policy, sets: list[Candidates], router: Router) -> Route:
    """The arcs from candidate to candidate; a skipped step drives nothing,
    as map_matching_2 does by default (concatenate_skips off)."""
    first: int = policy.chosen[0]
    arcs: list[int] = [int(sets[0].arc[first])]
    for j in range(len(policy.skipped)):
        c: int = policy.chosen[j]
        c_next: int = policy.chosen[j + 1]
        target: int = int(sets[j + 1].arc[c_next])
        if policy.skipped[j]:
            driven: list[int] = [target]
        else:
            arc: int = int(sets[j].arc[c])
            bound: float = router.bound(arc, sets[j + 1].arc,
                                        ROUTING_MAX_DISTANCE_FACTOR)
            driven = router.path(arc, float(sets[j].along[c]), target,
                                 float(sets[j + 1].along[c_next]), bound)
        for each in driven:
            if arcs[-1] != each:
                arcs.append(each)
    return Route(np.array(arcs, dtype=np.int64))
