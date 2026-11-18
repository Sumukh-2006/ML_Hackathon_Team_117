"""Matching one track with Yin et al.'s feature-based method."""

import numpy as np

from dataset import RoadGraph, Route, Track, great_circle
from segment.action_graph import ActionGraph
from segment.candidates import ArcGrid, nearest_arcs, stretch_distance
from segment.hyperparameters import (CANDIDATES, FIRST_RADIUS, GENTLE_TURN,
                                     GRID_CELL, KEY_TOLERANCE,
                                     MAX_TRAJECTORY_COST, OMEGA, SHARP_TURN,
                                     SIGMA)
from segment.simplify import key_points
from segment.viterbi import Stretch, decode


def match(graph: RoadGraph, track: Track) -> tuple[Route, dict]:
    """The route driven, and counts describing the run."""
    keys: list[int] = key_points(track, KEY_TOLERANCE)
    grid: ArcGrid = ArcGrid(graph, GRID_CELL)
    actions: ActionGraph = ActionGraph(graph, OMEGA, GENTLE_TURN, SHARP_TURN)

    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for key in keys:
        candidates.append(nearest_arcs(grid, float(track.x[key]),
                                       float(track.y[key]), CANDIDATES,
                                       FIRST_RADIUS))
    stretches: list[Stretch] = between_keys(track, keys, grid)

    trips: list[tuple[int, list[int]]] = decode(
        candidates, stretches, track.fix_count(), actions, SIGMA,
        MAX_TRAJECTORY_COST)
    route: Route = assemble(trips, stretches, actions)
    counts: dict = {"fixes": track.fix_count(), "keys": len(keys),
                    "trips": len(trips)}
    return route, counts


def between_keys(track: Track, keys: list[int],
                 grid: ArcGrid) -> list[Stretch]:
    """For each pair of consecutive key points: its fixes (both key points
    included), their distance, and dist(e, s) of the arcs near it."""
    stretches: list[Stretch] = []
    for i in range(len(keys) - 1):
        first, last = keys[i], keys[i + 1]
        distance: float = float(great_circle(track.lon[first],
                                             track.lat[first],
                                             track.lon[last],
                                             track.lat[last]))
        trajectory: dict[int, float] = stretch_distance(
            grid, track.x[first:last + 1], track.y[first:last + 1],
            MAX_TRAJECTORY_COST)
        stretches.append(Stretch(last - first + 1, distance, trajectory))
    return stretches


def assemble(trips: list[tuple[int, list[int]]], stretches: list[Stretch],
             actions: ActionGraph) -> Route:
    """The arcs of every trip: each chosen arc, joined to the next by the
    cheapest action sequence between them; trips one after another."""
    arcs: list[int] = []
    for first_key, chosen in trips:
        for step in range(len(chosen)):
            if step == 0:
                append_new(arcs, [chosen[0]])
                continue
            stretch: Stretch = stretches[first_key + step - 1]
            _, entry = actions.cheapest(chosen[step - 1], [chosen[step]],
                                        stretch.trajectory,
                                        MAX_TRAJECTORY_COST)
            append_new(arcs, action_path(chosen[step - 1], chosen[step],
                                         entry))
    return Route(np.array(arcs, dtype=np.int64))


def action_path(source: int, target: int, entry: dict[int, int]) -> list[int]:
    """The arcs from `source` to `target` along the search's entries."""
    path: list[int] = [target]
    while path[-1] != source:
        path.append(entry[path[-1]])
    path.reverse()
    return path


def append_new(arcs: list[int], path: list[int]) -> None:
    """Append, skipping an arc that repeats the one just before it."""
    for arc in path:
        if len(arcs) == 0 or arcs[-1] != arc:
            arcs.append(arc)
