"""Matching one track with MCM and turning its pairs into arcs driven.

OWN CHOICE: the paper does not say how separately matched stretches are
joined. Consecutive pairs are joined by the driving path between their arcs
when it is within EPSILON_TOPO (always the case inside one route);
otherwise the route simply continues from the next arc, leaving a gap.
"""

import numpy as np

from dataset import RoadGraph, Route, Track
from mcm.hyperparameters import EPSILON, EPSILON_TOPO, GRID_CELL
from mcm.routing import Predecessors
from mcm.score_matrix import offline
from mcm.similarity import ArcGrid, similarities


def match(graph: RoadGraph, track: Track) -> tuple[Route, dict]:
    """The route driven, and counts describing the run."""
    grid: ArcGrid = ArcGrid(graph, GRID_CELL)
    similarity: list[dict[int, float]] = similarities(graph, track, grid,
                                                      EPSILON)
    predecessors: Predecessors = Predecessors(graph, EPSILON_TOPO)

    pairs: list[tuple[int, int]] = offline(similarity, predecessors)
    route, gaps = assemble(pairs, predecessors)
    counts: dict = {"fixes": track.fix_count(), "matched": len(pairs),
                    "gaps": gaps}
    return route, counts


def assemble(pairs: list[tuple[int, int]],
             predecessors: Predecessors) -> tuple[Route, int]:
    """The arcs of the matched pairs in fix order, joined by their driving
    paths; also how many joins had no path within the bound."""
    arcs: list[int] = []
    gaps: int = 0
    for _, arc in pairs:
        if len(arcs) == 0:
            arcs.append(arc)
            continue
        driven: list[int] | None = predecessors.path(arcs[-1], arc)
        if driven is None:
            gaps += 1
            driven = [arcs[-1], arc]
        for each in driven[1:]:
            if arcs[-1] != each:
                arcs.append(each)
    return Route(np.array(arcs, dtype=np.int64)), gaps
