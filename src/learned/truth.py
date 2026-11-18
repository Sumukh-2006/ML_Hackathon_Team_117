"""The true route cut into pieces, one per step (derivation §4.1).

Each decision point is placed on an arc of the true route so that the
placements never move backwards along the route and their summed distance
to the points is least. A point is placed only on true arcs that are among
its own candidates' arcs, unless none is: a step can then err by zero only
when a route from a candidate on the placed arc exists, and placing a point
on an arc it has no candidate on would rule that out for both of its steps. With best[j][i] the least cost of placing points
0..j with point j on true arc i,
    best[j][i] = distance(j, i) + min over i' <= i of best[j - 1][i'],
a running minimum, so the program is exact and always has a solution.
Theorem 2 holds for any monotone placement; this one only makes the bound
as tight as the geometry allows.
"""

import numpy as np

from dataset import RoadGraph, Route
from learned.candidates import Candidates
from learned.track import DecisionTrack

# Added to the distance of a true arc the point has no candidate on, so it
# is used only where the point has no candidate on the true route at all.
NO_CANDIDATE: float = 1e7


def placement(graph: RoadGraph, truth: Route, track: DecisionTrack,
              points: list[Candidates]) -> np.ndarray:
    """For every decision point, the index into the true route of the arc
    it is placed on; non-decreasing along the track."""
    a, b = graph.arc_from[truth.arcs], graph.arc_to[truth.arcs]
    ax, ay = graph.node_x[a], graph.node_y[a]
    dx, dy = graph.node_x[b] - ax, graph.node_y[b] - ay
    squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)

    best: np.ndarray = np.zeros(len(truth.arcs))
    origins: list[np.ndarray] = []
    for j in range(track.point_count()):
        u = np.clip(((track.x[j] - ax) * dx + (track.y[j] - ay) * dy) / squared, 0, 1)
        distance = np.hypot(track.x[j] - ax - u * dx, track.y[j] - ay - u * dy)
        distance = distance + np.where(np.isin(truth.arcs, points[j].arc), 0.0,
                                       NO_CANDIDATE)
        if j == 0:
            best = distance
            origins.append(np.arange(len(best)))
            continue
        prefix_best, prefix_origin = running_minimum(best)
        best = distance + prefix_best
        origins.append(prefix_origin)

    chosen: np.ndarray = np.zeros(track.point_count(), dtype=np.int64)
    i: int = int(np.argmin(best))
    for j in range(track.point_count() - 1, -1, -1):
        chosen[j] = i
        i = int(origins[j][i])
    return chosen


def running_minimum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every i: the least of values[0..i] and the index where it is."""
    least: np.ndarray = np.minimum.accumulate(values)
    index: np.ndarray = np.zeros(len(values), dtype=np.int64)
    for i in range(1, len(values)):
        if values[i] < least[i - 1]:
            index[i] = i
        else:
            index[i] = index[i - 1]
    return least, index


def pieces(truth: Route, graph: RoadGraph, placed: np.ndarray) -> list[np.ndarray]:
    """The canonical true arcs of the start piece and of every step's piece:
    start = arcs up to the first point's arc; step j = arcs after point j's
    arc up to point j + 1's; the last step's piece runs to the route's end.
    With a single point the start piece is the whole route."""
    roads: np.ndarray = graph.canonical_arc[truth.arcs]
    n: int = len(placed)
    if n == 1:
        return [roads]
    result: list[np.ndarray] = [roads[:placed[0] + 1]]
    for j in range(n - 1):
        start: int = int(placed[j]) + 1
        stop: int = len(roads) if j == n - 2 else int(placed[j + 1]) + 1
        result.append(roads[start:stop])
    return result
