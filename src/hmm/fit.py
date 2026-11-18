"""Measuring sigma and beta from the training routes (paper section 5.2).

sigma = 1.4826 * median over fixes of the great-circle distance from a fix
to its true route. beta = median over consecutive thinned fixes of
|great circle - route distance along the true route| / ln 2, the median of
an exponential distribution being beta * ln 2.
"""

import numpy as np

from dataset import Corpus, RoadGraph, Route, Track, great_circle
from hmm.hyperparameters import MAD_TO_SIGMA, PREPROCESS_SIGMAS
from hmm.match import thin

# NOTE: route arcs are compared with fixes in blocks this large, which
# bounds memory; it does not change any value.
ARC_BLOCK: int = 500


def measure(corpus: Corpus, split: str) -> dict:
    """sigma and beta over every folder of `split`, with their sample sizes."""
    nearest: list[tuple[RoadGraph, Track, np.ndarray, np.ndarray]] = []
    distances: list[np.ndarray] = []
    for folder in corpus.folders(split):
        graph: RoadGraph = folder.graph()
        track: Track = folder.track(graph)
        distance, position = on_route(graph, track, folder.route(graph))
        nearest.append((graph, track, distance, position))
        distances.append(distance)
    sigma: float = MAD_TO_SIGMA * float(np.median(np.concatenate(distances)))

    differences: list[np.ndarray] = []
    for graph, track, distance, position in nearest:
        fixes: np.ndarray = np.array(thin(track, PREPROCESS_SIGMAS * sigma))
        straight: np.ndarray = great_circle(
            track.lon[fixes[:-1]], track.lat[fixes[:-1]],
            track.lon[fixes[1:]], track.lat[fixes[1:]])
        driven: np.ndarray = np.abs(np.diff(position[fixes]))
        differences.append(np.abs(straight - driven))
    difference: np.ndarray = np.concatenate(differences)
    beta: float = float(np.median(difference) / np.log(2))
    return {"sigma": sigma, "beta": beta, "split": split,
            "fixes": int(sum(len(d) for d in distances)),
            "transitions": int(len(difference))}


def on_route(graph: RoadGraph, track: Track,
             route: Route) -> tuple[np.ndarray, np.ndarray]:
    """For every fix: the great-circle distance to the nearest point of
    the route, and that point's position in metres along the route."""
    start: np.ndarray = np.concatenate(
        [[0.0], np.cumsum(graph.arc_length[route.arcs])[:-1]])
    best: np.ndarray = np.full(track.fix_count(), np.inf)
    distance: np.ndarray = np.zeros(track.fix_count())
    position: np.ndarray = np.zeros(track.fix_count())

    for first in range(0, len(route.arcs), ARC_BLOCK):
        arcs: np.ndarray = route.arcs[first:first + ARC_BLOCK]
        a, b = graph.arc_from[arcs], graph.arc_to[arcs]
        dx: np.ndarray = graph.node_x[b] - graph.node_x[a]
        dy: np.ndarray = graph.node_y[b] - graph.node_y[a]
        squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)
        u: np.ndarray = np.clip(((track.x[:, None] - graph.node_x[a]) * dx
                                 + (track.y[:, None] - graph.node_y[a]) * dy)
                                / squared, 0, 1)
        plane: np.ndarray = np.hypot(
            track.x[:, None] - graph.node_x[a] - u * dx,
            track.y[:, None] - graph.node_y[a] - u * dy)
        j: np.ndarray = plane.argmin(axis=1)
        rows: np.ndarray = np.arange(track.fix_count())
        closer: np.ndarray = plane[rows, j] < best
        best[closer] = plane[rows, j][closer]

        k: np.ndarray = first + j[closer]
        uk: np.ndarray = u[rows, j][closer]
        ak, bk = graph.arc_from[route.arcs[k]], graph.arc_to[route.arcs[k]]
        lon = graph.node_lon[ak] + uk * (graph.node_lon[bk] - graph.node_lon[ak])
        lat = graph.node_lat[ak] + uk * (graph.node_lat[bk] - graph.node_lat[ak])
        distance[closer] = great_circle(track.lon[closer], track.lat[closer],
                                        lon, lat)
        position[closer] = start[k] + uk * graph.arc_length[route.arcs[k]]
    return distance, position
