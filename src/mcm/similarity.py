"""Point-to-road similarity (Definition 5): EPSILON minus the distance from
a fix to an arc's segment when that distance is below EPSILON, else 0.

The distance is to the segment, as Definition 5 and Fig. 2 state: the
perpendicular foot when it falls on the segment, else the nearer end.
"""

import numpy as np

from dataset import RoadGraph, Track


class ArcGrid:
    """Arcs bucketed by the square cells their bounding boxes touch."""

    def __init__(self, graph: RoadGraph, cell: float) -> None:
        self.cell: float = cell
        ax, bx = graph.node_x[graph.arc_from], graph.node_x[graph.arc_to]
        ay, by = graph.node_y[graph.arc_from], graph.node_y[graph.arc_to]
        x0 = np.floor(np.minimum(ax, bx) / cell).astype(np.int64)
        x1 = np.floor(np.maximum(ax, bx) / cell).astype(np.int64)
        y0 = np.floor(np.minimum(ay, by) / cell).astype(np.int64)
        y1 = np.floor(np.maximum(ay, by) / cell).astype(np.int64)

        buckets: dict[tuple[int, int], list[int]] = {}
        for arc in range(graph.arc_count()):
            for i in range(x0[arc], x1[arc] + 1):
                for j in range(y0[arc], y1[arc] + 1):
                    buckets.setdefault((i, j), []).append(arc)
        self.buckets: dict[tuple[int, int], np.ndarray] = {}
        for key, arcs in buckets.items():
            self.buckets[key] = np.array(arcs, dtype=np.int64)

    def near(self, x: float, y: float, radius: float) -> np.ndarray:
        """Every arc whose bounding box may come within `radius`."""
        found: list[np.ndarray] = []
        for i in range(int(np.floor((x - radius) / self.cell)),
                       int(np.floor((x + radius) / self.cell)) + 1):
            for j in range(int(np.floor((y - radius) / self.cell)),
                           int(np.floor((y + radius) / self.cell)) + 1):
                if (i, j) in self.buckets:
                    found.append(self.buckets[(i, j)])
        if len(found) == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(found))


def similarities(graph: RoadGraph, track: Track, grid: ArcGrid,
                 epsilon: float) -> list[dict[int, float]]:
    """For every fix, the arcs with similarity above zero and its value."""
    result: list[dict[int, float]] = []
    for fix in range(track.fix_count()):
        x: float = float(track.x[fix])
        y: float = float(track.y[fix])
        arcs: np.ndarray = grid.near(x, y, epsilon)
        distance: np.ndarray = segment_distance(graph, arcs, x, y)
        close: np.ndarray = distance < epsilon

        values: dict[int, float] = {}
        for arc, d in zip(arcs[close].tolist(), distance[close].tolist()):
            values[arc] = epsilon - d
        result.append(values)
    return result


def segment_distance(graph: RoadGraph, arcs: np.ndarray, x: float,
                     y: float) -> np.ndarray:
    ax, ay = graph.node_x[graph.arc_from[arcs]], graph.node_y[graph.arc_from[arcs]]
    bx, by = graph.node_x[graph.arc_to[arcs]], graph.node_y[graph.arc_to[arcs]]
    dx, dy = bx - ax, by - ay
    squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)
    u: np.ndarray = np.clip(((x - ax) * dx + (y - ay) * dy) / squared, 0, 1)
    return np.hypot(x - ax - u * dx, y - ay - u * dy)
