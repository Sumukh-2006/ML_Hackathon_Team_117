"""The nearest road segments (arcs) of a key point, and the distances from
arcs to a stretch of fixes; both on the plane in metres."""

import numpy as np

from dataset import RoadGraph


class ArcGrid:
    """Arcs bucketed by the square cells their bounding boxes touch."""

    def __init__(self, graph: RoadGraph, cell: float) -> None:
        self.graph: RoadGraph = graph
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


def arc_distance(graph: RoadGraph, arcs: np.ndarray, x: float,
                 y: float) -> np.ndarray:
    """Distance from one point to each arc's segment."""
    ax, ay = graph.node_x[graph.arc_from[arcs]], graph.node_y[graph.arc_from[arcs]]
    bx, by = graph.node_x[graph.arc_to[arcs]], graph.node_y[graph.arc_to[arcs]]
    dx, dy = bx - ax, by - ay
    squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)
    u: np.ndarray = np.clip(((x - ax) * dx + (y - ay) * dy) / squared, 0, 1)
    return np.hypot(x - ax - u * dx, y - ay - u * dy)


def nearest_arcs(grid: ArcGrid, x: float, y: float, count: int,
                 radius: float) -> tuple[np.ndarray, np.ndarray]:
    """The `count` arcs nearest the point and their distances. The search
    radius doubles until `count` arcs lie within it, so the answer is exact."""
    while True:
        arcs: np.ndarray = grid.near(x, y, radius)
        distance: np.ndarray = arc_distance(grid.graph, arcs, x, y)
        inside: np.ndarray = distance <= radius
        if inside.sum() >= count or len(arcs) == grid.graph.arc_count():
            order: np.ndarray = np.argsort(distance)[:count]
            return arcs[order], distance[order]
        radius *= 2


def stretch_distance(grid: ArcGrid, xs: np.ndarray, ys: np.ndarray,
                     cap: float) -> dict[int, float]:
    """Section 4.1.1: dist(e, s) = min over the stretch's fixes of the
    distance to arc e, for every arc closer than `cap`; farther arcs are
    left out, and their trajectory cost is the cap itself (Eq. 2)."""
    nearest: dict[int, float] = {}
    for x, y in zip(xs.tolist(), ys.tolist()):
        arcs: np.ndarray = grid.near(x, y, cap)
        distance: np.ndarray = arc_distance(grid.graph, arcs, x, y)
        close: np.ndarray = distance < cap
        for arc, d in zip(arcs[close].tolist(), distance[close].tolist()):
            if d < nearest.get(arc, cap):
                nearest[arc] = d
    return nearest
