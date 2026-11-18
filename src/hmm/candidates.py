"""Match candidates of a fix: the nearest point of each road segment that
lies within the search radius (Newson & Krumm, section 3.1)."""

from dataclasses import dataclass

import numpy as np

from dataset import RoadGraph, great_circle


@dataclass(frozen=True)
class Candidates:
    """One candidate per road segment near a fix."""
    arc: np.ndarray       # the arc holding the candidate point
    along: np.ndarray     # metres from that arc's start to the point
    distance: np.ndarray  # great-circle metres from the fix to the point

    def count(self) -> int:
        return len(self.arc)


class GridIndex:
    """Arcs bucketed by the square cells their bounding boxes touch."""

    def __init__(self, graph: RoadGraph, cell: float) -> None:
        self.graph: RoadGraph = graph
        self.cell: float = cell
        buckets: dict[tuple[int, int], list[int]] = {}
        x0, x1, y0, y1 = arc_boxes(graph, cell)
        for arc in range(graph.arc_count()):
            for i in range(x0[arc], x1[arc] + 1):
                for j in range(y0[arc], y1[arc] + 1):
                    buckets.setdefault((i, j), []).append(arc)
        self.buckets: dict[tuple[int, int], np.ndarray] = {}
        for key, arcs in buckets.items():
            self.buckets[key] = np.array(arcs, dtype=np.int64)

    def arcs_near(self, x: float, y: float, radius: float) -> np.ndarray:
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


def arc_boxes(graph: RoadGraph, cell: float) -> tuple[np.ndarray, ...]:
    """The first and last cell column and row of every arc's box."""
    ax, bx = graph.node_x[graph.arc_from], graph.node_x[graph.arc_to]
    ay, by = graph.node_y[graph.arc_from], graph.node_y[graph.arc_to]
    x0 = np.floor(np.minimum(ax, bx) / cell).astype(np.int64)
    x1 = np.floor(np.maximum(ax, bx) / cell).astype(np.int64)
    y0 = np.floor(np.minimum(ay, by) / cell).astype(np.int64)
    y1 = np.floor(np.maximum(ay, by) / cell).astype(np.int64)
    return x0, x1, y0, y1


def candidates(index: GridIndex, segment: np.ndarray, lon: float, lat: float,
               x: float, y: float, radius: float) -> Candidates:
    """The nearest point of every road segment within `radius` of a fix."""
    graph: RoadGraph = index.graph
    arcs: np.ndarray = index.arcs_near(x, y, radius)
    a, b = graph.arc_from[arcs], graph.arc_to[arcs]

    dx: np.ndarray = graph.node_x[b] - graph.node_x[a]
    dy: np.ndarray = graph.node_y[b] - graph.node_y[a]
    squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)
    u: np.ndarray = np.clip(((x - graph.node_x[a]) * dx
                             + (y - graph.node_y[a]) * dy) / squared, 0, 1)

    point_lon = graph.node_lon[a] + u * (graph.node_lon[b] - graph.node_lon[a])
    point_lat = graph.node_lat[a] + u * (graph.node_lat[b] - graph.node_lat[a])
    distance: np.ndarray = great_circle(np.full(len(arcs), lon),
                                        np.full(len(arcs), lat),
                                        point_lon, point_lat)

    inside: np.ndarray = distance <= radius
    arcs, u, distance = arcs[inside], u[inside], distance[inside]
    order: np.ndarray = np.lexsort((distance, segment[arcs]))
    first: np.ndarray = np.ones(len(order), dtype=bool)
    first[1:] = segment[arcs[order[1:]]] != segment[arcs[order[:-1]]]
    best: np.ndarray = order[first]
    return Candidates(arcs[best], u[best] * graph.arc_length[arcs[best]],
                      distance[best])
