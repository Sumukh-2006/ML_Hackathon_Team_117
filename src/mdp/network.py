"""The road network as map_matching_2 sees it: edges are roads from junction
to junction (its "simplification" joins the arcs through inner nodes), and
a grid index finds the arcs near a point."""

import numpy as np

from dataset import RoadGraph


def road_edges(graph: RoadGraph) -> np.ndarray:
    """The road edge of every arc: a maximal chain of arcs whose inner nodes
    join exactly two neighbouring nodes. An arc that two arcs would continue
    into starts an edge of its own."""
    arc_from: list[int] = graph.arc_from.tolist()
    arc_to: list[int] = graph.arc_to.tolist()
    neighbours: list[set[int]] = [set() for _ in range(graph.node_count())]
    for arc in range(graph.arc_count()):
        if arc_from[arc] != arc_to[arc]:
            neighbours[arc_from[arc]].add(arc_to[arc])
            neighbours[arc_to[arc]].add(arc_from[arc])

    following: np.ndarray = np.full(graph.arc_count(), -1, dtype=np.int64)
    for arc in range(graph.arc_count()):
        node: int = arc_to[arc]
        if arc_from[arc] == node or len(neighbours[node]) != 2:
            continue
        onward: list[int] = [o for o in graph.out_arcs(node).tolist()
                             if arc_to[o] != arc_from[arc]]
        if len(onward) == 1:
            following[arc] = onward[0]
    entering: np.ndarray = np.bincount(following[following >= 0],
                                       minlength=graph.arc_count())
    following[(following >= 0) & (entering[following] > 1)] = -1
    return label_chains(following)


def label_chains(following: np.ndarray) -> np.ndarray:
    """Number every chain; chains start at arcs nothing continues into, and
    the arcs left over form closed loops."""
    edge: np.ndarray = np.full(len(following), -1, dtype=np.int64)
    has_previous: np.ndarray = np.zeros(len(following), dtype=bool)
    has_previous[following[following >= 0]] = True
    count: int = 0
    for first_pass in [True, False]:
        for arc in range(len(following)):
            if edge[arc] >= 0 or (first_pass and has_previous[arc]):
                continue
            current: int = arc
            while current >= 0 and edge[current] < 0:
                edge[current] = count
                current = int(following[current])
            count += 1
    return edge


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

    def projections(self, arcs: np.ndarray, x: float,
                    y: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For each arc: the share along it of the point's closest point,
        and that closest point's coordinates."""
        graph: RoadGraph = self.graph
        ax, ay = graph.node_x[graph.arc_from[arcs]], graph.node_y[graph.arc_from[arcs]]
        bx, by = graph.node_x[graph.arc_to[arcs]], graph.node_y[graph.arc_to[arcs]]
        dx, dy = bx - ax, by - ay
        squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)
        u: np.ndarray = np.clip(((x - ax) * dx + (y - ay) * dy) / squared,
                                0, 1)
        return u, ax + u * dx, ay + u * dy

    def nearest_road(self, x: float, y: float, limit: float) -> float:
        """Distance to the nearest arc, searching up to `limit`; `limit`
        itself when nothing is closer."""
        radius: float = min(self.cell, limit)
        while True:
            arcs: np.ndarray = self.near(x, y, radius)
            if len(arcs) > 0:
                _, px, py = self.projections(arcs, x, y)
                best: float = float(np.hypot(px - x, py - y).min())
                if best <= radius:
                    return best
            if radius >= limit:
                return limit
            radius = min(radius * 2, limit)
