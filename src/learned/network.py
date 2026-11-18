"""Layer 1: the road network as flat arrays, built once per folder.

Everything the compiled router reads is a contiguous NumPy array indexed by
node or arc id, so a search step is an array read, not a dictionary lookup.
"""

from dataclasses import dataclass

import numpy as np

from dataset import RoadGraph


@dataclass(frozen=True)
class Network:
    """Arcs, their geometry and connectivity, and the road each belongs to.

    NOTE: heading is the azimuth in degrees, clockwise from north, as in
    map_matching_2; `moves` is False for a zero-length arc, which has none.
    """
    graph: RoadGraph
    arc_from: np.ndarray
    arc_to: np.ndarray
    length: np.ndarray
    heading: np.ndarray
    moves: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    out_start: np.ndarray
    out_arc: np.ndarray
    road: np.ndarray

    @classmethod
    def from_graph(cls, graph: RoadGraph) -> "Network":
        dx: np.ndarray = graph.node_x[graph.arc_to] - graph.node_x[graph.arc_from]
        dy: np.ndarray = graph.node_y[graph.arc_to] - graph.node_y[graph.arc_from]
        heading: np.ndarray = np.degrees(np.arctan2(dx, dy)) % 360.0
        return cls(graph,
                   graph.arc_from.astype(np.int64), graph.arc_to.astype(np.int64),
                   graph.arc_length.astype(np.float64), heading,
                   np.hypot(dx, dy) > 0,
                   graph.node_x, graph.node_y,
                   graph.out_start.astype(np.int64),
                   graph.out_arc.astype(np.int64), roads(graph))

    def node_count(self) -> int:
        return len(self.node_x)


def roads(graph: RoadGraph) -> np.ndarray:
    """The junction-to-junction road of every arc: a maximal chain of arcs
    whose inner nodes join exactly two neighbouring nodes (map_matching_2's
    network simplification). An arc that two arcs would continue into starts
    a road of its own."""
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
        onward: list[int] = []
        for out in graph.out_arcs(node).tolist():
            if arc_to[out] != arc_from[arc]:
                onward.append(out)
        if len(onward) == 1:
            following[arc] = onward[0]
    entering: np.ndarray = np.bincount(following[following >= 0],
                                       minlength=graph.arc_count())
    following[(following >= 0) & (entering[following] > 1)] = -1
    return chain_labels(following)


def chain_labels(following: np.ndarray) -> np.ndarray:
    """Number the chains: first those starting at an arc nothing continues
    into, then the closed loops left over."""
    label: np.ndarray = np.full(len(following), -1, dtype=np.int64)
    has_previous: np.ndarray = np.zeros(len(following), dtype=bool)
    has_previous[following[following >= 0]] = True
    count: int = 0
    for first_pass in [True, False]:
        for arc in range(len(following)):
            if label[arc] >= 0 or (first_pass and has_previous[arc]):
                continue
            current: int = arc
            while current >= 0 and label[current] < 0:
                label[current] = count
                current = int(following[current])
            count += 1
    return label


class ArcGrid:
    """Arcs bucketed by the square cells their bounding boxes touch. Used
    only by candidate search, which runs a few hundred times per track."""

    def __init__(self, network: Network, cell: float) -> None:
        self.network: Network = network
        self.cell: float = cell
        ax, bx = network.node_x[network.arc_from], network.node_x[network.arc_to]
        ay, by = network.node_y[network.arc_from], network.node_y[network.arc_to]
        x0 = np.floor(np.minimum(ax, bx) / cell).astype(np.int64)
        x1 = np.floor(np.maximum(ax, bx) / cell).astype(np.int64)
        y0 = np.floor(np.minimum(ay, by) / cell).astype(np.int64)
        y1 = np.floor(np.maximum(ay, by) / cell).astype(np.int64)
        buckets: dict[tuple[int, int], list[int]] = {}
        for arc in range(len(ax)):
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
        """Share along each arc of the point's closest point, and that point."""
        net: Network = self.network
        ax, ay = net.node_x[net.arc_from[arcs]], net.node_y[net.arc_from[arcs]]
        bx, by = net.node_x[net.arc_to[arcs]], net.node_y[net.arc_to[arcs]]
        dx, dy = bx - ax, by - ay
        squared: np.ndarray = np.maximum(dx * dx + dy * dy, 1e-12)
        u: np.ndarray = np.clip(((x - ax) * dx + (y - ay) * dy) / squared,
                                0, 1)
        return u, ax + u * dx, ay + u * dy

    def nearest_road(self, x: float, y: float, limit: float) -> float:
        """Distance to the nearest arc within `limit`, else `limit`."""
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
