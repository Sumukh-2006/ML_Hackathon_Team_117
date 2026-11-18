"""The action graph (section 3.1, 4.1): nodes are road segments (arcs),
and an action is a turn from an arc into an arc that starts where it ends.

Taking action a into arc e costs (Eq. 5)
    C = C_traj * (C_len + omega * C_turn)
with C_traj = min(dist(e, s), 100) for the stretch s of track being
matched, C_len = length of e, and C_turn = 0, 1 or 2 by the turn angle.
The cheapest action sequence between two candidate arcs is found by
Dijkstra; the starting arc costs nothing, the arcs entered cost C each.
"""

import heapq
import math

import numpy as np

from dataset import RoadGraph


class ActionGraph:
    """Turn costs and Dijkstra over arcs for one road graph."""

    def __init__(self, graph: RoadGraph, omega: float, gentle: float,
                 sharp: float) -> None:
        self.omega: float = omega
        self.gentle: float = gentle
        self.sharp: float = sharp
        self.arc_to: list[int] = graph.arc_to.tolist()
        self.length: list[float] = graph.arc_length.tolist()
        self.out: list[list[int]] = []
        for node in range(graph.node_count()):
            self.out.append(graph.out_arcs(node).tolist())

        dx: np.ndarray = graph.node_x[graph.arc_to] - graph.node_x[graph.arc_from]
        dy: np.ndarray = graph.node_y[graph.arc_to] - graph.node_y[graph.arc_from]
        self.heading: list[float] = np.arctan2(dy, dx).tolist()
        self.moves: list[bool] = (np.hypot(dx, dy) > 0).tolist()
        self.piece: list[int] = weak_pieces(graph).tolist()

    def turn_cost(self, previous: int, arc: int) -> float:
        """Eq. 4. NOTE: a zero-length arc has no heading; turning onto or
        off it costs nothing."""
        if not (self.moves[previous] and self.moves[arc]):
            return 0.0
        angle: float = abs(self.heading[arc] - self.heading[previous])
        angle = min(angle, 2 * math.pi - angle)
        if angle < self.gentle:
            return 0.0
        if angle < self.sharp:
            return 1.0
        return 2.0

    def cheapest(self, source: int, targets: list[int],
                 trajectory: dict[int, float], cap: float
                 ) -> tuple[dict[int, float], dict[int, int]]:
        """Cost of the cheapest action sequence from `source` to every
        reachable target, and the arc each settled arc was entered from."""
        cost: dict[int, float] = {source: 0.0}
        entry: dict[int, int] = {}
        waiting: set[int] = set()
        for target in targets:
            if self.piece[target] == self.piece[source]:
                waiting.add(target)
        settled: set[int] = set()
        heap: list[tuple[float, int]] = [(0.0, source)]
        while len(heap) > 0 and len(waiting) > 0:
            here, arc = heapq.heappop(heap)
            if arc in settled:
                continue
            settled.add(arc)
            waiting.discard(arc)
            for following in self.out[self.arc_to[arc]]:
                step: float = trajectory.get(following, cap) * (
                    self.length[following]
                    + self.omega * self.turn_cost(arc, following))
                if here + step < cost.get(following, np.inf):
                    cost[following] = here + step
                    entry[following] = arc
                    heapq.heappush(heap, (here + step, following))

        reached: dict[int, float] = {}
        for target in targets:
            if target in settled:
                reached[target] = cost[target]
        return reached, entry


def weak_pieces(graph: RoadGraph) -> np.ndarray:
    """For every arc, a label shared by all arcs joined to it through roads
    in either direction. Arcs with different labels cannot reach each other."""
    parent: list[int] = list(range(graph.node_count()))
    for a, b in zip(graph.arc_from.tolist(), graph.arc_to.tolist()):
        root_a: int = find_root(parent, a)
        root_b: int = find_root(parent, b)
        if root_a != root_b:
            parent[root_a] = root_b
    roots: np.ndarray = np.array([find_root(parent, node)
                                  for node in range(graph.node_count())])
    return roots[graph.arc_from]


def find_root(parent: list[int], node: int) -> int:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node
