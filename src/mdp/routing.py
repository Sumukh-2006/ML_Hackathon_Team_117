"""Candidate routes (map_matching_2 router::candidate_route) and the sum of
absolute direction changes along them.

A route from candidate (arc a, along s) to candidate (arc b, along t) is
the rest of a, the shortest node path from a's end to b's start, and b up
to t; on the same arc going forward it is the piece between s and t. Turns
happen only at nodes. The direction changes of a route are the turns
between its consecutive arcs, accumulated along the shortest path found.
"""

import heapq

import numpy as np

from dataset import RoadGraph
from mdp.geometry import angle_diff, azimuth


class Router:
    """Bounded Dijkstra searches started from the end of a candidate's arc."""

    def __init__(self, graph: RoadGraph) -> None:
        self.arc_from: list[int] = graph.arc_from.tolist()
        self.arc_to: list[int] = graph.arc_to.tolist()
        self.length: list[float] = graph.arc_length.tolist()
        self.node_x: np.ndarray = graph.node_x
        self.node_y: np.ndarray = graph.node_y
        self.out: list[list[int]] = []
        for node in range(graph.node_count()):
            self.out.append(graph.out_arcs(node).tolist())
        dx: np.ndarray = graph.node_x[graph.arc_to] - graph.node_x[graph.arc_from]
        dy: np.ndarray = graph.node_y[graph.arc_to] - graph.node_y[graph.arc_from]
        self.heading: list[float] = azimuth(dx, dy).tolist()
        self.moves: list[bool] = (np.hypot(dx, dy) > 0).tolist()
        # NOTE: searches are kept per start arc with the bound they ran to. A
        # search to a larger bound settles every node within a smaller one
        # in the same order, with the same distance, entry arc and turning,
        # so it answers the smaller search once nodes past the bound are
        # ignored.
        self.searches: dict[int, tuple[float, dict[int, float],
                                       dict[int, int], dict[int, float]]] = {}

    def turn(self, first: int, second: int) -> float:
        """|direction change| from arc `first` onto arc `second`, in degrees.
        NOTE: a zero-length arc has no heading and adds no turn."""
        if not (self.moves[first] and self.moves[second]):
            return 0.0
        return abs(float(angle_diff(self.heading[first], self.heading[second])))

    def search(self, arc: int, bound: float
               ) -> tuple[dict[int, float], dict[int, int], dict[int, float]]:
        """From the end node of `arc`: distance to every node within `bound`
        (and possibly farther ones, from a cached larger search), the arc
        each is entered by, and the turning accumulated on the way."""
        if arc in self.searches and self.searches[arc][0] >= bound:
            _, cached_distance, cached_entry, cached_turning = self.searches[arc]
            return cached_distance, cached_entry, cached_turning
        start: int = self.arc_to[arc]
        distance: dict[int, float] = {start: 0.0}
        entry: dict[int, int] = {start: arc}
        turning: dict[int, float] = {start: 0.0}
        heap: list[tuple[float, int]] = [(0.0, start)]
        while len(heap) > 0:
            here, node = heapq.heappop(heap)
            if here > distance[node]:
                continue
            for following in self.out[node]:
                there: float = here + self.length[following]
                target: int = self.arc_to[following]
                if there > bound or there >= distance.get(target, np.inf):
                    continue
                distance[target] = there
                entry[target] = following
                turning[target] = turning[node] + self.turn(entry[node],
                                                            following)
                heapq.heappush(heap, (there, target))
        self.searches[arc] = (bound, distance, entry, turning)
        return distance, entry, turning

    def bound(self, arc: int, targets: np.ndarray, factor: float) -> float:
        """factor times the longest straight distance from the end node of
        `arc` to the start nodes of the target arcs."""
        start: int = self.arc_to[arc]
        goals: np.ndarray = np.array([self.arc_from[t] for t in targets.tolist()])
        gaps: np.ndarray = np.hypot(self.node_x[goals] - self.node_x[start],
                                    self.node_y[goals] - self.node_y[start])
        return factor * float(gaps.max())

    def routes(self, arc: int, along: float, targets: np.ndarray,
               alongs: np.ndarray, bound: float
               ) -> tuple[np.ndarray, np.ndarray]:
        """Length and direction changes of the route to every target;
        infinite length where no route exists within the bound."""
        head: float = self.length[arc] - along
        distance, entry, turning = self.search(arc, bound)
        lengths: np.ndarray = np.full(len(targets), np.inf)
        changes: np.ndarray = np.zeros(len(targets))
        for k in range(len(targets)):
            target: int = int(targets[k])
            if target == arc and alongs[k] >= along:
                lengths[k] = alongs[k] - along
                continue
            node: int = self.arc_from[target]
            if node in distance and distance[node] <= bound:
                lengths[k] = head + distance[node] + alongs[k]
                changes[k] = turning[node] + self.turn(entry[node], target)
        return lengths, changes

    def path(self, arc: int, along: float, target: int, target_along: float,
             bound: float) -> list[int]:
        """The arcs of the route from one candidate to another, both ends
        included; the route must exist within `bound`."""
        if target == arc and target_along >= along:
            return [arc]
        _, entry, _ = self.search(arc, bound)
        between: list[int] = []
        node: int = self.arc_from[target]
        while node != self.arc_to[arc]:
            between.append(entry[node])
            node = self.arc_from[entry[node]]
        between.reverse()
        return [arc] + between + [target]
