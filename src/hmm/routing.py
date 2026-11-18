"""Shortest driving distances between match candidates, as the paper's
"conventional route planner configured to give the route with the shortest
distance" (Newson & Krumm, section 3.2).

A candidate is a point on an arc: (arc, metres along it). Driving from one
to another follows the rest of the first arc, then a shortest node-to-node
path, then the start of the second arc; on the same arc moving forwards it
is just the difference. Searches are bounded, since routes longer than the
bound have probability zero, and kept per start node for reuse.
"""

import heapq

import numpy as np

from dataset import RoadGraph


class Router:
    """Bounded Dijkstra searches on one road graph, cached by start node."""

    def __init__(self, graph: RoadGraph) -> None:
        self.arc_to: list[int] = graph.arc_to.tolist()
        self.arc_from: list[int] = graph.arc_from.tolist()
        self.length: list[float] = graph.arc_length.tolist()
        self.out: list[list[int]] = []
        for node in range(graph.node_count()):
            self.out.append(graph.out_arcs(node).tolist())
        self.searches: dict[int, tuple[float, dict[int, float],
                                       dict[int, int]]] = {}

    def reach(self, source: int,
              bound: float) -> tuple[dict[int, float], dict[int, int]]:
        """Distances to every node within `bound` of `source`, and the arc
        each one is entered by on its shortest path."""
        if source in self.searches and self.searches[source][0] >= bound:
            return self.searches[source][1], self.searches[source][2]

        distance: dict[int, float] = {source: 0.0}
        entry: dict[int, int] = {}
        heap: list[tuple[float, int]] = [(0.0, source)]
        while len(heap) > 0:
            here, node = heapq.heappop(heap)
            if here > distance[node]:
                continue
            for arc in self.out[node]:
                there: float = here + self.length[arc]
                target: int = self.arc_to[arc]
                if there > bound or there >= distance.get(target, np.inf):
                    continue
                distance[target] = there
                entry[target] = arc
                heapq.heappush(heap, (there, target))

        self.searches[source] = (bound, distance, entry)
        return distance, entry

    def lengths(self, arc: int, along: float, arcs: np.ndarray,
                alongs: np.ndarray, bound: float) -> np.ndarray:
        """Route length from one candidate to each of several; infinity
        where no route within `bound` exists."""
        head: float = self.length[arc] - along
        distance, _ = self.reach(self.arc_to[arc], bound - head)
        result: np.ndarray = np.full(len(arcs), np.inf)
        for index in range(len(arcs)):
            target: int = int(arcs[index])
            if target == arc and alongs[index] >= along:
                result[index] = alongs[index] - along
                continue
            start: int = self.arc_from[target]
            if start in distance:
                result[index] = head + distance[start] + alongs[index]
        result[result > bound] = np.inf
        return result

    def path(self, arc: int, along: float, target: int,
             target_along: float) -> list[int]:
        """The arcs driven from one candidate to another, both included.
        The route must have been found by `lengths` before."""
        if target == arc and target_along >= along:
            return [arc]
        _, _, entry = self.searches[self.arc_to[arc]]
        between: list[int] = []
        node: int = self.arc_from[target]
        while node != self.arc_to[arc]:
            between.append(entry[node])
            node = self.arc_from[entry[node]]
        between.reverse()
        return [arc] + between + [target]
