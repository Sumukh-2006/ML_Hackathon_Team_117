"""Which arcs can lead into an arc within EPSILON_TOPO of driving.

near() in the paper asks, for an alive route ending on arc s, whether arc
l is within EPSILON_TOPO of driving. Asked from l's side it becomes one
search per arc: walk backwards from l's start node along incoming arcs up
to the bound; every alive arc ending at a reached node may precede l.

OWN CHOICE: the driving distance is directed, from the end of s to the
start of l (or s == l). The authors' code takes the least of four
endpoint-to-endpoint distances, which ignores direction; our output is
directed arcs, so direction has to come from somewhere.
"""

import heapq

import numpy as np

from dataset import RoadGraph


class Predecessors:
    """Bounded backward searches, cached by the node they start from."""

    def __init__(self, graph: RoadGraph, bound: float) -> None:
        self.bound: float = bound
        self.arc_from: list[int] = graph.arc_from.tolist()
        self.arc_to: list[int] = graph.arc_to.tolist()
        self.length: list[float] = graph.arc_length.tolist()
        incoming: list[list[int]] = [[] for _ in range(graph.node_count())]
        for arc in range(graph.arc_count()):
            incoming[self.arc_to[arc]].append(arc)
        self.incoming: list[list[int]] = incoming
        self.searches: dict[int, tuple[dict[int, float], dict[int, int]]] = {}

    def into(self, node: int) -> tuple[dict[int, float], dict[int, int]]:
        """Nodes that reach `node` by driving less than the bound, with that
        distance and the first arc of the shortest way from each."""
        if node in self.searches:
            return self.searches[node]
        distance: dict[int, float] = {node: 0.0}
        onward: dict[int, int] = {}
        heap: list[tuple[float, int]] = [(0.0, node)]
        while len(heap) > 0:
            here, current = heapq.heappop(heap)
            if here > distance[current]:
                continue
            for arc in self.incoming[current]:
                there: float = here + self.length[arc]
                source: int = self.arc_from[arc]
                if there >= self.bound or there >= distance.get(source, np.inf):
                    continue
                distance[source] = there
                onward[source] = arc
                heapq.heappush(heap, (there, source))
        self.searches[node] = (distance, onward)
        return distance, onward

    def path(self, previous: int, arc: int) -> list[int] | None:
        """The arcs driven from `previous` into `arc` (both included), or
        None when `arc` is not within the bound of `previous`."""
        if previous == arc:
            return [arc]
        _, onward = self.into(self.arc_from[arc])
        node: int = self.arc_to[previous]
        if node not in onward and node != self.arc_from[arc]:
            return None
        driven: list[int] = [previous]
        while node != self.arc_from[arc]:
            driven.append(onward[node])
            node = self.arc_to[onward[node]]
        driven.append(arc)
        return driven
