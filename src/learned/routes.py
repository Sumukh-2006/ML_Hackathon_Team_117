"""Layer 3's hot loop: routes between the candidates of neighbouring points.

For each candidate of point j, one bounded Dijkstra from the end node of its
arc reaches every candidate of point j + 1. Along each shortest path it
accumulates the length, the total direction change (degrees), and the
sharpest turn per metre (radians per metre: a turn's angle over the shorter
of its two arcs). The search is the same as mdp/routing.py's, including its
tie-breaking (heap order by distance then node, strict improvement only),
so lengths and direction changes are identical.

The functions are plain loops over arrays, compiled by Numba; with
NUMBA_DISABLE_JIT=1 they run as ordinary Python and give the same arrays.
"""

import heapq
import math

import numpy as np
from numba import njit

from learned.network import Network


@njit(cache=True)
def turn_degrees(first: int, second: int, heading: np.ndarray,
                 moves: np.ndarray) -> float:
    """|direction change| from one arc onto the next; 0 if either arc has
    no length (map_matching_2's convention)."""
    if not (moves[first] and moves[second]):
        return 0.0
    return abs((heading[second] - heading[first] + 180.0) % 360.0 - 180.0)


@njit(cache=True)
def turn_sharpness(first: int, second: int, degrees: float,
                   length: np.ndarray) -> float:
    """The turn's angle in radians over the shorter of its two arcs."""
    if degrees == 0.0:
        return 0.0
    return math.radians(degrees) / min(length[first], length[second])


@njit(cache=True)
def search(source: int, bound: float, arc_to: np.ndarray, length: np.ndarray,
           heading: np.ndarray, moves: np.ndarray, out_start: np.ndarray,
           out_arc: np.ndarray, distance: np.ndarray, entry: np.ndarray,
           turning: np.ndarray, sharpest: np.ndarray,
           touched: np.ndarray) -> int:
    """Dijkstra from the end node of arc `source` up to `bound`. Fills the
    scratch arrays for every node it reaches and returns how many nodes it
    touched (listed in `touched`), so the caller can reset just those."""
    start: int = arc_to[source]
    distance[start] = 0.0
    entry[start] = source
    turning[start] = 0.0
    sharpest[start] = 0.0
    touched[0] = start
    count: int = 1
    heap = [(0.0, start)]
    while len(heap) > 0:
        here, node = heapq.heappop(heap)
        if here > distance[node]:
            continue
        for k in range(out_start[node], out_start[node + 1]):
            arc: int = out_arc[k]
            there: float = here + length[arc]
            target: int = arc_to[arc]
            if there > bound or there >= distance[target]:
                continue
            if distance[target] == np.inf:
                touched[count] = target
                count += 1
            distance[target] = there
            degrees: float = turn_degrees(entry[node], arc, heading, moves)
            turning[target] = turning[node] + degrees
            sharpest[target] = max(sharpest[node],
                                   turn_sharpness(entry[node], arc, degrees,
                                                  length))
            entry[target] = arc
            heapq.heappush(heap, (there, target))
    return count


@njit(cache=True)
def step_routes(sources: np.ndarray, source_along: np.ndarray,
                targets: np.ndarray, target_along: np.ndarray, factor: float,
                arc_from: np.ndarray, arc_to: np.ndarray, length: np.ndarray,
                heading: np.ndarray, moves: np.ndarray, node_x: np.ndarray,
                node_y: np.ndarray, out_start: np.ndarray, out_arc: np.ndarray,
                distance: np.ndarray, entry: np.ndarray, turning: np.ndarray,
                sharpest: np.ndarray, touched: np.ndarray):
    """Route length, direction change and sharpest turn from every source
    candidate (rows) to every target candidate (columns); length is
    infinite where no route exists within the bound.

    NOTE: a search depends only on the source arc (its bound depends only on
    the arc and the targets), and adoption puts many candidates on the same
    arc; so rows are visited grouped by arc and each arc is searched once."""
    rows: int = len(sources)
    columns: int = len(targets)
    lengths = np.full((rows, columns), np.inf)
    changes = np.zeros((rows, columns))
    sharp = np.zeros((rows, columns))
    order = np.argsort(sources, kind="mergesort")
    first: int = 0
    while first < rows:
        source: int = sources[order[first]]
        last: int = first
        while last + 1 < rows and sources[order[last + 1]] == source:
            last += 1
        start: int = arc_to[source]
        farthest: float = 0.0
        for c in range(columns):
            goal: int = arc_from[targets[c]]
            gap: float = math.hypot(node_x[goal] - node_x[start],
                                    node_y[goal] - node_y[start])
            farthest = max(farthest, gap)
        bound: float = factor * farthest
        count: int = search(source, bound, arc_to, length, heading, moves,
                            out_start, out_arc, distance, entry, turning,
                            sharpest, touched)
        for k in range(first, last + 1):
            fill_row(order[k], source, source_along, targets, target_along,
                     bound, arc_from, length, heading, moves, distance,
                     entry, turning, sharpest, lengths, changes, sharp)
        for k in range(count):
            distance[touched[k]] = np.inf
        first = last + 1
    return lengths, changes, sharp


@njit(cache=True)
def fill_row(r: int, source: int, source_along: np.ndarray,
             targets: np.ndarray, target_along: np.ndarray, bound: float,
             arc_from: np.ndarray, length: np.ndarray, heading: np.ndarray,
             moves: np.ndarray, distance: np.ndarray, entry: np.ndarray,
             turning: np.ndarray, sharpest: np.ndarray, lengths: np.ndarray,
             changes: np.ndarray, sharp: np.ndarray) -> None:
    """Read one source candidate's routes out of the finished search."""
    head: float = length[source] - source_along[r]
    for c in range(len(targets)):
        target: int = targets[c]
        if target == source and target_along[c] >= source_along[r]:
            lengths[r, c] = target_along[c] - source_along[r]
            continue
        node: int = arc_from[target]
        if distance[node] <= bound:
            degrees: float = turn_degrees(entry[node], target, heading, moves)
            lengths[r, c] = head + distance[node] + target_along[c]
            changes[r, c] = turning[node] + degrees
            sharp[r, c] = max(sharpest[node],
                              turn_sharpness(entry[node], target, degrees,
                                             length))


@njit(cache=True)
def route_arcs(source: int, source_along: float, target: int,
               target_along: float, bound: float, arc_to: np.ndarray,
               arc_from: np.ndarray, length: np.ndarray, heading: np.ndarray,
               moves: np.ndarray, out_start: np.ndarray, out_arc: np.ndarray,
               distance: np.ndarray, entry: np.ndarray, turning: np.ndarray,
               sharpest: np.ndarray, touched: np.ndarray) -> np.ndarray:
    """The arcs driven from one candidate to another, both ends included;
    empty if no route exists within the bound."""
    if target == source and target_along >= source_along:
        return np.array([source])
    count: int = search(source, bound, arc_to, length, heading, moves,
                        out_start, out_arc, distance, entry, turning,
                        sharpest, touched)
    node: int = arc_from[target]
    if distance[node] > bound:
        for k in range(count):
            distance[touched[k]] = np.inf
        return np.zeros(0, dtype=np.int64)
    between = []
    while node != arc_to[source]:
        between.append(entry[node])
        node = arc_from[entry[node]]
    for k in range(count):
        distance[touched[k]] = np.inf
    path = np.empty(len(between) + 2, dtype=np.int64)
    path[0] = source
    for i in range(len(between)):
        path[i + 1] = between[len(between) - 1 - i]
    path[len(between) + 1] = target
    return path


class Router:
    """The scratch arrays a search writes into, allocated once per network
    and reset after every search by clearing only the nodes it touched."""

    def __init__(self, network: Network) -> None:
        self.network: Network = network
        nodes: int = network.node_count()
        self.distance: np.ndarray = np.full(nodes, np.inf)
        self.entry: np.ndarray = np.zeros(nodes, dtype=np.int64)
        self.turning: np.ndarray = np.zeros(nodes)
        self.sharpest: np.ndarray = np.zeros(nodes)
        self.touched: np.ndarray = np.zeros(nodes, dtype=np.int64)

    def step(self, sources: np.ndarray, source_along: np.ndarray,
             targets: np.ndarray, target_along: np.ndarray,
             factor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        net: Network = self.network
        return step_routes(sources, source_along, targets, target_along,
                           factor, net.arc_from, net.arc_to, net.length,
                           net.heading, net.moves, net.node_x, net.node_y,
                           net.out_start, net.out_arc, self.distance,
                           self.entry, self.turning, self.sharpest,
                           self.touched)

    def bound(self, source: int, targets: np.ndarray, factor: float) -> float:
        net: Network = self.network
        start: int = int(net.arc_to[source])
        goals: np.ndarray = net.arc_from[targets]
        gaps: np.ndarray = np.hypot(net.node_x[goals] - net.node_x[start],
                                    net.node_y[goals] - net.node_y[start])
        return factor * float(gaps.max())

    def arcs(self, source: int, source_along: float, target: int,
             target_along: float, bound: float) -> np.ndarray:
        net: Network = self.network
        return route_arcs(source, source_along, target, target_along, bound,
                          net.arc_to, net.arc_from, net.length, net.heading,
                          net.moves, net.out_start, net.out_arc,
                          self.distance, self.entry, self.turning,
                          self.sharpest, self.touched)
