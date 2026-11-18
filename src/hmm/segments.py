"""Grouping arcs into road segments, the paper's unit of match candidate.

OWN READING: Newson & Krumm match a fix to "road segments" of a commercial
map, where a segment runs from one junction to the next. Our map is cut into
short straight arcs, so a road segment here is a maximal chain of arcs whose
inner nodes join exactly two roads. A fix gets at most one candidate per
segment, at the segment's point nearest to it.
"""

import numpy as np

from dataset import RoadGraph


def arc_segments(graph: RoadGraph) -> np.ndarray:
    """The road segment of every arc, numbered from 0."""
    following: np.ndarray = continuations(graph)
    segment: np.ndarray = np.full(graph.arc_count(), -1, dtype=np.int64)
    has_previous: np.ndarray = np.zeros(graph.arc_count(), dtype=bool)
    has_previous[following[following >= 0]] = True

    count: int = 0
    for arc in range(graph.arc_count()):
        if has_previous[arc]:
            continue
        label_chain(arc, following, segment, count)
        count += 1

    # NOTE: what is left are closed loops of inner nodes (no junction).
    for arc in range(graph.arc_count()):
        if segment[arc] >= 0:
            continue
        label_chain(arc, following, segment, count)
        count += 1
    return segment


def continuations(graph: RoadGraph) -> np.ndarray:
    """following[a]: the single arc that carries arc a on through an inner
    node (one with exactly two neighbouring nodes), or -1 at a junction.
    An arc that two arcs would continue into starts a segment of its own."""
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

    predecessors: np.ndarray = np.bincount(following[following >= 0],
                                           minlength=graph.arc_count())
    shared: np.ndarray = (following >= 0) & (predecessors[following] > 1)
    following[shared] = -1
    return following


def label_chain(first: int, following: np.ndarray, segment: np.ndarray,
                label: int) -> None:
    """Give `label` to `first` and the arcs that follow it, stopping at a
    junction or where the chain closes on itself."""
    arc: int = first
    while arc >= 0 and segment[arc] < 0:
        segment[arc] = label
        arc = int(following[arc])
