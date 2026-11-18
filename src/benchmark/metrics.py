"""How much of a predicted route is right, measured in metres of road.

A route is read as a multiset of roads: c(a) is how often it drives arc a,
with twin arcs (same two nodes) counted as one road. With l(a) the length
of arc a, a prediction P and the truth T:

    overlap O = sum_a min(c_P(a), c_T(a)) l(a)
    added   A = sum_a max(0, c_P(a) - c_T(a)) l(a)
    missed  M = sum_a max(0, c_T(a) - c_P(a)) l(a)

The correct fraction O / max(L_P, L_T) is Woeltche's measure, made exact
on arcs; the route mismatch fraction (A + M) / L_T is Newson & Krumm's.
"""

from dataclasses import dataclass

import numpy as np

from dataset import RoadGraph, Route


@dataclass(frozen=True)
class RouteScore:
    """The five lengths, in metres, that every measure is built from."""
    overlap: float
    added: float
    missed: float
    predicted_length: float
    true_length: float

    def correct_fraction(self) -> float:
        return self.overlap / max(self.predicted_length, self.true_length)

    def mismatch_fraction(self) -> float:
        return (self.added + self.missed) / self.true_length

    def precision(self) -> float:
        # NOTE: an empty prediction has precision 0: nothing in it is right.
        if self.predicted_length == 0.0:
            return 0.0
        return self.overlap / self.predicted_length

    def recall(self) -> float:
        return self.overlap / self.true_length


def score_route(graph: RoadGraph, predicted: Route,
                truth: Route) -> RouteScore:
    """Compare two routes on the same graph."""
    predicted_count: np.ndarray = road_count(graph, predicted)
    true_count: np.ndarray = road_count(graph, truth)
    length: np.ndarray = graph.arc_length

    overlap: float = float(
        (np.minimum(predicted_count, true_count) * length).sum())
    added: float = float(
        (np.maximum(predicted_count - true_count, 0) * length).sum())
    missed: float = float(
        (np.maximum(true_count - predicted_count, 0) * length).sum())
    return RouteScore(overlap, added, missed,
                      float((predicted_count * length).sum()),
                      float((true_count * length).sum()))


def road_count(graph: RoadGraph, route: Route) -> np.ndarray:
    """How often the route drives each road, indexed by canonical arc."""
    return np.bincount(graph.canonical_arc[route.arcs],
                       minlength=graph.arc_count())


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
