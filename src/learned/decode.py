"""Exact decoding over the lattice (derivation Theorem 1), and driving the
chosen candidates into arcs.

With step costs s_j(c, c'), skip masks, fresh-start costs f_j(c) and a
discount g (1 for the learned model; Woeltche's 0.99 only for the check
against mdp/), swept from the last point back:
    V_{n-1}(c) = 0
    F_j        = min_c [ f_j(c) + g V_j(c) ]
    V_j(c)     = min_c' [ s_j(c, c') + g (F_{j+1} if skip else V_{j+1}(c')) ]
and the labelling follows the minimising choices forward from F_0. Ties
go to the first index, as np.argmax does in mdp/.
"""

from dataclasses import dataclass

import numpy as np

from dataset import Route
from learned.candidates import Candidates
from learned.hyperparameters import ROUTING_MAX_DISTANCE_FACTOR
from learned.routes import Router


@dataclass(frozen=True)
class Labelling:
    """The candidate at every point, whether step j skips, and the least
    total cost (F_0)."""
    chosen: np.ndarray
    skipped: np.ndarray
    value: float


def decode(step_costs: list[np.ndarray], skips: list[np.ndarray],
           fresh_costs: list[np.ndarray], discount: float) -> Labelling:
    n: int = len(fresh_costs)
    values: list[np.ndarray] = [np.zeros(0)] * n
    values[n - 1] = np.zeros(len(fresh_costs[n - 1]))
    best: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * n
    skipping: list[np.ndarray] = [np.zeros(0, dtype=bool)] * n
    fresh_best: list[int] = [0] * n
    for j in range(n - 2, -1, -1):
        fresh: np.ndarray = fresh_costs[j + 1] + discount * values[j + 1]
        fresh_best[j + 1] = int(np.argmin(fresh))
        onward: np.ndarray = np.where(skips[j], fresh[fresh_best[j + 1]],
                                      values[j + 1][None, :])
        total: np.ndarray = step_costs[j] + discount * onward
        best[j] = np.argmin(total, axis=1)
        rows: np.ndarray = np.arange(len(total))
        values[j] = total[rows, best[j]]
        skipping[j] = skips[j][rows, best[j]]

    start: np.ndarray = fresh_costs[0] + discount * values[0]
    c: int = int(np.argmin(start))
    least: float = float(start[c])
    chosen: np.ndarray = np.zeros(n, dtype=np.int64)
    skipped: np.ndarray = np.zeros(n - 1, dtype=bool)
    chosen[0] = c
    for j in range(n - 1):
        if skipping[j][c]:
            c = fresh_best[j + 1]
            skipped[j] = True
        else:
            c = int(best[j][c])
        chosen[j + 1] = c
    return Labelling(chosen, skipped, least)


def drive(labelling: Labelling, points: list[Candidates], router: Router) -> Route:
    """The arcs from candidate to candidate; a skip drives nothing but the
    next candidate's arc. A repeated arc at a join is written once."""
    first: int = int(labelling.chosen[0])
    arcs: list[int] = [int(points[0].arc[first])]
    for j in range(len(labelling.skipped)):
        c: int = int(labelling.chosen[j])
        following: int = int(labelling.chosen[j + 1])
        target: int = int(points[j + 1].arc[following])
        driven: list[int] = [target]
        if not labelling.skipped[j]:
            source: int = int(points[j].arc[c])
            bound: float = router.bound(source, points[j + 1].arc,
                                        ROUTING_MAX_DISTANCE_FACTOR)
            driven = [int(a) for a in router.arcs(
                source, float(points[j].along[c]), target,
                float(points[j + 1].along[following]), bound)]
        for arc in driven:
            if arcs[-1] != arc:
                arcs.append(arc)
    return Route(np.array(arcs, dtype=np.int64))
