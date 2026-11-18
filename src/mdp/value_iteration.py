"""Value iteration on the MDP, and the policy it gives.

States (single.hpp, state_size 2): S(j, c) = at point j on candidate c,
heading for point j + 1; and F(j) = at point j with no candidate yet (the
start, or right after a skip). From F(j) the agent picks a candidate c of
point j, paying the distance times 3 at the first point and times 0.1
after a skip, and lands in S(j, c). From S(j, c) it picks a candidate of
point j + 1 and lands in S(j + 1, c'), or in F(j + 1) when the transition
skips. S(last, c) ends the episode.

Every transition moves one point forward, so the states form a layered
acyclic graph: value iteration, swept from the last point backwards,
converges in that one sweep to
    V(S(j, c)) = max_c' [ r + gamma V(next) ],
the same values map_matching_2's in-place iteration reaches.
"""

from dataclasses import dataclass

import numpy as np

from mdp.candidates import Candidates
from mdp.geometry import TrackShape
from mdp.hyperparameters import (AFTER_SKIP_DISTANCE, DISCOUNT,
                                 DISTANCE_WEIGHT, FIRST_DISTANCE)
from mdp.rewards import fail_errors, step_rewards
from mdp.routing import Router


@dataclass(frozen=True)
class Policy:
    """The candidate chosen at every point, and whether the step from point
    j to point j + 1 skips (no route is driven between them)."""
    chosen: list[int]
    skipped: list[bool]


def fresh_value(candidates: Candidates, values: np.ndarray,
                weight: float) -> tuple[float, int]:
    """V(F(j)) and its best candidate."""
    total: np.ndarray = (-DISTANCE_WEIGHT * weight * candidates.distance
                         + DISCOUNT * values)
    best: int = int(np.argmax(total))
    return float(total[best]), best


def solve(sets: list[Candidates], shape: TrackShape,
          router: Router) -> Policy:
    """One backward sweep for the values, then the greedy policy from F(0)."""
    n: int = len(sets)
    fail: np.ndarray = fail_errors(shape)
    values: list[np.ndarray] = [np.zeros(0)] * n
    values[n - 1] = np.zeros(sets[n - 1].count())
    best: list[np.ndarray] = [np.zeros(0, dtype=np.int64)] * n
    skips: list[np.ndarray] = [np.zeros(0, dtype=bool)] * n

    for j in range(n - 2, -1, -1):
        rewards, skip = step_rewards(j, sets[j], sets[j + 1], shape, router,
                                     float(fail[j]))
        fresh, _ = fresh_value(sets[j + 1], values[j + 1],
                               AFTER_SKIP_DISTANCE)
        onward: np.ndarray = np.where(skip, fresh, values[j + 1][None, :])
        total: np.ndarray = rewards + DISCOUNT * onward
        best[j] = np.argmax(total, axis=1)
        rows: np.ndarray = np.arange(sets[j].count())
        values[j] = total[rows, best[j]]
        skips[j] = skip[rows, best[j]]

    _, c = fresh_value(sets[0], values[0], FIRST_DISTANCE)
    chosen: list[int] = [c]
    skipped: list[bool] = []
    for j in range(n - 1):
        if skips[j][c]:
            _, c = fresh_value(sets[j + 1], values[j + 1],
                               AFTER_SKIP_DISTANCE)
            skipped.append(True)
        else:
            c = int(best[j][c])
            skipped.append(False)
        chosen.append(c)
    return Policy(chosen, skipped)
