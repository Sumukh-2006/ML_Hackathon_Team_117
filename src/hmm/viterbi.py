"""Viterbi decoding of the candidate lattice, with the paper's repair of
breaks (Newson & Krumm, sections 3 and 4.2).

A column holds, for every candidate of one kept fix, the log probability of
the best path ending there and that path's candidate in the column before.
A break is a column that cannot be reached from the previous one: no
candidates, or no route of non-zero probability.
"""

from dataclasses import dataclass

import numpy as np

from dataset import Track, great_circle
from hmm.candidates import Candidates
from hmm.routing import Router


@dataclass(frozen=True)
class Model:
    sigma: float
    beta: float
    max_detour: float
    max_speed: float
    max_break: float


@dataclass(frozen=True)
class Column:
    fix: int
    candidates: Candidates
    score: np.ndarray
    back: np.ndarray


def emission(candidates: Candidates, sigma: float) -> np.ndarray:
    """log N(distance; 0, sigma), the paper's equation 1."""
    return (-0.5 * (candidates.distance / sigma) ** 2
            - np.log(np.sqrt(2 * np.pi) * sigma))


def connect(previous: Column, fix: int, candidates: Candidates, track: Track,
            router: Router, model: Model) -> Column | None:
    """The column of `fix` reached from `previous`, or None at a break."""
    if candidates.count() == 0:
        return None
    straight: float = float(great_circle(
        track.lon[previous.fix], track.lat[previous.fix],
        track.lon[fix], track.lat[fix]))
    elapsed: float = float(track.t[fix] - track.t[previous.fix])
    bound: float = min(straight + model.max_detour, model.max_speed * elapsed)

    best: np.ndarray = np.full(candidates.count(), -np.inf)
    back: np.ndarray = np.full(candidates.count(), -1, dtype=np.int64)
    for i in range(previous.candidates.count()):
        if previous.score[i] == -np.inf:
            continue
        route: np.ndarray = router.lengths(
            int(previous.candidates.arc[i]), float(previous.candidates.along[i]),
            candidates.arc, candidates.along, bound)
        # The paper's equation 2: an exponential in |straight - route|.
        transition: np.ndarray = (-np.log(model.beta)
                                  - np.abs(straight - route) / model.beta)
        total: np.ndarray = previous.score[i] + transition
        better: np.ndarray = total > best
        best[better] = total[better]
        back[better] = i

    if np.all(best == -np.inf):
        return None
    return Column(fix, candidates, best + emission(candidates, model.sigma),
                  back)


def decode(fixes: list[int], candidates: list[Candidates], track: Track,
           router: Router, model: Model) -> list[list[tuple[int, float]]]:
    """The best candidate path through every trip, as (arc, along) pairs.
    A break that cannot be repaired ends a trip; the next one starts at
    the fix after the break."""
    trips: list[list[tuple[int, float]]] = []
    stack: list[Column] = []
    i: int = 0
    while i < len(fixes):
        if len(stack) == 0:
            if candidates[i].count() > 0:
                stack.append(Column(fixes[i], candidates[i],
                                    emission(candidates[i], model.sigma),
                                    np.full(candidates[i].count(), -1)))
            i += 1
            continue

        column: Column | None = connect(stack[-1], fixes[i], candidates[i],
                                        track, router, model)
        if column is not None:
            stack.append(column)
            i += 1
            continue

        resume: int = heal(stack, i, fixes, candidates, track, router, model)
        if resume >= 0:
            i = resume
            continue
        trips.append(backtrack(stack))
        stack = []

    if len(stack) > 0:
        trips.append(backtrack(stack))
    return trips


def heal(stack: list[Column], broken: int, fixes: list[int],
         candidates: list[Candidates], track: Track, router: Router,
         model: Model) -> int:
    """Remove fixes on both sides of the break between the stack's last
    column and fix `broken` until the two sides reconnect (the index to go
    on from is returned) or the gap spans more than `max_break` seconds
    (-1 is returned and the stack is left as it was)."""
    removed: list[Column] = []
    right: int = broken
    while len(stack) > 1 and right + 1 < len(fixes):
        removed.append(stack.pop())
        right += 1
        gap: float = float(track.t[fixes[right]] - track.t[stack[-1].fix])
        if gap > model.max_break:
            break
        column: Column | None = connect(stack[-1], fixes[right],
                                        candidates[right], track, router,
                                        model)
        if column is not None:
            stack.append(column)
            return right + 1

    for column in reversed(removed):
        stack.append(column)
    return -1


def backtrack(stack: list[Column]) -> list[tuple[int, float]]:
    """The best path through the stacked columns, first column first."""
    path: list[tuple[int, float]] = []
    j: int = int(np.argmax(stack[-1].score))
    for column in reversed(stack):
        path.append((int(column.candidates.arc[j]),
                     float(column.candidates.along[j])))
        j = int(column.back[j])
    path.reverse()
    return path
