"""Path recovery over key points (section 4.2.2, Algorithm 1).

The likelihood of a global candidate path is the product of the candidates'
emissions p(e) (Gaussian in distance, Eq. 6) and the transitions
    p(e_f, e_t) = exp(-(N / n) * C* / dist(p_i, p_i+1))      (Eq. 8)
where C* is the cheapest action cost from e_f to e_t, N the number of
fixes from key point i to key point i+1, and n the number of fixes. The
best path is found by dynamic programming in log space.
"""

from dataclasses import dataclass

import numpy as np

from segment.action_graph import ActionGraph


@dataclass(frozen=True)
class Stretch:
    """What a transition between two consecutive key points needs."""
    fixes: int                    # N: fixes from one key point to the next
    distance: float               # metres between the two key points
    trajectory: dict[int, float]  # dist(e, s) for arcs closer than the cap


def emission(distance: np.ndarray, sigma: float) -> np.ndarray:
    return (-0.5 * (distance / sigma) ** 2
            - np.log(np.sqrt(2 * np.pi) * sigma))


def transitions(sources: np.ndarray, targets: np.ndarray, stretch: Stretch,
                total: int, actions: ActionGraph, cap: float) -> np.ndarray:
    """log p(e_f, e_t) for every source (row) and target (column); minus
    infinity where no action sequence exists.
    OWN CHOICE: the key-point distance is floored at 1 m, as Eq. 8 divides
    by it and two key points may coincide."""
    result: np.ndarray = np.full((len(sources), len(targets)), -np.inf)
    weight: float = (stretch.fixes / total) / max(stretch.distance, 1.0)
    target_list: list[int] = targets.tolist()
    for row in range(len(sources)):
        reached, _ = actions.cheapest(int(sources[row]), target_list,
                                      stretch.trajectory, cap)
        for column in range(len(targets)):
            if target_list[column] in reached:
                result[row, column] = -weight * reached[target_list[column]]
    return result


def decode(candidates: list[tuple[np.ndarray, np.ndarray]],
           stretches: list[Stretch], total: int, actions: ActionGraph,
           sigma: float, cap: float) -> list[tuple[int, list[int]]]:
    """The best candidate arc of every key point, trip by trip, each trip
    with the index of its first key point.
    OWN CHOICE: when no candidate of a key point can be reached from the
    previous one, the trip ends there and a new one starts (the paper does
    not cover this case)."""
    trips: list[tuple[int, list[int]]] = []
    first_key: int = 0
    arcs, distance = candidates[0]
    score: np.ndarray = emission(distance, sigma)
    backs: list[np.ndarray] = [np.full(len(arcs), -1)]
    chosen_arcs: list[np.ndarray] = [arcs]

    for key in range(1, len(candidates)):
        arcs, distance = candidates[key]
        step: np.ndarray = transitions(chosen_arcs[-1], arcs,
                                       stretches[key - 1], total, actions,
                                       cap)
        total_score: np.ndarray = score[:, None] + step
        best: np.ndarray = total_score.max(axis=0)
        if np.all(best == -np.inf):
            trips.append((first_key, backtrack(score, backs, chosen_arcs)))
            first_key = key
            score = emission(distance, sigma)
            backs = [np.full(len(arcs), -1)]
            chosen_arcs = [arcs]
            continue
        backs.append(total_score.argmax(axis=0))
        chosen_arcs.append(arcs)
        score = best + emission(distance, sigma)

    trips.append((first_key, backtrack(score, backs, chosen_arcs)))
    return trips


def backtrack(score: np.ndarray, backs: list[np.ndarray],
              chosen_arcs: list[np.ndarray]) -> list[int]:
    path: list[int] = []
    j: int = int(np.argmax(score))
    for column in range(len(backs) - 1, -1, -1):
        path.append(int(chosen_arcs[column][j]))
        j = int(backs[column][j])
    path.reverse()
    return path
