"""Candidate search and candidate adoption (paper 2.2-2.3, thesis 3.1.2-3.1.3).

A candidate is a road position for a track point: the closest point of one
road edge, kept as (arc, metres along it, x, y) with its distance to the
point. Adoption copies other points' candidates into a point's set; an
adopted candidate keeps the other point's road position and only gets a new
distance, so that several points can collapse onto one road position.
"""

from dataclasses import dataclass

import numpy as np

from mdp.network import ArcGrid


@dataclass(frozen=True)
class Candidates:
    arc: np.ndarray
    along: np.ndarray
    px: np.ndarray
    py: np.ndarray
    distance: np.ndarray
    adopted: np.ndarray

    def count(self) -> int:
        return len(self.arc)


def search(grid: ArcGrid, edges: np.ndarray, x: float, y: float, k: int,
           radius: float, upper: float) -> Candidates:
    """The closest point of each of the k nearest road edges within the
    radius; the radius doubles up to `upper` while none is found."""
    while True:
        arcs: np.ndarray = grid.near(x, y, radius)
        u, px, py = grid.projections(arcs, x, y)
        distance: np.ndarray = np.hypot(px - x, py - y)
        inside: np.ndarray = distance <= radius
        arcs, u, px, py, distance = (arcs[inside], u[inside], px[inside],
                                     py[inside], distance[inside])
        order: np.ndarray = np.lexsort((distance, edges[arcs]))
        first: np.ndarray = np.ones(len(order), dtype=bool)
        first[1:] = edges[arcs[order[1:]]] != edges[arcs[order[:-1]]]
        best: np.ndarray = order[first]
        best = best[np.argsort(distance[best], kind="stable")][:k]
        if len(best) > 0 or radius >= upper:
            along: np.ndarray = u[best] * grid.graph.arc_length[arcs[best]]
            return Candidates(arcs[best], along, px[best], py[best],
                              distance[best], np.zeros(len(best), dtype=bool))
        radius = min(radius * 2, upper)


def adopt(own: list[Candidates], x: np.ndarray, y: np.ndarray,
          cell: float) -> list[Candidates]:
    """CA nearby, then CA siblings, each candidate adopted at most once."""
    gathered: list[list[tuple]] = []
    for i in range(len(own)):
        gathered.append(rows(own[i]))
    adopt_nearby(own, x, y, cell, gathered)
    for i in range(len(own)):
        for j in [i - 1, i + 1]:
            if 0 <= j < len(own):
                add_from(own[j], x[i], y[i], gathered[i])
    return [as_candidates(rows_of_point) for rows_of_point in gathered]


def adopt_nearby(own: list[Candidates], x: np.ndarray, y: np.ndarray,
                 cell: float, gathered: list[list[tuple]]) -> None:
    """Every own candidate whose segment from its point to its road position
    passes within point i's farthest own candidate distance is adopted by i."""
    links: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for j in range(len(own)):
        for c in range(own[j].count()):
            x0, x1 = sorted([x[j], own[j].px[c]])
            y0, y1 = sorted([y[j], own[j].py[c]])
            for gx in range(int(np.floor(x0 / cell)), int(np.floor(x1 / cell)) + 1):
                for gy in range(int(np.floor(y0 / cell)), int(np.floor(y1 / cell)) + 1):
                    links.setdefault((gx, gy), []).append((j, c))

    for i in range(len(own)):
        if own[i].count() == 0:
            continue
        reach: float = float(own[i].distance.max())
        for (j, c) in near_links(links, x[i], y[i], reach, cell):
            if j == i:
                continue
            gap: float = segment_gap(x[i], y[i], x[j], y[j],
                                     own[j].px[c], own[j].py[c])
            if gap <= reach:
                add_one(own[j], c, x[i], y[i], gathered[i])


def near_links(links: dict[tuple[int, int], list[tuple[int, int]]],
               x: float, y: float, reach: float,
               cell: float) -> set[tuple[int, int]]:
    found: set[tuple[int, int]] = set()
    for gx in range(int(np.floor((x - reach) / cell)),
                    int(np.floor((x + reach) / cell)) + 1):
        for gy in range(int(np.floor((y - reach) / cell)),
                        int(np.floor((y + reach) / cell)) + 1):
            found.update(links.get((gx, gy), []))
    return found


def segment_gap(x: float, y: float, ax: float, ay: float, bx: float,
                by: float) -> float:
    dx, dy = bx - ax, by - ay
    squared: float = max(dx * dx + dy * dy, 1e-12)
    u: float = min(max(((x - ax) * dx + (y - ay) * dy) / squared, 0.0), 1.0)
    return float(np.hypot(x - ax - u * dx, y - ay - u * dy))


def rows(candidates: Candidates) -> list[tuple]:
    result: list[tuple] = []
    for c in range(candidates.count()):
        result.append((int(candidates.arc[c]), float(candidates.along[c]),
                       float(candidates.px[c]), float(candidates.py[c]),
                       float(candidates.distance[c]), False))
    return result


def add_from(source: Candidates, x: float, y: float,
             target: list[tuple]) -> None:
    for c in range(source.count()):
        add_one(source, c, x, y, target)


def add_one(source: Candidates, c: int, x: float, y: float,
            target: list[tuple]) -> None:
    """Adopt one road position unless the set already holds it."""
    arc: int = int(source.arc[c])
    along: float = float(source.along[c])
    for row in target:
        if row[0] == arc and row[1] == along:
            return
    distance: float = float(np.hypot(source.px[c] - x, source.py[c] - y))
    target.append((arc, along, float(source.px[c]), float(source.py[c]),
                   distance, True))


def as_candidates(rows_of_point: list[tuple]) -> Candidates:
    if len(rows_of_point) == 0:
        empty: np.ndarray = np.zeros(0)
        return Candidates(empty.astype(np.int64), empty, empty, empty, empty,
                          empty.astype(bool))
    columns = list(zip(*rows_of_point))
    return Candidates(np.array(columns[0], dtype=np.int64),
                      np.array(columns[1]), np.array(columns[2]),
                      np.array(columns[3]), np.array(columns[4]),
                      np.array(columns[5], dtype=bool))
