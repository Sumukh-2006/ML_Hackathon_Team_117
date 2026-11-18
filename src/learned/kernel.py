"""The compiled per-step kernel: everything about every route of one step.

For each candidate pair (c at point j, c' at point j + 1) it returns the
route's length, total direction change and sharpest turn (as routes.py), the
two coverage measures (derivation §5.2), and, when the true piece is given,
the step error Delta_j (derivation §4): the metres by which the step's arcs
after c's arc differ from the true piece, counted per road.

The route's polyline is c's position, then the end node of c's arc, the
nodes of the shortest path, the start node of c''s arc, and c''s position;
on the same arc moving forward it is just the two positions.
"""

import math

import numpy as np
from numba import njit

from learned.routes import search, turn_degrees, turn_sharpness


@njit(cache=True)
def point_segment(x: float, y: float, ax: float, ay: float, bx: float,
                  by: float) -> float:
    dx: float = bx - ax
    dy: float = by - ay
    squared: float = max(dx * dx + dy * dy, 1e-12)
    u: float = ((x - ax) * dx + (y - ay) * dy) / squared
    u = min(max(u, 0.0), 1.0)
    return math.hypot(x - ax - u * dx, y - ay - u * dy)


@njit(cache=True)
def coverage(vx: np.ndarray, vy: np.ndarray, count: int, fix_x: np.ndarray,
             fix_y: np.ndarray) -> tuple:
    """Mean distance from the fixes to the polyline vx[:count], vy[:count],
    and mean distance from its vertices to the nearest fix."""
    to_route: float = 0.0
    for f in range(len(fix_x)):
        nearest: float = np.inf
        for k in range(count - 1):
            nearest = min(nearest, point_segment(fix_x[f], fix_y[f], vx[k],
                                                 vy[k], vx[k + 1], vy[k + 1]))
        if count == 1:
            nearest = math.hypot(fix_x[f] - vx[0], fix_y[f] - vy[0])
        to_route += nearest
    to_fixes: float = 0.0
    for k in range(count):
        nearest = np.inf
        for f in range(len(fix_x)):
            nearest = min(nearest, math.hypot(vx[k] - fix_x[f], vy[k] - fix_y[f]))
        to_fixes += nearest
    return to_route / max(len(fix_x), 1), to_fixes / count


@njit(cache=True)
def piece_error(path: np.ndarray, count: int, piece: np.ndarray,
                canonical: np.ndarray, length: np.ndarray,
                tally: np.ndarray) -> float:
    """Sum over roads of |count in path - count in piece| x road length.
    `tally` is a zeroed scratch array per road; it is left zeroed."""
    for k in range(count):
        tally[canonical[path[k]]] += 1
    for k in range(len(piece)):
        tally[piece[k]] -= 1
    total: float = 0.0
    for k in range(count):
        road: int = canonical[path[k]]
        total += abs(tally[road]) * length[road]
        tally[road] = 0
    for k in range(len(piece)):
        road = piece[k]
        total += abs(tally[road]) * length[road]
        tally[road] = 0
    return total


@njit(cache=True)
def step_lattice(sources, source_along, source_px, source_py, targets,
                 target_along, target_px, target_py, factor, fix_x, fix_y,
                 piece, with_truth, canonical, arc_from, arc_to, length,
                 heading, moves, node_x, node_y, out_start, out_arc,
                 distance, entry, turning, sharpest, touched, tally,
                 path, vx, vy):
    """Rows are source candidates, columns target candidates. A length of
    infinity means no route within the bound. Each source arc is searched
    once; the path to each target is read once and shared by the rows on
    that arc."""
    rows: int = len(sources)
    columns: int = len(targets)
    lengths = np.full((rows, columns), np.inf)
    changes = np.zeros((rows, columns))
    sharp = np.zeros((rows, columns))
    cover_fix = np.zeros((rows, columns))
    cover_route = np.zeros((rows, columns))
    delta = np.zeros((rows, columns))
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
            farthest = max(farthest, math.hypot(node_x[goal] - node_x[start],
                                                node_y[goal] - node_y[start]))
        bound: float = factor * farthest
        touched_count: int = search(source, bound, arc_to, length, heading,
                                    moves, out_start, out_arc, distance,
                                    entry, turning, sharpest, touched)
        for c in range(columns):
            path_count: int = shortest_path(source, targets[c], bound,
                                            arc_from, arc_to, distance,
                                            entry, path)
            for k in range(first, last + 1):
                fill_pair(order[k], c, source, path, path_count,
                          source_along, source_px, source_py, targets,
                          target_along, target_px, target_py, fix_x, fix_y,
                          piece, with_truth, canonical, arc_from, arc_to,
                          length, heading, moves, node_x, node_y, distance,
                          entry, turning, sharpest, tally, vx, vy, lengths,
                          changes, sharp, cover_fix, cover_route, delta)
        for k in range(touched_count):
            distance[touched[k]] = np.inf
        first = last + 1
    return lengths, changes, sharp, cover_fix, cover_route, delta


@njit(cache=True)
def shortest_path(source, target, bound, arc_from, arc_to, distance, entry,
                  path) -> int:
    """Write into `path` the arcs after the source arc on the shortest route
    to `target`, the target arc last; return how many. -1 if unreachable."""
    node: int = arc_from[target]
    if distance[node] > bound:
        return -1
    count: int = 0
    current: int = node
    while current != arc_to[source]:
        count += 1
        current = arc_from[entry[current]]
    current = node
    for k in range(count - 1, -1, -1):
        path[k] = entry[current]
        current = arc_from[entry[current]]
    path[count] = target
    return count + 1


@njit(cache=True)
def fill_pair(r, c, source, path, path_count, source_along, source_px,
              source_py, targets, target_along, target_px, target_py, fix_x,
              fix_y, piece, with_truth, canonical, arc_from, arc_to, length,
              heading, moves, node_x, node_y, distance, entry, turning,
              sharpest, tally, vx, vy, lengths, changes, sharp, cover_fix,
              cover_route, delta) -> None:
    """All measures of one candidate pair, given the path to its target."""
    target: int = targets[c]
    same_arc: bool = target == source and target_along[c] >= source_along[r]
    if not same_arc and path_count < 0:
        return
    vx[0] = source_px[r]
    vy[0] = source_py[r]
    count: int = 1
    driven: int = 0
    if same_arc:
        lengths[r, c] = target_along[c] - source_along[r]
    else:
        node: int = arc_from[target]
        degrees: float = turn_degrees(entry[node], target, heading, moves)
        lengths[r, c] = (length[source] - source_along[r] + distance[node]
                         + target_along[c])
        changes[r, c] = turning[node] + degrees
        sharp[r, c] = max(sharpest[node],
                          turn_sharpness(entry[node], target, degrees, length))
        vx[count] = node_x[arc_to[source]]
        vy[count] = node_y[arc_to[source]]
        count += 1
        for k in range(path_count - 1):
            vx[count] = node_x[arc_to[path[k]]]
            vy[count] = node_y[arc_to[path[k]]]
            count += 1
        driven = path_count
    vx[count] = target_px[c]
    vy[count] = target_py[c]
    count += 1
    to_route, to_fixes = coverage(vx, vy, count, fix_x, fix_y)
    cover_fix[r, c] = to_route
    cover_route[r, c] = to_fixes
    if with_truth:
        delta[r, c] = piece_error(path, driven, piece, canonical, length,
                                  tally)
