"""Trajectory simplification (paper 2.1, thesis 3.1.1), in its four steps:
spatial duplicates, warps, Douglas-Peucker, median merge."""

import numpy as np

from dataset import Track


def without_duplicates(x: np.ndarray, y: np.ndarray,
                       t: np.ndarray) -> np.ndarray:
    """Indices of the points that differ from the point before them."""
    moved: np.ndarray = np.ones(len(x), dtype=bool)
    moved[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    return np.nonzero(moved)[0]


def without_warps(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                  speed: float) -> np.ndarray:
    """Indices of the points left after dropping each point that would need
    more than `speed` to reach from the last kept point.
    OWN READING: the thesis keeps "the shortest remaining segment"; here
    the later point of the impossible jump is the one dropped."""
    kept: list[int] = [0]
    for i in range(1, len(x)):
        last: int = kept[-1]
        elapsed: float = float(t[i] - t[last])
        distance: float = float(np.hypot(x[i] - x[last], y[i] - y[last]))
        if elapsed > 0 and distance / elapsed > speed:
            continue
        kept.append(i)
    return np.array(kept, dtype=np.int64)


def douglas_peucker(x: np.ndarray, y: np.ndarray,
                    tolerance: float) -> np.ndarray:
    """Indices Douglas-Peucker keeps: the ends, and recursively the point
    farthest from the chord while it lies more than `tolerance` away. Points
    of a reverse movement within the tolerance go too (the default)."""
    keep: np.ndarray = np.zeros(len(x), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(x) - 1)]
    while len(stack) > 0:
        first, last = stack.pop()
        if last - first < 2:
            continue
        ax, ay, bx, by = x[first], y[first], x[last], y[last]
        px, py = x[first + 1:last], y[first + 1:last]
        dx, dy = bx - ax, by - ay
        squared: float = max(dx * dx + dy * dy, 1e-12)
        u: np.ndarray = np.clip(((px - ax) * dx + (py - ay) * dy) / squared,
                                0, 1)
        distance: np.ndarray = np.hypot(px - ax - u * dx, py - ay - u * dy)
        farthest: int = int(np.argmax(distance))
        if distance[farthest] > tolerance:
            keep[first + 1 + farthest] = True
            stack.append((first, first + 1 + farthest))
            stack.append((first + 1 + farthest, last))
    return np.nonzero(keep)[0]


def median_merge(x: np.ndarray, y: np.ndarray,
                 tolerance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each run of directly connected points that lie within the
    first point's tolerance of it into the median of the run's coordinates
    (thesis Def. 3.1). A later revisit of the place is not merged."""
    merged_x: list[float] = []
    merged_y: list[float] = []
    i: int = 0
    while i < len(x):
        j: int = i
        while (j + 1 < len(x)
               and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= tolerance[i]):
            j += 1
        merged_x.append(float(np.median(x[i:j + 1])))
        merged_y.append(float(np.median(y[i:j + 1])))
        i = j + 1
    return np.array(merged_x), np.array(merged_y)


def simplified(track: Track, warp_speed: float,
               tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """The first three steps: duplicates, warps and Douglas-Peucker. Median
    merge needs the road network for its adaptive radius and runs after."""
    keep: np.ndarray = without_duplicates(track.x, track.y, track.t)
    x, y, t = track.x[keep], track.y[keep], track.t[keep]
    keep = without_warps(x, y, t, warp_speed)
    x, y = x[keep], y[keep]
    keep = douglas_peucker(x, y, tolerance)
    return x[keep], y[keep]
