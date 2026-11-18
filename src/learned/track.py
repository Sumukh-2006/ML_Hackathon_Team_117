"""Layer 2: the track model (derivation §3.1, §5.3).

Raw fixes become decision points by Woeltche's four cleaning steps, with a
measured thinning tolerance. Unlike his version every decision point keeps
its time and the span of raw fixes it stands for, and every raw fix gets a
smoothed speed from a least-squares fit over a time window.
"""

from dataclasses import dataclass

import numpy as np

from dataset import Track
from learned.network import ArcGrid


@dataclass(frozen=True)
class DecisionTrack:
    """Decision points (x, y, t), each standing for raw fixes first..last;
    the raw fixes with their smoothed speeds; and per step j (point j to
    j + 1) the measures the features need."""
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    first_fix: np.ndarray
    last_fix: np.ndarray
    fix_x: np.ndarray
    fix_y: np.ndarray
    fix_speed: np.ndarray
    length: np.ndarray      # straight distance, point j to j + 1
    azimuth: np.ndarray     # degrees clockwise from north
    turn: np.ndarray        # direction change at point j + 1
    elapsed: np.ndarray     # seconds, point j to j + 1
    speed: np.ndarray       # mean smoothed speed over the step's fixes

    def point_count(self) -> int:
        return len(self.x)

    def step_fixes(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """The raw fixes from point j's last fix to point j + 1's first."""
        span: slice = slice(int(self.last_fix[j]), int(self.first_fix[j + 1]) + 1)
        return self.fix_x[span], self.fix_y[span]


def smoothed_speed(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                   half_window: float) -> np.ndarray:
    """At every fix, the speed of the least-squares line through the fixes
    within `half_window` seconds of it (Proposition 4). Prefix sums make
    every window's fit O(1)."""
    first: np.ndarray = np.searchsorted(t, t - half_window, side="left")
    last: np.ndarray = np.searchsorted(t, t + half_window, side="right")
    tc: np.ndarray = t - t[0]
    sums: list[np.ndarray] = []
    for values in [np.ones_like(tc), tc, tc * tc, x, y, tc * x, tc * y]:
        sums.append(np.concatenate([[0.0], np.cumsum(values)]))
    n, s_t, s_tt, s_x, s_y, s_tx, s_ty = [s[last] - s[first] for s in sums]
    spread: np.ndarray = s_tt - s_t * s_t / np.maximum(n, 1)
    usable: np.ndarray = (n >= 2) & (spread > 1e-9)
    safe: np.ndarray = np.where(usable, spread, 1.0)
    vx: np.ndarray = (s_tx - s_t * s_x / np.maximum(n, 1)) / safe
    vy: np.ndarray = (s_ty - s_t * s_y / np.maximum(n, 1)) / safe
    return np.where(usable, np.hypot(vx, vy), 0.0)


def without_duplicates(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the fixes that differ from the fix before them."""
    moved: np.ndarray = np.ones(len(x), dtype=bool)
    moved[1:] = (np.diff(x) != 0) | (np.diff(y) != 0)
    return np.nonzero(moved)[0]


def without_warps(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                  warp_speed: float) -> np.ndarray:
    """Indices left after dropping each fix that would need more than
    `warp_speed` from the last kept one (the later fix of the jump goes)."""
    kept: list[int] = [0]
    for i in range(1, len(x)):
        last: int = kept[-1]
        elapsed: float = float(t[i] - t[last])
        distance: float = float(np.hypot(x[i] - x[last], y[i] - y[last]))
        if elapsed > 0 and distance / elapsed > warp_speed:
            continue
        kept.append(i)
    return np.array(kept, dtype=np.int64)


def douglas_peucker(x: np.ndarray, y: np.ndarray, tolerance: float) -> np.ndarray:
    """Positions kept by Douglas-Peucker (reverse movements within the
    tolerance are dropped, the default)."""
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
        u: np.ndarray = np.clip(((px - ax) * dx + (py - ay) * dy) / squared, 0, 1)
        distance: np.ndarray = np.hypot(px - ax - u * dx, py - ay - u * dy)
        farthest: int = int(np.argmax(distance))
        if distance[farthest] > tolerance:
            keep[first + 1 + farthest] = True
            stack.append((first, first + 1 + farthest))
            stack.append((first + 1 + farthest, last))
    return np.nonzero(keep)[0]


def decision_points(track: Track, grid: ArcGrid, tolerance: float,
                    warp_speed: float, merge_tolerance: float,
                    road_share: float, search_limit: float
                    ) -> tuple[np.ndarray, ...]:
    """x, y, t, first fix and last fix of every decision point. The median
    merge collapses runs of consecutive points within the run's first
    point's tolerance (adaptive: half the distance to the nearest road when
    larger) into their median position and median time."""
    kept: np.ndarray = without_duplicates(track.x, track.y)
    kept = kept[without_warps(track.x[kept], track.y[kept], track.t[kept],
                              warp_speed)]
    kept = kept[douglas_peucker(track.x[kept], track.y[kept], tolerance)]
    x, y, t = track.x[kept], track.y[kept], track.t[kept]

    out: list[tuple[float, float, float, int, int]] = []
    i: int = 0
    while i < len(kept):
        road: float = grid.nearest_road(float(x[i]), float(y[i]), search_limit)
        radius: float = max(merge_tolerance, road_share * road)
        j: int = i
        while j + 1 < len(kept) and np.hypot(x[j + 1] - x[i], y[j + 1] - y[i]) <= radius:
            j += 1
        out.append((float(np.median(x[i:j + 1])), float(np.median(y[i:j + 1])),
                    float(np.median(t[i:j + 1])), int(kept[i]), int(kept[j])))
        i = j + 1
    columns = list(zip(*out))
    return (np.array(columns[0]), np.array(columns[1]), np.array(columns[2]),
            np.array(columns[3], dtype=np.int64),
            np.array(columns[4], dtype=np.int64))


def decision_track(track: Track, grid: ArcGrid, tolerance: float,
                   half_window: float, warp_speed: float,
                   merge_tolerance: float, road_share: float,
                   search_limit: float) -> DecisionTrack:
    x, y, t, first_fix, last_fix = decision_points(
        track, grid, tolerance, warp_speed, merge_tolerance, road_share,
        search_limit)
    fix_speed: np.ndarray = smoothed_speed(track.t, track.x, track.y,
                                           half_window)
    dx, dy = np.diff(x), np.diff(y)
    heading: np.ndarray = np.degrees(np.arctan2(dx, dy)) % 360.0
    turn: np.ndarray = (heading[1:] - heading[:-1] + 180.0) % 360.0 - 180.0

    speed: np.ndarray = np.zeros(len(x) - 1)
    for j in range(len(x) - 1):
        span: slice = slice(int(last_fix[j]), int(first_fix[j + 1]) + 1)
        speed[j] = float(fix_speed[span].mean())
    return DecisionTrack(x, y, t, first_fix, last_fix, track.x, track.y,
                         fix_speed, np.hypot(dx, dy), heading, turn,
                         np.diff(t), speed)
