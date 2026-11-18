"""Key points by the Douglas-Peucker algorithm (section 4.2.1), applied to
the fixes' longitude and latitude as the paper's tolerance is in degrees."""

import numpy as np

from dataset import Track


def key_points(track: Track, tolerance: float) -> list[int]:
    """Indices of the fixes Douglas-Peucker keeps: the first and the last,
    and recursively the farthest fix from the chord whenever it lies more
    than `tolerance` away."""
    keep: np.ndarray = np.zeros(track.fix_count(), dtype=bool)
    keep[0] = True
    keep[-1] = True
    stack: list[tuple[int, int]] = [(0, track.fix_count() - 1)]
    while len(stack) > 0:
        first, last = stack.pop()
        if last - first < 2:
            continue
        farthest, distance = farthest_from_chord(track, first, last)
        if distance > tolerance:
            keep[farthest] = True
            stack.append((first, farthest))
            stack.append((farthest, last))
    return np.nonzero(keep)[0].tolist()


def farthest_from_chord(track: Track, first: int,
                        last: int) -> tuple[int, float]:
    """The fix strictly between first and last farthest from the segment
    joining them, and its distance, in degrees."""
    ax, ay = track.lon[first], track.lat[first]
    bx, by = track.lon[last], track.lat[last]
    px: np.ndarray = track.lon[first + 1:last]
    py: np.ndarray = track.lat[first + 1:last]
    dx, dy = bx - ax, by - ay
    squared: float = max(dx * dx + dy * dy, 1e-24)
    u: np.ndarray = np.clip(((px - ax) * dx + (py - ay) * dy) / squared, 0, 1)
    distance: np.ndarray = np.hypot(px - ax - u * dx, py - ay - u * dy)
    best: int = int(np.argmax(distance))
    return first + 1 + best, float(distance[best])
