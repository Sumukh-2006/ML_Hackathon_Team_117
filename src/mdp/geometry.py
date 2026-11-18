"""Angles as map_matching_2 defines them (thesis Defs. 3.6-3.8).

An azimuth is measured clockwise from north, in degrees in [0, 360). The
difference of two azimuths is wrapped into [-180, 180), so 355 and 5 differ
by 10. A direction change (or turn) is the difference between the azimuths
of two consecutive segments, so 0 means straight on.
"""

from dataclasses import dataclass

import numpy as np


def azimuth(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Azimuth of the vector (dx east, dy north)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """b - a wrapped into [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class TrackShape:
    """The simplified track's points and the measures of its segments:
    segment i joins point i to point i + 1, and turn i is the direction
    change at point i + 1, from segment i to segment i + 1."""
    x: np.ndarray
    y: np.ndarray
    length: np.ndarray
    azimuth: np.ndarray
    turn: np.ndarray

    @classmethod
    def from_points(cls, x: np.ndarray, y: np.ndarray) -> "TrackShape":
        dx: np.ndarray = np.diff(x)
        dy: np.ndarray = np.diff(y)
        heading: np.ndarray = azimuth(dx, dy)
        return cls(x, y, np.hypot(dx, dy), heading,
                   angle_diff(heading[:-1], heading[1:]))

    def point_count(self) -> int:
        return len(self.x)
