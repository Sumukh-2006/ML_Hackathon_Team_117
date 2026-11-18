"""The GPS track of one corpus folder."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.projection import Projection
from dataset.table import DatasetError, read_table


@dataclass(frozen=True)
class Track:
    """Fixes in time order, on the road graph's plane.

    NOTE: `.track` lines are `lon lat t`, longitude first, with t in
    seconds. The positions are projected with the graph's projection, so a
    fix and a road share one coordinate system.
    """
    lon: np.ndarray
    lat: np.ndarray
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray

    @classmethod
    def from_file(cls, path: Path, projection: Projection) -> "Track":
        """Read a `.track` file; time must strictly increase."""
        lon, lat, t = read_table(path, [float, float, float])
        check_time_order(path, t)
        x, y = projection.to_xy(lon, lat)
        return cls(lon, lat, x, y, t)

    def fix_count(self) -> int:
        return len(self.t)


def check_time_order(path: Path, t: np.ndarray) -> None:
    """Fixes must come in strictly increasing time."""
    bad: np.ndarray = np.nonzero(np.diff(t) <= 0)[0]
    if len(bad) > 0:
        line: int = int(bad[0]) + 2
        raise DatasetError(
            f"{path}:{line}: time {t[bad[0] + 1]} does not come after "
            f"{t[bad[0]]} ({len(bad)} such places in the file)")
