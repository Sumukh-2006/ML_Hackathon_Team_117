"""Longitude and latitude to metres, for local geometry and for lengths."""

from dataclasses import dataclass

import numpy as np

# NOTE: a physical constant, not a hyperparameter: the IUGG mean Earth radius.
EARTH_RADIUS: float = 6371008.8


@dataclass(frozen=True)
class Projection:
    """Equirectangular projection around one origin, in metres.

    x grows east and y grows north. The east-west scale is exact only at
    the origin's latitude; the worst error over the corpus is 1.05% (folder
    00000086, 0.51 degrees of latitude), so the plane is used for local
    geometry (a fix's distance to a road) and never for route lengths.
    """
    origin_lon: float
    origin_lat: float

    @classmethod
    def around(cls, lon: np.ndarray, lat: np.ndarray) -> "Projection":
        """The projection centred on the mean of the given positions."""
        return cls(float(np.radians(lon.mean())),
                   float(np.radians(lat.mean())))

    def to_xy(self, lon: np.ndarray,
              lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Plane coordinates of positions given in degrees."""
        east: np.ndarray = np.radians(lon) - self.origin_lon
        north: np.ndarray = np.radians(lat) - self.origin_lat
        x: np.ndarray = EARTH_RADIUS * east * np.cos(self.origin_lat)
        y: np.ndarray = EARTH_RADIUS * north
        return x, y


def great_circle(lon_a: np.ndarray, lat_a: np.ndarray,
                 lon_b: np.ndarray, lat_b: np.ndarray) -> np.ndarray:
    """Haversine distance in metres between positions given in degrees."""
    phi_a: np.ndarray = np.radians(lat_a)
    phi_b: np.ndarray = np.radians(lat_b)
    half_dphi: np.ndarray = (phi_b - phi_a) / 2
    half_dlambda: np.ndarray = np.radians(lon_b - lon_a) / 2

    h: np.ndarray = (np.sin(half_dphi) ** 2
                     + np.cos(phi_a) * np.cos(phi_b)
                     * np.sin(half_dlambda) ** 2)
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.minimum(h, 1.0)))
