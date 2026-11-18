"""Features of every step (derivation §5), from the kernel's arrays.

A step from candidate c at point j to c' at point j + 1 is a skip when no
route exists or the route is more than SKIP_ROUTE_FACTOR times the straight
distance between the points. The stored arrays are the raw step features
and the skip mask; the cost (decode.py) gives a skip its own weight instead
of the step features. Fresh starts (the first point, and the point after a
skip) are priced by the candidate's distance, with their own weights.
"""

import numpy as np

from learned.candidates import Candidates
from learned.track import DecisionTrack

STEP_FEATURES: list[str] = [
    "distance", "last_distance", "length", "heading", "turn", "turning",
    "cover_fix", "cover_route", "speed", "too_fast", "turn_force"]
# The weights: one per step feature, then the skip, the first point's
# distance and the distance at a fresh start after a skip.
WEIGHTS: list[str] = STEP_FEATURES + ["skip", "first_distance",
                                      "restart_distance"]


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (b - a + 180.0) % 360.0 - 180.0


def step_features(j: int, track: DecisionTrack, current: Candidates,
                  following: Candidates, lengths: np.ndarray,
                  changes: np.ndarray, sharp: np.ndarray,
                  cover_fix: np.ndarray, cover_route: np.ndarray,
                  too_fast_speed: float, skip_factor: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The (K, K', 11) raw feature array of step j and its skip mask."""
    straight: float = float(track.length[j])
    routed: np.ndarray = np.where(np.isfinite(lengths), lengths, 0.0)
    skip: np.ndarray = ~np.isfinite(lengths)
    if straight > 0:
        skip |= routed > skip_factor * straight

    out: np.ndarray = np.zeros(lengths.shape + (len(STEP_FEATURES),))
    out[..., 0] = current.distance[:, None] + following.distance[None, :]
    if j == track.point_count() - 2:
        out[..., 1] = 2.0 * following.distance[None, :]
    factor: np.ndarray = np.ones_like(routed)
    if straight > 0:
        factor = np.where(routed > 0, routed / straight, 1.0)
    out[..., 2] = np.abs(straight - routed) * factor

    dx: np.ndarray = following.px[None, :] - current.px[:, None]
    dy: np.ndarray = following.py[None, :] - current.py[:, None]
    has_heading: np.ndarray = (routed > 0) & (np.hypot(dx, dy) > 0)
    heading: np.ndarray = np.degrees(np.arctan2(dx, dy)) % 360.0
    out[..., 3] = np.where(has_heading,
                           np.abs(angle_diff(track.azimuth[j], heading)), 0.0)
    if j >= 1:
        route_turn: np.ndarray = angle_diff(track.azimuth[j - 1], heading)
        mismatch: np.ndarray = np.abs(angle_diff(track.turn[j - 1], route_turn))
        out[..., 4] = np.where(has_heading, mismatch, 0.0)
    out[..., 5] = changes
    out[..., 6] = cover_fix
    out[..., 7] = cover_route

    elapsed: float = float(track.elapsed[j])
    speed: float = float(track.speed[j])
    out[..., 8] = np.abs(routed - speed * elapsed)
    if elapsed > 0:
        out[..., 9] = np.maximum(0.0, routed / elapsed - too_fast_speed)
    out[..., 10] = speed * speed * sharp
    return out, skip


def skip_errors(following: Candidates, piece: np.ndarray, canonical: np.ndarray,
                length: np.ndarray) -> np.ndarray:
    """Delta_j of a skip into each target: its route is just the target arc,
    so the error is the piece length, minus the target's length when the
    piece drives that road, plus it otherwise."""
    piece_length: float = float(length[piece].sum())
    roads: np.ndarray = canonical[following.arc]
    on_piece: np.ndarray = np.isin(roads, piece)
    return np.where(on_piece, piece_length - length[roads],
                    piece_length + length[roads])


def start_errors(first: Candidates, piece: np.ndarray, canonical: np.ndarray,
                 length: np.ndarray) -> np.ndarray:
    """Delta_0 of every candidate of the first point, the same way."""
    return skip_errors(first, piece, canonical, length)
