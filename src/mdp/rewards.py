"""The MDP's rewards, as map_matching_2's single.hpp computes them (the
user chose the code over the paper's Eq. 3.3).

For the step from point j (candidate c) to point j + 1 (candidate c'):
    r = -[ w_d (d_c + d_c')  (+ w_d * 2 d_c' when j + 1 is the last point)
         + w_l |l_t - l_r| * (l_r / l_t)
         + w_a |azimuth(track j) - azimuth(route)|
         + w_t |turn(track at j) - turn(track j-1 -> route)|     (j >= 1)
         + w_adc (direction changes along the route) ]
A missing route costs the fail error and skips to a fresh start at j + 1;
a route over SKIP_ROUTE_FACTOR times the track segment keeps its reward but
skips too. Two candidates at the same position on equal points cost 0.
"""

import numpy as np

from mdp.candidates import Candidates
from mdp.geometry import TrackShape, angle_diff, azimuth
from mdp.hyperparameters import (AZIMUTH_WEIGHT, DIRECTION_WEIGHT,
                                 DISTANCE_WEIGHT, FAIL_BASE, FAIL_WEIGHT_MAX,
                                 FAIL_WEIGHT_MIN, FAIL_WEIGHT_SIGMA,
                                 LAST_EXTRA_DISTANCE, LENGTH_WEIGHT,
                                 ROUTING_MAX_DISTANCE_FACTOR,
                                 SKIP_ROUTE_FACTOR, TURN_WEIGHT)
from mdp.routing import Router


def fail_errors(shape: TrackShape) -> np.ndarray:
    """single.hpp _fail_error for every point j: the base, 3 x (track length
    + total absolute turning), plus the sum over points i of
    (2 + 3 exp(-(i - j)^2 / 18)) x (segment length i + |turn i|).
    NOTE: the Gaussian part is summed within 40 points of j; beyond that
    its weight is below 1e-38 of the rest."""
    n: int = shape.point_count()
    value: np.ndarray = np.zeros(n)
    value[:n - 1] += shape.length
    value[:max(n - 2, 0)] += np.abs(shape.turn)
    base: float = FAIL_BASE * float(shape.length.sum()
                                    + np.abs(shape.turn).sum())
    offsets: np.ndarray = np.arange(-40, 41)
    kernel: np.ndarray = np.exp(-offsets ** 2 / (2 * FAIL_WEIGHT_SIGMA ** 2))
    local: np.ndarray = (FAIL_WEIGHT_MIN * value.sum()
                         + (FAIL_WEIGHT_MAX - FAIL_WEIGHT_MIN)
                         * np.convolve(value, kernel, mode="same"))
    return base + local


def step_rewards(j: int, current: Candidates, following: Candidates,
                 shape: TrackShape, router: Router, fail: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Rewards from every candidate of point j (rows) to every candidate of
    point j + 1 (columns), and which of those transitions skip."""
    rewards: np.ndarray = np.zeros((current.count(), following.count()))
    skips: np.ndarray = np.zeros((current.count(), following.count()),
                                 dtype=bool)
    last: bool = j + 2 == shape.point_count()
    track_length: float = float(shape.length[j])
    for c in range(current.count()):
        arc: int = int(current.arc[c])
        bound: float = router.bound(arc, following.arc,
                                    ROUTING_MAX_DISTANCE_FACTOR)
        lengths, changes = router.routes(arc, float(current.along[c]),
                                         following.arc, following.along,
                                         bound)
        row, skip = row_rewards(j, c, current, following, shape, lengths,
                                changes, track_length, last)
        missing: np.ndarray = ~np.isfinite(lengths)
        row[missing] = -fail
        skip[missing] = True
        rewards[c] = row
        skips[c] = skip
    return rewards, skips


def row_rewards(j: int, c: int, current: Candidates, following: Candidates,
                shape: TrackShape, lengths: np.ndarray, changes: np.ndarray,
                track_length: float, last: bool
                ) -> tuple[np.ndarray, np.ndarray]:
    """The rewards from one candidate, before missing routes are handled."""
    finite: np.ndarray = np.where(np.isfinite(lengths), lengths, 0.0)
    penalty: np.ndarray = DISTANCE_WEIGHT * (following.distance
                                             + current.distance[c])
    if last:
        penalty = penalty + DISTANCE_WEIGHT * LAST_EXTRA_DISTANCE * following.distance

    difference: np.ndarray = np.abs(track_length - finite)
    if track_length > 0:
        factor: np.ndarray = np.where(finite > 0, finite / track_length, 1.0)
    else:
        factor = np.ones(len(finite))
    penalty = penalty + LENGTH_WEIGHT * difference * factor
    skip: np.ndarray = (finite > 0) & (track_length > 0) & (factor > SKIP_ROUTE_FACTOR)

    dx: np.ndarray = following.px - current.px[c]
    dy: np.ndarray = following.py - current.py[c]
    has_heading: np.ndarray = (finite > 0) & (np.hypot(dx, dy) > 0)
    heading: np.ndarray = azimuth(dx, dy)
    off_course: np.ndarray = np.abs(angle_diff(shape.azimuth[j], heading))
    penalty = penalty + np.where(has_heading, AZIMUTH_WEIGHT * off_course, 0)
    if j >= 1:
        route_turn: np.ndarray = angle_diff(shape.azimuth[j - 1], heading)
        mismatch: np.ndarray = np.abs(angle_diff(shape.turn[j - 1], route_turn))
        penalty = penalty + np.where(has_heading, TURN_WEIGHT * mismatch, 0)
    penalty = penalty + DIRECTION_WEIGHT * changes

    standing: np.ndarray = (track_length == 0) & (finite == 0)
    return np.where(standing, 0.0, -penalty), skip
