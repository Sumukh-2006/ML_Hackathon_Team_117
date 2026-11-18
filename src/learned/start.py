"""The starting weights w0 (derivation §6): Woeltche's for his terms, 0 for
the new ones.

His fresh starts cost 0.8 x 3 x distance at the first point and 0.8 x 0.1
x distance after a skip. His skip has no single weight: a missing route
costs the fail error of single.hpp, 3 x (track length + total absolute
turning) plus a local term, which grows with the track. Our skip weight
starts at the median of that fail error over the train tracks' points
(OWN CHOICE: a constant has to stand in for a cost that varies).
"""

import numpy as np

from learned.features import STEP_FEATURES
from learned.track import DecisionTrack

WOELTCHE_STEP: dict[str, float] = {
    "distance": 0.8, "last_distance": 0.8, "length": 0.3, "heading": 0.5,
    "turn": 0.5, "turning": 0.6}
FIRST_DISTANCE: float = 0.8 * 3.0
RESTART_DISTANCE: float = 0.8 * 0.1


def fail_errors(track: DecisionTrack) -> np.ndarray:
    """single.hpp's fail error at every point: 3 x (track length + total
    |turn|) plus, summed over points i, (2 + 3 exp(-(i - j)^2 / 18)) x
    (segment i's length + |turn i|), within 40 points of j."""
    n: int = track.point_count()
    value: np.ndarray = np.zeros(n)
    value[:n - 1] += track.length
    value[:max(n - 2, 0)] += np.abs(track.turn)
    base: float = 3.0 * float(track.length.sum() + np.abs(track.turn).sum())
    offsets: np.ndarray = np.arange(-40, 41)
    kernel: np.ndarray = np.exp(-offsets ** 2 / (2 * 3.0 ** 2))
    return base + 2.0 * value.sum() + 3.0 * np.convolve(value, kernel, mode="same")


def starting_weights(tracks: list[DecisionTrack]) -> np.ndarray:
    step: list[float] = [WOELTCHE_STEP.get(name, 0.0) for name in STEP_FEATURES]
    skip: float = float(np.median(np.concatenate([fail_errors(t) for t in tracks])))
    return np.array(step + [skip, FIRST_DISTANCE, RESTART_DISTANCE])
