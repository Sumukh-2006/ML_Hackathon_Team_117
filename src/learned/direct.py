"""Layer 5, second version: direct minimisation of the decoded error.

Woeltche set his five weights by running his matcher about 1000 times
under Bayesian optimisation of its accuracy on the Hof data (thesis 5.2,
Table 5.4): the objective is the score of the routes the weights actually
produce, and every weight is a non-negative penalty. The ramp loss (Do et
al. 2008; McAllester and Keshet 2011), min_y C_w - min_y [C_w - Delta],
tends to exactly Delta(y_hat(w)) as w is scaled up (the direct loss of
McAllester, Hazan and Keshet 2010). Both point at the error of what w decodes to. The step-error version,
sum_t Delta(y_hat_t(w)) / sum_t L_t, bounds 1 - CF (Theorem 2) but
measured 3.7% against a real 1.4% on train, and minimising it moved the
real score by only 0.11 points; so, as Woeltche does, the objective is
the real one: the weighted 1 - CF of the driven routes. With 14 weights
it can be minimised directly. It is
piecewise constant in w, so the search is derivative-free: a coordinate
pattern search over non-negative weights (OWN CHOICE: scikit-optimize's
gp_minimize, which Woeltche used, is not installed).
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dataset import Folder
from learned.evaluate import score, weighted_correct_fraction
from learned.hyperparameters import (SEARCH_FACTORS, SEARCH_SHARES,
                                     SEARCH_SWEEPS)
from learned.train import TrainingSet, decode_all


class RealError:
    """Weighted 1 - CF of the routes `weights` drive on the given folders'
    own lattices, with one worker pool kept for the whole search."""

    def __init__(self, folders: list[Folder], paths: list[Path], pool: Pool) -> None:
        self.folders: list[Folder] = folders
        self.paths: list[Path] = paths
        self.pool: Pool = pool

    def __call__(self, weights: np.ndarray) -> float:
        jobs: list[tuple] = [(f, p, weights) for f, p in zip(self.folders, self.paths)]
        return 1.0 - weighted_correct_fraction(self.pool.map(score, jobs))


def search(objective, data: TrainingSet, start: np.ndarray,
           frozen: np.ndarray, log) -> np.ndarray:
    """Coordinate pattern search from `start`. A weight above 0 tries
    itself times SEARCH_FACTORS (0 included); a weight at 0 tries values
    whose cost per metre of route is SEARCH_SHARES of the distance term's
    at `start`. Weights in `frozen` stay at their start. Sweeps repeat
    until one changes nothing."""
    w: np.ndarray = start.copy()
    best: float = objective(w)
    log(f"start: error {best:.6f}")
    per_metre: np.ndarray = feature_per_metre(data, start)
    distance_share: float = start[0] * per_metre[0]
    for sweep in range(SEARCH_SWEEPS):
        improved: bool = False
        for k in range(len(w)):
            if frozen[k]:
                continue
            if w[k] > 0:
                trials: list[float] = [w[k] * f for f in SEARCH_FACTORS]
            else:
                trials = [share * distance_share / max(per_metre[k], 1e-12)
                          for share in SEARCH_SHARES]
            for value in trials:
                trial: np.ndarray = w.copy()
                trial[k] = value
                error: float = objective(trial)
                if error < best - 1e-9:
                    best, w, improved = error, trial, True
        log(f"sweep {sweep}: error {best:.6f}, "
            f"w {np.round(w, 4).tolist()}")
        if not improved:
            break
    return w


def feature_per_metre(data: TrainingSet, weights: np.ndarray) -> np.ndarray:
    """|phi_k| per metre of true route along the labellings `weights`
    decodes to: the scale on which a new weight's trial values are set."""
    features, _ = decode_all(data.batches, weights, 0.0, 1.0)
    return np.abs(features).sum(axis=0) / data.true_length.sum()
