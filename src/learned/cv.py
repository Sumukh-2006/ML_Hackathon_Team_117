"""Grouped cross-validation (derivation §8): a folder's own track and its
noisy copies are always on the same side of a split. Every score is the
weighted correct fraction of held-out folders' own (organisers') tracks.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dataset import Folder
from learned.evaluate import score
from learned.hyperparameters import (EXCLUDED_COPIES, FOLDS, FOLD_SEED,
                                     WORKERS)
from learned.train import TrainingSet, train, training_set, weight_scale


def folds(names: list[str]) -> list[list[str]]:
    order: np.ndarray = np.random.default_rng(FOLD_SEED).permutation(sorted(names))
    return [sorted(order[k::FOLDS].tolist()) for k in range(FOLDS)]


def lattice_paths(root: Path, names: list[str], copies: bool) -> list[Path]:
    """Own lattices of `names`, and their copies when asked (never those
    of EXCLUDED_COPIES)."""
    paths: list[Path] = []
    for name in names:
        paths.append(root / f"{name}_own.npz")
        if copies and name not in EXCLUDED_COPIES:
            paths.extend(sorted(root.glob(f"{name}_copy*.npz")))
    return paths


def held_out_rows(folders: dict[str, Folder], root: Path, names: list[str],
                  weights: np.ndarray) -> list[dict]:
    jobs: list[tuple] = [(folders[n], root / f"{n}_own.npz", weights) for n in names]
    with Pool(WORKERS) as pool:
        return pool.map(score, jobs)


def fit(data: TrainingSet, start: np.ndarray, strength: float,
        fixed_zero: np.ndarray, log) -> np.ndarray:
    """The structured hinge (first version; kept for comparison)."""
    s: np.ndarray = weight_scale(data, start)
    return train(data, start, strength, s, fixed_zero, log)


def fit_direct(folders: dict[str, Folder], root: Path, names: list[str],
               start: np.ndarray, scale: np.ndarray, frozen: np.ndarray,
               log) -> np.ndarray:
    """Direct search on the real error of `names`' own tracks."""
    from learned.direct import RealError, search
    paths: list[Path] = [root / f"{n}_own.npz" for n in names]
    data: TrainingSet = training_set(paths, scale)
    with Pool(WORKERS) as pool:
        objective = RealError([folders[n] for n in names], paths, pool)
        return search(objective, data, start, frozen, log)
