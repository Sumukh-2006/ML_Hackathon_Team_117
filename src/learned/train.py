"""Layer 5: training by the structured hinge (derivation §6), with the
1-slack cutting-plane method of structural SVMs (Joachims, Finley and Yu,
Machine Learning 77, 2009).

For track t with true length L_t, oracle labelling y_t and weights w, the
loss is
    h_t(w) = max_y [ Delta(y) - Delta(y_t) + w . (phi(y_t) - phi(y)) ] >= 0,
and training minimises the derivation's J (eq. objective), divided by the
total true length L = sum_t L_t so that lambda does not depend on how many
tracks there are:
    J(w) = lambda sum_k s_k^2 (w_k - w0_k)^2 + sum_t h_t(w) / L.
Being a sum of metres over a sum of metres, the hinge term bounds the
length-weighted 1 - CF the benchmark reports (Theorem 4, summed).
The derivation names Adam; a proximal bundle method (Kiwiel, Math.
Programming 46, 1990) reaches the same unique minimiser in far fewer
decoding passes and says when it has (OWN CHOICE). Each pass decodes every
track loss-augmented at a trial w; the maximisers give one plane
a + g . w below sum_t h_t / L, exact at w. The planes give a lower model of
the hinge term; the next trial minimises lambda R(w) + model(w) plus a pull
mu/2 sum s^2 (w - c)^2 toward the best point c so far. A trial that lowers
J by at least a tenth of what the model predicted becomes the new c. It
stops when the predicted decrease is within TOLERANCE of J(c).

The oracle y_t is a labelling of least Delta; the derivation allows any.
Many tie: where a step's true piece is only the next candidate's arc, a
skip into that candidate errs exactly as little as driving the route. An
oracle that breaks ties by index skips often, and training then has to
make skips cheap. So y_t is the least-Delta labelling that is cheapest
under the current w (minimise Delta + epsilon C_w, in float64), and is
chosen again whenever w changes: the CCCP scheme for latent structural
SVMs (Yu and Joachims, ICML 2009). Each inner problem is the convex one of
Theorem 3; the outer loop stops when the oracles stop changing.

s_k puts every weight on one scale: the mean of |phi_k| per metre of true
route over the oracle and the starting labellings (OWN CHOICE: the
derivation writes the plain ||w - w0||^2; any positive scaling keeps the
minimiser unique and Theorem 4 unchanged).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from learned.batch import Batch, Decoded, build, decode
from learned.hyperparameters import (BATCH_PAIRS, CUTTING_ROUNDS,
                                     CUTTING_TOLERANCE, DUAL_ITERATIONS,
                                     ORACLE_ROUNDS, ORACLE_TIE)


@dataclass
class TrainingSet:
    batches: list[Batch]          # pinned host memory
    true_length: np.ndarray       # (tracks,)
    oracle_features: np.ndarray   # (tracks, W), set by choose_oracle
    oracle_error: np.ndarray      # (tracks,)


def pair_count(path: Path) -> int:
    sizes: np.ndarray = np.load(path)["sizes"].astype(np.int64)
    return int((sizes[:-1] * sizes[1:]).sum())


def batches_of(paths: list[Path], limit: int) -> list[list[Path]]:
    """Consecutive groups of lattices with at most `limit` pairs each (a
    single larger lattice gets a group of its own)."""
    groups: list[list[Path]] = [[]]
    filled: int = 0
    for path in paths:
        pairs: int = pair_count(path)
        if filled + pairs > limit and len(groups[-1]) > 0:
            groups.append([])
            filled = 0
        groups[-1].append(path)
        filled += pairs
    return groups


def decode_all(batches: list[Batch], weights: np.ndarray, error_weight: float,
               cost_weight: float, dtype: torch.dtype = torch.float32
               ) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    w: torch.Tensor = torch.tensor(weights, dtype=torch.float64)
    for batch in batches:
        on_gpu: Batch = batch.to("cuda")
        decoded: Decoded = decode(on_gpu, w, error_weight, cost_weight, dtype)
        features.append(decoded.features.cpu().numpy())
        errors.append(decoded.error.cpu().numpy())
        del on_gpu
    return np.concatenate(features), np.concatenate(errors)


def training_set(paths: list[Path], scale: np.ndarray) -> TrainingSet:
    groups: list[list[Path]] = batches_of(paths, BATCH_PAIRS)
    batches: list[Batch] = [build(g, scale, "pinned") for g in groups]
    lengths: np.ndarray = np.concatenate([b.true_length.numpy() for b in batches])
    empty: np.ndarray = np.zeros(0)
    return TrainingSet(batches, lengths, empty, empty)


def choose_oracle(data: TrainingSet, weights: np.ndarray) -> None:
    """y_t = argmin Delta + epsilon C_w, with epsilon so small that the cost
    term of a whole track stays under ORACLE_TIE metres (measured on the
    labellings w decodes to)."""
    decoded, _ = decode_all(data.batches, weights, 0.0, 1.0)
    magnitude: float = float((np.abs(decoded) @ np.abs(weights)).max())
    epsilon: float = ORACLE_TIE / max(magnitude, 1e-12)
    features, error = decode_all(data.batches, weights, -1.0, epsilon,
                                 torch.float64)
    data.oracle_features = features
    data.oracle_error = error


def weight_scale(data: TrainingSet, start: np.ndarray) -> np.ndarray:
    """s_k: |phi_k| per metre of true route, over the oracle labellings and
    the labellings at w0 (choosing the oracles at w0)."""
    choose_oracle(data, start)
    at_start, _ = decode_all(data.batches, start, 0.0, 1.0)
    s: np.ndarray = (np.abs(data.oracle_features).sum(axis=0)
                     + np.abs(at_start).sum(axis=0)) / (2 * data.true_length.sum())
    floor: float = 1e-3 * float(s.max())
    return np.maximum(s, floor)


def plane(data: TrainingSet, weights: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The plane a + g . w that the maximisers at `weights` give, and each
    track's hinge h_t (metres) there."""
    features, error = decode_all(data.batches, weights, 1.0, 1.0)
    total: float = float(data.true_length.sum())
    a_t: np.ndarray = error - data.oracle_error
    g_t: np.ndarray = data.oracle_features - features
    hinge: np.ndarray = a_t + g_t @ weights
    return float(a_t.sum()) / total, g_t.sum(axis=0) / total, hinge


def solve_planes(a: np.ndarray, g: np.ndarray, centre: np.ndarray,
                 strength: float, s: np.ndarray) -> np.ndarray:
    """argmin_w strength sum s^2 (w - centre)^2 + max_i (a_i + g_i . w),
    through its dual over the simplex of plane weights alpha:
        w(alpha) = centre - (g^T alpha) / (2 strength s^2)
        D(alpha) = alpha . (a + g centre) - |g^T alpha / s|^2 / (4 strength),
    maximised by accelerated projected gradient (FISTA)."""
    b: np.ndarray = a + g @ centre
    scaled: np.ndarray = g / s
    gram: np.ndarray = scaled @ scaled.T / (2 * strength)
    lipschitz: float = max(float(np.linalg.eigvalsh(gram).max()), 1e-12)
    alpha: np.ndarray = np.full(len(a), 1.0 / len(a))
    ahead: np.ndarray = alpha.copy()
    momentum: float = 1.0
    for iteration in range(DUAL_ITERATIONS):
        gradient: np.ndarray = b - gram @ ahead
        moved: np.ndarray = onto_simplex(ahead + gradient / lipschitz)
        following: float = (1 + np.sqrt(1 + 4 * momentum * momentum)) / 2
        ahead = moved + (momentum - 1) / following * (moved - alpha)
        alpha, momentum = moved, following
        # Stationary when a plain projected-gradient step from alpha itself
        # stays put (the accelerated iterate can pause without being there).
        if iteration % 50 == 49:
            step: np.ndarray = onto_simplex(alpha + (b - gram @ alpha) / lipschitz)
            if np.abs(step - alpha).max() < 1e-12:
                break
    return centre - (g.T @ alpha) / (2 * strength * s * s)


def onto_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {x >= 0, sum x = 1} (Duchi et al. 2008)."""
    u: np.ndarray = np.sort(v)[::-1]
    cumulative: np.ndarray = np.cumsum(u) - 1
    index: np.ndarray = np.arange(1, len(v) + 1)
    rho: int = int(np.nonzero(u - cumulative / index > 0)[0][-1])
    return np.maximum(v - cumulative[rho] / (rho + 1), 0.0)


def train(data: TrainingSet, start: np.ndarray, strength: float,
          s: np.ndarray, fixed_zero: np.ndarray, log) -> np.ndarray:
    """CCCP from w0 = `start`: choose the oracles at w, solve the convex
    problem with them by cutting planes, repeat. Weights in `fixed_zero`
    are held at 0 (for ablations)."""
    keep: np.ndarray = ~fixed_zero
    origin: np.ndarray = np.where(keep, start, 0.0)
    w: np.ndarray = origin
    for outer in range(ORACLE_ROUNDS):
        previous: np.ndarray = data.oracle_features
        choose_oracle(data, w)
        if outer > 0 and np.allclose(
                previous, data.oracle_features, rtol=1e-9, atol=1e-6):
            break
        log(f"oracle {outer}: Delta per metre "
            f"{data.oracle_error.sum() / data.true_length.sum():.6f}, skips "
            f"{int(data.oracle_features[:, -3].sum())}")
        w = cutting_planes(data, origin, w, strength, s, keep, log)
    return w


def cutting_planes(data: TrainingSet, origin: np.ndarray, w: np.ndarray,
                   strength: float, s: np.ndarray, keep: np.ndarray,
                   log) -> np.ndarray:
    """The convex problem for fixed oracles, from `w`, by proximal bundle."""
    total: float = float(data.true_length.sum())
    a_list: list[float] = [0.0]
    g_list: list[np.ndarray] = [np.zeros(len(origin))]
    pull: float = strength

    def regulariser(x: np.ndarray) -> float:
        return float(strength * np.sum((s * (x - origin)) ** 2))

    a, g, hinge = plane(data, w)
    a_list.append(a)
    g_list.append(np.where(keep, g, 0.0))
    centre, centre_j = w, regulariser(w) + hinge.sum() / total
    for round_ in range(CUTTING_ROUNDS):
        planes_a, planes_g = np.array(a_list), np.array(g_list)
        joint: float = strength + pull / 2
        middle: np.ndarray = (strength * origin + pull / 2 * centre) / joint
        trial: np.ndarray = np.where(keep, solve_planes(planes_a, planes_g, middle, joint, s), 0.0)
        modelled: float = regulariser(trial) + float((planes_a + planes_g @ trial).max())
        predicted: float = centre_j - modelled
        if predicted <= CUTTING_TOLERANCE * max(abs(centre_j), 1e-12):
            log(f"  converged after {round_ + 1} passes: J {centre_j:.6f}")
            return centre
        a, g, hinge = plane(data, trial)
        a_list.append(a)
        g_list.append(np.where(keep, g, 0.0))
        trial_j: float = regulariser(trial) + hinge.sum() / total
        if centre_j - trial_j >= 0.1 * predicted:
            centre, centre_j = trial, trial_j
            pull = max(pull / 2, 1e-3 * strength)
        else:
            pull = min(pull * 2, 1e3 * strength)
        if round_ % 10 == 0:
            log(f"  pass {round_}: J {centre_j:.6f}, predicted decrease "
                f"{predicted:.2e}, hinge per metre {hinge.sum() / total:.6f}")
    log(f"  stopped after {CUTTING_ROUNDS} passes: J {centre_j:.6f}")
    return centre
