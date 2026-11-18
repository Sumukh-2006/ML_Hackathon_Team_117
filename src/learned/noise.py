"""Layer 6: the organisers' added noise, fitted and regenerated (derivation §7).

Per axis the added offset is e = g + o: g Gaussian with a fresh part
(variance (1 - a) s_g^2) and a mean-reverting drift (variance a s_g^2,
correlation phi per second), and o a rare outlier (probability p, Gaussian
with standard deviation s_o). Fitted on train folders only, from
(our track - original track).
"""

from dataclasses import asdict, dataclass

import numpy as np

from dataset import Track
from learned.hyperparameters import FIT_LAGS


@dataclass(frozen=True)
class NoiseModel:
    core_sigma: float      # s_g: std of the Gaussian part, metres per axis
    drift_share: float     # a: share of the Gaussian variance that drifts
    drift_phi: float       # phi: drift correlation after one second
    outlier_chance: float  # p
    outlier_sigma: float   # s_o, metres per axis

    def to_dict(self) -> dict:
        return asdict(self)


def offsets(ours: Track, original: Track) -> tuple[np.ndarray, np.ndarray]:
    return ours.x - original.x, ours.y - original.y


def fit(pairs: list[tuple[Track, Track]]) -> NoiseModel:
    """Fit the model to the offsets of (ours, original) track pairs."""
    values: list[np.ndarray] = []
    for ours, original in pairs:
        dx, dy = offsets(ours, original)
        values.append(dx)
        values.append(dy)
    e: np.ndarray = np.concatenate(values)
    core: float = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    variance: float = float(e.var())
    fourth: float = float(np.mean((e - e.mean()) ** 4))
    outlier_part: float = max(variance - core * core, 0.0)   # p s_o^2
    if outlier_part > 0:
        spread: float = (fourth - 3 * core ** 4 - 6 * core ** 2 * outlier_part) / (3 * outlier_part)
        outlier_sigma: float = float(np.sqrt(max(spread, 0.0)))
        chance: float = outlier_part / max(spread, 1e-9)
    else:
        outlier_sigma, chance = 0.0, 0.0
    share, phi = fit_drift(pairs)
    # NOTE: the autocorrelation is measured on the total offset, outliers
    # included; the drift is a share of the Gaussian core only.
    core_share: float = min(share * variance / (core * core), 1.0)
    return NoiseModel(core, core_share, phi, min(chance, 1.0), outlier_sigma)


def fit_drift(pairs: list[tuple[Track, Track]]) -> tuple[float, float]:
    """a and phi from corr(k) = a phi^k, fitted in log space over FIT_LAGS
    on runs of fixes one second apart."""
    correlations: list[float] = []
    for lag in FIT_LAGS:
        products: list[np.ndarray] = []
        squares: list[np.ndarray] = []
        for ours, original in pairs:
            for e in offsets(ours, original):
                gap: np.ndarray = ours.t[lag:] - ours.t[:-lag]
                usable: np.ndarray = gap == lag
                centred: np.ndarray = e - e.mean()
                products.append((centred[lag:] * centred[:-lag])[usable])
                squares.append(centred * centred)
        correlations.append(float(np.concatenate(products).mean()
                                  / np.concatenate(squares).mean()))
    slope, intercept = np.polyfit(np.array(FIT_LAGS, dtype=float),
                                  np.log(np.maximum(correlations, 1e-6)), 1)
    return float(np.exp(intercept)), float(np.exp(slope))


def generate(t: np.ndarray, model: NoiseModel,
             random: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Fresh offsets for fixes at times t, one draw per axis. The drift is
    the exact Ornstein-Uhlenbeck step over each gap between fixes."""
    axes: list[np.ndarray] = []
    for _ in range(2):
        fresh: np.ndarray = random.normal(0.0, model.core_sigma
                                          * np.sqrt(1 - model.drift_share), len(t))
        drift_sigma: float = model.core_sigma * np.sqrt(model.drift_share)
        drift: np.ndarray = np.zeros(len(t))
        drift[0] = random.normal(0.0, drift_sigma)
        for i in range(1, len(t)):
            keep: float = model.drift_phi ** float(t[i] - t[i - 1])
            drift[i] = keep * drift[i - 1] + random.normal(
                0.0, drift_sigma * np.sqrt(1 - keep * keep))
        outlier: np.ndarray = np.where(random.random(len(t)) < model.outlier_chance,
                                       random.normal(0.0, model.outlier_sigma, len(t)), 0.0)
        axes.append(fresh + drift + outlier)
    return axes[0], axes[1]


def noisy_copy(original: Track, model: NoiseModel,
               random: np.random.Generator) -> Track:
    """The original with fresh noise, in the plane coordinates the matcher
    reads. NOTE: lon/lat are left as the original's; nothing downstream
    reads them."""
    dx, dy = generate(original.t, model, random)
    return Track(original.lon, original.lat, original.x + dx, original.y + dy,
                 original.t)


def describe(pairs: list[tuple[Track, Track]]) -> dict:
    """The statistics the fit is checked against: std, kurtosis, and the
    autocorrelation at 1, 2, 3, 5 and 10 s."""
    values: list[np.ndarray] = []
    for ours, original in pairs:
        dx, dy = offsets(ours, original)
        values.append(dx)
        values.append(dy)
    e: np.ndarray = np.concatenate(values)
    centred: np.ndarray = e - e.mean()
    kurtosis: float = float(np.mean(centred ** 4) / np.mean(centred ** 2) ** 2)
    acf: dict[str, float] = {}
    for lag in [1, 2, 3, 5, 10]:
        products: list[np.ndarray] = []
        squares: list[np.ndarray] = []
        for ours, original in pairs:
            for axis in offsets(ours, original):
                c = axis - axis.mean()
                usable = (ours.t[lag:] - ours.t[:-lag]) == lag
                products.append((c[lag:] * c[:-lag])[usable])
                squares.append(c * c)
        acf[str(lag)] = float(np.concatenate(products).mean() / np.concatenate(squares).mean())
    return {"std": float(e.std()), "kurtosis": kurtosis, "acf": acf}
