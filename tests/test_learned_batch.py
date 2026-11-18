"""The batched GPU decoder reaches the same optimum as the reference decoder
in decode.py, plain, loss-augmented and as the oracle. Both read the same
float16-rounded features; the GPU sums in float32, so values agree to a
relative 1e-5."""

from pathlib import Path

import numpy as np
import pytest
import torch

from learned.batch import FLOAT16_LIMIT, build, decode as decode_batch
from learned.decode import decode
from learned.lattice import load

LATTICES: Path = Path(__file__).parent.parent / "results" / "learned" / "lattices" / "train"
NAMES: list[str] = ["00000003_own", "00000020_copy1", "00000043_own", "00000010_copy5"]


def reference(path: Path, w: np.ndarray, scale: np.ndarray, error_weight: float,
              cost_weight: float) -> float:
    lattice = load(path)
    costs, skips = [], []
    for j in range(lattice.step_count()):
        rounded = np.clip(lattice.features[j] / scale, -FLOAT16_LIMIT, FLOAT16_LIMIT)
        rounded = rounded.astype(np.float16).astype(np.float64) * scale
        step = np.where(lattice.skips[j], w[-3], rounded @ w[:-3])
        error = np.where(lattice.skips[j], 0.0, lattice.errors[j])
        costs.append(cost_weight * step - error_weight * error)
        skips.append(lattice.skips[j])
    fresh = []
    for i, p in enumerate(lattice.points):
        weight = w[-2] if i == 0 else w[-1]
        error = lattice.start_errors if i == 0 else lattice.skip_errors[i - 1]
        fresh.append(cost_weight * weight * p.distance - error_weight * error)
    return decode(costs, skips, fresh, 1.0).value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
def test_batched_optimum_equals_reference() -> None:
    paths = [LATTICES / f"{n}.npz" for n in NAMES]
    scale = np.array([10.0, 10.0, 100.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    batch = build(paths, scale, "cuda")
    random = np.random.default_rng(1)
    for error_weight, cost_weight in [(0.0, 1.0), (1.0, 1.0), (-1.0, 0.0)]:
        w = np.abs(random.normal(size=14))
        decoded = decode_batch(batch, torch.tensor(w), error_weight, cost_weight)
        ours = (decoded.features.cpu().numpy() @ w) * cost_weight \
            - error_weight * decoded.error.cpu().numpy()
        for t, path in enumerate(paths):
            theirs = reference(path, w, scale, error_weight, cost_weight)
            assert ours[t] == pytest.approx(theirs, rel=1e-5, abs=1e-3), (path.name, error_weight)
