"""Step errors (derivation §4), the decoder's exactness (Theorem 1) and
the noise model's fit (§7)."""

import itertools
from pathlib import Path

import numpy as np

from benchmark.metrics import score_route
from dataset import Corpus, Track
from learned.decode import Labelling, decode, drive
from learned.kernel import piece_error
from learned.lattice import load
from learned.network import Network
from learned.noise import NoiseModel, fit, generate
from learned.routes import Router

ROOT: Path = Path(__file__).parent.parent
LATTICES: Path = ROOT / "results" / "learned" / "lattices" / "train"


def test_piece_error_counts_metres_per_road() -> None:
    length = np.array([100.0, 40.0, 70.0, 25.0])
    canonical = np.array([0, 1, 2, 3])
    tally = np.zeros(4, dtype=np.int64)
    # driven 1, 2, 1 against true 1, 3: road 1 once too often (40), road 2
    # added (70), road 3 missed (25).
    path = np.array([1, 2, 1])
    piece = np.array([1, 3])
    assert piece_error(path, 3, piece, canonical, length, tally) == 135.0
    assert not tally.any()


def test_step_errors_never_undercount() -> None:
    """Theorem 2 on real lattices: for random labellings without skips, the
    summed step errors are at least the driven route's real error."""
    folders = {f.name(): f for f in Corpus.from_root(ROOT / "data" / "corpus_fixed").folders("train")}
    random = np.random.default_rng(0)
    for name in ["00000003", "00000020"]:
        lattice = load(LATTICES / f"{name}_own.npz")
        graph = folders[name].graph()
        truth = folders[name].route(graph)
        router = Router(Network.from_graph(graph))
        for _ in range(5):
            chosen = [int(random.integers(lattice.points[0].count()))]
            bound = float(lattice.start_errors[chosen[0]])
            for j in range(lattice.step_count()):
                ordinary = np.nonzero(~lattice.skips[j][chosen[-1]])[0]
                if len(ordinary) == 0:
                    break
                d = int(random.choice(ordinary))
                bound += float(lattice.errors[j][chosen[-1], d])
                chosen.append(d)
            if len(chosen) < len(lattice.points):
                continue
            labelling = Labelling(np.array(chosen), np.zeros(len(chosen) - 1, bool), 0.0)
            route = drive(labelling, lattice.points, router)
            score = score_route(graph, route, truth)
            real = score.added + score.missed
            assert bound >= real - 1e-6, (name, bound, real)


def test_decoder_equals_brute_force() -> None:
    random = np.random.default_rng(3)
    for _ in range(20):
        sizes = [3, 2, 3, 2]
        costs = [random.uniform(0, 10, (sizes[j], sizes[j + 1])) for j in range(3)]
        skips = [random.random((sizes[j], sizes[j + 1])) < 0.3 for j in range(3)]
        fresh = [random.uniform(0, 5, k) for k in sizes]
        best = np.inf
        for y in itertools.product(*[range(k) for k in sizes]):
            total = fresh[0][y[0]]
            for j in range(3):
                options = []
                if not skips[j][y[j], y[j + 1]]:
                    options.append(costs[j][y[j], y[j + 1]])
                if skips[j][y[j]].any():
                    options.append(costs[j][y[j]][skips[j][y[j]]].min() + fresh[j + 1][y[j + 1]])
                total += min(options) if options else np.inf
            best = min(best, total)
        assert decode(costs, skips, fresh, 1.0).value == best or abs(decode(costs, skips, fresh, 1.0).value - best) < 1e-9


def test_noise_fit_recovers_known_model() -> None:
    truth = NoiseModel(5.0, 0.35, 0.85, 0.01, 15.0)
    random = np.random.default_rng(7)
    pairs = []
    for _ in range(40):
        t = np.arange(3000, dtype=float)
        dx, dy = generate(t, truth, random)
        zero = np.zeros_like(t)
        pairs.append((Track(zero, zero, dx, dy, t), Track(zero, zero, zero, zero, t)))
    fitted = fit(pairs)
    assert abs(fitted.core_sigma - 5.0) < 0.15
    assert abs(fitted.drift_share - 0.35) < 0.05
    assert abs(fitted.drift_phi - 0.85) < 0.02
