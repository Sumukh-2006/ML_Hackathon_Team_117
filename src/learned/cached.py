"""Layer 5, third version: the direct search on own tracks AND noisy copies.

The second version (direct.py) searched the real error of the 64 own
tracks of a fold and overfitted: in-sample error fell in every fold while
the pooled held-out CF went from 98.448% to 98.385% (results/learned/
cv.log). Woeltche tuned his weights on "hundreds of tracks to compensate
for individual tendencies" (thesis 5.2). The noisy copies of layer 6 were
built for exactly that, but the search never used them. Here the
objective is the weighted 1 - CF over every own track of the fold plus
COPIES_IN_SEARCH noisy copies of each (user's choice, 2026-11-18: 3, for
memory), and the search procedure is unchanged.

Each trial decodes every lattice. Reloading lattices and rebuilding road
networks per trial (direct.RealError) costs seconds per track, so here a
fixed set of worker processes each loads its share once (all lattices of
a folder in the same worker, sharing its graph and router) and answers
trial weights over a pipe.
"""

from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

from benchmark.metrics import score_route
from dataset import Folder, Route
from learned.decode import decode, drive
from learned.evaluate import labelling
from learned.hyperparameters import (EXCLUDED_COPIES, ROUTING_MAX_DISTANCE_FACTOR, SEARCH_FACTORS,
                                     SEARCH_SHARES, SEARCH_SWEEPS, WORKERS)
from learned.lattice import Lattice, load
from learned.network import Network
from learned.routes import Router

COPIES_IN_SEARCH: int = 3


def lattice_paths(root: Path, name: str, copies: int) -> list[Path]:
    paths: list[Path] = [root / f"{name}_own.npz"]
    if name not in EXCLUDED_COPIES:
        paths += [root / f"{name}_copy{k}.npz" for k in range(copies)]
    return paths


def chosen_features(lattice: Lattice, chosen: np.ndarray, skipped: np.ndarray) -> np.ndarray:
    """|phi| summed along a labelling (step features only; skips add none)."""
    total: np.ndarray = np.zeros(lattice.features[0].shape[-1] if lattice.features else 11)
    for j in range(len(skipped)):
        if not skipped[j]:
            total += np.abs(lattice.features[j][chosen[j], chosen[j + 1]])
    return total


def path_vector(lattice: Lattice, chosen: np.ndarray, skipped: np.ndarray) -> np.ndarray:
    """Phi(y): the 14 numbers whose dot product with the weights is the
    labelling's cost (evaluate.labelling): step features of driven steps,
    the number of skips, the first point's distance, and the distance of
    every fresh start after a skip."""
    phi: np.ndarray = np.zeros(lattice.features[0].shape[-1] + 3)
    phi[-2] = lattice.points[0].distance[chosen[0]]
    for j in range(len(skipped)):
        if skipped[j]:
            phi[-3] += 1.0
            phi[-1] += lattice.points[j + 1].distance[chosen[j + 1]]
        else:
            phi[:-3] += lattice.features[j][chosen[j], chosen[j + 1]]
    return phi


_STEP_ROUTES: dict = {}
_STORED_ARCS: list[int] = [0]
# OWN CHOICE: each worker's route cache is emptied when it passes either
# limit. Routes are kept as int32 arrays: kept as tuples of Python ints, the
# long routes of track 33 (hives) grew one worker past 11 GB on 2026-11-18
# and the host ran out of memory twice.
STEP_ROUTE_LIMIT: int = 300_000      # cached steps per worker
STEP_ARC_LIMIT: int = 10_000_000     # cached arcs per worker (40 MB)
# OWN CHOICE: above this many candidate pairs (track 33's own lattice) the
# line search does not keep float64 base and slope arrays for every pair;
# with them, that one worker reached 11.5 GB.
LARGE_LATTICE: int = 10**12  # light path off by default; lower it if memory runs short


def drive_cached(name: str, labelling, points, router: Router) -> Route:
    """decode.drive, with every step's route kept per worker: the lattices
    never change, so a step (j, c, c') always drives the same arcs, and the
    many labellings of a search share most of their steps."""
    first: int = int(labelling.chosen[0])
    arcs: list[int] = [int(points[0].arc[first])]
    for j in range(len(labelling.skipped)):
        c: int = int(labelling.chosen[j])
        following: int = int(labelling.chosen[j + 1])
        target: int = int(points[j + 1].arc[following])
        driven: list[int] = [target]
        if not labelling.skipped[j]:
            key = (name, j, c, following)
            cached = _STEP_ROUTES.get(key)
            if cached is None:
                source: int = int(points[j].arc[c])
                bound: float = router.bound(source, points[j + 1].arc,
                                            ROUTING_MAX_DISTANCE_FACTOR)
                cached = np.asarray(router.arcs(
                    source, float(points[j].along[c]), target,
                    float(points[j + 1].along[following]), bound), dtype=np.int32)
                if (len(_STEP_ROUTES) >= STEP_ROUTE_LIMIT
                        or _STORED_ARCS[0] + len(cached) > STEP_ARC_LIMIT):
                    _STEP_ROUTES.clear()
                    _STORED_ARCS[0] = 0
                _STEP_ROUTES[key] = cached
                _STORED_ARCS[0] += len(cached)
            driven = cached.tolist()
        for arc in driven:
            if arcs[-1] != arc:
                arcs.append(arc)
    return Route(np.array(arcs, dtype=np.int64))


def line_surface(name: str, lattice: Lattice, router: Router, graph, truth, weights: np.ndarray,
                 k: int, low: float, high: float) -> tuple[list[float], list[tuple[float, float]]]:
    """MERT's exact line search for one track (Och 2003; Macherey et al.
    2008; Dyer 2013). Along w(t) = weights + t e_k, every labelling's cost
    is a line a + b t, and the decoder follows the lower envelope of those
    lines. The envelope is found by repeated exact decoding: if the decoder
    returns the same line at both ends of an interval, that line is the
    envelope on the whole interval (the envelope is concave and never above
    any line); otherwise the two lines' crossing is decoded, and is either
    a breakpoint or reveals a new line. Returns the breakpoints low = t_0 <
    ... < t_m = high and, per segment, the (overlap, longer) of the route the
    decoder drives there."""
    pairs: int = sum(int(f.shape[0] * f.shape[1]) for f in lattice.features)
    if pairs > LARGE_LATTICE:
        # Memory before speed: decode from the stored features each time.
        def at(t: float):
            w = weights.copy()
            w[k] += t
            chosen = labelling(lattice, w)
            return chosen, path_vector(lattice, chosen.chosen, chosen.skipped)
        return envelope(name, lattice, router, graph, truth, weights, k, low, high, at)
    # Along the line every cost is base + t * slope: computed once here,
    # so each decode below is an axpy and the dynamic program.
    steps: int = lattice.step_count()
    base: list[np.ndarray] = []
    slope: list[np.ndarray] = []
    for j in range(steps):
        f: np.ndarray = lattice.features[j]
        base.append(np.where(lattice.skips[j], weights[-3],
                             f.astype(np.float64) @ weights[:-3]))
        if k < f.shape[-1]:
            slope.append(np.where(lattice.skips[j], 0.0, f[..., k].astype(np.float64)))
        elif k == f.shape[-1]:
            slope.append(lattice.skips[j].astype(np.float64))
        else:
            slope.append(np.zeros(lattice.skips[j].shape))
    distances: list[np.ndarray] = [p.distance.astype(np.float64) for p in lattice.points]
    fresh_base: list[np.ndarray] = [weights[-1] * d for d in distances]
    fresh_base[0] = weights[-2] * distances[0]
    fresh_slope: list[np.ndarray] = [np.zeros(len(d)) for d in distances]
    if k == len(weights) - 2:
        fresh_slope[0] = distances[0]
    elif k == len(weights) - 1:
        fresh_slope = [d for d in distances]
        fresh_slope[0] = np.zeros(len(distances[0]))

    def at(t: float):
        chosen = decode([b + t * s for b, s in zip(base, slope)], lattice.skips,
                        [b + t * s for b, s in zip(fresh_base, fresh_slope)], 1.0)
        return chosen, path_vector(lattice, chosen.chosen, chosen.skipped)
    return envelope(name, lattice, router, graph, truth, weights, k, low, high, at)


def envelope(name: str, lattice: Lattice, router: Router, graph, truth, weights: np.ndarray,
             k: int, low: float, high: float, at) -> tuple[list[float], list[tuple[float, float]]]:
    """The lower envelope along the line, from an exact decoder `at(t)`."""

    def same(a, b) -> bool:
        return (np.array_equal(a[0].chosen, b[0].chosen)
                and np.array_equal(a[0].skipped, b[0].skipped))

    segments: list[tuple[float, float, object]] = []

    def split(t1: float, y1, t2: float, y2, depth: int) -> None:
        if same(y1, y2) or depth > 60:
            segments.append((t1, t2, y1))
            return
        a1, b1 = float(y1[1] @ weights), float(y1[1][k])
        a2, b2 = float(y2[1] @ weights), float(y2[1][k])
        if abs(b1 - b2) < 1e-12:          # one line: a tie on the whole interval
            segments.append((t1, t2, y1))
            return
        cross: float = (a2 - a1) / (b1 - b2)
        cross = min(max(cross, t1), t2)
        y = at(cross)
        value: float = float(y[1] @ weights + cross * y[1][k])
        line: float = a1 + b1 * cross
        if value < line - 1e-9 * max(1.0, abs(line)) and not same(y, y1) and not same(y, y2):
            split(t1, y1, cross, y, depth + 1)
            split(cross, y, t2, y2, depth + 1)
        else:
            segments.append((t1, cross, y1))
            segments.append((cross, t2, y2))

    split(low, at(low), high, at(high), 0)
    breaks: list[float] = [segments[0][0]]
    scores: list[tuple[float, float]] = []
    cache: dict[bytes, tuple[float, float]] = {}
    for t1, t2, y in segments:
        if t2 <= breaks[-1] and scores:
            continue
        key: bytes = y[0].chosen.tobytes() + y[0].skipped.tobytes()
        if key not in cache:
            s = score_route(graph, drive_cached(name, y[0], lattice.points, router), truth)
            cache[key] = (s.overlap, max(s.predicted_length, s.true_length))
        breaks.append(t2)
        scores.append(cache[key])
    return breaks, scores


def serve(shard: list[tuple[Folder, list[Path]]], pipe: Connection) -> None:
    items: list[tuple[str, Lattice, Router, object, object]] = []
    for folder, paths in shard:
        graph = folder.graph()
        router: Router = Router(Network.from_graph(graph))
        truth = folder.route(graph)
        for path in paths:
            items.append((path.stem, load(path), router, graph, truth))
    pipe.send("ready")
    while True:
        message = pipe.recv()
        if message is None:
            return
        if isinstance(message[0], str) and message[0] == "line":
            _, weights, k, low, high = message
            pipe.send([line_surface(name, lattice, router, graph, truth, weights, k, low, high)
                       for name, lattice, router, graph, truth in items])
            continue
        weights, with_features = message
        rows: list[dict] = []
        for name, lattice, router, graph, truth in items:
            chosen = labelling(lattice, weights)
            route = drive_cached(name, chosen, lattice.points, router)
            s = score_route(graph, route, truth)
            row: dict = {"lattice": name, "overlap": s.overlap,
                         "predicted_length": s.predicted_length,
                         "true_length": s.true_length}
            if with_features:
                row["features"] = chosen_features(lattice, chosen.chosen, chosen.skipped)
            rows.append(row)
        pipe.send(rows)


class Evaluator:
    """The real error of trial weights on a fixed set of lattices."""

    def __init__(self, folders: list[Folder], root: Path, copies: int) -> None:
        shards: list[list] = [[] for _ in range(WORKERS)]
        sizes: list[int] = [0] * WORKERS
        jobs = [(f, lattice_paths(root, f.name(), copies)) for f in folders]
        # Biggest folders first, each to the least loaded worker.
        jobs.sort(key=lambda job: -sum(p.stat().st_size for p in job[1]))
        for job in jobs:
            k: int = int(np.argmin(sizes))
            shards[k].append(job)
            sizes[k] += sum(p.stat().st_size for p in job[1])
        self.pipes: list[Connection] = []
        self.processes: list[Process] = []
        for shard in shards:
            if not shard:
                continue
            ours, theirs = Pipe()
            process = Process(target=serve, args=(shard, theirs), daemon=True)
            process.start()
            self.pipes.append(ours)
            self.processes.append(process)
        for pipe in self.pipes:
            assert pipe.recv() == "ready"
        self.lattices: int = sum(len(job[1]) for job in jobs)

    def rows(self, weights: np.ndarray, with_features: bool = False) -> list[dict]:
        for pipe in self.pipes:
            pipe.send((weights, with_features))
        out: list[dict] = []
        for pipe in self.pipes:
            out += pipe.recv()
        return out

    def __call__(self, weights: np.ndarray) -> float:
        rows: list[dict] = self.rows(weights)
        overlap: float = sum(r["overlap"] for r in rows)
        longer: float = sum(max(r["predicted_length"], r["true_length"]) for r in rows)
        return 1.0 - overlap / longer

    def line(self, weights: np.ndarray, k: int, low: float, high: float) -> tuple[float, float]:
        """The exact minimiser of the weighted 1 - CF along weight k over
        t in [low, high]: every track's error surface is piecewise constant;
        summing them over the union of breakpoints gives the global surface,
        and the best segment's middle is returned with its error (ties go
        to the segment nearest t = 0, i.e. the smallest change)."""
        for pipe in self.pipes:
            pipe.send(("line", weights, k, low, high))
        surfaces: list = []
        for pipe in self.pipes:
            surfaces += pipe.recv()
        cuts: np.ndarray = np.unique(np.concatenate([np.array(b) for b, _ in surfaces]))
        middles: np.ndarray = (cuts[:-1] + cuts[1:]) / 2
        overlap: np.ndarray = np.zeros(len(middles))
        longer: np.ndarray = np.zeros(len(middles))
        for breaks, scores in surfaces:
            index: np.ndarray = np.clip(np.searchsorted(np.array(breaks), middles) - 1, 0, len(scores) - 1)
            values: np.ndarray = np.array(scores)
            overlap += values[index, 0]
            longer += values[index, 1]
        error: np.ndarray = 1.0 - overlap / longer
        best: float = float(error.min())
        ties: np.ndarray = np.nonzero(error <= best + 1e-12)[0]
        pick: int = int(ties[np.argmin(np.abs(middles[ties]))])
        return float(middles[pick]), float(error[pick])

    def feature_per_metre(self, weights: np.ndarray) -> np.ndarray:
        rows: list[dict] = self.rows(weights, True)
        return (sum(r["features"] for r in rows)
                / sum(r["true_length"] for r in rows))

    def close(self) -> None:
        for pipe in self.pipes:
            pipe.send(None)
        for process in self.processes:
            process.join()


def search(objective: Evaluator, start: np.ndarray, frozen: np.ndarray, log) -> np.ndarray:
    """direct.search's coordinate pattern search, unchanged, except that
    the scale for trying a zero weight comes from the evaluator. Weights
    are 11 step weights then skip, first distance, restart distance; the
    trial scale of a zero step weight is a share of the distance term's
    cost per metre."""
    w: np.ndarray = start.copy()
    best: float = objective(w)
    log(f"start: error {best:.6f} on {objective.lattices} lattices")
    per_metre: np.ndarray = objective.feature_per_metre(start)
    distance_share: float = start[0] * per_metre[0]
    for sweep in range(SEARCH_SWEEPS):
        improved: bool = False
        for k in range(len(w)):
            if frozen[k]:
                continue
            if w[k] > 0:
                trials: list[float] = [w[k] * f for f in SEARCH_FACTORS]
            elif k < len(per_metre):
                trials = [share * distance_share / max(per_metre[k], 1e-12)
                          for share in SEARCH_SHARES]
            else:
                continue
            for value in trials:
                trial: np.ndarray = w.copy()
                trial[k] = value
                error: float = objective(trial)
                if error < best - 1e-9:
                    best, w, improved = error, trial, True
        log(f"sweep {sweep}: error {best:.6f}, w {np.round(w, 4).tolist()}")
        if not improved:
            break
    return w


def exact_search(objective: Evaluator, start: np.ndarray, frozen: np.ndarray, log) -> np.ndarray:
    """Coordinate descent with MERT's exact line search: each weight in
    turn moves to the exact best value over the same range the pattern
    search sampled (0 to 4 times a non-zero weight; 0 to the largest
    trial value for a zero step weight), until a sweep changes nothing."""
    w: np.ndarray = start.copy()
    best: float = objective(w)
    log(f"start: error {best:.6f} on {objective.lattices} lattices")
    per_metre: np.ndarray = objective.feature_per_metre(start)
    distance_share: float = start[0] * per_metre[0]
    for sweep in range(SEARCH_SWEEPS):
        improved: bool = False
        for k in range(len(w)):
            if frozen[k]:
                continue
            if w[k] > 0:
                top: float = w[k] * max(SEARCH_FACTORS)
            elif k < len(per_metre):
                top = max(SEARCH_SHARES) * distance_share / max(per_metre[k], 1e-12)
            else:
                continue
            t, error = objective.line(w, k, -w[k], top - w[k])
            if error < best - 1e-9:
                w = w.copy()
                w[k] += t
                best, improved = error, True
        log(f"sweep {sweep}: error {best:.6f}, w {np.round(w, 4).tolist()}")
        if not improved:
            break
    return w
