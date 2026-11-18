"""The measure stage: every value approach 5 takes from data, from train only.

- tolerance: the Douglas-Peucker tolerance at which the noisy train tracks
  keep as many decision points as their clean originals keep at Woeltche's
  5 m (derivation §3.1);
- half_window: the speed fit's half window that best reproduces true travel
  speed over the steps (derivation §5.3), true speed being the original
  track's path length over a step, sampled every 5 s, over its duration;
- too_fast_speed: the 99.9th percentile of those true step speeds;
- noise: the fitted noise model and its check against the measured noise.
"""

from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dataset import Folder, RoadGraph, Track
from learned.hyperparameters import (BISECTION_STEPS, GRID_CELL,
                                     HALF_WINDOWS, MEDIAN_MERGE_ROAD_SHARE,
                                     MEDIAN_MERGE_TOLERANCE, RADIUS_UPPER,
                                     REFERENCE_TOLERANCE, TOLERANCE_RANGE,
                                     TOO_FAST_QUANTILE, TRUE_SPEED_SAMPLE,
                                     WARP_SPEED, WORKERS)
from learned.network import ArcGrid, Network
from learned.noise import describe, fit, noisy_copy
from learned.track import decision_points, smoothed_speed


def original_track(folder: Folder, originals: str, graph: RoadGraph) -> Track:
    """The clean original of a folder's track, on the same plane."""
    path: Path = Path(originals) / folder.split / folder.name() / f"{folder.name()}.track"
    return Track.from_file(path, graph.projection)


def point_counts(job: tuple) -> tuple[int, int]:
    """Decision points of (original at the reference, ours at `tolerance`)."""
    folder, originals, tolerance = job
    graph = folder.graph()
    grid = ArcGrid(Network.from_graph(graph), GRID_CELL)
    ours = folder.track(graph)
    original = original_track(folder, originals, graph)
    reference = decision_points(original, grid, REFERENCE_TOLERANCE, WARP_SPEED,
                                MEDIAN_MERGE_TOLERANCE, MEDIAN_MERGE_ROAD_SHARE,
                                RADIUS_UPPER)
    noisy = decision_points(ours, grid, tolerance, WARP_SPEED,
                            MEDIAN_MERGE_TOLERANCE, MEDIAN_MERGE_ROAD_SHARE,
                            RADIUS_UPPER)
    return len(reference[0]), len(noisy[0])


def measure_tolerance(folders: list[Folder], originals: str) -> dict:
    """Bisection on the tolerance until the noisy total matches the
    reference total (the noisy count falls as the tolerance grows)."""
    low, high = TOLERANCE_RANGE
    history: list[tuple[float, int, int]] = []
    with Pool(WORKERS) as pool:
        for _ in range(BISECTION_STEPS):
            middle: float = (low + high) / 2
            counts = pool.map(point_counts, [(f, originals, middle) for f in folders])
            reference: int = sum(c[0] for c in counts)
            noisy: int = sum(c[1] for c in counts)
            history.append((middle, reference, noisy))
            if noisy > reference:
                low = middle
            else:
                high = middle
    return {"tolerance": (low + high) / 2, "reference_points": history[-1][1],
            "history": history}


def step_speeds(job: tuple) -> tuple[list, list]:
    """For every step of one track: the smoothed mean speed for each half
    window, and the true speed from the original."""
    folder, originals, tolerance = job
    graph = folder.graph()
    grid = ArcGrid(Network.from_graph(graph), GRID_CELL)
    ours = folder.track(graph)
    original = original_track(folder, originals, graph)
    _, _, _, first_fix, last_fix = decision_points(
        ours, grid, tolerance, WARP_SPEED, MEDIAN_MERGE_TOLERANCE,
        MEDIAN_MERGE_ROAD_SHARE, RADIUS_UPPER)
    smoothed: list[np.ndarray] = [smoothed_speed(ours.t, ours.x, ours.y, h)
                                  for h in HALF_WINDOWS]
    estimates: list[list[float]] = []
    truths: list[float] = []
    for j in range(len(first_fix) - 1):
        start, stop = int(last_fix[j]), int(first_fix[j + 1])
        duration: float = float(ours.t[stop] - ours.t[start])
        if duration <= 0:
            continue
        sample: np.ndarray = np.unique(np.concatenate([
            np.arange(start, stop, TRUE_SPEED_SAMPLE), [stop]]))
        travel: float = float(np.hypot(np.diff(original.x[sample]),
                                       np.diff(original.y[sample])).sum())
        truths.append(travel / duration)
        estimates.append([float(s[start:stop + 1].mean()) for s in smoothed])
    return estimates, truths


def measure_speed(folders: list[Folder], originals: str, tolerance: float) -> dict:
    with Pool(WORKERS) as pool:
        results = pool.map(step_speeds, [(f, originals, tolerance) for f in folders])
    estimates: np.ndarray = np.array([e for r in results for e in r[0]])
    truths: np.ndarray = np.array([t for r in results for t in r[1]])
    errors: np.ndarray = np.median(np.abs(estimates - truths[:, None]), axis=0)
    best: int = int(np.argmin(errors))
    return {"half_window": HALF_WINDOWS[best],
            "speed_error_by_half_window": dict(zip(map(str, HALF_WINDOWS), errors.tolist())),
            "too_fast_speed": float(np.percentile(truths, TOO_FAST_QUANTILE)),
            "steps": int(len(truths))}


def measure_noise(folders: list[Folder], originals: str, seed: int) -> dict:
    pairs: list[tuple[Track, Track]] = []
    for folder in folders:
        graph = folder.graph()
        pairs.append((folder.track(graph), original_track(folder, originals, graph)))
    model = fit(pairs)
    random = np.random.default_rng(seed)
    generated: list[tuple[Track, Track]] = []
    for _, original in pairs:
        generated.append((noisy_copy(original, model, random), original))
    return {"model": model.to_dict(), "measured": describe(pairs),
            "generated": describe(generated)}
