"""The held-out 10 under the same conditions as the given tracks.

The executor test (serve.executor_test) matched the 10 held-out folders on
Kubicka's clean originals, but every given track carries the organisers'
added noise. Here each held-out original gets DRAWS fresh draws of the
noise model fitted on train (learned.noise, results/learned/parameters.json),
written as ordinary corpus folders, and every model matches all of them.

Usage:
  uv run python -m serve.held_out_noisy make  <executor_test folder> <out root>
  uv run python -m serve.held_out_noisy match <out root>
  then, per model: uv run python -m benchmark <out root> results/held_out_noisy <model>

<out root>/noisy<k>/<id>/ holds symlinks to the folder's nodes, arcs and
route, and the noisy track. Draw k of folder id uses the seed
(NOISE_SEED, id, k), the same scheme as the training copies; the held-out
ids are never train ids, so no draw repeats a training copy's.
"""

import csv
import json
import sys
import time
from multiprocessing import Pool, Process, Queue
from pathlib import Path

import numpy as np

from dataset import Corpus, RoadGraph, Track
from dataset.projection import EARTH_RADIUS
from learned.hyperparameters import NOISE_SEED
from learned.noise import NoiseModel, describe, generate

ROOT: Path = Path(__file__).resolve().parents[2]
OUT: Path = ROOT / "results" / "held_out_noisy"
# Same count as the training copies (user's choice, 2026-11-18).
DRAWS: int = 8
MODELS: list[str] = ["hmm", "segment", "mcm", "learned", "mdp"]
WORKERS: int = 14
# OWN CHOICE: an MDP match running longer than this is stopped and redone
# at MDP_FALLBACK_TOLERANCE, the tolerance the user chose for track 33 when
# the 5 m one did not finish (results/mdp/deviation.md).
MDP_TIMEOUT: float = 1200.0  # seconds
MDP_FALLBACK_TOLERANCE: float = 20.0  # metres


def to_lonlat(graph: RoadGraph, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of Projection.to_xy."""
    p = graph.projection
    lon = np.degrees(p.origin_lon + x / (EARTH_RADIUS * np.cos(p.origin_lat)))
    lat = np.degrees(p.origin_lat + y / EARTH_RADIUS)
    return lon, lat


def make(source: Path, out: Path) -> None:
    model = NoiseModel(**json.loads(
        (ROOT / "results" / "learned" / "parameters.json").read_text())["noise"])
    pairs: list[tuple[Track, Track]] = []
    for folder in Corpus.from_root(source.parent).folders(source.name):
        graph: RoadGraph = folder.graph()
        original: Track = folder.track(graph)
        for k in range(DRAWS):
            random = np.random.default_rng([NOISE_SEED, int(folder.name()), k])
            dx, dy = generate(original.t, model, random)
            lon, lat = to_lonlat(graph, original.x + dx, original.y + dy)
            target: Path = out / f"noisy{k}" / folder.name()
            target.mkdir(parents=True, exist_ok=True)
            for suffix in ["nodes", "arcs", "route"]:
                link: Path = target / f"{folder.name()}.{suffix}"
                if not link.exists():
                    link.symlink_to((folder.path / f"{folder.name()}.{suffix}").resolve())
            with (target / f"{folder.name()}.track").open("w") as handle:
                for a, b, t in zip(lon, lat, original.t):
                    handle.write(f"{a:.8f}\t{b:.8f}\t{t:.6f}\n")
            # Read back through the normal reader, as the matchers will.
            pairs.append((Track.from_file(target / f"{folder.name()}.track",
                                          graph.projection), original))
    print("generated noise:", json.dumps(describe(pairs)))


def run(job: tuple) -> tuple[str, str, str, float, str]:
    model, folder, tolerance = job
    from serve.models import run_batch
    began: float = time.perf_counter()
    try:
        if model == "mdp" and tolerance is not None:
            import mdp.match
            mdp.match.SIMPLIFY_TOLERANCE = tolerance
        graph = folder.graph()
        track = folder.track(graph)
        fixes = list(zip(track.lon.tolist(), track.lat.tolist(), track.t.tolist()))
        result = run_batch(model, str(folder.path), str(ROOT / "results"), fixes, None)
        target: Path = OUT / model / "routes" / folder.split / f"{folder.name()}.route"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(f"{a}\n" for a in result["route"]))
        note: str = "" if tolerance is None else f"tolerance {tolerance:g} m"
        return model, folder.split, folder.name(), round(time.perf_counter() - began, 1), note
    except Exception as error:
        return model, folder.split, folder.name(), round(time.perf_counter() - began, 1), repr(error)[:200]


def run_into(job: tuple, queue: Queue) -> None:
    queue.put(run(job))


def run_mdp(folders: list, log) -> list[tuple]:
    """MDP jobs, each in its own process so that one can be stopped."""
    rows: list[tuple] = []
    waiting: list[tuple] = [("mdp", f, None) for f in folders]
    running: list[tuple] = []
    while waiting or running:
        while waiting and len(running) < WORKERS:
            job = waiting.pop(0)
            queue: Queue = Queue()
            process = Process(target=run_into, args=(job, queue))
            process.start()
            running.append((job, process, queue, time.perf_counter()))
        time.sleep(1.0)
        for item in list(running):
            job, process, queue, began = item
            if not queue.empty():
                rows.append(queue.get())
                process.join()
                running.remove(item)
                log(rows[-1])
            elif time.perf_counter() - began > MDP_TIMEOUT:
                process.kill()
                process.join()
                running.remove(item)
                log(("mdp", job[1].split, job[1].name(), "timeout, retry"))
                waiting.append(("mdp", job[1], MDP_FALLBACK_TOLERANCE))
            elif not process.is_alive() and queue.empty():
                running.remove(item)
                rows.append(("mdp", job[1].split, job[1].name(), 0.0, "process died"))
                log(rows[-1])
    return rows


def match(root: Path) -> None:
    corpus: Corpus = Corpus.from_root(root)
    folders = [f for split in corpus.splits for f in corpus.folders(split)]
    # Longest tracks first, so the pool stays busy at the end.
    folders.sort(key=lambda f: -(f.path / f"{f.name()}.track").stat().st_size)

    def log(row) -> None:
        print(*row, flush=True)

    rows: list[tuple] = []
    with Pool(WORKERS) as pool:
        jobs = [(m, f, None) for m in MODELS if m != "mdp" for f in folders]
        for row in pool.imap_unordered(run, jobs):
            rows.append(row)
            log(row)
    rows += run_mdp(folders, log)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "timing.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "split", "id", "seconds", "note"])
        writer.writerows(sorted(rows))
    print("problems:", [r for r in rows if r[4] and not r[4].startswith("tolerance")])


if __name__ == "__main__":
    stage: str = sys.argv[1]
    if stage == "make":
        make(Path(sys.argv[2]), Path(sys.argv[3]))
    elif stage == "match":
        match(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
