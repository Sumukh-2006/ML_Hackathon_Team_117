"""Approach 5, in stages; each saves its output before the next reads it.

    python -m learned measure  <corpus> <originals> <results>
    python -m learned lattices <corpus> <originals> <results>
    python -m learned prepare  <corpus> <results>
    python -m learned freeze   <results>      (weights.json = the starting weights)
    python -m learned cv       <corpus> <results>
    python -m learned train    <corpus> <results>
    python -m learned match    <corpus> <results>

Only the train split is ever measured or trained on.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from dataset import Corpus
from learned.hyperparameters import COPIES, NOISE_SEED, SCALE_SAMPLE, WORKERS


def measure(corpus: Path, originals: Path, results: Path) -> None:
    from learned.measure import measure_noise, measure_speed, measure_tolerance
    folders = Corpus.from_root(corpus).folders("train")
    began: float = time.perf_counter()
    tolerance: dict = measure_tolerance(folders, str(originals))
    print(f"tolerance {tolerance['tolerance']:.2f} m "
          f"({time.perf_counter() - began:.0f} s)")
    speed: dict = measure_speed(folders, str(originals), tolerance["tolerance"])
    print(f"half window {speed['half_window']} s, too fast above "
          f"{speed['too_fast_speed']:.1f} m/s")
    noise: dict = measure_noise(folders, str(originals), NOISE_SEED)
    print(f"noise {noise['model']}")
    parameters: dict = {"tolerance": tolerance["tolerance"],
                        "half_window": speed["half_window"],
                        "too_fast_speed": speed["too_fast_speed"],
                        "noise": noise["model"]}
    out: Path = results / "learned"
    out.mkdir(parents=True, exist_ok=True)
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2))
    (out / "measurements.json").write_text(json.dumps(
        {"tolerance": tolerance, "speed": speed, "noise": noise}, indent=2))


def lattice_job(job: tuple) -> tuple[str, int, float]:
    """Build and save one lattice: (folder, originals, copy, out, parameters).
    copy -1 is the folder's own track; copy k >= 0 is the train original
    with noise draw k. Truth is kept for train only."""
    from learned.lattice import Builder, save
    from learned.measure import original_track
    from learned.noise import NoiseModel, noisy_copy
    folder, originals, copy, out, parameters = job
    began: float = time.perf_counter()
    graph = folder.graph()
    if copy < 0:
        track = folder.track(graph)
    else:
        random = np.random.default_rng([NOISE_SEED, int(folder.name()), copy])
        track = noisy_copy(original_track(folder, originals, graph),
                           NoiseModel(**parameters["noise"]), random)
    truth = folder.route(graph) if folder.split == "train" else None
    built = Builder(graph).lattice(track, parameters, truth)
    save(built, out)
    pairs: int = sum(int(f.shape[0] * f.shape[1]) for f in built.features)
    return out.name, pairs, time.perf_counter() - began


def lattices(corpus: Path, originals: Path, results: Path) -> None:
    from multiprocessing import Pool
    parameters: dict = json.loads((results / "learned" / "parameters.json").read_text())
    root: Path = results / "learned" / "lattices"
    jobs: list[tuple] = []
    for split in ["train", "student_test"]:
        for folder in Corpus.from_root(corpus).folders(split):
            copies: list[int] = [-1]
            if split == "train":
                copies += list(range(COPIES))
            for copy in copies:
                tag: str = "own" if copy < 0 else f"copy{copy}"
                out: Path = root / split / f"{folder.name()}_{tag}.npz"
                if not out.exists():
                    jobs.append((folder, str(originals), copy, out, parameters))
    jobs.sort(key=lambda job: job[2])     # every own track before any copy
    began: float = time.perf_counter()
    total: int = 0
    with Pool(WORKERS, maxtasksperchild=8) as pool:
        for name, pairs, took in pool.imap_unordered(lattice_job, jobs):
            total += pairs
            print(f"{name}: {pairs / 1e6:.2f}M pairs, {took:.1f} s", flush=True)
    print(f"{len(jobs)} lattices, {total / 1e6:.0f}M pairs, "
          f"{time.perf_counter() - began:.0f} s")


def prepare(corpus: Path, results: Path) -> None:
    """The float16 storage scale of every step feature (its 99.9th
    percentile magnitude over the train own lattices, / 1000) and the
    starting weights, into parameters.json."""
    from learned.lattice import Builder
    from learned.start import starting_weights
    out: Path = results / "learned"
    parameters: dict = json.loads((out / "parameters.json").read_text())
    own: list[Path] = sorted((out / "lattices" / "train").glob("*_own.npz"))
    random = np.random.default_rng(0)
    samples: list[np.ndarray] = []
    for path in own:
        with np.load(path) as data:
            features: np.ndarray = data["features"]
        pick = random.integers(0, len(features), min(len(features), SCALE_SAMPLE))
        samples.append(np.abs(features[pick]))
    scale: np.ndarray = np.maximum(
        np.percentile(np.concatenate(samples), 99.9, axis=0) / 1000, 1e-3)
    tracks = []
    for folder in Corpus.from_root(corpus).folders("train"):
        graph = folder.graph()
        tracks.append(Builder(graph).decision_track(folder.track(graph), parameters))
    parameters["feature_scale"] = scale.tolist()
    parameters["start_weights"] = starting_weights(tracks).tolist()
    (out / "parameters.json").write_text(json.dumps(parameters, indent=2))
    print("scale", np.round(scale, 3).tolist())
    print("w0", np.round(parameters["start_weights"], 3).tolist())


def cross_validate(corpus: Path, results: Path) -> None:
    """Grouped 5-fold CV of the direct search: the starting weights, the
    search over Woeltche's terms only, and over all weights. Every score is
    on held-out folders' own tracks. Rows: (setting, fold, folder)."""
    import csv
    from learned.cv import fit_direct, folds, held_out_rows
    from learned.evaluate import weighted_correct_fraction
    from learned.features import WEIGHTS
    out: Path = results / "learned"
    parameters: dict = json.loads((out / "parameters.json").read_text())
    scale: np.ndarray = np.array(parameters["feature_scale"])
    start: np.ndarray = np.array(parameters["start_weights"])
    root: Path = out / "lattices" / "train"
    folders = {f.name(): f for f in Corpus.from_root(corpus).folders("train")}
    log_file = (out / "cv.log").open("w")

    def log(text: str) -> None:
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    new_terms: list[str] = ["cover_fix", "cover_route", "speed", "too_fast", "turn_force"]
    settings: dict[str, np.ndarray | None] = {
        "start": None,
        "search_woeltche_terms": np.array([n in new_terms for n in WEIGHTS]),
        "search_all": np.zeros(len(WEIGHTS), dtype=bool)}
    rows: list[dict] = []
    for setting, frozen in settings.items():
        for k, test in enumerate(folds(list(folders))):
            train_names: list[str] = [n for n in sorted(folders) if n not in test]
            began: float = time.perf_counter()
            weights: np.ndarray = start
            if frozen is not None:
                weights = fit_direct(folders, root, train_names, start, scale, frozen, log)
            held = held_out_rows(folders, root, test, weights)
            rows.extend({"setting": setting, "fold": k, **r} for r in held)
            log(f"{setting} fold {k}: held-out CF {weighted_correct_fraction(held):.5f} "
                f"({time.perf_counter() - began:.0f} s)")
        mine = [r for r in rows if r["setting"] == setting]
        log(f"== {setting}: pooled held-out CF {weighted_correct_fraction(mine):.5f}")
    with (out / "cv_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cross_validate_copies(corpus: Path, results: Path, method: str = "grid") -> None:
    """Grouped 5-fold CV of the direct search with noisy copies in the
    objective (cached.py). Same folds and same held-out scoring (own tracks
    only) as `cv`, so the pooled numbers compare with cv.log's."""
    import csv
    from learned.cached import COPIES_IN_SEARCH, Evaluator, exact_search, search
    from learned.cv import folds
    run = exact_search if method == "exact" else search
    prefix: str = "exact" if method == "exact" else "copies"
    from learned.evaluate import weighted_correct_fraction
    from learned.features import WEIGHTS
    out: Path = results / "learned"
    parameters: dict = json.loads((out / "parameters.json").read_text())
    start: np.ndarray = np.array(parameters["start_weights"])
    root: Path = out / "lattices" / "train"
    folders = {f.name(): f for f in Corpus.from_root(corpus).folders("train")}
    log_file = (out / f"cv_{prefix}.log").open("a")

    def log(text: str) -> None:
        print(text, flush=True)
        log_file.write(text + "\n")
        log_file.flush()

    new_terms: list[str] = ["cover_fix", "cover_route", "speed", "too_fast", "turn_force"]
    settings: dict[str, np.ndarray] = {
        f"{prefix}_woeltche_terms": np.array([n in new_terms for n in WEIGHTS]),
        f"{prefix}_all": np.zeros(len(WEIGHTS), dtype=bool)}
    rows: list[dict] = []
    weights_out: dict = {}
    for setting, frozen in settings.items():
        for k, test in enumerate(folds(list(folders))):
            began: float = time.perf_counter()
            train_names: list[str] = [n for n in sorted(folders) if n not in test]
            evaluator = Evaluator([folders[n] for n in train_names], root, COPIES_IN_SEARCH)
            weights: np.ndarray = run(evaluator, start, frozen, log)
            evaluator.close()
            held = Evaluator([folders[n] for n in test], root, 0)
            held_rows = held.rows(weights)
            held.close()
            for r in held_rows:
                r["id"] = r.pop("lattice").replace("_own", "")
            rows.extend({"setting": setting, "fold": k, **r} for r in held_rows)
            weights_out[f"{setting}/{k}"] = weights.tolist()
            log(f"{setting} fold {k}: held-out CF {weighted_correct_fraction(held_rows):.5f} "
                f"({time.perf_counter() - began:.0f} s)")
        mine = [r for r in rows if r["setting"] == setting]
        log(f"== {setting}: pooled held-out CF {weighted_correct_fraction(mine):.5f}")
        with (out / f"cv_{prefix}_rows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (out / f"cv_{prefix}_weights.json").write_text(json.dumps(weights_out, indent=2))


def resume_exact(corpus: Path, results: Path) -> None:
    """Finish exact_woeltche_terms after the run was stopped in its last
    fold: folds whose search finished keep the weights of their last logged
    sweep (only their held-out rows are recomputed); the others are searched
    again. Same folds, start and scoring as cv_exact."""
    import csv
    import re
    from learned.cached import COPIES_IN_SEARCH, Evaluator, exact_search
    from learned.cv import folds
    from learned.evaluate import weighted_correct_fraction
    from learned.features import WEIGHTS
    out: Path = results / "learned"
    setting: str = "exact_woeltche_terms"
    text: str = (out / "cv_exact.log").read_text()
    done: dict[int, np.ndarray] = {}
    for sweeps, k in re.findall(r"((?:sweep \d: error [\d.]+, w \[[^\]]+\]\n)+)"
                                + setting + r" fold (\d): held-out", text):
        last: str = re.findall(r"w \[([^\]]+)\]", sweeps)[-1]
        done[int(k)] = np.array([float(v) for v in last.split(",")])
    parameters: dict = json.loads((out / "parameters.json").read_text())
    start: np.ndarray = np.array(parameters["start_weights"])
    root: Path = out / "lattices" / "train"
    folders = {f.name(): f for f in Corpus.from_root(corpus).folders("train")}
    new_terms: list[str] = ["cover_fix", "cover_route", "speed", "too_fast", "turn_force"]
    frozen: np.ndarray = np.array([n in new_terms for n in WEIGHTS])
    log_file = (out / "cv_exact.log").open("a")

    def log(line: str) -> None:
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"-- resumed: folds {sorted(done)} kept from the log")
    rows: list[dict] = []
    weights_out: dict = {}
    for k, test in enumerate(folds(list(folders))):
        began: float = time.perf_counter()
        if k in done:
            weights = done[k]
        else:
            train_names = [n for n in sorted(folders) if n not in test]
            evaluator = Evaluator([folders[n] for n in train_names], root, COPIES_IN_SEARCH)
            weights = exact_search(evaluator, start, frozen, log)
            evaluator.close()
        held = Evaluator([folders[n] for n in test], root, 0)
        held_rows = held.rows(weights)
        held.close()
        for r in held_rows:
            r["id"] = r.pop("lattice").replace("_own", "")
        rows.extend({"setting": setting, "fold": k, **r} for r in held_rows)
        weights_out[f"{setting}/{k}"] = weights.tolist()
        log(f"{setting} fold {k}: held-out CF {weighted_correct_fraction(held_rows):.5f} "
            f"({time.perf_counter() - began:.0f} s){' [kept]' if k in done else ''}")
    log(f"== {setting}: pooled held-out CF {weighted_correct_fraction(rows):.5f}")
    with (out / "cv_exact_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "cv_exact_weights.json").write_text(json.dumps(weights_out, indent=2))


def final_train_copies(corpus: Path, results: Path, setting: str, method: str = "grid") -> None:
    """The search of `setting` on all 80 train folders and their copies;
    the result goes to trained_weights.json (weights.json is only replaced
    by hand, once cross-validation says the search helps)."""
    from learned.cached import COPIES_IN_SEARCH, Evaluator, exact_search, search
    from learned.features import WEIGHTS
    run = exact_search if method == "exact" else search
    out: Path = results / "learned"
    parameters: dict = json.loads((out / "parameters.json").read_text())
    new_terms: list[str] = ["cover_fix", "cover_route", "speed", "too_fast", "turn_force"]
    frozen: np.ndarray = (np.array([n in new_terms for n in WEIGHTS])
                          if setting.endswith("woeltche_terms")
                          else np.zeros(len(WEIGHTS), dtype=bool))
    folders = list(Corpus.from_root(corpus).folders("train"))
    evaluator = Evaluator(folders, out / "lattices" / "train", COPIES_IN_SEARCH)
    weights: np.ndarray = run(evaluator, np.array(parameters["start_weights"]), frozen, print)
    evaluator.close()
    (out / "trained_weights.json").write_text(json.dumps(
        {"setting": setting, "weights": dict(zip(WEIGHTS, weights.tolist()))}, indent=2))


def final_train(corpus: Path, results: Path) -> None:
    """The direct search over all weights on all 80 train folders."""
    from learned.cv import fit_direct
    from learned.features import WEIGHTS
    out: Path = results / "learned"
    parameters: dict = json.loads((out / "parameters.json").read_text())
    folders = {f.name(): f for f in Corpus.from_root(corpus).folders("train")}
    weights: np.ndarray = fit_direct(
        folders, out / "lattices" / "train", sorted(folders),
        np.array(parameters["start_weights"]), np.array(parameters["feature_scale"]),
        np.zeros(len(WEIGHTS), dtype=bool), print)
    (out / "weights.json").write_text(json.dumps(dict(zip(WEIGHTS, weights.tolist())), indent=2))


def freeze(results: Path) -> None:
    """weights.json = the starting weights, untrained (the prototype)."""
    from learned.features import WEIGHTS
    out: Path = results / "learned"
    parameters: dict = json.loads((out / "parameters.json").read_text())
    (out / "weights.json").write_text(json.dumps(
        dict(zip(WEIGHTS, parameters["start_weights"])), indent=2))


def match_job(job: tuple) -> str:
    from learned.evaluate import match
    folder, path, weights, out = job
    match(folder, path, weights).write(out)
    return folder.name()


def match_all(corpus: Path, results: Path) -> None:
    """Routes for every folder from the final weights, where the benchmark
    reads them: results/learned/routes/<split>/<id>.route."""
    from multiprocessing import Pool
    out: Path = results / "learned"
    weights: np.ndarray = np.array(list(json.loads((out / "weights.json").read_text()).values()))
    jobs: list[tuple] = []
    for split in ["train", "student_test"]:
        target: Path = out / "routes" / split
        target.mkdir(parents=True, exist_ok=True)
        for folder in Corpus.from_root(corpus).folders(split):
            jobs.append((folder, out / "lattices" / split / f"{folder.name()}_own.npz",
                         weights, target / f"{folder.name()}.route"))
    with Pool(WORKERS) as pool:
        for name in pool.imap_unordered(match_job, jobs):
            print(name, flush=True)


def main(arguments: list[str]) -> None:
    stage: str = arguments[0]
    if stage == "measure":
        measure(Path(arguments[1]), Path(arguments[2]), Path(arguments[3]))
    elif stage == "lattices":
        lattices(Path(arguments[1]), Path(arguments[2]), Path(arguments[3]))
    elif stage == "prepare":
        prepare(Path(arguments[1]), Path(arguments[2]))
    elif stage == "freeze":
        freeze(Path(arguments[1]))
    elif stage == "cv":
        cross_validate(Path(arguments[1]), Path(arguments[2]))
    elif stage == "cv_copies":
        cross_validate_copies(Path(arguments[1]), Path(arguments[2]))
    elif stage == "cv_exact":
        cross_validate_copies(Path(arguments[1]), Path(arguments[2]), "exact")
    elif stage == "train_copies":
        final_train_copies(Path(arguments[1]), Path(arguments[2]), arguments[3])
    elif stage == "resume_exact":
        resume_exact(Path(arguments[1]), Path(arguments[2]))
    elif stage == "train_exact":
        final_train_copies(Path(arguments[1]), Path(arguments[2]), arguments[3], "exact")
    elif stage == "train":
        final_train(Path(arguments[1]), Path(arguments[2]))
    elif stage == "match":
        match_all(Path(arguments[1]), Path(arguments[2]))
    else:
        raise SystemExit(f"unknown stage {stage!r}; see the module docstring")


if __name__ == "__main__":
    main(sys.argv[1:])
