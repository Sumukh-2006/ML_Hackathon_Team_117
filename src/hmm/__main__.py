"""Newson & Krumm's HMM matcher: measure sigma and beta on train, then match
every folder of every split.

Usage: uv run python -m hmm <corpus root> <results root>

Writes <results root>/hmm/parameters.json, routes/<split>/<id>.route and
timing.csv, each before anything is printed.
"""

import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from dataset import Corpus, Folder, RoadGraph, Route
from hmm.fit import measure
from hmm.hyperparameters import WORKERS
from hmm.match import match


def main() -> None:
    corpus_root, output = arguments(sys.argv[1:])
    corpus: Corpus = Corpus.from_root(corpus_root)

    parameters: dict = measure(corpus, "train")
    output.mkdir(parents=True, exist_ok=True)
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2))

    jobs: list[tuple] = folder_jobs(corpus, parameters, output)
    with Pool(WORKERS) as pool:
        rows: list[dict] = pool.map(match_folder, jobs, chunksize=1)
    write_timing(rows, output / "timing.csv")
    print(f"sigma {parameters['sigma']:.2f} m, beta {parameters['beta']:.2f} m;"
          f" matched {len(rows)} folders in {output}")


def arguments(words: list[str]) -> tuple[Path, Path]:
    if len(words) != 2:
        sys.exit("usage: python -m hmm <corpus root> <results root>")
    return Path(words[0]), Path(words[1]) / "hmm"


def folder_jobs(corpus: Corpus, parameters: dict,
                output: Path) -> list[tuple]:
    """One job per folder, longest track first so the pool stays busy."""
    jobs: list[tuple] = []
    for split in corpus.splits:
        for folder in corpus.folders(split):
            size: int = (folder.path / f"{folder.name()}.track").stat().st_size
            target: Path = output / "routes" / split / f"{folder.name()}.route"
            jobs.append((size, folder, parameters["sigma"],
                         parameters["beta"], target))
    jobs.sort(key=job_size, reverse=True)
    return jobs


def job_size(job: tuple) -> int:
    return job[0]


def match_folder(job: tuple) -> dict:
    """Match one folder, write its route, and time it."""
    _, folder, sigma, beta, target = job
    started: float = time.perf_counter()
    graph: RoadGraph = folder.graph()
    route, counts = match(graph, folder.track(graph), sigma, beta)
    route.write(target)
    counts["seconds"] = round(time.perf_counter() - started, 2)
    counts["split"] = folder.split
    counts["id"] = folder.name()
    return counts


def write_timing(rows: list[dict], path: Path) -> None:
    columns: list[str] = ["split", "id", "fixes", "kept", "matched", "trips",
                          "seconds"]
    with path.open("w", newline="") as handle:
        writer: csv.DictWriter = csv.DictWriter(handle, columns)
        writer.writeheader()
        for row in sorted(rows, key=row_name):
            writer.writerow(row)


def row_name(row: dict) -> str:
    return row["split"] + row["id"]


if __name__ == "__main__":
    main()
