"""Routes from results/learned/trained_weights.json, written beside (not
over) the shipped model's, so the two can be benchmarked side by side:

    uv run python -m learned.score_trained <corpus_fixed> <held_out_noisy root> results
    then: python -m benchmark <corpus_fixed> results learned_trained
          python -m benchmark <held_out_noisy root> results/held_out_noisy learned_trained

train and student_test use their stored lattices; the noisy held-out tracks
have none, so their lattices are built in memory as learned.serving does.
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from dataset import Corpus
from learned.hyperparameters import WORKERS

NAME: str = "learned_trained"


def weights(results: Path) -> np.ndarray:
    data: dict = json.loads((results / "learned" / "trained_weights.json").read_text())
    return np.array(list(data["weights"].values()))


def stored(job: tuple) -> str:
    from learned.evaluate import match
    folder, lattice, w, out = job
    match(folder, lattice, w).write(out)
    return folder.name()


def built(job: tuple) -> str:
    from learned.decode import drive
    from learned.evaluate import labelling
    from learned.lattice import Builder
    from learned.network import Network
    from learned.routes import Router
    folder, parameters, w, out = job
    graph = folder.graph()
    lattice = Builder(graph).lattice(folder.track(graph), parameters, None)
    route = drive(labelling(lattice, w), lattice.points, Router(Network.from_graph(graph)))
    out.parent.mkdir(parents=True, exist_ok=True)
    route.write(out)
    return folder.name()


def main(corpus: Path, held_out: Path, results: Path) -> None:
    w: np.ndarray = weights(results)
    parameters: dict = json.loads((results / "learned" / "parameters.json").read_text())
    jobs_stored: list[tuple] = []
    for split in ["train", "student_test"]:
        target: Path = results / NAME / "routes" / split
        target.mkdir(parents=True, exist_ok=True)
        for folder in Corpus.from_root(corpus).folders(split):
            jobs_stored.append((folder, results / "learned" / "lattices" / split
                                / f"{folder.name()}_own.npz", w, target / f"{folder.name()}.route"))
    noisy = Corpus.from_root(held_out)
    jobs_built: list[tuple] = [
        (f, parameters, w, results / "held_out_noisy" / NAME / "routes" / s / f"{f.name()}.route")
        for s in noisy.splits for f in noisy.folders(s)]
    with Pool(WORKERS) as pool:
        print(len(pool.map(stored, jobs_stored)), "stored-lattice routes")
        print(len(pool.map(built, jobs_built)), "held-out routes")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
