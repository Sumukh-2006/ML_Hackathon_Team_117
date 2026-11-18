"""Score an approach's predicted routes against the true routes.

Usage:
  uv run python -m benchmark <corpus root> <results root> <approach>
  uv run python -m benchmark <corpus root> <results root> --truth

Predictions are read from <results root>/<approach>/routes/<split>/<id>.route;
`--truth` scores every true route against itself instead. The per-track
rows and the totals are written to <results root>/benchmark/<approach>/
before the totals are printed. A folder with no prediction is scored as an
empty route and counted as missing; a folder with no true route (held out)
cannot be scored and is left out. Scores are grouped by the original
dataset's hand tags, which only a fixed corpus (`python -m fixes`) carries.
"""

import sys
from pathlib import Path

import numpy as np

from benchmark.metrics import RouteScore, score_route
from benchmark.report import (SUMMARY_COLUMNS, TRACK_COLUMNS, print_summary,
                              summary_rows, tag_text, track_row, write_csv)
from dataset import Corpus, Folder, RoadGraph, Route


def main() -> None:
    corpus_root, results_root, approach = arguments(sys.argv[1:])
    corpus: Corpus = Corpus.from_root(corpus_root)

    rows: list[dict] = score_corpus(corpus, results_root, approach)
    summary: list[dict] = summary_rows(rows)

    output: Path = results_root / "benchmark" / approach
    write_csv(rows, TRACK_COLUMNS, output / "per_track.csv")
    write_csv(summary, SUMMARY_COLUMNS, output / "summary.csv")
    print_summary(summary)


def arguments(words: list[str]) -> tuple[Path, Path, str]:
    if len(words) != 3:
        sys.exit("usage: python -m benchmark <corpus root> <results root> "
                 "<approach | --truth>")
    approach: str = "truth" if words[2] == "--truth" else words[2]
    return Path(words[0]), Path(words[1]), approach


def score_corpus(corpus: Corpus, results_root: Path,
                 approach: str) -> list[dict]:
    rows: list[dict] = []
    for split in corpus.splits:
        for folder in corpus.folders(split):
            if not folder.has_route():
                continue
            rows.append(score_folder(folder, results_root, approach))
    return rows


def score_folder(folder: Folder, results_root: Path, approach: str) -> dict:
    """Score one folder's prediction, labelled with its hand tags."""
    graph: RoadGraph = folder.graph()
    truth: Route = folder.route(graph)

    predicted, missing = prediction(folder, graph, truth, results_root,
                                    approach)
    score: RouteScore = score_route(graph, predicted, truth)

    tags: list[str] | None = None
    if folder.has_tags():
        tags = folder.tags()
    return track_row(folder.split, folder.name(), tag_text(tags), score,
                     missing, len(predicted.breaks(graph)))


def prediction(folder: Folder, graph: RoadGraph, truth: Route,
               results_root: Path, approach: str) -> tuple[Route, bool]:
    """The predicted route, and whether it was missing."""
    if approach == "truth":
        return truth, False
    path: Path = (results_root / approach / "routes" / folder.split
                  / f"{folder.name()}.route")
    if not path.exists():
        return Route(np.zeros(0, dtype=np.int64)), True
    return Route.from_file(path, graph), False


if __name__ == "__main__":
    main()
