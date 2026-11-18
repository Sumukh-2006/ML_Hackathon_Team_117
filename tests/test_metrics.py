"""The scorer gives exact lengths on routes small enough to check by hand."""

from pathlib import Path

import numpy as np
import pytest

from benchmark.metrics import score_route
from dataset import RoadGraph, Route


def line_graph(tmp_path: Path) -> RoadGraph:
    """Nodes 0-1-2-3 east along the equator, 0.001 degrees (~111 m) apart.
    Arcs: 0:0>1  1:1>2  2:2>3  3:1>0  4:0>1 (a twin of arc 0)."""
    folder: Path = tmp_path / "1"
    folder.mkdir()
    (folder / "1.nodes").write_text("0 0\n0.001 0\n0.002 0\n0.003 0\n")
    (folder / "1.arcs").write_text("0 1\n1 2\n2 3\n1 0\n0 1\n")
    (folder / "1.track").write_text("0 0 1\n")
    return RoadGraph.from_folder(folder)


def test_identical_routes_score_perfectly(tmp_path: Path) -> None:
    graph: RoadGraph = line_graph(tmp_path)
    route: Route = Route(np.array([0, 1, 2]))
    score = score_route(graph, route, route)
    assert score.correct_fraction() == 1.0
    assert score.mismatch_fraction() == 0.0


def test_added_and_missed_lengths(tmp_path: Path) -> None:
    graph: RoadGraph = line_graph(tmp_path)
    arc: float = float(graph.arc_length[0])
    truth: Route = Route(np.array([0, 1, 2]))
    predicted: Route = Route(np.array([0, 1, 3]))
    score = score_route(graph, predicted, truth)
    assert score.overlap == pytest.approx(2 * arc)
    assert score.added == pytest.approx(arc)
    assert score.missed == pytest.approx(arc)
    assert score.correct_fraction() == pytest.approx(2 / 3)
    assert score.mismatch_fraction() == pytest.approx(2 / 3)


def test_twins_are_the_same_road_and_repeats_count(tmp_path: Path) -> None:
    graph: RoadGraph = line_graph(tmp_path)
    truth: Route = Route(np.array([0, 3, 0]))
    predicted: Route = Route(np.array([4, 3, 4]))
    assert score_route(graph, predicted, truth).correct_fraction() == 1.0
    once: Route = Route(np.array([0]))
    assert score_route(graph, once, truth).recall() == pytest.approx(1 / 3)


def test_empty_prediction_scores_zero(tmp_path: Path) -> None:
    graph: RoadGraph = line_graph(tmp_path)
    score = score_route(graph, Route(np.zeros(0, dtype=np.int64)),
                        Route(np.array([0, 1])))
    assert score.correct_fraction() == 0.0
    assert score.precision() == 0.0
    assert score.mismatch_fraction() == 1.0
