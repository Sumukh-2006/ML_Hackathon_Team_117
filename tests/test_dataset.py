"""The parser reads well-formed files exactly and names the line of a bad one."""

from pathlib import Path

import numpy as np
import pytest

from dataset import Corpus, DatasetError, Folder, RoadGraph, Route
from dataset.table import read_table

CORPUS_ROOT: Path = Path(__file__).parent.parent / "data" / "corpus"


def write_folder(root: Path, name: str, nodes: str, arcs: str,
                 track: str) -> Path:
    folder: Path = root / name
    folder.mkdir()
    (folder / f"{name}.nodes").write_text(nodes)
    (folder / f"{name}.arcs").write_text(arcs)
    (folder / f"{name}.track").write_text(track)
    return folder


def test_wrong_field_count_names_the_line(tmp_path: Path) -> None:
    path: Path = tmp_path / "x.arcs"
    path.write_text("0\t1\n1\n")
    with pytest.raises(DatasetError, match=r"x\.arcs:2: expected 2 fields"):
        read_table(path, [int, int])


def test_unparsable_field_names_the_line(tmp_path: Path) -> None:
    path: Path = tmp_path / "x.nodes"
    path.write_text("8.5\t49.8\n8.6\tabc\n")
    with pytest.raises(DatasetError, match=r"x\.nodes:2: field 2"):
        read_table(path, [float, float])


def test_arc_to_missing_node_is_rejected(tmp_path: Path) -> None:
    folder: Path = write_folder(tmp_path, "7", "8.5 49.8\n8.6 49.8\n",
                                "0 1\n1 5\n", "8.5 49.8 1\n")
    with pytest.raises(DatasetError, match=r"7\.arcs:2: arc joins node 1"):
        RoadGraph.from_folder(folder)


def test_twins_and_reverses(tmp_path: Path) -> None:
    folder: Path = write_folder(tmp_path, "7",
                                "8.5 49.8\n8.6 49.8\n8.6 49.9\n",
                                "0 1\n1 0\n0 1\n1 2\n", "8.5 49.8 1\n")
    graph: RoadGraph = RoadGraph.from_folder(folder)
    assert graph.canonical_arc.tolist() == [0, 1, 0, 3]
    assert graph.reverse_arc.tolist() == [1, 0, 1, -1]
    assert sorted(graph.out_arcs(0).tolist()) == [0, 2]


def test_route_breaks_and_length(tmp_path: Path) -> None:
    folder: Path = write_folder(tmp_path, "7",
                                "0 0\n0.001 0\n0.002 0\n",
                                "0 1\n1 2\n2 1\n", "0 0 1\n")
    graph: RoadGraph = RoadGraph.from_folder(folder)
    route: Route = Route(np.array([0, 1, 0]))
    assert route.breaks(graph).tolist() == [1]
    assert route.length(graph) == pytest.approx(3 * 111.195, rel=1e-3)


def test_whole_corpus_parses() -> None:
    corpus: Corpus = Corpus.from_root(CORPUS_ROOT)
    count: int = 0
    for split in corpus.splits:
        for folder in corpus.folders(split):
            graph: RoadGraph = folder.graph()
            folder.track(graph)
            folder.route(graph)
            count += 1
    assert count == 90


def test_hand_tags_are_read_and_checked(tmp_path: Path) -> None:
    folder_path: Path = write_folder(tmp_path, "7", "8.5 49.8\n",
                                     "", "8.5 49.8 1\n")
    folder: Folder = Folder("train", folder_path)
    (folder_path / "7.tags").write_text("feature=gaps\nfeature=loops\n")
    assert folder.tags() == ["feature=gaps", "feature=loops"]
    (folder_path / "7.tags").write_text("")
    assert folder.tags() == []
    (folder_path / "7.tags").write_text("gaps\n")
    with pytest.raises(DatasetError, match=r"7\.tags:1: expected key=value"):
        folder.tags()
