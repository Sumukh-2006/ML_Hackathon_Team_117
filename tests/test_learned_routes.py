"""learned/ rebuilds Woeltche's candidates and routes; with his settings and
his decision points it must reproduce mdp/ exactly, pair for pair."""

from pathlib import Path

import numpy as np

import mdp.candidates
import mdp.match
import mdp.network
import mdp.routing
from dataset import Corpus, Folder
from learned.candidates import adopt, search
from learned.hyperparameters import (GRID_CELL, K_NEAREST, RADIUS,
                                     RADIUS_UPPER, ROUTING_MAX_DISTANCE_FACTOR)
from learned.network import ArcGrid, Network
from learned.routes import Router

CORPUS: Path = Path(__file__).parent.parent / "data" / "corpus_fixed"
TRACKS: list[tuple[str, str]] = [("train", "00000003"),
                                 ("student_test", "00000007"),
                                 ("train", "00000020")]


def folder(split: str, name: str) -> Folder:
    for each in Corpus.from_root(CORPUS).folders(split):
        if each.name() == name:
            return each
    raise LookupError(name)


def test_candidates_and_routes_equal_mdp() -> None:
    for split, name in TRACKS:
        graph = folder(split, name).graph()
        track = folder(split, name).track(graph)
        old_grid = mdp.network.ArcGrid(graph, GRID_CELL)
        shape = mdp.match.cleaned_shape(track, old_grid)
        old_edges = mdp.network.road_edges(graph)

        network = Network.from_graph(graph)
        assert np.array_equal(network.road, old_edges)
        grid = ArcGrid(network, GRID_CELL)
        new_own, old_own = [], []
        for i in range(shape.point_count()):
            x, y = float(shape.x[i]), float(shape.y[i])
            new_own.append(search(grid, x, y, K_NEAREST, RADIUS, RADIUS_UPPER))
            old_own.append(mdp.candidates.search(old_grid, old_edges, x, y,
                                                 K_NEAREST, RADIUS,
                                                 RADIUS_UPPER))
        new_sets = adopt(new_own, shape.x, shape.y, GRID_CELL)
        old_sets = mdp.candidates.adopt(old_own, shape.x, shape.y, GRID_CELL)

        router = Router(network)
        old_router = mdp.routing.Router(graph)
        for j in range(shape.point_count() - 1):
            a, b = new_sets[j], new_sets[j + 1]
            assert np.array_equal(a.arc, old_sets[j].arc)
            assert np.array_equal(a.along, old_sets[j].along)
            lengths, changes, _ = router.step(a.arc, a.along, b.arc, b.along,
                                              ROUTING_MAX_DISTANCE_FACTOR)
            for c in range(a.count()):
                bound = old_router.bound(int(a.arc[c]), b.arc,
                                         ROUTING_MAX_DISTANCE_FACTOR)
                old_lengths, old_changes = old_router.routes(
                    int(a.arc[c]), float(a.along[c]), b.arc, b.along, bound)
                assert np.array_equal(lengths[c], old_lengths), (name, j, c)
                assert np.array_equal(changes[c], old_changes), (name, j, c)

            c, d = 0, int(np.argmin(lengths[0]))
            if np.isfinite(lengths[0, d]):
                bound = router.bound(int(a.arc[c]), b.arc,
                                     ROUTING_MAX_DISTANCE_FACTOR)
                new_path = router.arcs(int(a.arc[c]), float(a.along[c]),
                                       int(b.arc[d]), float(b.along[d]), bound)
                old_path = old_router.path(int(a.arc[c]), float(a.along[c]),
                                           int(b.arc[d]), float(b.along[d]),
                                           bound)
                assert new_path.tolist() == old_path, (name, j)
