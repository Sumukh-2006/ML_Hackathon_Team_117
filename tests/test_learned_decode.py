"""With Woeltche's weights, fail cost, fresh-start factors and discount,
learned/'s cleaning, lattice and decoder must give mdp/'s routes."""

from pathlib import Path

import numpy as np

import mdp.match
from dataset import Corpus, Folder
from learned.decode import decode, drive
from learned.features import step_features
from learned.hyperparameters import (ROUTING_MAX_DISTANCE_FACTOR,
                                     SKIP_ROUTE_FACTOR)
from learned.kernel import step_lattice
from learned.lattice import Builder
from mdp.geometry import TrackShape
from mdp.rewards import fail_errors

CORPUS: Path = Path(__file__).parent.parent / "data" / "corpus_fixed"
TRACKS: list[tuple[str, str]] = [("train", "00000003"),
                                 ("student_test", "00000007"),
                                 ("train", "00000020")]
# distance, last_distance, length, heading, turn, turning; the rest are 0.
WOELTCHE: np.ndarray = np.array([0.8, 0.8, 0.3, 0.5, 0.5, 0.6, 0, 0, 0, 0, 0])
PARAMETERS: dict = {"tolerance": 5.0, "half_window": 5.0,
                    "too_fast_speed": 1e9}


def folder(split: str, name: str) -> Folder:
    for each in Corpus.from_root(CORPUS).folders(split):
        if each.name() == name:
            return each
    raise LookupError(name)


def test_woeltche_weights_reproduce_mdp() -> None:
    for split, name in TRACKS:
        graph = folder(split, name).graph()
        track = folder(split, name).track(graph)
        builder = Builder(graph)
        decided = builder.decision_track(track, PARAMETERS)
        points = builder.candidates(decided)
        fail = fail_errors(TrackShape.from_points(decided.x, decided.y))
        net, router = builder.network, builder.router

        costs, skips = [], []
        for j in range(decided.point_count() - 1):
            a, b = points[j], points[j + 1]
            fix_x, fix_y = decided.step_fixes(j)
            lengths, changes, sharp, cover_fix, cover_route, _ = step_lattice(
                a.arc, a.along, a.px, a.py, b.arc, b.along, b.px, b.py,
                ROUTING_MAX_DISTANCE_FACTOR, fix_x, fix_y,
                np.zeros(0, dtype=np.int64), False, graph.canonical_arc,
                net.arc_from, net.arc_to, net.length, net.heading, net.moves,
                net.node_x, net.node_y, net.out_start, net.out_arc,
                router.distance, router.entry, router.turning,
                router.sharpest, router.touched, builder.tally, builder.path,
                builder.vx, builder.vy)
            features, skip = step_features(j, decided, a, b, lengths, changes,
                                           sharp, cover_fix, cover_route,
                                           PARAMETERS["too_fast_speed"],
                                           SKIP_ROUTE_FACTOR)
            cost = features @ WOELTCHE
            costs.append(np.where(np.isfinite(lengths), cost, fail[j]))
            skips.append(skip)
        fresh = [0.8 * 0.1 * p.distance for p in points]
        fresh[0] = 0.8 * 3.0 * points[0].distance

        ours = drive(decode(costs, skips, fresh, 0.99), points, router)
        theirs, _ = mdp.match.match(graph, track)
        assert np.array_equal(ours.arcs, theirs.arcs), name
