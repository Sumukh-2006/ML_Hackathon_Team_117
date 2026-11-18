"""Streaming HMM == batch HMM: every fix pushed one at a time, then
finish, gives the route `python -m hmm` wrote to results/hmm/routes."""

import json
from pathlib import Path

import numpy as np
import pytest

from dataset import Corpus, Route
from hmm.online import OnlineHMM

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
RESULTS = ROOT / "results" / "hmm"
# Folders with breaks the repair cannot heal (10, 2, 185 and 16 trips),
# hives (33), and one clean track.
FOLDERS = ["00000001", "00000002", "00000032", "00000081", "00000033",
           "00000005"]


def folder(name: str):
    for each in Corpus.from_root(CORPUS).folders("train"):
        if each.name() == name:
            return each
    raise KeyError(name)


def stream(name: str, chunk: int = 1) -> tuple[list[int], list[int]]:
    """(streamed route, committed arcs as they arrived)."""
    parameters = json.loads((RESULTS / "parameters.json").read_text())
    f = folder(name)
    graph = f.graph()
    track = f.track(graph)
    online = OnlineHMM(graph, parameters["sigma"], parameters["beta"])
    seen: list[int] = []
    fixes = list(zip(track.lon.tolist(), track.lat.tolist(), track.t.tolist()))
    for start in range(0, len(fixes), chunk):
        update = online.push(fixes[start:start + chunk])
        seen += update.committed
    route, _ = online.finish()
    return route.arcs.tolist(), seen


@pytest.mark.skipif(not (RESULTS / "parameters.json").exists(),
                    reason="run `python -m hmm` first")
@pytest.mark.parametrize("name", FOLDERS)
def test_streaming_equals_batch(name: str) -> None:
    batch = (RESULTS / "routes" / "train" / f"{name}.route").read_text().split()
    route, seen = stream(name)
    assert route == [int(a) for a in batch]
    # What was committed during streaming is a prefix of the final route.
    assert seen == route[:len(seen)]
