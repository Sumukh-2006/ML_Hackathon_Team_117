"""Invariants that caught two real bugs during development. Run: python -m app.test_smoke

Each assert is a claim the UI makes on screen. If one fails, the sheet is lying.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from . import config, match
from .build_payloads import route_flags


def check(trip_id: str, partition: str) -> dict:
    base = config.TEST_DIR if partition == "test" else config.TRAIN_DIR
    trip = match.build_trip(os.path.join(base, trip_id))
    pred, dist = match.match(trip)
    m = match.score(trip, pred, dist)
    on, join = route_flags(trip, pred)

    a, p = m["arcs"], m["point"]

    # 1. counting roads either way round can only be more generous than strict
    assert a["recall_dir"] >= a["recall_strict"], (trip_id, a)
    assert p["top1_dir"] >= p["top1_strict"], (trip_id, p)

    # 2. you cannot miss more roads than segments: the road-level count is a
    #    coarser partition, so it must be <= the strict count
    assert a["missed_arcs_dir"] <= a["missed_arcs"], (trip_id, a)
    assert a["wrong_way_only"] == a["missed_arcs"] - a["missed_arcs_dir"]

    # 3. per-fix direction flags agree with the arc scores
    n_dir = int(((on == 1) | (on == 2)).sum())
    assert abs(n_dir / len(on) - p["top1_dir"]) < 0.005, (trip_id, n_dir, p["top1_dir"])
    n_strict = int((on == 1).sum())
    assert abs(n_strict / len(on) - p["top1_strict"]) < 0.005, (trip_id, n_strict)

    # 4. a correct-direction fix must exist for every route arc counted as hit
    assert (on == 1).any() or a["recall_strict"] == 0

    # 5. continuity breaks are a subset of "not continuous", never negative
    assert 0 <= m["continuity"]["breaks"] < len(pred)

    print(f"{trip_id} {partition:5s} "
          f"strict={a['recall_strict']:.3f} dir={a['recall_dir']:.3f} "
          f"missed={a['missed_arcs_dir']}/{a['missed_arcs']} "
          f"breaks={m['continuity']['breaks']} OK")
    return m


def check_payload(trip_id: str, expected: dict) -> None:
    path = os.path.join(config.DATA_DIR, f"trip_{trip_id}.json")
    with open(path, encoding="utf-8") as f:
        pay = json.load(f)
    assert pay["metrics"]["arcs"] == expected["arcs"], f"{path} is stale; rebuild payloads"
    assert pay["pred"]["on"], "payload missing per-fix direction flags"
    assert len(pay["network"]["xy"]) == pay["network"]["n"] * 4
    assert len(pay["pred"]["arc"]) == pay["track"]["n"]
    print(f"payload {trip_id}: {os.path.getsize(path)//1024} KB OK")


def main() -> int:
    results = {
        "00000001": check("00000001", "train"),
        "00000007": check("00000007", "test"),
    }
    for tid, expected in results.items():
        path = os.path.join(config.DATA_DIR, f"trip_{tid}.json")
        if os.path.exists(path):
            check_payload(tid, expected)
        else:
            print(f"payload {tid} missing (run build_payloads)")
    print("all invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
