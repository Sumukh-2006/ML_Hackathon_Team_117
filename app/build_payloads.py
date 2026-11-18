"""Precompute one JSON payload per trip for the UI.

Rationale: the browser must never hold a 307k-arc network, so the payload carries
only the arcs within 250 m of the drive plus every arc the reference route uses.
Matching itself still runs against every arc within 1 km of the trace, so the
narrower view changes what is drawn, not what the number means.

Run:  python -m app.build_payloads              # all 90 trips
      python -m app.build_payloads --only 00000001,00000007
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from . import config, match


def corridor_keep(trip: match.Trip) -> np.ndarray:
    """Arcs with either endpoint within CORRIDOR of a fix, plus every route arc.

    ⚠️ assumption: the drawn network is a display subset, not the matching graph.
    """
    lon0, lat0 = trip.lon0, trip.lat0
    nx, ny = match.to_xy(trip.nodes, lon0, lat0)
    A = np.column_stack([nx[trip.arcs[:, 0]], ny[trip.arcs[:, 0]]])
    B = np.column_stack([nx[trip.arcs[:, 1]], ny[trip.arcs[:, 1]]])
    tree = cKDTree(trip.track_xy)
    near = np.zeros(len(trip.arcs), bool)
    for pts in (A, B):
        d, _ = tree.query(pts, k=1, workers=-1)
        near |= d < config.CORRIDOR
    near[np.unique(trip.route)] = True
    return near


def r(x, dp: int = config.COORD_DP):
    return [round(float(v), dp) for v in x]


def route_flags(trip: match.Trip, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-fix direction status and connectivity, so the UI never re-derives labels.

    on   1 = the snapped arc is a reference arc in travel direction
         2 = it is the reverse twin of a reference arc (right road, wrong way)
         0 = neither
    join 1 = this snap shares a node with the previous one (one continuous drive)
    """
    route_set = np.unique(trip.route)
    N = len(trip.nodes)
    key = trip.arcs[:, 0] * N + trip.arcs[:, 1]
    rev = trip.arcs[:, 1] * N + trip.arcs[:, 0]
    route_key = np.unique(key[route_set])
    # route_both is the road set: every route direction plus its reverse
    route_both = np.unique(np.concatenate([route_key, rev[route_set]]))
    on = np.where(
        np.isin(pred, route_set), 1, np.where(np.isin(key[pred], route_both), 2, 0)
    ).astype(np.int8)
    join = np.ones(len(pred), bool)
    if len(pred) > 1:
        join[1:] = (trip.arcs[pred[:-1], 1] == trip.arcs[pred[1:], 0]) | (pred[:-1] == pred[1:])
    return on, join.astype(np.int8)


def place_lookup() -> dict:
    """City/country per trip from the EDA reverse-geocode. Cosmetic only."""
    p = os.path.join(config.ROOT, "notes", "locations.csv")
    out = {}
    if os.path.exists(p):
        import csv

        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["instance"]] = {
                    "city": row.get("city") or "",
                    "country": row.get("country") or "",
                }
    return out


def build_one(trip_id: str, partition: str, out_dir: str) -> dict:
    t0 = time.time()
    base = config.TEST_DIR if partition == "test" else config.TRAIN_DIR
    trip = match.build_trip(os.path.join(base, trip_id))
    pred, dist = match.match(trip)
    on, join = route_flags(trip, pred)
    snap, _ = match.snap_to_arc(trip.track_xy, trip.A, trip.B, pred)

    # back to lon/lat for everything the renderer draws
    lon0, lat0 = trip.lon0, trip.lat0
    k = np.cos(np.radians(lat0))

    def to_lonlat(xy):
        return np.column_stack([lon0 + xy[:, 0] / (111320.0 * k), lat0 + xy[:, 1] / 111320.0])

    track_ll = trip.track[:, :2]
    snap_ll = to_lonlat(snap)

    keep = corridor_keep(trip)
    nx, ny = match.to_xy(trip.nodes, lon0, lat0)
    ends = np.column_stack([nx, ny])[trip.arcs[keep].ravel()]
    net_ll = to_lonlat(ends).reshape(-1, 4)  # (a_lon,a_lat,b_lon,b_lat)

    t_off = np.rint(trip.track[:, 2] - trip.track[0, 2]).astype(np.int64)
    dt = np.diff(trip.track[:, 2])
    track_bbox = [
        float(track_ll[:, 0].min()), float(track_ll[:, 1].min()),
        float(track_ll[:, 0].max()), float(track_ll[:, 1].max()),
    ]
    place = place_lookup().get(trip_id, {})

    payload = {
        "id": trip_id,
        "partition": partition,
        "method": config.METHOD,
        "bbox": track_bbox,
        "track": {
            "n": int(len(track_ll)),
            "t": t_off.tolist(),
            "xy": r(track_ll.ravel()),
        },
        "network": {
            "n": int(keep.sum()),
            "ids": np.flatnonzero(keep).tolist(),
            "xy": r(net_ll.ravel()),
        },
        "route": trip.route.tolist(),
        "pred": {
            "arc": pred.tolist(),
            "snap": r(snap_ll.ravel()),
            "dist": [round(float(v), 1) for v in dist],
            "on": on.tolist(),
            "join": join.tolist(),
        },
        "metrics": match.score(trip, pred, dist),
        "meta": {
            "city": place.get("city", ""),
            "country": place.get("country", ""),
            "drive_m": round(float(np.sum(match.hav(
                track_ll[:-1, 0], track_ll[:-1, 1], track_ll[1:, 0], track_ll[1:, 1]))), 1),
            "duration_s": int(t_off[-1]) if len(t_off) else 0,
            "max_gap_s": int(dt.max()) if len(dt) else 0,
            "network_arcs_total": int(len(trip.arcs)),
            "matched_in_ms": None,
            "build_ms": None,
        },
    }
    payload["meta"]["build_ms"] = int((time.time() - t0) * 1000)
    with open(os.path.join(out_dir, f"trip_{trip_id}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return payload


def summary(p: dict, size_kb: int) -> dict:
    m, meta = p["metrics"], p["meta"]
    return {
        "id": p["id"],
        "partition": p["partition"],
        "city": p["meta"].get("city", ""),
        "fixes": m["counts"]["fixes"],
        "route_arcs": m["counts"]["route_arcs"],
        "network_arcs": meta["network_arcs_total"],
        "drawn_arcs": p["network"]["n"],
        "drive_km": round(meta["drive_m"] / 1000, 1),
        "duration_s": meta["duration_s"],
        "max_gap_s": meta["max_gap_s"],
        "f1_strict": m["arcs"]["f1_strict"],
        "recall_strict": m["arcs"]["recall_strict"],
        "recall_dir": m["arcs"]["recall_dir"],
        "top1_dir": m["point"]["top1_dir"],
        "missed_arcs": m["arcs"]["missed_arcs"],
        "missed_dir": m["arcs"]["missed_arcs_dir"],
        "breaks": m["continuity"]["breaks"],
        "median_offset_m": m["point"]["median_offset_m"],
        "size_kb": size_kb,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="comma-separated trip ids")
    ap.add_argument("--out", default=config.DATA_DIR)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    trips = [(t, p) for t, p in match.trip_dirs() if not wanted or t in wanted]
    if not trips:
        sys.exit(f"no trips matched {wanted}")

    rows, t0 = [], time.time()
    for i, (tid, part) in enumerate(trips, 1):
        payload = build_one(tid, part, args.out)
        path = os.path.join(args.out, f"trip_{tid}.json")
        rows.append(summary(payload, os.path.getsize(path) // 1024))
        print(f"[{i:2d}/{len(trips)}] {tid} {part:5s} "
              f"recall={payload['metrics']['arcs']['recall_strict']:.3f} "
              f"dir={payload['metrics']['arcs']['recall_dir']:.3f} "
              f"{os.path.getsize(path)//1024:5d} KB", flush=True)

    idx_path = os.path.join(args.out, "index.json")
    old = {}
    if os.path.exists(idx_path):
        with open(idx_path, encoding="utf-8") as f:
            old = {t["id"]: t for t in json.load(f)["trips"]}
    for row in rows:
        old[row["id"]] = row
    payload = {
        "generated_by": "app/build_payloads.py",
        "method": config.METHOD,
        "trips": sorted(old.values(), key=lambda t: t["id"]),
    }
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"wrote {len(rows)} payload(s) in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
