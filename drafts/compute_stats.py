"""Measure every folder of the corpus once and save the numbers the
figures in dataset.typ are drawn from.

Run from the project root:  .venv/bin/python drafts/compute_stats.py
Writes drafts/stats/{per_folder.csv, pooled.npz, summary.json}.
"""

import csv
import json
from pathlib import Path

import numpy as np

from dataset import Corpus, RoadGraph, Track, great_circle

ROOT: Path = Path(__file__).resolve().parent.parent
CORPUS: Path = ROOT / "data" / "corpus"
ORIGINAL: Path = ROOT / "data" / "corpus_original"
LABELS: Path = ROOT / "data" / "hand_labels" / "kubicka_metadata_90.csv"
OUT: Path = Path(__file__).resolve().parent / "stats"

# Histogram bins (display only, not used to classify anything).
ARC_BINS: np.ndarray = np.logspace(-1, 4, 101)          # 0.1 m .. 10 km
NOISE_LAGS: np.ndarray = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60])


def point_to_route(x: np.ndarray, y: np.ndarray, graph: RoadGraph,
                   arcs: np.ndarray) -> np.ndarray:
    """Distance from each fix to the nearest straight arc of the route."""
    ax = graph.node_x[graph.arc_from[arcs]]
    ay = graph.node_y[graph.arc_from[arcs]]
    bx = graph.node_x[graph.arc_to[arcs]]
    by = graph.node_y[graph.arc_to[arcs]]
    dx, dy = bx - ax, by - ay
    length2 = np.maximum(dx * dx + dy * dy, 1e-12)
    out = np.empty(len(x))
    for start in range(0, len(x), 400):
        px = x[start:start + 400, None]
        py = y[start:start + 400, None]
        s = np.clip(((px - ax) * dx + (py - ay) * dy) / length2, 0.0, 1.0)
        ex = ax + s * dx - px
        ey = ay + s * dy - py
        out[start:start + 400] = np.sqrt(ex * ex + ey * ey).min(axis=1)
    return out


def speeds(track: Track) -> np.ndarray:
    step = great_circle(track.lon[:-1], track.lat[:-1],
                        track.lon[1:], track.lat[1:])
    return step / np.diff(track.t)


def lagged_products(t: np.ndarray, e: np.ndarray) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """Sum of e_i * e_j and pair count over pairs exactly `lag` s apart."""
    sums = np.zeros(len(NOISE_LAGS))
    counts = np.zeros(len(NOISE_LAGS))
    for k, lag in enumerate(NOISE_LAGS):
        j = np.searchsorted(t, t + lag)
        ok = (j < len(t))
        ok[ok] &= np.isclose(t[j[ok]], t[ok] + lag)
        sums[k] = float((e[ok] * e[j[ok]]).sum())
        counts[k] = float(ok.sum())
    return sums, counts


def read_labels() -> dict[str, dict]:
    with LABELS.open() as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    labels = read_labels()
    corpus = Corpus.from_root(CORPUS)

    rows: list[dict] = []
    pooled: dict[str, list] = {k: [] for k in [
        "dt", "speed_noisy", "speed_orig", "dist_noisy", "dist_orig",
        "noise_e", "noise_n", "route_arc_len", "split_of_fix"]}
    map_arc_hist = np.zeros(len(ARC_BINS) - 1)
    out_degree = np.zeros(20, dtype=np.int64)
    roads_at_node = np.zeros(20, dtype=np.int64)
    ac_sum = np.zeros(len(NOISE_LAGS))
    ac_cnt = np.zeros(len(NOISE_LAGS))

    for split in ["train", "student_test"]:
        for folder in corpus.folders(split):
            name = folder.name()
            graph = folder.graph()
            track = folder.track(graph)
            original = Track.from_file(
                ORIGINAL / split / name / f"{name}.track", graph.projection)
            route = folder.route(graph).arcs

            dt = np.diff(track.t)
            d_noisy = point_to_route(track.x, track.y, graph, route)
            d_orig = point_to_route(original.x, original.y, graph, route)
            e = track.x - original.x
            n = track.y - original.y
            for axis in (e, n):
                s, c = lagged_products(track.t, axis)
                ac_sum += s
                ac_cnt += c

            canon = graph.canonical_arc[route]
            uniq, counts = np.unique(canon, return_counts=True)
            repeated_m = float(((counts - 1) * graph.arc_length[uniq]).sum())
            # reverse_arc already names the lowest-id twin of the reverse.
            back = graph.reverse_arc[route[:-1]]
            u_turns = int(((back >= 0) & (back == canon[1:])).sum())
            breaks = int((graph.arc_to[route[:-1]]
                          != graph.arc_from[route[1:]]).sum())

            degree = np.diff(graph.out_start)
            out_degree += np.bincount(np.minimum(degree, 19), minlength=20)
            map_arc_hist += np.histogram(graph.arc_length, ARC_BINS)[0]
            # Distinct neighbouring nodes, ignoring direction: 1 = dead end,
            # 2 = a point along one road, 3+ = a junction.
            a = np.minimum(graph.arc_from, graph.arc_to)
            b = np.maximum(graph.arc_from, graph.arc_to)
            pairs = np.unique(a * graph.node_count() + b)
            ends = np.r_[pairs // graph.node_count(), pairs % graph.node_count()]
            neighbours = np.bincount(ends, minlength=graph.node_count())
            roads_at_node += np.bincount(np.minimum(neighbours, 19), minlength=20)

            label = labels[name]
            tags = [t.split("=")[1] for t in label["tags"].split(";") if t]
            v_orig = speeds(original)
            rows.append({
                "split": split, "id": name, "tags": ";".join(tags),
                "mismatch_mad": float(label["mismatch_mad"]),
                "mismatch_max": float(label["mismatch_max"]),
                "lon": float(np.median(track.lon)),
                "lat": float(np.median(track.lat)),
                "nodes": graph.node_count(), "arcs": graph.arc_count(),
                "twin_arcs": int((graph.canonical_arc
                                  != np.arange(graph.arc_count())).sum()),
                "two_way_frac": float((graph.reverse_arc >= 0).mean()),
                "map_w_km": float(np.ptp(graph.node_x) / 1000),
                "map_h_km": float(np.ptp(graph.node_y) / 1000),
                "fixes": track.fix_count(),
                "duration_s": float(track.t[-1] - track.t[0]),
                "dt_max": float(dt.max()),
                "frac_dt_1": float(np.isclose(dt, 1.0).mean()),
                "missing_s": float((dt - 1).clip(min=0).sum()),
                "route_arcs": len(route),
                "route_km": float(graph.arc_length[route].sum() / 1000),
                "route_breaks": breaks,
                "route_u_turns": u_turns,
                "route_repeat_m": repeated_m,
                "track_km_noisy": float(speeds(track).dot(dt) / 1000),
                "track_km_orig": float(v_orig.dot(dt) / 1000),
                "speed_median": float(np.median(v_orig)),
                "speed_p95": float(np.percentile(v_orig, 95)),
                "dist_orig_median": float(np.median(d_orig)),
                "dist_orig_p95": float(np.percentile(d_orig, 95)),
                "dist_orig_max": float(d_orig.max()),
                "dist_noisy_median": float(np.median(d_noisy)),
                "dist_noisy_p95": float(np.percentile(d_noisy, 95)),
            })
            pooled["dt"].append(dt)
            pooled["speed_noisy"].append(speeds(track))
            pooled["speed_orig"].append(v_orig)
            pooled["dist_noisy"].append(d_noisy)
            pooled["dist_orig"].append(d_orig)
            pooled["noise_e"].append(e)
            pooled["noise_n"].append(n)
            pooled["route_arc_len"].append(graph.arc_length[route])
            pooled["split_of_fix"].append(
                np.full(len(d_noisy), split == "train"))
            print(f"{split}/{name}: {track.fix_count()} fixes, "
                  f"{graph.arc_count()} arcs, route {len(route)} arcs")

    with (OUT / "per_folder.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    arrays = {k: np.concatenate(v) for k, v in pooled.items()}
    noise_var = float(np.mean(np.r_[arrays["noise_e"], arrays["noise_n"]] ** 2))
    np.savez_compressed(
        OUT / "pooled.npz", **arrays, map_arc_hist=map_arc_hist,
        arc_bins=ARC_BINS, out_degree=out_degree,
        roads_at_node=roads_at_node, noise_lags=NOISE_LAGS,
        noise_autocorr=(ac_sum / ac_cnt) / noise_var)

    present = {int(r["id"]) for r in rows}
    summary = {
        "folders": len(rows),
        "train": sum(r["split"] == "train" for r in rows),
        "student_test": sum(r["split"] == "student_test" for r in rows),
        "unseen_ids": sorted(set(range(100)) - present),
        "fixes": int(sum(r["fixes"] for r in rows)),
        "hours": sum(r["duration_s"] for r in rows) / 3600,
        "route_km": sum(r["route_km"] for r in rows),
        "arcs_total": int(sum(r["arcs"] for r in rows)),
        "frac_dt_1": float(np.isclose(arrays["dt"], 1.0).mean()),
        "noise_std_e": float(arrays["noise_e"].std()),
        "noise_std_n": float(arrays["noise_n"].std()),
        "dist_orig_median": float(np.median(arrays["dist_orig"])),
        "dist_noisy_median": float(np.median(arrays["dist_noisy"])),
        "dist_noisy_p99": float(np.percentile(arrays["dist_noisy"], 99)),
        "route_breaks_total": int(sum(r["route_breaks"] for r in rows)),
        "folders_with_breaks": int(sum(r["route_breaks"] > 0 for r in rows)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
