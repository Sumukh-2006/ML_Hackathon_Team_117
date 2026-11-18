"""Nearest-arc map matching, plus the scoring rules the UI quotes.

Rationale: this is the floor, not the product. It snaps each fix to the nearest
road segment on its own - no history, no connectivity, no smoothing - so any
matcher that cannot beat it is not doing map matching. The UI shows it because
the gap between it and the reference route is the story: 89% of fixes land on a
road the car really drove, but only 57% of the driven segments are touched, and
32 points of that gap is pure direction.

Alternative that lost: nearest-node snapping. It scores 98% recall but 67% F1,
because a snapped node is credited with every incident arc it never localised.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np

from . import config

R_EARTH = 6371000.0


# ----------------------------------------------------------------- file loading
def trip_dirs() -> list[tuple[str, str]]:
    """[(trip_id, partition)] for every shipped trip, train first."""
    out = []
    for part, base in (("train", config.TRAIN_DIR), ("test", config.TEST_DIR)):
        for d in sorted(glob.glob(os.path.join(base, "*"))):
            if os.path.isdir(d):
                out.append((os.path.basename(d), part))
    return out


def load_trip(inst_dir: str) -> dict:
    """Read the four headerless files of one trip. Empty files return empty arrays."""
    name = os.path.basename(inst_dir)

    def rd(ext, ncol):
        p = os.path.join(inst_dir, f"{name}.{ext}")
        a = np.loadtxt(p, ndmin=2)
        return np.atleast_2d(a)[:, :ncol] if ncol is not None else np.atleast_2d(a)

    nodes = rd("nodes", 2)
    arcs = rd("arcs", 2).astype(np.int64)
    route = np.atleast_1d(
        np.loadtxt(os.path.join(inst_dir, f"{name}.route"), dtype=np.int64)
    )
    track = rd("track", None)
    return {"nodes": nodes, "arcs": arcs, "route": route, "track": track}


# ------------------------------------------------------------------ geometry
def hav(lon1, lat1, lon2, lat2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dphi = la2 - la1
    dl = np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def to_xy(p, lon0, lat0):
    """Local metric plane (equirectangular at the trip centre).

    A single global projection is invalid here: the 90 drives span 137 degrees of
    longitude and 25 UTM zones. Within one trip this is < 0.1% scale error, which
    is far below the 5 m GPS noise it is being compared against.
    """
    k = np.cos(np.radians(lat0))
    return (p[:, 0] - lon0) * 111320.0 * k, (p[:, 1] - lat0) * 111320.0


def seg_dist_pairs(P, A, B):
    """Elementwise point-to-segment distance for aligned (k,2) arrays -> (k,)."""
    dx = B[:, 0] - A[:, 0]
    dy = B[:, 1] - A[:, 1]
    L2 = np.maximum(dx * dx + dy * dy, 1e-6)
    t = np.clip(((P[:, 0] - A[:, 0]) * dx + (P[:, 1] - A[:, 1]) * dy) / L2, 0, 1)
    return np.hypot(P[:, 0] - (A[:, 0] + t * dx), P[:, 1] - (A[:, 1] + t * dy))


def snap_to_arc(P, A, B, arc):
    """Closest point on each already-chosen arc, and its offset distance."""
    a, b = A[arc], B[arc]
    dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    L2 = np.maximum(dx * dx + dy * dy, 1e-6)
    t = np.clip(((P[:, 0] - a[:, 0]) * dx + (P[:, 1] - a[:, 1]) * dy) / L2, 0, 1)
    q = a + t[:, None] * np.column_stack([dx, dy])
    return q, np.hypot(*(P - q).T)


def seg_dist_lean(P, A, B):
    """P:(n,2) A:(m,2) B:(m,2) -> (n,m) point-to-segment distance, float32.

    Fewer temporaries than the textbook form; this sits in the inner loop of
    every candidate search, and the run is memory-bandwidth bound, not FLOP bound.
    """
    if P.dtype != np.float32:
        P = P.astype(np.float32)
    ax, ay = A[:, 0][None, :], A[:, 1][None, :]
    dx = (B[:, 0] - A[:, 0])[None, :]
    dy = (B[:, 1] - A[:, 1])[None, :]
    L2 = dx * dx + dy * dy
    np.maximum(L2, 1e-6, out=L2)
    px, py = P[:, 0][:, None], P[:, 1][:, None]
    t = (px - ax) * dx + (py - ay) * dy
    t /= L2
    np.clip(t, 0, 1, out=t)
    qx = t * dx
    qx += ax
    qy = t * dy
    qy += ay
    return np.hypot(px - qx, py - qy)


def build_grid(A, B, lo, hi, cell=config.CELL):
    """Spatial hash: each arc is inserted into every cell its bounding box touches."""
    m = len(A)
    ncx = int((hi[0] - lo[0]) // cell) + 2
    ncy = int((hi[1] - lo[1]) // cell) + 2
    amin, amax = np.minimum(A, B), np.maximum(A, B)
    x0 = np.clip(((amin[:, 0] - lo[0]) // cell).astype(np.int64), 0, ncx - 1)
    x1 = np.clip(((amax[:, 0] - lo[0]) // cell).astype(np.int64), 0, ncx - 1)
    y0 = np.clip(((amin[:, 1] - lo[1]) // cell).astype(np.int64), 0, ncy - 1)
    y1 = np.clip(((amax[:, 1] - lo[1]) // cell).astype(np.int64), 0, ncy - 1)
    w = x1 - x0 + 1
    cnt = w * (y1 - y0 + 1)
    tot = int(cnt.sum())
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    wr = np.repeat(w, cnt)
    keys = np.repeat(x0 + y0 * ncx, cnt) + (off % wr) + (off // wr) * ncx
    order = np.argsort(keys, kind="stable")
    cell_arc = np.repeat(np.arange(m), cnt)[order]
    cell_ptr = np.concatenate([[0], np.cumsum(np.bincount(keys, minlength=ncx * ncy))])
    return {
        "lo": lo,
        "ncx": ncx,
        "cell_ptr": cell_ptr.astype(np.int64),
        "cell_arc": cell_arc.astype(np.int64),
    }


def nearest_arc_brute(P, A, B):
    """Exact nearest arc over every arc, chunked over arcs -> (dist, arc index)."""
    P32 = P.astype(np.float32, copy=False)
    m, n = len(A), len(P32)
    best_d = np.full(n, np.inf, np.float32)
    best_a = np.zeros(n, np.int64)
    block = max(1, int(min(2.0e6 / max(n, 1), m)))
    rows = np.arange(n)
    for s in range(0, m, block):
        e = min(m, s + block)
        d = seg_dist_lean(P32, A[s:e], B[s:e])
        j = d.argmin(axis=1)
        dm = d[rows, j]
        upd = dm < best_d
        best_d[upd] = dm[upd]
        best_a[upd] = s + j[upd]
    return best_d, best_a


def nearest_arc_grid(P, A, B, grid, ring=config.RING, chunk=4.0e5):
    """Nearest arc via the spatial hash; exact whenever the result is < ring*CELL."""
    g_lo, ncx, ptr, carc = grid["lo"], grid["ncx"], grid["cell_ptr"], grid["cell_arc"]
    ncy = len(ptr) // ncx
    offs = np.array(
        [(dx, dy) for dy in range(-ring, ring + 1) for dx in range(-ring, ring + 1)],
        np.int64,
    )
    K = len(offs)
    n = len(P)
    dist = np.full(n, np.inf, np.float32)
    arc = np.zeros(n, np.int64)

    pcx = np.clip(((P[:, 0] - g_lo[0]) // config.CELL).astype(np.int64), 0, ncx - 1)
    pcy = np.clip(((P[:, 1] - g_lo[1]) // config.CELL).astype(np.int64), 0, ncy - 1)
    keys = np.clip(pcy[:, None] + offs[None, :, 1], 0, ncy - 1) * ncx + np.clip(
        pcx[:, None] + offs[None, :, 0], 0, ncx - 1
    )
    cnts = ptr[keys + 1] - ptr[keys]

    step = max(1, int(chunk // max(1, K)))
    for s in range(0, n, step):
        e = min(n, s + step)
        c = cnts[s:e]
        tot = int(c.sum())
        if tot == 0:
            continue
        lens = c.ravel()
        in0 = ptr[keys[s:e]].ravel()
        ids = carc[
            np.repeat(in0, lens)
            + (np.arange(tot) - np.repeat(np.cumsum(lens) - lens, lens))
        ]
        rows = np.repeat(np.repeat(np.arange(e - s), K), lens)
        d = seg_dist_pairs(P[s:e][rows], A[ids], B[ids])
        grp = np.ones(tot, bool)
        grp[1:] = rows[1:] != rows[:-1]
        st = np.flatnonzero(grp)
        elem_min = np.repeat(np.minimum.reduceat(d, st), np.diff(np.append(st, tot)))
        js = np.flatnonzero(d <= elem_min + 1e-3)  # ties within 1 mm kept
        js = js[np.lexsort((ids[js], rows[js]))]   # deterministic: lowest arc id wins
        first = np.ones(len(js), bool)
        first[1:] = rows[js][1:] != rows[js][:-1]
        js = js[first]
        dist[s + rows[js]] = d[js]
        arc[s + rows[js]] = ids[js]
    return dist, arc


# ---------------------------------------------------------------------- trips
@dataclass
class Trip:
    """One drive with its network already projected and pre-culled."""

    trip_id: str
    partition: str
    nodes: np.ndarray
    arcs: np.ndarray
    route: np.ndarray
    track: np.ndarray
    lon0: float
    lat0: float
    A: np.ndarray = field(repr=False)
    B: np.ndarray = field(repr=False)
    keep: np.ndarray = field(repr=False)
    grid: dict = field(repr=False)
    track_xy: np.ndarray = field(repr=False)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)


def build_trip(inst_dir: str) -> Trip:
    part = "test" if os.path.basename(os.path.dirname(inst_dir)) == "student_test" else "train"
    d = load_trip(inst_dir)
    nodes, arcs, route, track = d["nodes"], d["arcs"], d["route"], d["track"]
    lon0, lat0 = float(nodes[:, 0].mean()), float(nodes[:, 1].mean())
    nx, ny = to_xy(nodes, lon0, lat0)
    tx, ty = to_xy(track, lon0, lat0)
    P = np.column_stack([tx, ty])
    A = np.column_stack([nx[arcs[:, 0]], ny[arcs[:, 0]]])
    B = np.column_stack([nx[arcs[:, 1]], ny[arcs[:, 1]]])

    # Cull arcs whose box misses the track box + margin: they cannot be the argmin
    # for any fix we score, and dropping them is what makes 307k-arc networks usable.
    lo, hi = P.min(0) - config.BBOX_MARGIN, P.max(0) + config.BBOX_MARGIN
    amin, amax = np.minimum(A, B), np.maximum(A, B)
    keep = np.flatnonzero(
        (amax[:, 0] >= lo[0]) & (amin[:, 0] <= hi[0])
        & (amax[:, 1] >= lo[1]) & (amin[:, 1] <= hi[1])
    )
    Ag, Bg = A[keep], B[keep]
    return Trip(
        trip_id=os.path.basename(inst_dir),
        partition=part,
        nodes=nodes,
        arcs=arcs,
        route=route,
        track=track,
        lon0=lon0,
        lat0=lat0,
        A=A,
        B=B,
        keep=keep,
        grid=build_grid(Ag, Bg, lo, hi),
        track_xy=P,
    )


def match(trip: Trip, P_xy: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Snap each fix independently -> (arc id, offset metres). No state between fixes."""
    P = trip.track_xy if P_xy is None else P_xy
    Ag, Bg = trip.A[trip.keep], trip.B[trip.keep]
    dist, local = nearest_arc_grid(P, Ag, Bg, trip.grid)
    far = np.flatnonzero(dist > config.CELL * config.RING)  # distance not certified
    if len(far):
        d2, a2 = nearest_arc_brute(P[far], Ag, Bg)
        dist[far], local[far] = d2, a2
    return trip.keep[local].astype(np.int64), dist.astype(np.float64)


def score(trip: Trip, pred: np.ndarray, dist: np.ndarray) -> dict:
    """Arc-set scores in both direction rules, plus continuity.

    Two counting rules, deliberately:
      strict          - the snapped arc must equal the route arc, direction included.
                        This is the literal task (output is an arc sequence).
      direction-blind - the arc or its reverse twin counts. 92% of these arcs have a
                        twin at the same geometry, so strict depends on the tie rule;
                        this view isolates 'found the road' from 'faced the right way'.
    """
    route_set = np.unique(trip.route)
    pred_set = np.unique(pred)
    N = len(trip.nodes)
    key = trip.arcs[:, 0] * N + trip.arcs[:, 1]
    rev = trip.arcs[:, 1] * N + trip.arcs[:, 0]

    route_key = np.unique(key[route_set])
    route_both = np.unique(np.concatenate([route_key, rev[route_set]]))
    pred_key = np.unique(key[pred_set])
    pred_both = np.unique(np.concatenate([pred_key, rev[pred_set]]))

    def prf(a, b):
        inter = int(np.isin(a, b).sum())
        rec = inter / max(len(a), 1)
        prec = inter / max(len(b), 1)
        return rec, prec, 2 * rec * prec / max(rec + prec, 1e-12), inter

    rec_s, prec_s, f1_s, hit_s = prf(route_set, pred_set)
    # direction-ignored: does the *road* appear in the prediction, either way round?
    # Denominator stays one entry per route arc, so this can never exceed strict.
    rec_d = float(np.isin(route_key, pred_both).mean())
    prec_d = float(np.isin(pred_both, route_both).mean())
    f1_d = 2 * rec_d * prec_d / max(rec_d + prec_d, 1e-12)
    hit_d = int(np.isin(route_key, pred_both).sum())

    # continuity: two consecutive snaps that share no node cannot be one drive
    from_node = trip.arcs[pred, 0]
    to_node = trip.arcs[pred, 1]
    breaks = int(np.sum((to_node[:-1] != from_node[1:]) & (pred[:-1] != pred[1:])))

    missed_strict = int(len(route_set) - hit_s)
    missed_dir = int(len(route_key) - hit_d)
    on_route_dir = np.isin(key[pred], route_both)

    return {
        "arcs": {
            "recall_strict": round(rec_s, 4),
            "precision_strict": round(prec_s, 4),
            "f1_strict": round(f1_s, 4),
            "recall_dir": round(rec_d, 4),
            "precision_dir": round(prec_d, 4),
            "f1_dir": round(f1_d, 4),
            "missed_arcs": missed_strict,
            "missed_arcs_dir": missed_dir,
            "wrong_way_only": max(0, missed_strict - missed_dir),
        },
        "point": {
            "top1_strict": round(float(np.isin(pred, route_set).mean()), 4),
            "top1_dir": round(float(on_route_dir.mean()), 4),
            "median_offset_m": round(float(np.median(dist)), 2),
            "p95_offset_m": round(float(np.percentile(dist, 95)), 2),
        },
        "continuity": {"breaks": breaks, "per_1000_fixes": round(1000 * breaks / max(len(pred), 1), 2)},
        "counts": {
            "fixes": int(len(pred)),
            "route_arcs": int(len(route_set)),
            "network_arcs": int(len(trip.arcs)),
            "pred_arcs": int(len(pred_set)),
        },
    }
