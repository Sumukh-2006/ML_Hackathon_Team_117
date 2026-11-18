"""
Data-quality + leakage audit for the map-matching dataset (90 drives).

Answers two questions that a faculty panel will ask:
  1. LEAKAGE   - is a random train/validation split honest, or do validation drives
                 reuse the same road network (and the same roads) as training drives?
  2. LABELS    - which drives have a broken / incomplete ground-truth route, and which
                 "broken" drives are actually just long parking periods with GPS jitter?

Design decisions (each one is a panel question waiting to happen):

* Network identity is measured on the NODE COORDINATE SET (rounded to 1e-5 deg ~ 1 m),
  not on node ids. Node ids are line numbers, they are renumbered per instance, so an
  arc id cannot be transferred between instances; the *geometry* can.
* Label coverage = share of GPS points within 50 m / 200 m of the labelled route.
  Segments of the drawn polyline that JUMP the gap (arc i ends at v, arc i+1 starts at
  w != v) are removed.  Removing "all segments longer than 2 km" instead - the rule used
  in the first pass - is wrong here: all 11 gap chords in this dataset are 15-596 m, so
  the 2 km rule deletes zero chords and instead deletes genuine 2-5 km highway arcs.
  Both variants are reported so the difference is auditable.
* "Parked" is detected with a 10 s displacement window (>10 km/h = moving). A 1 s
  step is useless: GPS jitter while stationary looks like 70 km/h.

Run:  python notes/quality_audit.py
Out:  notes/quality_audit.csv, notes/figs/09_leakage_quality.png
"""
import os, sys, glob, csv, hashlib, itertools
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, hav, to_xy, seg_dist, route_polyline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
os.makedirs(FIGS, exist_ok=True)

# Okabe-Ito colourblind-safe palette
C_TRAIN, C_TEST = "#0072B2", "#D55E00"     # blue / vermillion
C_BAD, C_OK = "#D55E00", "#56B4E9"
C_DARK, C_GREY = "#222222", "#999999"

COV_BAD = 0.90          # a drive is "poorly covered" below this
PARK_CONC = 0.50        # >50% of off-label points in one 100 m cell = parked, not mislabelled


# ------------------------------------------------------------------ helpers
def min_dist(P, A, B, chunk=256):
    """min point-to-segment distance, chunked over points to bound memory"""
    out = np.empty(len(P))
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = seg_dist(P[i:i + chunk], A, B).min(axis=1)
    return out


def moving_mask(P, t, w=5, kmh=10.0):
    """moving = displacement over a +-w sample window exceeds kmh (jitter-robust)"""
    n = len(P)
    if n <= 2 * w + 2:
        return np.ones(n, bool)
    d = np.hypot(*(P[2 * w:] - P[:-2 * w]).T)
    dt = np.maximum(t[2 * w:] - t[:-2 * w], 1e-6)
    v = np.zeros(n)
    v[w:w + len(d)] = d / dt * 3.6
    v[:w], v[-w:] = v[w], v[w - 1]
    return v > kmh


def arc_len(nodes, arcs):
    return hav(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1],
               nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])


def analyse(inst, split):
    """all per-instance metrics; the one place where a number is defined"""
    nodes, arcs, route, track = load(inst)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    chain, poly = route_polyline(nodes, arcs, route)        # ordered node chain of the label
    S = np.column_stack(to_xy(poly, lon0, lat0))            # route polyline, metres
    P = np.column_stack(to_xy(track, lon0, lat0))           # GPS track, metres

    # ---- route gaps: arc i ends somewhere else than arc i+1 starts
    bend = arcs[route[:-1], 1] != arcs[route[1:], 0]
    gaps = np.where(bend)[0]
    gap_m = hav(nodes[arcs[route[gaps], 1], 0], nodes[arcs[route[gaps], 1], 1],
                nodes[arcs[route[gaps + 1], 0], 0], nodes[arcs[route[gaps + 1], 0], 1]) \
        if len(gaps) else np.zeros(0)

    # ---- label coverage. seg j runs poly[j] -> poly[j+1]; seg g+1 spans a gap chord
    seglen = np.hypot(*(S[1:] - S[:-1]).T)
    chord = np.zeros(len(seglen), bool)
    chord[gaps + 1] = True
    keep = ~chord
    d = min_dist(P, S[:-1][keep], S[1:][keep])               # distance to the honest label
    d_2km = min_dist(P, S[:-1][seglen <= 2000], S[1:][seglen <= 2000])  # the 2 km rule
    mov = moving_mask(P, track[:, 2])

    # ---- track-session anomalies
    dt = np.diff(track[:, 2])
    step = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    spd = np.where(dt > 0, step / np.maximum(dt, 1e-9) * 3.6, np.inf)
    dup = int(np.sum((np.diff(track[:, 0]) == 0) & (np.diff(track[:, 1]) == 0)))

    # ---- where are the off-label points? parked blob vs distributed (label) error
    off = d > 50
    if off.sum() > 5:
        cell = np.floor(P[off] / 100).astype(int)            # 100 m grid
        _, cnt = np.unique(cell, axis=0, return_counts=True)
        conc = float(cnt.max() / off.sum())
    else:
        conc = 0.0

    # distance from a subsample of GPS points to the nearest arc of the whole network:
    # separates "GPS sits on a road the label missed" from "GPS sits in a car park"
    sub = P[::max(1, len(P) // 600)]
    A = np.column_stack(to_xy(nodes[arcs[:, 0]], lon0, lat0))
    B = np.column_stack(to_xy(nodes[arcs[:, 1]], lon0, lat0))
    dn = min_dist(sub, A, B, chunk=32)

    return dict(
        instance=os.path.basename(inst), split=split,
        lat=lat0, lon=lon0, n_nodes=len(nodes), n_arcs=len(arcs),
        n_route=len(route), n_track=len(track), dur_s=float(track[-1, 2] - track[0, 2]),
        route_gaps=int(len(gaps)), gap_max_m=float(gap_m.max()) if len(gaps) else 0.0,
        cov50=float(np.mean(d < 50)), cov200=float(np.mean(d < 200)),
        cov50_moving=float(np.mean(d[mov] < 50)),
        cov50_2km=float(np.mean(d_2km < 50)), cov200_2km=float(np.mean(d_2km < 200)),
        n_off50=int(off.sum()), off_top_cell=conc, med_dev_m=float(np.median(d)),
        med_dev_net_m=float(np.median(dn)),
        n_dt30=int(np.sum(dt > 30)), dt_max_s=float(dt.max()),
        n_jump200=int(np.sum(step > 200)), jump_max_m=float(step.max()),
        n_fast200=int(np.sum(spd > 200)), vmax_kmh=float(min(spd.max(), 9999.0)),
        n_dup=dup,
        # kept for the figure, not for the CSV
        _poly=S, _poly_ll=poly, _track=P, _d=d, _mov=mov, _nodes=nodes, _arcs=arcs,
        _route=route, _track_ll=track, _chain=chain,
    )


# ------------------------------------------------------------------ leakage
def leakage(rows):
    """exact node-coordinate overlap for all 90*89/2 pairs, + route overlap for the
    pairs that share any nodes at all (the only ones that can share driven roads)"""
    per = {}
    coord2inst = defaultdict(list)
    for r in rows:
        nodes = r["_nodes"]
        key = np.round(nodes, 5)                              # 1e-5 deg ~ 1.1 m
        s = set(map(tuple, key))
        per[r["instance"]] = s
        for c in s:
            coord2inst[c].append(r["instance"])

    shared = defaultdict(int)
    for c, lst in coord2inst.items():                         # count only shared coords
        if len(lst) > 1:
            for a, b in itertools.combinations(sorted(lst), 2):
                shared[(a, b)] += 1

    n = {r["instance"]: len(per[r["instance"]]) for r in rows}
    jac = {}
    cont = {}
    for (a, b), k in shared.items():
        jac[(a, b)] = k / (n[a] + n[b] - k)
        cont[(a, b)] = k / min(n[a], n[b])                    # 1.0 = one extract nests inside the other

    by = {r["instance"]: r for r in rows}

    def route_overlap(a, b):
        """fraction of the shorter labelled route that lies within 10 m of the other's"""
        A, B = by[a], by[b]
        lon0 = (A["lon"] + B["lon"]) / 2
        lat0 = (A["lat"] + B["lat"]) / 2
        pa = np.column_stack(to_xy(A["_poly_ll"], lon0, lat0))
        pb = np.column_stack(to_xy(B["_poly_ll"], lon0, lat0))
        ab = np.mean(min_dist(pa, pb[:-1], pb[1:]) < 10)
        ba = np.mean(min_dist(pb, pa[:-1], pa[1:]) < 10)
        if len(pa) <= len(pb):
            return float(ab), a
        return float(ba), b

    road = {}
    for (a, b), j in jac.items():
        if j >= 0.02 or cont[(a, b)] >= 0.50:
            road[(a, b)] = route_overlap(a, b)

    split = {r["instance"]: r["split"] for r in rows}
    best_same = defaultdict(float)
    best_cross = defaultdict(float)
    partner = defaultdict(lambda: (0.0, ""))
    roadpartner = defaultdict(lambda: (0.0, ""))
    for (a, b), j in jac.items():
        for x, y in ((a, b), (b, a)):
            if split[x] == split[y]:
                best_same[x] = max(best_same[x], j)
            else:
                best_cross[x] = max(best_cross[x], j)
            if j > partner[x][0]:
                partner[x] = (j, y)
            if (a, b) in road and road[(a, b)][0] > roadpartner[x][0]:
                roadpartner[x] = road[(a, b)]

    # union-find on "shares a network": >=10% node Jaccard, or one extract nested in the other
    par = {r["instance"]: r["instance"] for r in rows}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for (a, b), j in jac.items():
        if j >= 0.10 or cont[(a, b)] >= 0.90:
            par[find(a)] = find(b)
    groups = defaultdict(list)
    for r in rows:
        groups[find(r["instance"])].append(r["instance"])
    groups = {k: sorted(v) for k, v in groups.items()}

    cluster = {}
    order = sorted([v for v in groups.values() if len(v) > 1], key=lambda m: -len(m))
    for i, members in enumerate(order, 1):
        for m in members:
            cluster[m] = f"C{i}"
    for r in rows:
        cluster.setdefault(r["instance"], "")

    for r in rows:
        nm = r["instance"]
        r["max_jaccard_any"] = round(partner[nm][0], 3)
        r["jaccard_partner"] = partner[nm][1]
        r["max_jaccard_same_split"] = round(best_same[nm], 3)
        r["max_jaccard_cross_split"] = round(best_cross[nm], 3)
        r["max_route_overlap"] = round(roadpartner[nm][0], 3)
        r["route_overlap_partner"] = roadpartner[nm][1]
        r["network_cluster"] = cluster[nm]
    return dict(jac=jac, cont=cont, road=road, groups=groups, shared=shared, per=per, order=order)


# ------------------------------------------------------------------ figure
def figure(rows, leaks):
    jac, cont, road, groups, shared, per, order = (leaks["jac"], leaks["cont"], leaks["road"],
                                                  leaks["groups"], leaks["shared"],
                                                  leaks["per"], leaks["order"])
    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(14, 4.6), dpi=120, constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1], wspace=0.06)
    gs1 = gs[0, 0].subgridspec(2, 1, height_ratios=[0.85, 1.0], hspace=0.55)
    aw = fig.add_subplot(gs1[0])          # world: one point per drive
    az = fig.add_subplot(gs1[1])          # Europe/Urals zoom: the 5 shared networks
    by = {r["instance"]: r for r in rows}

    # ---------- panel 1a: world map, train vs test
    for sp, col, mk, lab in (("train", C_TRAIN, "o", "train (80)"),
                             ("test", C_TEST, "^", "test (10)")):
        aw.scatter([r["lon"] for r in rows if r["split"] == sp],
                   [r["lat"] for r in rows if r["split"] == sp],
                   s=20, c=col, marker=mk, lw=.4, edgecolor="white", label=lab, zorder=3)
    aw.add_patch(plt.Rectangle((-12, 36), 84, 28, fill=False, ec="0.35", lw=.8, ls="--", zorder=5))
    aw.set_xlim(-118, 148); aw.set_ylim(-44, 68)
    aw.set_aspect("equal")
    aw.set_ylabel("latitude (deg)", fontsize=9.5)
    aw.set_xlabel("longitude (deg)", fontsize=9.5)
    aw.tick_params(labelsize=7)
    aw.grid(alpha=.22, lw=.4)
    aw.legend(fontsize=8.2, loc="lower left", framealpha=.95, borderpad=.25,
              handletextpad=.4, labelspacing=.3)
    aw.set_title("No test drive shares a road\nnetwork with train; 5 groups do",
                 fontsize=10)

    # ---------- panel 1b: zoom on the only place where the sharing happens
    for sp, col, mk in (("train", C_TRAIN, "o"), ("test", C_TEST, "^")):
        az.scatter([r["lon"] for r in rows if r["split"] == sp],
                   [r["lat"] for r in rows if r["split"] == sp],
                   s=26, c=col, marker=mk, lw=.5, edgecolor="white", zorder=3)
    noff = {1: (0, 11), 2: (0, -12), 3: (0, -12), 4: (11, 0), 5: (0, 11)}
    for i, mem in enumerate(order, 1):
        xs = [by[m]["lon"] for m in mem]; ys = [by[m]["lat"] for m in mem]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        ro = max((road.get((min(x, y), max(x, y)), (0.0,))[0]
                  for x, y in itertools.combinations(mem, 2)), default=0.0)
        edge = C_BAD if ro >= 0.15 else C_DARK
        # ring each member POINT (not an area) so the mark carries no radius meaning
        az.scatter(xs, ys, s=70, facecolor="none", edgecolor=edge, lw=1.3, zorder=4)
        az.annotate(f"{i}", xy=(cx, cy), xytext=noff.get(i, (0, 11)),
                    textcoords="offset points", fontsize=8.0, fontweight="bold",
                    color=edge, zorder=6, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.85))
    az.set_xlim(-12, 72); az.set_ylim(34, 64)
    az.set_aspect("equal")
    az.set_xlabel("longitude (deg)", fontsize=9.5)
    az.set_ylabel("latitude (deg)", fontsize=9.5)
    az.tick_params(labelsize=7.5)
    az.grid(alpha=.22, lw=.4)
    az.set_title("zoom: ringed = same road network\n(red = 15%+ of the route shared)\n"
                 + "  ".join(f"{i}:{'/'.join(m[-2:] for m in mem)}"
                             for i, mem in enumerate(order, 1)), fontsize=7.4)


    # ---------- panel 2: label coverage, sorted
    a = fig.add_subplot(gs[0, 1])
    srt = sorted(rows, key=lambda r: r["cov50"])
    y = np.arange(len(srt))
    col = [C_BAD if r["cov50"] < COV_BAD else C_OK for r in srt]
    a.barh(y, [r["cov50"] for r in srt], color=col, height=.85, lw=0)
    a.scatter([r["cov50_moving"] for r in srt], y, s=5, c=C_DARK, zorder=4)
    a.axvline(COV_BAD, c=C_DARK, ls="--", lw=.8)
    a.set_yticks([])
    a.set_xlim(0, 2.3)
    a.set_xticks([0, .5, 1.0])
    a.set_xlabel("GPS points within 50 m of the label\n(bar; dot = same after parked samples dropped)",
                 fontsize=9.5)
    a.set_ylabel("90 drives, worst at the bottom", fontsize=9.5)
    a.text(1.06, 4.0, "worst 3:\n"
                      " 00000054  0.13  parked\n"
                      " 00000033  0.41  parked\n"
                      " 00000002  0.88  label",
           fontsize=7.8, va="center", ha="left", color=C_BAD, linespacing=1.4)
    a.tick_params(axis="x", labelsize=8)
    a.set_title("3 of 90 drives label <90% of their points\nat 50 m; 2 are a parked car, 1 is real",
                fontsize=10)

    # ---------- panel 3: how many drives have each defect
    a = fig.add_subplot(gs[0, 2])
    cats = [("route gaps\n(disconnected label)", sum(1 for r in rows if r["route_gaps"])),
            ("time gap > 30 s", sum(1 for r in rows if r["n_dt30"])),
            ("position jump > 200 m", sum(1 for r in rows if r["n_jump200"])),
            ("GPS speed > 200 km/h", sum(1 for r in rows if r["n_fast200"])),
            ("duplicate points", sum(1 for r in rows if r["n_dup"])),
            ("label coverage < 90%", sum(1 for r in rows if r["cov50"] < COV_BAD)),
            ("parked blob > 50% of\nits off-label points",
             sum(1 for r in rows if r["off_top_cell"] > PARK_CONC and r["n_off50"] >= 30))]
    yy = np.arange(len(cats))
    a.barh(yy, [c[1] for c in cats], color=C_TRAIN, height=.7)
    for i, (lab, v) in enumerate(cats):
        a.text(v + 0.6, i, f"{v} / 90", va="center", fontsize=9.5)
    a.set_yticks(yy); a.set_yticklabels([c[0] for c in cats], fontsize=9)
    a.invert_yaxis(); a.set_xlim(0, max(c[1] for c in cats) * 1.30)
    a.set_xlabel("drives affected (of 90)")
    a.tick_params(axis="x", labelsize=9)
    a.grid(axis="x", alpha=.25, lw=.5)
    a.set_title("Defects are common but small: 30 drives have\na >30 s time gap, 11 a broken label",
                fontsize=10)

    # ---------- panel 4: the worst single anomaly
    a = fig.add_subplot(gs[0, 3])
    r = by["00000054"]
    S, P, mov = r["_poly"], r["_track"], r["_mov"]
    parked = ~mov
    bx, byy = np.median(P[parked, 0]), np.median(P[parked, 1])
    near = np.hypot(*(S - [bx, byy]).T).argmin()
    half = 520
    cx = (bx + S[near, 0]) / 2
    cy = (byy + S[near, 1]) / 2
    a.plot(S[:, 0], S[:, 1], "-", c=C_TEST, lw=2.4, zorder=3, label="labelled route")
    a.scatter(P[parked, 0], P[parked, 1], s=1.2, c=C_GREY, lw=0, zorder=2,
              label=f"GPS below 10 km/h ({parked.sum()} pts)")
    a.scatter(P[mov, 0], P[mov, 1], s=2, c=C_TRAIN, lw=0, zorder=2,
              label=f"GPS moving ({mov.sum()} pts)")
    a.set_xlim(cx - half, cx + half); a.set_ylim(cy - half, cy + half)
    a.set_aspect("equal")
    a.set_xlabel("metres east", fontsize=9.5); a.set_ylabel("metres north", fontsize=9.5)
    a.tick_params(labelsize=7.5)
    a.legend(fontsize=8.0, loc="lower left", framealpha=.95, borderpad=.3)
    a.annotate("1 parked car, 4.7 h:\n87% of all 19,313 samples,\n"
               "GPS up to 250 m off the road,\ncoverage 0.13 -> 0.88 once dropped",
               xy=(bx + 70, byy + 40), xytext=(cx - half * 0.96, cy + half * 0.88),
               fontsize=7.8, ha="left", va="top", color=C_DARK,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", lw=.6, alpha=.95),
               arrowprops=dict(arrowstyle="->", lw=.9, color="0.25"))
    a.set_title("00000054: coverage 0.13 is a 4.7 h\nparked tail, not a bad label", fontsize=10)

    out = os.path.join(FIGS, "09_leakage_quality.png")
    fig.savefig(out)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
def main():
    insts = []
    for sp, folder in (("train", "train"), ("test", "student_test")):
        for d in sorted(glob.glob(os.path.join(ROOT, folder, "*"))):
            if os.path.isdir(d):
                insts.append((d, sp))
    print(f"analysing {len(insts)} drives ...", flush=True)
    rows = []
    for i, (d, sp) in enumerate(insts):
        rows.append(analyse(d, sp))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(insts)}", flush=True)

    leaks = leakage(rows)
    jac, cont, road, groups = leaks["jac"], leaks["cont"], leaks["road"], leaks["groups"]
    per, order = leaks["per"], leaks["order"]

    # ---- flags
    for r in rows:
        f = []
        if r["route_gaps"]:
            f.append("gap")
        if r["n_dt30"]:
            f.append("dt>30s")
        if r["n_jump200"]:
            f.append("jump>200m")
        if r["n_fast200"]:
            f.append("v>200kmh")
        if r["n_dup"]:
            f.append("dup")
        if r["cov50"] < COV_BAD:
            f.append("cov<90%")
        if r["off_top_cell"] > PARK_CONC and r["n_off50"] >= 30:
            f.append("parked")
        r["flags"] = ";".join(f)

    # ---- CSV
    cols = ["instance", "split", "lat", "lon", "n_nodes", "n_arcs", "n_route", "n_track",
            "dur_s", "network_cluster", "max_jaccard_any", "jaccard_partner",
            "max_jaccard_same_split", "max_jaccard_cross_split",
            "max_route_overlap", "route_overlap_partner",
            "route_gaps", "gap_max_m", "cov50", "cov50_moving", "cov200",
            "cov50_2km", "cov200_2km", "n_off50", "off_top_cell", "med_dev_m",
            "med_dev_net_m", "n_dt30", "dt_max_s", "n_jump200", "jump_max_m",
            "n_fast200", "vmax_kmh", "n_dup", "flags"]
    with open(os.path.join(ROOT, "notes", "quality_audit.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["instance"]):
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in cols})

    # ---- console summary (the numbers quoted in quality_findings.md)
    print("\n--- leakage ---")
    print("  pairs sharing >=10% of their nodes:")
    for (a, b), j in sorted(jac.items(), key=lambda kv: -kv[1]):
        if j >= 0.10:
            print(f"    {a} {b}  jaccard={j:.2f} containment={cont[(a, b)]:.2f} "
                  f"shared_nodes={leaks['shared'][(a, b)]} route_overlap={road.get((a, b), (0,))[0]:.2f}")
    print("  pairs where one extract nests in the other (containment>=0.90):")
    for (a, b), c in cont.items():
        if c >= 0.90 and jac[(a, b)] < 0.10:
            print(f"    {a} {b}  containment={c:.2f} shared_nodes={leaks['shared'][(a, b)]} "
                  f"route_overlap={road.get((a, b), (0,))[0]:.2f}")
    print(f"  max node Jaccard over ALL train-test pairs: "
          f"{max(r['max_jaccard_cross_split'] for r in rows):.4f}")
    print("  pairs that share driven roads (>=15% of the shorter route within 10 m of the other):")
    for (a, b), (v, short) in sorted(road.items(), key=lambda kv: -kv[1][0]):
        if v >= 0.15:
            ra, rb = next(x for x in rows if x["instance"] == a), next(x for x in rows if x["instance"] == b)
            print(f"    {a} {b}  route_overlap={v:.2f} nodes_jaccard={jac[(a, b)]:.2f} "
                  f"gap_between_centroids="
                  f"{hav(ra['lon'], ra['lat'], rb['lon'], rb['lat']) / 1000:.1f} km")
    print(f"  max route overlap over ALL train-test pairs: "
          f"{max(r['max_route_overlap'] for r in rows if r['split'] == 'test'):.4f}")

    tr = [r for r in rows if r['split'] == 'train']
    te = [r for r in rows if r['split'] == 'test']
    sets = {r['instance']: per[r['instance']] for r in rows}
    print("\n  shared nodes between each test drive and every train drive:")
    for t in te:
        tot = sum(len(sets[t['instance']] & sets[r['instance']]) for r in tr)
        print(f"    {t['instance']}: {tot} nodes shared with all 80 train drives")

    print("\n--- route gaps ---")
    for r in rows:
        if r["route_gaps"]:
            print(f"  {r['instance']} ({r['split']}): {r['route_gaps']} gap(s), "
                  f"max jump {r['gap_max_m']:.0f} m")
    print("\n--- poor coverage (cov50 < 0.90) ---")
    for r in sorted(rows, key=lambda r: r["cov50"])[:5]:
        print(f"  {r['instance']} ({r['split']}): cov50={r['cov50']:.3f} "
              f"moving={r['cov50_moving']:.3f} cov200={r['cov200']:.3f} "
              f"2km-rule={r['cov50_2km']:.3f} parked_share={r['off_top_cell']:.2f} "
              f"net_med={r['med_dev_net_m']:.0f} m")
    print("\n--- low coverage that is a parked blob (>50% of off points in one 100 m cell) ---")
    print("  " + ", ".join(r["instance"] for r in rows
                           if r["off_top_cell"] > PARK_CONC and r["n_off50"] >= 30))
    print("\n--- anomaly counts ---")
    for k, lab in (("route_gaps", "route gaps"), ("n_dt30", "dt>30 s"),
                   ("n_jump200", "jump>200 m"), ("n_fast200", "speed>200 km/h"),
                   ("n_dup", "duplicate points")):
        print(f"  {lab:18s} {sum(1 for r in rows if r[k]):2d} drives")
    print(f"  worst dt {max(r['dt_max_s'] for r in rows):.0f} s ({max(rows, key=lambda r: r['dt_max_s'])['instance']}), "
          f"worst jump {max(r['jump_max_m'] for r in rows):.0f} m "
          f"({max(rows, key=lambda r: r['jump_max_m'])['instance']})")

    print("\n--- duplicate instances? ---")
    rh = defaultdict(list)
    for r in rows:
        rh[hashlib.md5(r["_route"].tobytes()).hexdigest()].append(r["instance"])
    print("  identical route-arc sequences:", [v for v in rh.values() if len(v) > 1] or "none")
    gh = defaultdict(list)
    for r in rows:
        gh[hashlib.md5(np.round(r["_poly"], 5).tobytes()).hexdigest()].append(r["instance"])
    print("  identical route geometry:", [v for v in gh.values() if len(v) > 1] or "none")
    th = defaultdict(list)
    for r in rows:
        t = r["_track_ll"]
        th[(round(t[0, 0], 3), round(t[0, 1], 3), round(t[-1, 0], 3), round(t[-1, 1], 3), len(t))].append(r["instance"])
    print("  identical track signature:", [v for v in th.values() if len(v) > 1] or "none")

    out = figure(rows, leaks)
    print(f"\nwrote notes/quality_audit.csv ({len(rows)} rows) and {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
