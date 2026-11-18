"""
EDA for the map-matching dataset (one folder = one drive, 4 files).

Rationale: EDA here is not "describe the dataframe". It is answering 3 questions that
decide the whole solution: (1) how big is the GPS noise vs the road spacing,
(2) how ambiguous is snapping (how many candidate roads inside the noise radius),
(3) does the test set look like the train set. Every plot below answers one of those.

Run:  python notes/eda.py
Out:  notes/figs/*.png  +  notes/eda_audit.csv
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
os.makedirs(FIGS, exist_ok=True)
R_EARTH = 6371000.0


# ---------------------------------------------------------------- loading
def load(inst_dir):
    name = os.path.basename(inst_dir)

    def rd(ext, ncol):
        p = os.path.join(inst_dir, f"{name}.{ext}")
        if not os.path.exists(p):
            return None
        a = np.loadtxt(p, ndmin=2)
        return a if ncol is None else a[:, :ncol]

    nodes = rd("nodes", 2)                                  # lon, lat
    arcs = rd("arcs", 2).astype(int)                        # from_node, to_node
    route = np.atleast_1d(np.loadtxt(os.path.join(inst_dir, f"{name}.route"))).astype(int)
    track = rd("track", None)                               # lon, lat, t
    return nodes, arcs, route, track


# ---------------------------------------------------------------- geo helpers
def hav(lon1, lat1, lon2, lat2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = la2 - la1, np.radians(lon2 - lon1)
    h = np.sin(dphi / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def to_xy(p, lon0, lat0):
    """local metric plane (metres) - fine at city scale"""
    k = np.cos(np.radians(lat0))
    return (p[:, 0] - lon0) * 111320.0 * k, (p[:, 1] - lat0) * 111320.0


def seg_dist(P, A, B):
    """P:(n,2) A:(m,2) B:(m,2) metric -> dist (n,m) point-to-segment"""
    ax, ay = A[:, 0][None, :], A[:, 1][None, :]
    bx, by = B[:, 0][None, :], B[:, 1][None, :]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-12)
    t = np.clip(((P[:, 0][:, None] - ax) * dx + (P[:, 1][:, None] - ay) * dy) / L2, 0, 1)
    qx, qy = ax + t * dx, ay + t * dy
    return np.hypot(P[:, 0][:, None] - qx, P[:, 1][:, None] - qy)


def route_polyline(nodes, arcs, route):
    """ordered node chain of the ground-truth route.

    WRONG (earlier): [from nodes...] + [to nodes...]  -> runs the path, then jumps
    back to the start and re-traces it, drawing one phantom chord.
    RIGHT: start at the first arc's source, then each arc contributes its target.
    """
    chain = [int(arcs[route[0]][0])] + [int(arcs[r][1]) for r in route]
    return np.array(chain), nodes[np.array(chain)]


# ---------------------------------------------------------------- 1. the money plot
def fig_overview(inst, tag):
    nodes, arcs, route, track = load(inst)
    lon0 = nodes[:, 0].mean(); lat0 = nodes[:, 1].mean()
    nx, ny = to_xy(nodes, lon0, lat0)
    chain, poly = route_polyline(nodes, arcs, route)
    px, py = to_xy(poly, lon0, lat0)
    tx, ty = to_xy(track, lon0, lat0)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))

    # (a) network + true route
    ax[0].scatter(nx, ny, s=1, c="0.75", lw=0)
    ax[0].plot(px, py, "-", c="crimson", lw=1.4)
    ax[0].scatter(px[0], py[0], c="lime", s=60, marker="o", zorder=5, label="start")
    ax[0].scatter(px[-1], py[-1], c="blue", s=60, marker="s", zorder=5, label="end")
    ax[0].set_title(f"{tag}: road network + true route ({len(route)} arcs)")
    ax[0].legend(fontsize=8); ax[0].set_aspect("equal"); ax[0].tick_params(labelsize=7)

    # (b) zoom: raw GPS vs route - the actual problem
    cx, cy = np.median(px), np.median(py)
    near = (np.abs(tx - cx) < 1200) & (np.abs(ty - cy) < 1200)
    ax[1].plot(px, py, "-", c="crimson", lw=2, label="true route")
    ax[1].plot(tx[near], ty[near], ".-", c="0.2", ms=2, lw=.5, label="raw GPS")
    ax[1].set_xlim(cx - 1200, cx + 1200); ax[1].set_ylim(cy - 1200, cy + 1200)
    ax[1].set_title("zoom: GPS noise vs ground truth"); ax[1].legend(fontsize=8)
    ax[1].set_aspect("equal"); ax[1].tick_params(labelsize=7)

    # (c) noise histogram = residual from the TRUE route
    step = max(1, len(track) // 500)
    P = np.column_stack(to_xy(track[::step], lon0, lat0))
    segs = np.column_stack(to_xy(poly, lon0, lat0))
    d = seg_dist(P, segs[:-1], segs[1:]).min(axis=1)
    ax[2].hist(d, bins=50, color="steelblue")
    ax[2].axvline(np.median(d), c="k", ls="--", label=f"median {np.median(d):.0f} m")
    ax[2].axvline(np.percentile(d, 95), c="r", ls=":", label=f"p95 {np.percentile(d,95):.0f} m")
    ax[2].set_title("GPS error vs true route (sets the snap radius)")
    ax[2].set_xlabel("metres"); ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"01_overview_{tag}.png"), dpi=110)
    plt.close(fig)
    return np.median(d), np.percentile(d, 95)


# ---------------------------------------------------------------- 2. ambiguity: how many roads inside the noise radius
def fig_ambiguity(inst, tag, radius=60.0):
    nodes, arcs, route, track = load(inst)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    segs = np.column_stack(to_xy(nodes, lon0, lat0))
    A, B = segs[arcs[:, 0]], segs[arcs[:, 1]]
    seglen = np.hypot(*(B - A).T)

    step = max(1, len(track) // 300)
    P = np.column_stack(to_xy(track[::step], lon0, lat0))
    d = seg_dist(P, A, B)                      # (points, arcs)
    ncand = (d < radius).sum(axis=1)

    truth = np.zeros(len(P), int)
    _, poly = route_polyline(nodes, arcs, route)
    pt = np.column_stack(to_xy(poly, lon0, lat0))
    dtrue = seg_dist(P, pt[:-1], pt[1:]).min(axis=1)

    # naive baseline: snap each point to the NEAREST ARC EVER (no connectivity)
    naive_arc = d.argmin(axis=1)
    naive_err = np.array([min(abs(naive_arc[k] - i) for i in route[:1]) for k in range(0)]) if False else None
    route_set = set(route.tolist())
    naive_on_route = np.array([a in route_set for a in naive_arc])
    naive_right_scan = np.mean([(d[k, a] < d[k, a] + 1e-9) and (a in route_set) for k, a in enumerate(naive_arc)])

    # nearest NODE baseline
    nxy = segs
    dn = hav(nodes[None, :, 0], nodes[None, :, 1],
             track[::step][:, None, 0], track[::step][:, None, 1])
    nearest_node = dn.argmin(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(ncand, bins=np.arange(0, ncand.max() + 2) - .5, color="darkorange")
    ax[0].set_title(f"candidate arcs within {radius:.0f} m of a GPS point")
    ax[0].set_xlabel("# candidate arcs"); ax[0].set_ylabel("# points")
    ax[1].hist(seglen, bins=60, color="seagreen")
    ax[1].axvline(np.median(np.diff(track[:, 2])) * 100, c="k", ls="--",
                  label="point spacing (m/sample)")
    ax[1].set_title("arc length distribution vs GPS sampling spacing")
    ax[1].set_xlabel("arc length (m)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, f"02_ambiguity_{tag}.png"), dpi=110)
    plt.close(fig)
    return dict(mean_cand=ncand.mean(), max_cand=int(ncand.max()),
                naive_arc_hitrate=float(naive_right_scan),
                noise_med=float(np.median(dtrue)))


# ---------------------------------------------------------------- 3. dataset-level audit
def audit(inst):
    nodes, arcs, route, track = load(inst)
    name = os.path.basename(inst)
    seg = hav(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1], nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    rlen = seg[route].sum()
    step = np.diff(track[:, 2])
    tlen = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1]).sum()
    gaps = int(np.sum(arcs[route[:-1], 1] != arcs[route[1:], 0]))
    rs = set(route.tolist())
    tin = np.mean([a in rs for a in range(len(arcs))])
    return dict(instance=name, n_nodes=len(nodes), n_arcs=len(arcs), n_route=len(route),
                n_track=len(track), route_km=rlen / 1000, drive_km=tlen / 1000,
                ratio=rlen / max(tlen, 1), dt_med=float(np.median(step)),
                dt_max=float(step.max()), speed_kmh=float(tlen / track[-1, 2] * 3.6),
                route_gaps=gaps, route_frac_of_graph=tin)


def fig_dataset(rows):
    import csv
    with open(os.path.join(ROOT, "notes", "eda_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for lab, key in (("n_arcs", "n_arcs"), ("n_track", "n_track"), ("speed_kmh", "speed_kmh"), ("ratio", "ratio")):
        v = [r[key] for r in rows]
        ax[list(("n_arcs", "n_track", "speed_kmh", "ratio")).index(key)].hist(v, bins=25)
        ax[list(("n_arcs", "n_track", "speed_kmh", "ratio")).index(key)].set_title(lab)
    fig.suptitle("per-instance distributions (all 90 drives)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "03_dataset.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    train = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
    test = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))

    print("--- single-instance deep dives ---")
    for d in train[:3] + test[:2]:
        tag = ("test_" if "student_test" in d else "train_") + os.path.basename(d)
        med, p95 = fig_overview(d, tag)
        amb = fig_ambiguity(d, tag)
        print(f"{tag}: noise med {med:.0f} m p95 {p95:.0f} m | {amb}")

    print("\n--- dataset-wide audit ---")
    rows = [audit(d) for d in train + test]
    fig_dataset(rows)
    print(f"wrote notes/eda_audit.csv ({len(rows)} rows) and notes/figs/*.png")
