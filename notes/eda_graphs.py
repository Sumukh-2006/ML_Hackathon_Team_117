"""Figures for the EDA graph document.

Computes the one thing the report could not state - network connectivity - and caches
it to connectivity.csv, then renders nine figures to notes/figs_graphs/.

Run: python notes/eda_graphs.py
"""
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notes", "figs_graphs")
CACHE = os.path.join(OUT, "connectivity.csv")

INK = "#1a1a1a"
MUTED = "#8a8a8a"
ACCENT = "#a8322d"
BLUE = "#2f5d8a"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 9,
    "axes.edgecolor": "#555555",
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def trips():
    out = []
    for split in ("train", "student_test"):
        for d in sorted(glob.glob(os.path.join(ROOT, split, "*"))):
            if os.path.isdir(d):
                out.append((os.path.basename(d), split, d))
    return out


def load(d, rid, ext):
    return np.loadtxt(os.path.join(d, f"{rid}.{ext}"), ndmin=2)


def connectivity():
    """Weakly connected components and isolated nodes, per trip."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return list(csv.DictReader(f))
    rows = []
    for rid, split, d in trips():
        arcs = load(d, rid, "arcs").astype(np.int64)
        n = len(load(d, rid, "nodes"))
        used = np.unique(arcs)
        m = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
        ncomp, lab = connected_components(m, directed=False)
        sizes = np.bincount(lab)
        isolated = n - len(used)
        # components that contain at least one arc (a lone node counts as its own)
        used_lab = np.unique(lab[used])
        rows.append({
            "trip": rid, "split": split, "nodes": n, "arcs": len(arcs),
            "n_components": int(ncomp),
            "components_with_arcs": int(len(used_lab)),
            "largest_share": float(sizes.max() / n),
            "isolated_nodes": int(isolated),
            "isolated_share": float(isolated / n),
            "arcs_outside_largest": None,
        })
        # share of arcs outside the largest component among arc-bearing components
        big = used_lab[np.argmax(sizes[used_lab])]
        outside = int((lab[arcs[:, 0]] != big).sum())
        rows[-1]["arcs_outside_largest"] = float(outside / len(arcs))
    os.makedirs(OUT, exist_ok=True)
    with open(CACHE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def fig_world():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    lons, lats = [], []
    for rid, split, d in trips():
        t = load(d, rid, "track")
        c = ACCENT if split == "student_test" else BLUE
        ax.plot(t[:, 0].mean(), t[:, 1].mean(), "o", ms=5, color=c,
                alpha=0.85, markeredgecolor="white", markeredgewidth=0.5)
        lons.append(t[:, 0])
        lats.append(t[:, 1])
    lon = np.concatenate(lons)
    lat = np.concatenate(lats)
    ax.set_xlabel("longitude (degrees)")
    ax.set_ylabel("latitude (degrees)")
    ax.set_title("One dot per trip")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.axhline(0, color=MUTED, lw=0.5, ls=":")
    ax.plot([], [], "o", color=BLUE, ms=5, label="80 training")
    ax.plot([], [], "o", color=ACCENT, ms=5, label="10 student-testing")
    ax.legend(loc="lower left", frameon=False, fontsize=7.5)
    hb = ax2.hexbin(lon, lat, gridsize=70, bins="log", cmap="Blues", mincnt=1)
    ax2.set_xlabel("longitude (degrees)")
    ax2.set_title(f"All {len(lon):,} fixes")
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-60, 80)
    ax2.axhline(0, color=MUTED, lw=0.5, ls=":")
    cb = fig.colorbar(hb, ax=ax2, pad=0.02, fraction=0.045)
    cb.set_label("fixes (log)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Figure 1  Where the 90 trips are", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g1_world.png"))
    plt.close(fig)


def fig_anatomy():
    picks = ["00000001", "00000003", "00000046", "00000009"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    for ax, rid in zip(axes.ravel(), picks):
        split, d = next((s, dd) for r, s, dd in trips() if r == rid)
        nodes, arcs = load(d, rid, "nodes"), load(d, rid, "arcs").astype(int)
        t, route = load(d, rid, "track"), load(d, rid, "route").astype(int).ravel()
        seg = nodes[arcs[:, :2]].mean(1)
        ax.plot(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1], ".",
                ms=0.25, color="#c9c9c9", zorder=1)
        for a in route:
            u, v = arcs[a]
            ax.plot(nodes[[u, v], 0], nodes[[u, v], 1], color=ACCENT, lw=1.1, zorder=2)
        ax.plot(t[:, 0], t[:, 1], ".", ms=1.1, color=BLUE, zorder=3)
        ax.set_title(f"trip {rid}   {len(t):,} fixes, {len(arcs):,} arcs", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Figure 2  Anatomy of a trip: fixes (blue), network (grey), reference route (red)",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g2_anatomy.png"))
    plt.close(fig)


def fig_sizes():
    fixes, km = [], []
    for rid, split, d in trips():
        t = load(d, rid, "track")
        fixes.append(len(t))
        seg = 111320 * np.cos(np.radians(t[:, 1]))
        dx = np.diff(t[:, 0]) * seg[:-1]
        dy = np.diff(t[:, 1]) * 110540
        km.append(np.hypot(dx, dy).sum() / 1000)
    order = np.argsort(fixes)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    axes[0].bar(range(len(fixes)), np.array(fixes)[order], color=BLUE, width=0.85)
    axes[0].set_yscale("log")
    axes[0].set_title("Fixes per trip (sorted, log scale)")
    axes[0].set_ylabel("GPS fixes")
    axes[0].axhline(np.median(fixes), color=ACCENT, lw=1)
    axes[0].text(len(fixes) * 0.02, np.median(fixes) * 1.25,
                 f"median {np.median(fixes):,.0f}", color=ACCENT, fontsize=7.5)
    axes[0].set_xlabel("trips")
    axes[1].bar(range(len(km)), np.sort(km), color=BLUE, width=0.85)
    axes[1].set_title("Drive length per trip (sorted)")
    axes[1].set_ylabel("km")
    axes[1].set_xlabel("trips")
    axes[1].axhline(np.median(km), color=ACCENT, lw=1)
    axes[1].text(len(km) * 0.02, np.median(km) * 1.5,
                 f"median {np.median(km):.1f} km", color=ACCENT, fontsize=7.5)
    fig.suptitle("Figure 3  Trip size and drive length", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g3_sizes.png"))
    plt.close(fig)


def fig_sampling():
    dts, frac = [], []
    for rid, split, d in trips():
        t = load(d, rid, "track")
        dt = np.diff(t[:, 2])
        dts.append(dt)
        frac.append(100 * (dt == 1.0).mean())
    dt = np.concatenate(dts)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    keep = dt[dt <= 60]
    axes[0].hist(keep, bins=np.arange(0.5, 61.5, 1), color=BLUE)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("seconds between consecutive fixes (capped at 60)")
    axes[0].set_ylabel("steps (log)")
    axes[0].set_title("Interval between fixes")
    axes[0].annotate(f"{100 * (dt == 1.0).mean():.2f}% of all steps are exactly 1 s",
                     xy=(4, 2e4), fontsize=7.5, color=ACCENT)
    order = np.argsort(frac)
    axes[1].step(np.arange(len(frac)), np.array(frac)[order], where="mid", color=BLUE)
    axes[1].axhline(np.median(frac), color=ACCENT, lw=1)
    axes[1].text(2, np.median(frac) - 6, f"median trip {np.median(frac):.1f}%",
                 color=ACCENT, fontsize=7.5)
    axes[1].set_xlabel("trips (sorted)")
    axes[1].set_ylabel("% of steps at 1 s")
    axes[1].set_title("Per-trip regularity")
    axes[1].annotate(f"worst {min(frac):.1f}%", xy=(0, min(frac)),
                     xytext=(6, min(frac) + 8), fontsize=7.5, color=ACCENT,
                     arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))
    fig.suptitle("Figure 4  Sampling is bimodal, not uniformly noisy",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g4_sampling.png"))
    plt.close(fig)


def fig_network():
    n, a = [], []
    for rid, split, d in trips():
        n.append(len(load(d, rid, "nodes")))
        a.append(len(load(d, rid, "arcs")))
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(n, a, s=16, color=BLUE, alpha=0.8, edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("road nodes in the trip's network")
    ax.set_ylabel("directed arcs")
    ax.set_title("Figure 5  Network size is set per trip")
    lo = min(min(n), min(a))
    hi = max(max(n), max(a))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.7, ls=":")
    ax.text(0.03, 0.95, f"{min(n):,}–{max(n):,} nodes\n{min(a):,}–{max(a):,} arcs",
            transform=ax.transAxes, va="top", fontsize=7.5, color=MUTED)
    fig.savefig(os.path.join(OUT, "g6_tmp.png")) if False else None
    fig.savefig(os.path.join(OUT, "g5_network.png"))
    plt.close(fig)


def fig_connectivity(rows):
    large = np.array([float(r["largest_share"]) for r in rows]) * 100
    iso = np.array([float(r["isolated_share"]) for r in rows]) * 100
    outside = np.array([float(r["arcs_outside_largest"]) for r in rows]) * 100
    comps = np.array([float(r["components_with_arcs"]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    axes[0].hist(large, bins=20, color=BLUE)
    axes[0].set_title("Largest component")
    axes[0].set_xlabel("% of nodes")
    axes[0].set_ylabel("trips")
    axes[0].axvline(large.min(), color=ACCENT, lw=1)
    axes[0].text(0.05, 0.92, f"min {large.min():.1f}%", transform=axes[0].transAxes,
                 color=ACCENT, fontsize=7.5, va="top")
    axes[1].hist(np.log10(comps), bins=15, color=BLUE)
    axes[1].set_title("Components per trip")
    axes[1].set_xlabel("log10(count)")
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{10 ** v:.0f}"))
    axes[1].text(0.97, 0.92, f"median {np.median(comps):.0f}", transform=axes[1].transAxes,
                 fontsize=7.5, color=ACCENT, va="top", ha="right")
    axes[2].hist(outside, bins=20, color=BLUE)
    axes[2].set_title("Arcs outside largest")
    axes[2].set_xlabel("% of arcs")
    axes[2].axvline(np.median(outside), color=ACCENT, lw=1)
    axes[2].text(0.97, 0.92, f"median {np.median(outside):.1f}%", transform=axes[2].transAxes,
                 color=ACCENT, fontsize=7.5, va="top", ha="right")
    fig.suptitle("Figure 6  Every network is a pasted tile, not a clean island",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g6_connectivity.png"))
    plt.close(fig)


def fig_candidates():
    src = os.path.join(ROOT, "notes", "candidates_stats.csv")
    if not os.path.exists(src):
        return
    rows = list(csv.DictReader(open(src)))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for k, style in ((5, ":"), (10, "-"), (20, "--")):
        ys = []
        for r in (25, 50, 75):
            col = f"hit_r{r}_k{k}"
            ys.append(100 * np.mean([float(x[col]) for x in rows]))
        ax.plot([25, 50, 75], ys, style, marker="o", ms=4,
                color=BLUE if k == 10 else MUTED,
                label=f"K = {k}" + ("  (chosen)" if k == 10 else ""))
        for x, y in zip((25, 50, 75), ys):
            if k == 10:
                ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points",
                            xytext=(4, -10), fontsize=7.5, color=BLUE)
    ax.set_xlabel("search radius (m)")
    ax.set_ylabel("% of fixes whose reference arc is in the candidate set")
    ax.set_title("Figure 7  Candidate recall against the reference route")
    ax.set_xticks([25, 50, 75])
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(OUT, "g7_candidates.png"))
    plt.close(fig)


def fig_residual():
    per_trip, pooled = [], []
    for rid, split, d in trips():
        nodes, arcs, t, route = (load(d, rid, "nodes"), load(d, rid, "arcs").astype(int),
                                 load(d, rid, "track"), load(d, rid, "route").astype(int).ravel())
        k = 111320 * np.cos(np.radians(t[:, 1].mean()))
        P = np.c_[t[:, 0] * k, t[:, 1] * 110540]
        A = np.c_[nodes[arcs[route][:, 0], 0] * k, nodes[arcs[route][:, 0], 1] * 110540]
        B = np.c_[nodes[arcs[route][:, 1], 0] * k, nodes[arcs[route][:, 1], 1] * 110540]
        ab = B - A
        L2 = np.maximum((ab ** 2).sum(1), 1e-18)
        best = np.full(len(P), np.inf)
        for s in range(0, len(P), 1500):
            p = P[s:s + 1500]
            ap = p[:, None, :] - A[None]
            tt = np.clip((ap * ab[None]).sum(2) / L2[None], 0, 1)
            pr = A[None] + tt[:, :, None] * ab[None]
            best[s:s + 1500] = np.sqrt(((p[:, None] - pr) ** 2).sum(2)).min(1)
        per_trip.append(np.median(best))
        pooled.append(best)
    pooled = np.concatenate(pooled)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    axes[0].hist(np.clip(pooled, 0, 50), bins=50, color=BLUE)
    axes[0].set_xlabel("distance from fix to its reference arc (m, capped at 50)")
    axes[0].set_ylabel("fixes")
    axes[0].set_title("Pooled residual")
    axes[0].axvline(np.median(pooled), color=ACCENT, lw=1)
    axes[0].text(np.median(pooled) + 1, axes[0].get_ylim()[1] * 0.85,
                 f"median {np.median(pooled):.1f} m", color=ACCENT, fontsize=7.5)
    pt = np.sort(per_trip)
    axes[1].plot(np.arange(len(pt)), pt, ".", ms=4, color=BLUE)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("trips (sorted)")
    axes[1].set_ylabel("median residual (m)")
    axes[1].set_title("Per-trip median residual")
    axes[1].annotate(f"{pt.max():.0f} m", xy=(len(pt) - 1, pt.max()),
                     xytext=(-30, -4), textcoords="offset points",
                     fontsize=7.5, color=ACCENT)
    fig.suptitle("Figure 8  GPS noise: one trip's labels sit two orders of magnitude worse",
                 fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g8_residual.png"))
    plt.close(fig)


def fig_geometry():
    lens, twin = [], []
    for rid, split, d in trips():
        nodes = load(d, rid, "nodes")
        arcs = load(d, rid, "arcs").astype(np.int64)
        route = load(d, rid, "route").astype(np.int64).ravel()
        dxy = (nodes[arcs[:, 0]] - nodes[arcs[:, 1]]) * np.array([111320.0, 110540.0])
        L = np.hypot(dxy[:, 0], dxy[:, 1])
        lens.append(L)
        key = {(int(u), int(v)) for u, v in arcs}
        twin.append(100 * np.mean([(int(arcs[a][1]), int(arcs[a][0])) in key for a in route]))
    L = np.concatenate(lens)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    axes[0].hist(np.clip(L, 0.1, 2000), bins=np.logspace(-1, 3.3, 60), color=BLUE)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("segment length (m, log)")
    axes[0].set_ylabel("arcs")
    axes[0].set_title("Segment length")
    axes[0].axvline(np.median(L), color=ACCENT, lw=1)
    axes[0].text(np.median(L) * 1.2, axes[0].get_ylim()[1] * 0.8,
                 f"median {np.median(L):.1f} m", color=ACCENT, fontsize=7.5)
    axes[1].hist(twin, bins=20, color=BLUE)
    axes[1].set_xlabel("% of reference arcs with a reverse twin")
    axes[1].set_ylabel("trips")
    axes[1].set_title("Direction ambiguity per trip")
    axes[1].annotate(f"{np.mean(twin):.1f}% mean", xy=(np.mean(twin), 0),
                     xytext=(-52, 18), textcoords="offset points",
                     fontsize=7.5, color=ACCENT)
    fig.suptitle("Figure 9  Geometry scale and direction ambiguity", fontsize=9.5,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "g9_geometry.png"))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = connectivity()
    fig_world()
    fig_anatomy()
    fig_sizes()
    fig_sampling()
    fig_network()
    fig_connectivity(rows)
    fig_candidates()
    fig_residual()
    fig_geometry()
    print(f"wrote figures to {OUT}")
    print(f"connectivity: largest-component share min "
          f"{min(float(r['largest_share']) for r in rows) * 100:.1f}%"
          f"  median components {np.median([float(r['components_with_arcs']) for r in rows]):.0f}"
          f"  median isolated "
          f"{np.median([float(r['isolated_share']) for r in rows]) * 100:.2f}%")


if __name__ == "__main__":
    main()
