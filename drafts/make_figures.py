"""Draw the figures of dataset.typ from drafts/stats (compute_stats.py).

Run from the project root:  .venv/bin/python drafts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from dataset import Corpus, Track

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent
FIG: Path = HERE / "figures"
STATS: Path = HERE / "stats"

# Validated categorical slots (dataviz reference palette, light mode).
TRAIN, TEST = "#2a78d6", "#eb6834"
ORIG, NOISY = "#1baf7a", "#6250d6"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0"
NOT_TAGGED = "#b9b8b2"
SPLIT_COLOR = {"train": TRAIN, "student_test": TEST}
SPLIT_LABEL = {"train": "train (80)", "student_test": "student_test (10)"}
TAGS = ["gaps", "u-turns", "loops", "hives", "congruence-issues"]
WIDTH = 6.7  # inches, the Typst text width

plt.rcParams.update({
    "font.family": ["Noto Sans", "DejaVu Sans"], "font.size": 7.5,
    "axes.titlesize": 8, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "legend.frameon": False, "legend.fontsize": 7,
    "legend.handlelength": 1.2, "figure.dpi": 150,
    "svg.fonttype": "path", "lines.linewidth": 1.4,
})


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def dot(color, label, ms=4.5):
    return Line2D([], [], ls="", marker="o", ms=ms, color=color, label=label)


def split_legend(ax, loc="best", **kw):
    ax.legend(handles=[dot(c, SPLIT_LABEL[s]) for s, c in SPLIT_COLOR.items()], loc=loc, **kw)


def has_tag(df: pd.DataFrame, tag: str) -> np.ndarray:
    return df.tags.fillna("").str.split(";").apply(lambda t: tag in t).values


def note(ax, text, xy, xytext, **kw):
    ax.annotate(text, xy, xytext, fontsize=6.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6), **kw)


# ---------------------------------------------------------------- one folder
def window_arcs(graph, box):
    x0, x1, y0, y1 = box
    fx, fy = graph.node_x[graph.arc_from], graph.node_y[graph.arc_from]
    tx, ty = graph.node_x[graph.arc_to], graph.node_y[graph.arc_to]
    keep = ((np.maximum(fx, tx) > x0) & (np.minimum(fx, tx) < x1)
            & (np.maximum(fy, ty) > y0) & (np.minimum(fy, ty) < y1))
    keep &= graph.canonical_arc == np.arange(graph.arc_count())
    return np.stack([np.c_[fx[keep], fy[keep]], np.c_[tx[keep], ty[keep]]], axis=1)


def draw_scene(ax, graph, track, orig, route, box, fixes=True):
    ax.add_collection(LineCollection(window_arcs(graph, box), colors="#c9c8c2", lw=0.7, zorder=1))
    seg = np.stack([np.c_[graph.node_x[graph.arc_from[route]], graph.node_y[graph.arc_from[route]]],
                    np.c_[graph.node_x[graph.arc_to[route]], graph.node_y[graph.arc_to[route]]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#f2b705", lw=4.5, alpha=0.55, zorder=2,
                                     capstyle="round"))
    ax.plot(orig.x, orig.y, color=ORIG, lw=1.0, zorder=3)
    if fixes:
        ax.scatter(track.x, track.y, s=5, color=NOISY, lw=0, zorder=4)
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(GRID)


def scale_bar(ax, metres, box):
    x0, x1, y0, y1 = box
    xe, ys = x1 - 0.05 * (x1 - x0), y0 + 0.06 * (y1 - y0)
    ax.add_patch(Rectangle((xe - metres - 0.03 * (x1 - x0), y0 + 0.02 * (y1 - y0)),
                           metres + 0.06 * (x1 - x0), 0.13 * (y1 - y0), fc="white", ec="none",
                           alpha=0.85, zorder=5))
    ax.plot([xe - metres, xe], [ys, ys], color=INK, lw=1.2, zorder=6, solid_capstyle="butt")
    ax.text(xe - metres / 2, ys + 0.015 * (y1 - y0), f"{metres:g} m", ha="center",
            va="bottom", fontsize=6.3, zorder=6)


def square_box(cx, cy, half):
    return (cx - half, cx + half, cy - half, cy + half)


def fig_example() -> None:
    corpus = Corpus.from_root(ROOT / "data" / "corpus")
    folders = {f.name(): f for s in ["train", "student_test"] for f in corpus.folders(s)}

    def load(name):
        f = folders[name]
        graph = f.graph()
        orig = Track.from_file(ROOT / "data" / "corpus_original" / f.split / name / f"{name}.track",
                               graph.projection)
        return graph, f.track(graph), orig, f.route(graph).arcs

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.75))
    graph, track, orig, route = load("00000007")
    cx, cy = (orig.x.min() + orig.x.max()) / 2, (orig.y.min() + orig.y.max()) / 2
    box = square_box(cx, cy, max(np.ptp(orig.x), np.ptp(orig.y)) / 2 + 300)
    draw_scene(axes[0], graph, track, orig, route, box, fixes=False)
    scale_bar(axes[0], 2000, box)
    axes[0].set_title("a  A whole drive (test folder 07)")

    c = len(orig.x) // 3
    box = square_box(orig.x[c], orig.y[c], 110)
    draw_scene(axes[1], graph, track, orig, route, box)
    scale_bar(axes[1], 50, box)
    axes[1].set_title("b  Zoom: one fix per second")

    graph, track, orig, route = load("00000054")
    sub = slice(None, None, 5)
    d = np.hypot(orig.x[sub, None] - orig.x[None, sub], orig.y[sub, None] - orig.y[None, sub])
    c = int(np.argmax((d < 30).sum(axis=1))) * 5     # densest spot, display only
    box = square_box(orig.x[c], orig.y[c], 140)
    draw_scene(axes[2], graph, track, orig, route, box)
    scale_bar(axes[2], 50, box)
    axes[2].set_title("c  A “hive”: car standing (train 54)")

    handles = [Line2D([], [], color="#c9c8c2", lw=1, label="road graph (the map)"),
               Line2D([], [], color="#f2b705", lw=4, alpha=0.55, label="true route (the answer)"),
               Line2D([], [], color=ORIG, lw=1.2, label="original GPS track"),
               dot(NOISY, "given track: the input (noisy)", ms=3)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.8)
    save(fig, "example")


# ---------------------------------------------------------------- world map
def land(ax, xlim, ylim) -> None:
    geo = json.loads((HERE / "geo" / "ne_110m_land.geojson").read_text())
    for feat in geo["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, fc="#eeede9",
                                 ec="#d6d5cf", lw=0.4, zorder=0))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_world(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={"width_ratios": [1.9, 1]})
    europe = (-11, 32, 34, 66)
    for ax, (xl, yl), title in [
        (axes[0], ((-170, 180), (-58, 80)), "a  Where the drives took place"),
        (axes[1], (europe[:2], europe[2:]), "b  Europe, enlarged"),
    ]:
        land(ax, xl, yl)
        for split in ["train", "student_test"]:
            d = df[df.split == split]
            ax.scatter(d.lon, d.lat, s=14, color=SPLIT_COLOR[split], ec="white", lw=0.6, zorder=3)
        ax.set_title(title)
    axes[0].add_patch(Rectangle((europe[0], europe[2]), europe[1] - europe[0], europe[3] - europe[2],
                                fill=False, ec=INK2, lw=0.6, ls=(0, (2, 2)), zorder=4))
    inside = (df.lon.between(*europe[:2]) & df.lat.between(*europe[2:])).sum()
    axes[1].text(0.02, 0.02, f"{inside} of {len(df)} folders", transform=axes[1].transAxes,
                 fontsize=6.5, color=INK2)
    split_legend(axes[0], loc="lower left")
    fig.tight_layout(w_pad=0.5)
    save(fig, "world")


# ---------------------------------------------------------------- tags
def fig_tags(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH * 0.62, 1.9))
    names = TAGS + ["(no tag)"]
    y = np.arange(len(names))[::-1]
    for split, off in [("train", 0.2), ("student_test", -0.2)]:
        m = (df.split == split).values
        flags = [has_tag(df, t)[m] for t in TAGS]
        flags.append(df.tags.fillna("").values[m] == "")
        share = np.array([f.mean() for f in flags]) * 100
        ax.barh(y + off, share, height=0.38, color=SPLIT_COLOR[split], label=SPLIT_LABEL[split])
        for yy, v, f in zip(y + off, share, flags):
            ax.text(v + 1.2, yy, f"{f.sum()}", va="center", fontsize=5.8, color=INK2)
    ax.set_yticks(y, names)
    ax.set_xlabel("folders with the tag (%); numbers at bar ends are folder counts")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "tags")


# ---------------------------------------------------------------- sizes
def strip(ax, df, col, scale, label, title):
    rng = np.random.default_rng(0)
    for k, split in enumerate(["train", "student_test"]):
        v = df.loc[df.split == split, col].values * scale
        ax.scatter(v, k + rng.uniform(-0.22, 0.22, len(v)), s=9, color=SPLIT_COLOR[split],
                   ec="white", lw=0.4, zorder=3)
        med = np.median(v)
        ax.plot([med, med], [k - 0.34, k + 0.34], color=INK, lw=1.3, zorder=4)
        ax.text(med, k - 0.42, f"{med:,.0f}", ha="center", va="bottom", fontsize=6, color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["train", "test"])
    ax.set_ylim(1.55, -0.85)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(label)
    ax.set_title(title)


def fig_sizes(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.75))
    strip(axes[0], df, "fixes", 1, "GPS fixes (≈ seconds of driving)", "a  Track length")
    strip(axes[1], df, "route_km", 1, "kilometres of road", "b  True route length")
    strip(axes[2], df, "arcs", 1, "arcs (road pieces) in the map", "c  Road graph size")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    axes[2].legend(handles=[dot(TRAIN, "train", 4), dot(TEST, "student_test", 4),
                            Line2D([], [], color=INK, lw=1.3, label="median")],
                   loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "sizes")


# ---------------------------------------------------------------- sampling
def fig_sampling(df: pd.DataFrame, pooled) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    dt = pooled["dt"]
    ax = axes[0]
    bins = np.r_[0.5, 1.5, np.logspace(np.log10(1.5), np.log10(dt.max() * 1.2), 40)[1:]]
    h, e = np.histogram(dt, bins)
    ax.bar(e[:-1], h, width=np.diff(e), align="edge", color=TRAIN, ec="white", lw=0.4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("time between consecutive fixes (s)")
    ax.set_ylabel("number of intervals")
    note(ax, f"{np.isclose(dt, 1).mean():.1%} are exactly 1 s", (1.5, h[0] * 0.7), (6, h[0] * 0.15))
    note(ax, f"one gap of {dt.max() / 3600:.1f} h", (dt.max(), 1.4), (dt.max() / 300, 30))
    ax.set_title("a  Sampling interval, all tracks pooled")

    ax = axes[1]
    rng = np.random.default_rng(2)
    tagged = has_tag(df, "gaps")
    for k, (flag, color) in enumerate([(True, TRAIN), (False, NOT_TAGGED)]):
        v = df.dt_max.values[tagged == flag]
        ax.scatter(v, k + rng.uniform(-0.22, 0.22, len(v)), s=11, color=color, ec="white",
                   lw=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks([0, 1], [f"tagged “gaps”\n({tagged.sum()} folders)",
                           f"not tagged\n({(~tagged).sum()} folders)"])
    ax.set_ylim(1.6, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("longest interval in the track (s)")
    note(ax, "one track with a 2 s interval", (2, 1.15), (6, 1.42))
    ax.set_title("b  The “gaps” tag = any interval ≥ 2 s")
    fig.tight_layout(w_pad=1.5)
    save(fig, "sampling")


# ---------------------------------------------------------------- noise
def fig_noise(pooled) -> None:
    e, n = pooled["noise_e"], pooled["noise_n"]
    sigma = 0.5 * (e.std() + n.std())
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.2), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    ax = axes[0]
    lim = 22
    ax.hist2d(e, n, bins=70, range=[[-lim, lim], [-lim, lim]], cmin=1,
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#cde2fb", "#3987e5", "#0d366b"]),
              norm=matplotlib.colors.LogNorm(), rasterized=True)
    for k in (1, 2, 3):
        ax.add_patch(plt.Circle((0, 0), k * sigma, fill=False, ec=INK, lw=0.5, ls=(0, (2, 2))))
    ax.text(0, -3 * sigma - 0.6, f"3σ = {3 * sigma:.0f} m", fontsize=6, color=INK, ha="center", va="top")
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlabel("east shift (m)"); ax.set_ylabel("north shift (m)")
    ax.set_title("a  Shift of each fix")

    ax = axes[1]
    r = np.hypot(e, n)
    ax.hist(r, bins=np.linspace(0, 30, 61), density=True, color=TRAIN, ec="white", lw=0.3, label="measured")
    rr = np.linspace(0, 30, 300)
    ax.plot(rr, rr / sigma**2 * np.exp(-rr**2 / (2 * sigma**2)), color=INK, lw=1.1,
            label=f"Gaussian,\nσ = {sigma:.1f} m")
    ax.set_xlabel("distance moved (m)"); ax.set_ylabel("share of fixes (density)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95))
    ax.set_title("b  How far fixes moved")

    ax = axes[2]
    lags, ac = pooled["noise_lags"], pooled["noise_autocorr"]
    ax.plot(lags, ac, color=TRAIN, marker="o", ms=3.5, label="measured")
    ax.plot(lags, ac[0] ** lags, color=INK, lw=1.0, ls=(0, (3, 2)), label="if it faded at a fixed rate")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 30, 60], ["1", "2", "5", "10", "30", "60"])
    ax.set_xlabel("seconds between two fixes")
    ax.set_ylabel("correlation of their shifts")
    ax.text(6.5, 0.165, "measured", fontsize=6.5, color=INK)
    ax.text(2.4, 0.05, "fixed-rate\nfade", fontsize=6.5, color=INK2)
    ax.set_title("c  Shifts persist for ~20 s")
    fig.tight_layout(w_pad=1.2)
    save(fig, "noise")


# ---------------------------------------------------------------- noise effects
def fig_noise_effect(df: pd.DataFrame, pooled) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25))
    ax = axes[0]
    for key, color, label, dx, dy, ha in [("dist_orig", ORIG, "original track", 1 / 1.4, 6, "right"),
                                          ("dist_noisy", NOISY, "given track", 1.4, -8, "left")]:
        d = np.sort(pooled[key])
        ax.plot(np.maximum(d, 0.01), np.linspace(0, 100, len(d)), color=color, label=label)
        med = np.median(d)
        ax.plot([med], [50], "o", ms=3.5, color=color)
        ax.text(med * dx, 50 + dy, f"median {med:.1f} m", fontsize=6.3, color=INK2, ha=ha, va="center")
    ax.set_xscale("log"); ax.set_xlim(0.05, 1000)
    ax.set_xlabel("distance from fix to true route (m)")
    ax.set_ylabel("fixes closer than that (%)")
    ax.legend(loc="lower right")
    ax.set_title("a  Fixes drift off the road")

    ax = axes[1]
    bins = np.linspace(0, 45, 91)
    for key, color, label in [("speed_orig", ORIG, "original track"), ("speed_noisy", NOISY, "given track")]:
        ax.hist(pooled[key] * 3.6, bins * 3.6, density=True, histtype="step", color=color, lw=1.3, label=label)
    note(ax, "stopped cars\n(spike runs off scale)", (2, 0.05), (28, 0.045))
    note(ax, "…appear to move\nat ~25 km/h", (24, 0.016), (70, 0.022))
    ax.set_xlabel("speed between consecutive fixes (km/h)")
    ax.set_ylabel("share of intervals (density)")
    ax.set_ylim(0, 0.09)
    ax.legend(loc="upper right")
    ax.set_title("b  Standing still looks like moving")

    ax = axes[2]
    ratio = df.track_km_noisy / df.track_km_orig
    for split in ["train", "student_test"]:
        m = df.split == split
        ax.scatter(df.speed_median[m] * 3.6, ratio[m], s=12, color=SPLIT_COLOR[split], ec="white", lw=0.4, zorder=3)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 5], ["1×", "1.5×", "2×", "3×", "5×"])
    ax.minorticks_off()
    ax.set_xlabel("typical speed of the drive (km/h)")
    ax.set_ylabel("track length, given ÷ original")
    split_legend(ax, loc="upper right", fontsize=6.3)
    ax.set_title("c  Slow drives look longer")
    fig.tight_layout(w_pad=1.2)
    save(fig, "noise_effect")


# ---------------------------------------------------------------- congruence
def fig_congruence(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 1.9))
    d = df.assign(cong=has_tag(df, "congruence-issues")).sort_values("mismatch_max", ascending=False)
    x = np.arange(len(d))
    ax.bar(x, d.mismatch_max, width=0.75, color=np.where(d.cong, NOISY, NOT_TAGGED))
    ax.set_yscale("log")
    ax.set_xlim(-1, len(d))
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("the 90 folders, sorted")
    ax.set_ylabel("farthest fix from\nthe true route (m)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=NOISY, label=f"tagged “congruence-issues” ({d.cong.sum()})"),
                       Rectangle((0, 0), 1, 1, color=NOT_TAGGED, label=f"not tagged ({(~d.cong).sum()})")],
              loc="upper right")
    fig.tight_layout()
    save(fig, "congruence")


# ---------------------------------------------------------------- road graph
def fig_graph(df: pd.DataFrame, pooled) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.25))
    ax = axes[0]
    bins = pooled["arc_bins"]
    centers = np.sqrt(bins[:-1] * bins[1:])
    hm = pooled["map_arc_hist"] / pooled["map_arc_hist"].sum()
    hr, _ = np.histogram(pooled["route_arc_len"], bins)
    hr = hr / hr.sum()
    ax.plot(centers, hm * 100, color=MUTED, label="all arcs in the maps")
    ax.plot(centers, hr * 100, color=TRAIN, label="arcs on true routes")
    med_map = np.interp(0.5, np.cumsum(hm), centers)
    med_route = np.median(pooled["route_arc_len"])
    ax.set_xscale("log"); ax.set_xlim(0.3, 3000)
    ax.set_xlabel("arc length (m)"); ax.set_ylabel("share of arcs (%)")
    ax.set_ylim(0, 6.2)
    ax.legend(loc="upper left", fontsize=6.3)
    ax.text(0.98, 0.42, f"median\nmap {med_map:.0f} m\nroute {med_route:.0f} m", transform=ax.transAxes,
            ha="right", fontsize=6.3, color=INK2)
    ax.set_title("a  Arcs are short straight pieces")

    ax = axes[1]
    deg = pooled["roads_at_node"][1:6].astype(float)
    share = deg / pooled["roads_at_node"].sum() * 100
    labels = ["1\ndead\nend", "2\nalong\na road", "3", "4", "5+"]
    share[-1] = 100 - share[:-1].sum()
    ax.bar(range(5), share, color=[MUTED, MUTED, TRAIN, TRAIN, TRAIN], width=0.7)
    for k, v in enumerate(share):
        ax.text(k, v + 1.5, f"{v:.0f}%" if v >= 1 else f"{v:.1f}%", ha="center", fontsize=6, color=INK2)
    ax.set_xticks(range(5), labels, fontsize=6.3)
    ax.text(3, 33, "junctions", ha="center", fontsize=6.5, color=TRAIN)
    ax.plot([2.6, 4.4], [31, 31], color=TRAIN, lw=0.8)
    ax.set_xlabel("roads meeting at a node")
    ax.set_ylabel("share of nodes (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("b  Most nodes are not junctions")

    ax = axes[2]
    rng = np.random.default_rng(3)
    for k, split in enumerate(["train", "student_test"]):
        m = (df.split == split).values
        v = 100 * df.route_arcs.values[m] / df.arcs.values[m]
        ax.scatter(v, k + rng.uniform(-0.22, 0.22, m.sum()), s=10, color=SPLIT_COLOR[split], ec="white", lw=0.4, zorder=3)
        med = np.median(v)
        ax.plot([med, med], [k - 0.34, k + 0.34], color=INK, lw=1.3, zorder=4)
        ax.text(med, k - 0.42, f"{med:.1f}%", ha="center", va="bottom", fontsize=6, color=INK, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xscale("log")
    ax.set_xticks([0.1, 1, 10], ["0.1%", "1%", "10%"])
    ax.set_yticks([0, 1], ["train", "test"])
    ax.set_ylim(1.55, -0.85)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("map arcs that are on the true route")
    ax.set_title("c  The route is a sliver of the map")
    fig.tight_layout(w_pad=1.2)
    save(fig, "graph")


# ---------------------------------------------------------------- route quirks
def fig_route(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.9), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    rows = [("u-turns", df.route_u_turns.values > 0, "route reverses onto\nthe same road (U-turn)"),
            ("loops", df.route_repeat_m.values > 0, "route drives a road\ntwice the same way")]
    y = np.arange(len(rows))[::-1].astype(float)
    for yy, (tag, seen, label) in zip(y, rows):
        t = has_tag(df, tag)
        for off, flag, color, name in [(0.18, True, TRAIN, f"tagged “{tag}”"), (-0.18, False, NOT_TAGGED, "not tagged")]:
            share = seen[t == flag].mean() * 100
            ax.barh(yy + off, share, height=0.34, color=color)
            ax.text(share + 1.5, yy + off, f"{seen[t == flag].sum()} of {(t == flag).sum()}  ({name})",
                    va="center", fontsize=6, color=INK2)
    ax.set_yticks(y, [r[2] for r in rows])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("folders where the true route shows it (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("a  Do the tags match the true routes?")

    ax = axes[1]
    d = df[df.route_breaks > 0].sort_values(["route_breaks", "id"])
    ax.barh(range(len(d)), d.route_breaks, color=TRAIN, height=0.65)
    ax.set_yticks(range(len(d)), [f"{'test' if s != 'train' else 'train'} {i:02d}" for s, i in zip(d.split, d.id)],
                  fontsize=6)
    ax.set_xlabel("jumps between unconnected arcs")
    ax.set_xticks([0, 1, 2, 3])
    ax.grid(axis="y", visible=False)
    ax.set_title(f"b  Routes with jumps ({len(d)} of {len(df)})")
    fig.tight_layout(w_pad=1.5)
    save(fig, "route")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    for old in ["corpus.svg"]:
        (FIG / old).unlink(missing_ok=True)
    df = pd.read_csv(STATS / "per_folder.csv", dtype={"tags": str})
    pooled = np.load(STATS / "pooled.npz")
    fig_example()
    fig_world(df)
    fig_tags(df)
    fig_sizes(df)
    fig_sampling(df, pooled)
    fig_noise(pooled)
    fig_noise_effect(df, pooled)
    fig_congruence(df)
    fig_graph(df, pooled)
    fig_route(df)


if __name__ == "__main__":
    main()
