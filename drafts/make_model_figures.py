"""Draw the data figures of the models part of report.typ.

Reads only saved results (results/benchmark, results/executor_test,
results/learned) and, for the thinning figure, track 00000082 and its
clean original. Run from the project root:

    .venv/bin/python drafts/make_model_figures.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (FIG, GRID, INK, INK2, MUTED, ROOT, WIDTH, NOISY,  # noqa: E402
                          save, scale_bar, window_arcs)

from dataset import Corpus, Track  # noqa: E402
from learned.track import douglas_peucker  # noqa: E402

RESULTS: Path = ROOT / "results"

# Categorical slots 1-5 of the validated palette, in fixed order
# (validate_palette.js: all checks pass in light mode; contrast WARN, so
# every chart labels its marks directly).
MODELS: list[str] = ["hmm", "segment", "mcm", "mdp", "learned"]
NAME: dict[str, str] = {"hmm": "1 HMM", "segment": "2 Segment (Yin)",
                        "mcm": "3 MCM", "mdp": "4 MDP (Wöltche)",
                        "learned": "5 Ours, Wöltche's weights"}
COLOR: dict[str, str] = {"hmm": "#2a78d6", "segment": "#eb6834", "mcm": "#1baf7a",
                         "mdp": "#eda100", "learned": "#e87ba4"}
TAG_ORDER: list[str] = ["clean", "gaps", "u-turns", "loops", "hives", "congruence-issues"]


def summary(model: str, executor: bool = False) -> pd.DataFrame:
    base: Path = RESULTS / ("executor_test/benchmark" if executor else "benchmark")
    return pd.read_csv(base / model / "summary.csv")


def per_track(model: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "benchmark" / model / "per_track.csv", dtype={"id": str})


# ------------------------------------------------------------ overall error
def fig_overall() -> None:
    """Error of every model on each noisy test set. Model 5 appears twice:
    with Wöltche's weights, and with the trained weights (hatched; on train
    its value is the cross-validated one, since in-sample would flatter it)."""
    stats = json.loads((ROOT / "drafts" / "stats" / "models.json").read_text())
    cf = stats["cf"]
    panels = [("train", "a  train: 80 noisy"),
              ("student_test", "b  student_test: 10 noisy"),
              ("held_out_noisy", "c  held-out: 10 × 8 noisy draws")]
    names = [NAME[m] for m in MODELS] + ["5 Ours, trained weights"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.05), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (name, title) in zip(axes, panels):
        errors = [100 - cf[name][m] for m in MODELS]
        trained = (stats["training"]["exact_copies"]["pooled"] if name == "train"
                   else cf[name]["learned_trained"])
        errors.append(100 - trained)
        bars = ax.barh(y, errors, color=[COLOR[m] for m in MODELS] + [COLOR["learned"]], height=0.62)
        bars[-1].set_hatch("/////"); bars[-1].set_edgecolor("white"); bars[-1].set_linewidth(0)
        top = max(errors)
        for yi, e in zip(y, errors):
            ax.text(e + top * 0.02, yi, f"{e:.2f}", va="center", fontsize=6.5, color=INK)
        ax.set_xlim(0, top * 1.22)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("100 − weighted CF (%)")
    axes[0].set_yticks(y, names)
    fig.tight_layout(w_pad=1.0)
    save(fig, "models_overall")


# ------------------------------------------------------------- per hand tag
def fig_tags() -> None:
    """A table-like heat map: error on train per hand tag."""
    data = np.zeros((len(MODELS), len(TAG_ORDER)))
    counts = []
    for i, m in enumerate(MODELS):
        s = summary(m)
        s = s[s.split == "train"]
        for j, tag in enumerate(TAG_ORDER):
            row = s[s.group == tag].iloc[0]
            data[i, j] = 100 * (1 - row.weighted_correct_fraction)
            if i == 0:
                counts.append(int(row.tracks))
    fig, ax = plt.subplots(figsize=(WIDTH, 1.75))
    # Sequential blue ramp, light -> dark, on a log scale of the error.
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
    ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=0.8, vmax=45), aspect="auto")
    for i in range(len(MODELS)):
        best = np.argmin(data[:, :], axis=0)
        for j in range(len(TAG_ORDER)):
            v = data[i, j]
            dark = v > 8
            weight = "bold" if best[j] == i else "normal"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if dark else INK, fontweight=weight)
    ax.set_xticks(range(len(TAG_ORDER)),
                  [f"{t.replace('-issues', '')}\n({c} tracks)" for t, c in zip(TAG_ORDER, counts)])
    ax.set_yticks(range(len(MODELS)), [NAME[m] for m in MODELS])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("100 − weighted CF (%) on train, by Kubicka's hand tag; best per column in bold")
    save(fig, "models_tags")


# ---------------------------------------------- MDP vs ours, track by track
def fig_pertrack() -> None:
    a = per_track("mdp").set_index("id")
    b = per_track("learned").set_index("id")
    t = a[a.split == "train"]
    ids = t.index
    ea = np.clip(100 * (1 - a.loc[ids, "correct_fraction"]), 0.02, None)
    eb = np.clip(100 * (1 - b.loc[ids, "correct_fraction"]), 0.02, None)
    km = a.loc[ids, "true_length"] / 1000
    lost_a = (1 - a.loc[ids, "correct_fraction"]) * a.loc[ids, "true_length"] / 1000
    lost_b = (1 - b.loc[ids, "correct_fraction"]) * b.loc[ids, "true_length"] / 1000

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 2.7),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    better = eb < ea
    for mask, color, label in [(better, COLOR["learned"], "ours better"),
                               (~better, COLOR["mdp"], "MDP better or equal")]:
        ax.scatter(ea[mask], eb[mask], s=6 + 2.2 * km[mask], color=color, alpha=0.8,
                   edgecolors="white", linewidths=0.6, label=label, zorder=3)
    lim = (0.015, 100)
    ax.plot(lim, lim, color=MUTED, lw=0.8, zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("MDP: 100 − CF (%)")
    ax.set_ylabel("Ours: 100 − CF (%)")
    ax.set_title("a  Each train track (dot area ∝ route length)")
    for name in ["00000082", "00000033"]:
        ax.annotate(name[-2:], (ea[name], eb[name]), (ea[name] * 1.6, eb[name] * 0.45),
                    fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.legend(loc="upper left", markerscale=0.8)

    # b: where the lost kilometres are: the ten tracks each loses most on.
    order = (lost_a + lost_b).sort_values(ascending=False).index[:10]
    yy = np.arange(len(order))[::-1]
    bx.barh(yy + 0.19, lost_a[order], height=0.36, color=COLOR["mdp"], label="MDP")
    bx.barh(yy - 0.19, lost_b[order], height=0.36, color=COLOR["learned"], label="Ours")
    bx.set_yticks(yy, [f"track {i[-2:]}" for i in order])
    bx.grid(axis="y", visible=False)
    bx.set_xlabel("route wrongly matched (km, ≈ (1 − CF) × length)")
    bx.set_title("b  Where the kilometres are lost")
    bx.legend(loc="lower right")
    fig.tight_layout(w_pad=1.5)
    save(fig, "models_pertrack")
    print(f"pertrack: ours better on {int(better.sum())}, worse on {int((eb > ea).sum())}, "
          f"MDP lost {lost_a.sum():.1f} km, ours {lost_b.sum():.1f} km")


# ------------------------------------------------------- thinning tolerance
def fig_thinning() -> None:
    corpus = Corpus.from_root(ROOT / "data" / "corpus_fixed")
    folder = {f.name(): f for f in corpus.folders("train")}["00000082"]
    graph = folder.graph()
    noisy: Track = folder.track(graph)
    orig = Track.from_file(ROOT / "data" / "corpus_original" / "train" / "00000082"
                           / "00000082.track", graph.projection)
    tolerance = json.loads((RESULTS / "learned" / "parameters.json").read_text())["tolerance"]
    route = folder.route(graph).arcs

    cases = [(orig, 5.0, f"a  Clean original, 5 m"),
             (noisy, 5.0, f"b  Noisy track, 5 m (Wöltche)"),
             (noisy, tolerance, f"c  Noisy track, {tolerance:.1f} m (measured)")]
    # A 600 m square in the middle of the drive.
    c = len(orig.x) // 2
    half = 300
    box = (orig.x[c] - half, orig.x[c] + half, orig.y[c] - half, orig.y[c] + half)
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.45))
    for ax, (track, tol, title) in zip(axes, cases):
        kept = douglas_peucker(track.x, track.y, tol)
        ax.add_collection(LineCollection(window_arcs(graph, box), colors="#c9c8c2",
                                         lw=0.7, zorder=1))
        seg = np.stack([np.c_[graph.node_x[graph.arc_from[route]], graph.node_y[graph.arc_from[route]]],
                        np.c_[graph.node_x[graph.arc_to[route]], graph.node_y[graph.arc_to[route]]]],
                       axis=1)
        ax.add_collection(LineCollection(seg, colors="#f2b705", lw=4.0, alpha=0.5, zorder=2,
                                         capstyle="round"))
        ax.scatter(track.x, track.y, s=2.2, color=MUTED, lw=0, zorder=3)
        ax.plot(track.x[kept], track.y[kept], color=NOISY, lw=0.8, zorder=4)
        ax.scatter(track.x[kept], track.y[kept], s=11, color=NOISY, edgecolors="white",
                   linewidths=0.5, zorder=5)
        inside = ((track.x[kept] > box[0]) & (track.x[kept] < box[1])
                  & (track.y[kept] > box[2]) & (track.y[kept] < box[3])).sum()
        ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
        ax.set_title(title)
        ax.text(0.03, 0.97, f"{len(kept)} points kept in the whole drive\n{inside} in this window",
                transform=ax.transAxes, va="top", fontsize=6.3, color=INK,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        print(f"thinning {title}: {len(kept)} kept")
    scale_bar(axes[0], 100, box)
    fig.tight_layout(w_pad=0.6)
    save(fig, "models_thinning")


# ----------------------------------------------- measured choices (layer 1)
def fig_measured() -> None:
    m = json.loads((RESULTS / "learned" / "measurements.json").read_text())
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(WIDTH, 1.95))

    hist = sorted(m["tolerance"]["history"])
    tol = [h[0] for h in hist]
    pts = [h[2] for h in hist]
    ref = m["tolerance"]["reference_points"]
    ax.plot(tol, pts, color=COLOR["hmm"], marker="o", ms=3.5)
    ax.axhline(ref, color=MUTED, lw=0.8, ls="--")
    ax.text(tol[-1], ref * 1.05, "clean originals at 5 m", ha="right", va="bottom",
            fontsize=6.3, color=INK2)
    t = m["tolerance"]["tolerance"]
    ax.annotate(f"{t:.2f} m", (t, ref), (t + 6, ref * 2.0), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_yscale("log")
    ax.set_xlabel("Douglas–Peucker tolerance (m)")
    ax.set_ylabel("decision points, 80 tracks")
    ax.set_title("a  Thinning tolerance (bisection)")

    sp = m["speed"]["speed_error_by_half_window"]
    w = [float(k) for k in sp]
    e = [sp[k] for k in sp]
    bx.plot(w, e, color=COLOR["hmm"], marker="o", ms=3.5)
    best = m["speed"]["half_window"]
    bx.annotate(f"{best:g} s: {sp[str(best)]:.2f} m/s", (best, sp[str(best)]),
                (best + 4, 0.75), fontsize=6.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    bx.set_ylim(0, 1.2)
    bx.set_xlabel("half window of the speed fit (s)")
    bx.set_ylabel("median speed error (m/s)")
    bx.set_title("b  Speed window")

    lags = [1, 2, 3, 5, 10]
    meas = [m["noise"]["measured"]["acf"][str(k)] for k in lags]
    gen = [m["noise"]["generated"]["acf"][str(k)] for k in lags]
    cx.plot(lags, meas, color=COLOR["hmm"], marker="o", ms=4, label="measured")
    cx.plot(lags, gen, color=COLOR["segment"], marker="s", ms=2.5, lw=1.0, ls="--",
            label="regenerated")
    cx.set_ylim(0, 0.32)
    cx.set_xlabel("lag (s)")
    cx.set_ylabel("autocorrelation")
    cx.set_title("c  Noise model check")
    cx.legend(loc="upper right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "models_measured")


# --------------------------------------------------- cross-validated search
def search_runs(text: str, setting: str) -> tuple[list[float], list[float], list[float]]:
    """(in-sample error at start, after the search, held-out CF) per fold."""
    runs = re.findall(r"start: error ([\d.]+).*\n((?:sweep \d: error [\d.]+.*\n)+)"
                      + setting + r" fold (\d): held-out CF ([\d.]+)", text)
    s0 = [100 * float(r[0]) for r in runs[:5]]
    s1 = [100 * float(re.findall(r"error ([\d.]+)", r[1])[-1]) for r in runs[:5]]
    held = [float(r[3]) for r in runs[:5]]
    return s0, s1, held


def fig_cv() -> None:
    """Held-out error per fold for every way of setting the weights
    (drafts/stats/models.json, from the CV logs)."""
    tr = json.loads((ROOT / "drafts" / "stats" / "models.json").read_text())["training"]
    series = [("hand", "Wöltche's weights", MUTED, "o"),
              ("grid_own", "grid search, own tracks", COLOR["mdp"], "s"),
              ("grid_copies", "grid search, + noisy copies", COLOR["segment"], "D"),
              ("exact_copies", "exact search, + noisy copies", COLOR["learned"], "^")]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    folds = np.arange(5)
    offset = np.linspace(-0.22, 0.22, len(series))
    for (key, label, color, marker), dx in zip(series, offset):
        values = tr[key]["folds"]
        pooled = tr[key]["pooled"]
        tag = f"{label}: pooled {pooled:.2f}%" if pooled else label
        ax.scatter([f + dx for f, v in zip(folds, values) if v is not None],
                   [100 - v for v in values if v is not None], color=color, s=24,
                   marker=marker, zorder=3, label=tag, edgecolors="white", linewidths=0.4)
    ax.set_xticks(folds, [f"fold {f + 1}" for f in folds])
    ax.set_ylabel("held-out 100 − CF (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, None)
    ax.set_title("Error on the 16 tracks each fold holds out (weights searched on the other 64)")
    ax.legend(loc="upper left", fontsize=6.3, bbox_to_anchor=(1.01, 1.0), title="weighted CF pooled over 80 tracks",
              title_fontsize=6.3)
    fig.tight_layout()
    save(fig, "models_cv")

if __name__ == "__main__":
    fig_overall()
    fig_tags()
    fig_pertrack()
    fig_thinning()
    fig_measured()
    fig_cv()
