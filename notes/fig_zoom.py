"""
Tight zooms: raw GPS fixes against the answer route, at three magnifications.

Data source: ONLY the trip's own files (.nodes, .arcs, .route, .track).
No OSM, no basemap, no tiles - the grey geometry is `arcs` drawn from `nodes`.
The city name in the title comes from notes/locations.csv and is only a label.

Run: python notes/fig_zoom.py
Out: notes/figs/11_zoom_detail.png
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, to_xy, seg_dist, route_polyline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
INST = os.path.join(ROOT, "train", "00000003")
LOC = {r["instance"]: r for r in csv.DictReader(open(os.path.join(ROOT, "notes", "locations.csv"), encoding="utf-8"))}


def pick_focus(nodes, arcs, route, track, err, win=60.0, vmin=25.0):
    """centre on a TYPICAL fix on a SINGLE-PASS straight stretch.

    Three things disqualify a window, each for a real reason:
      - the car is stopped  -> fixes collapse into a blob
      - the car passes the same junction repeatedly -> a knot of crossing lines
      - error far from the median -> unrepresentative of the dataset
    Straightness = |last-first| / path-length. ~1 means one clean pass.
    """
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    tx, ty = to_xy(track, lon0, lat0)
    v = np.zeros(len(tx))
    step = np.diff(track[:, 2])
    dd = np.hypot(np.diff(tx), np.diff(ty))
    v[:-1] = np.where(step > 0, dd / np.maximum(step, 1e-9) * 3.6, 0)

    best, best_score = 0, -1e9
    for i in range(2, len(tx) - 2):
        if not (vmin <= v[i] <= 140):
            continue
        m = np.where((np.abs(tx - tx[i]) < win) & (np.abs(ty - ty[i]) < win))[0]
        if not (6 <= len(m) <= 28):
            continue
        seg = np.hypot(np.diff(tx[m]), np.diff(ty[m])).sum()
        chord = np.hypot(tx[m[-1]] - tx[m[0]], ty[m[-1]] - ty[m[0]])
        straight = chord / max(seg, 1e-9)
        if straight < 0.85:
            continue
        score = straight - abs(err[i] - np.median(err)) / 10.0
        if score > best_score:
            best, best_score = i, score
    return best


def draw(ax, nodes, arcs, route, track, xy_node, xy_track, cx, cy, half, show_net=True,
         legend=False, title=""):
    if show_net:
        inside = (np.abs(xy_node[arcs[:, 0], 0] - cx) < half * 1.4) & \
                 (np.abs(xy_node[arcs[:, 0], 1] - cy) < half * 1.4)
        for a in arcs[inside]:
            u, v = int(a[0]), int(a[1])
            ax.plot(xy_node[[u, v], 0], xy_node[[u, v], 1], "-", c="0.80", lw=1.1, zorder=1,
                    solid_capstyle="round")

    _, poly = route_polyline(nodes, arcs, route)
    px, py = to_xy(poly, nodes[:, 0].mean(), nodes[:, 1].mean())
    ax.plot(px, py, "-", c="#c53030", lw=3.0, zorder=2, solid_capstyle="round")

    ax.plot(xy_track[:, 0], xy_track[:, 1], "-", c="0.15", lw=.9, alpha=.75, zorder=3)
    ax.plot(xy_track[:, 0], xy_track[:, 1], ".", c="0.10", ms=4.0, zorder=4)

    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.55")
    ax.set_title(title, fontsize=9.5, loc="left")
    # scale bar
    bar = 10 ** np.floor(np.log10(half * 0.9))
    x0, y0 = cx - half * 0.82, cy - half * 0.82
    ax.plot([x0, x0 + bar], [y0, y0], "-", c="k", lw=2.4, solid_capstyle="butt", zorder=6)
    ax.annotate(f"{bar:.0f} m", (x0 + bar / 2, y0), textcoords="offset points", xytext=(0, 4),
                ha="center", fontsize=7.5)
    if legend:
        ax.legend(handles=[
            Line2D([], [], color="#c53030", lw=3, label="answer route (.route)"),
            Line2D([], [], color="0.10", marker=".", ls="-", lw=.9, ms=4, label="raw GPS (.track)"),
            Line2D([], [], color="0.80", lw=1.1, label="roads (.arcs)")],
            fontsize=7.5, loc="upper left", framealpha=.93)


nodes, arcs, route, track = load(INST)
lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
xy_node = np.column_stack(to_xy(nodes, lon0, lat0))
xy_track = np.column_stack(to_xy(track, lon0, lat0))

# error of each fix against its own route, for the annotation
_, poly = route_polyline(nodes, arcs, route)
R = np.column_stack(to_xy(poly, lon0, lat0))
d = seg_dist(xy_track, R[:-1], R[1:]).min(axis=1)
print(f"trip 00000003: {len(track)} fixes, error median {np.median(d):.1f} m, "
      f"p95 {np.percentile(d,95):.1f} m, max {d.max():.1f} m")

focus = pick_focus(nodes, arcs, route, track, d)
cx, cy = xy_track[focus]
print(f"focus = fix {focus}, error there {d[focus]:.1f} m")

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
draw(ax[0], nodes, arcs, route, track, xy_node, xy_track, cx, cy, 250, legend=True,
     title=f"(a) 500 m across: the drive, the roads it could pick, the fixes\n"
           f"{len(track)} fixes over {track[-1,2]/60:.0f} min, "
           f"error median {np.median(d):.0f} m, p95 {np.percentile(d,95):.0f} m")
draw(ax[1], nodes, arcs, route, track, xy_node, xy_track, cx, cy, 45, show_net=False,
     legend=True,
     title="(b) 90 m across: one fix per second, 9 m apart\n"
           "this is the signal a matcher sees")
draw(ax[2], nodes, arcs, route, track, xy_node, xy_track, cx, cy, 15, show_net=False,
     title=f"(c) 30 m across: each dot is one fix\n"
           f"the fix nearest the centre is {d[focus]:.1f} m off the road")

fig.suptitle("Raw GPS against the supplied answer route - every line is drawn from the trip's own files "
             "(nodes, arcs, route, track); no external map data", fontsize=11, x=.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGS, "11_zoom_detail.png"), dpi=130, bbox_inches="tight")
print("wrote notes/figs/11_zoom_detail.png")
