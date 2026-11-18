"""
Per-BREAKPOINT detail (12_breakpoints_detail.png) - corrected.

Earlier version claimed every jump was "roads the label omitted" and drew the shortest
road path in green. That is only true for some of them. There are TWO kinds:

  TYPE A - LABEL SKIP.  An arc exists between the two ends (ratio detour/gap ~ 1.0).
          The road is in `.arcs`; only `.route` forgot to list it. Repair = insert it.

  TYPE B - NETWORK GAP.  No arc and no road along the line (detour/gap >> 1).
          The car crossed a road that is absent from `.arcs` altogether. Drawing a
          road path here is misleading: it is a detour the driver never took.

So the figure now draws what the car ACTUALLY did (its fixes across the jump, orange)
and only shows the network detour (green dashed) when it exists, labelled as not driven.

Run: python notes/fig_breaks_detail.py
Out: notes/figs/12_breakpoints_detail.png, notes/breakpoints_detail.csv
"""
import os, sys, glob, csv, heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, hav, seg_dist
from candidates import label_observations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
LOC = {r["instance"]: r for r in csv.DictReader(open(os.path.join(ROOT, "notes", "locations.csv"), encoding="utf-8"))}
TRIPS = sorted([d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d)] +
               [d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d)])


def dijkstra(adj, src, dst):
    dist = {src: 0.0}; prev = {}
    pq = [(0.0, src)]; seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, 1e18):
                dist[v] = nd; prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return None, None
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1], dist[dst]


def analyse(d):
    nodes, arcs, route, track = load(d)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    k = np.cos(np.radians(lat0))
    xy = np.column_stack([(nodes[:, 0] - lon0) * 111320 * k, (nodes[:, 1] - lat0) * 110540])
    trxy = np.column_stack([(track[:, 0] - lon0) * 111320 * k, (track[:, 1] - lat0) * 110540])

    seg = hav(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1], nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    route_len = float(seg[route].sum())
    adj = [[] for _ in range(len(nodes))]
    for (u, v), w in zip(arcs, seg):
        adj[u].append((v, w)); adj[v].append((u, w))

    # which route index does each fix belong to (monotone projection)
    lab, _ = label_observations(nodes, arcs, route, track)
    ridx = np.full(len(track), -1, int)
    for j, a in enumerate(lab):
        if a >= 0:
            ridx[j] = int(np.where(route == a)[0][0]) if a in route else -1

    out = []
    for i in range(len(route) - 1):
        v, u2 = int(arcs[route[i]][1]), int(arcs[route[i + 1]][0])
        if v == u2:
            continue
        gap = float(hav(nodes[v][0], nodes[v][1], nodes[u2][0], nodes[u2][1]))
        path, plen = dijkstra(adj, v, u2)
        direct = any((a == v and b == u2) or (a == u2 and b == v) for a, b in arcs)

        # what the car did: the fixes labelled to the two arcs either side of the jump
        m = np.where((ridx == i) | (ridx == i + 1))[0]
        if len(m) >= 2:
            driven = float(np.hypot(np.diff(trxy[m, 0]), np.diff(trxy[m, 1])).sum())
        else:
            driven = float("nan")
        # does ANY mapped road pass within 10 m of the jump midpoint?
        mid = (xy[v] + xy[u2]) / 2
        road_near = int((seg_dist(mid[None, :], xy[arcs[:, 0]], xy[arcs[:, 1]])[0] < 10.0).sum())
        ratio = (plen / gap) if (plen and gap > 0) else np.nan
        if direct or (ratio is not np.nan and ratio <= 1.15):
            kind = "label skip"
        elif ratio >= 1.5:
            kind = "network gap"
        else:
            kind = "unclear"
        out.append(dict(nodes=nodes, arcs=arcs, route=route, xy=xy, trxy=trxy, idx=i, v=v, u2=u2,
                        gap=gap, path=path, plen=plen, hops=(len(path) - 1) if path else 0,
                        direct=direct, driven=driven, ratio=ratio, kind=kind, road_near=road_near,
                        nfix=len(m), fixes_idx=m))
    return out, route_len, os.path.basename(d), ("student_test" in d)


breaks = []
for d in TRIPS:
    res, rlen, name, is_test = analyse(d)
    for b in res:
        b.update(inst=name, test=is_test, route_len=rlen)
        b["share"] = b["plen"] / (rlen + b["plen"]) if rlen else 0.0
        breaks.append(b)

breaks.sort(key=lambda b: ({"label skip": 0, "unclear": 1, "network gap": 2}[b["kind"]], -b["gap"]))

rows = [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in b.items()
         if k not in ("nodes", "arcs", "route", "xy", "trxy", "path", "fixes_idx")} for b in breaks]
with open(os.path.join(ROOT, "notes", "breakpoints_detail.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

nk = {k: sum(1 for b in breaks if b["kind"] == k) for k in ("label skip", "unclear", "network gap")}
print(f"{len(breaks)} breakpoints in {len({b['inst'] for b in breaks})} trips: {nk}")
for b in breaks:
    print(f"  {b['kind']:12s} {b['inst']} idx{b['idx']:4d} gap {b['gap']:6.0f} m  "
          f"graph path {b['plen']:7.0f} m ({b['hops']:2d} roads)  ratio {b['ratio']:5.1f}  "
          f"car covered {b['driven']:6.0f} m in {b['nfix']:3d} fixes")

ncol, n = 4, len(breaks)
nrow = int(np.ceil((n + 2) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 4.0 * nrow))
axs = np.atleast_2d(axs).ravel()
COLOUR = {"label skip": "#2f855a", "unclear": "#b7791f", "network gap": "#c53030"}

for ax, b in zip(axs, breaks):
    xy, arcs, route, trxy = b["xy"], b["arcs"], b["route"], b["trxy"]
    p, q = xy[b["v"]], xy[b["u2"]]
    span = max(b["gap"], min(b["plen"], 2500) if b["plen"] else b["gap"], 150.0)
    if b["kind"] == "network gap":
        span = max(b["gap"] * 2.2, 150.0)
    cx, cy = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    half = span * 1.2 + 100
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)

    inside = (xy[arcs[:, 0], 0] > cx - half) & (xy[arcs[:, 0], 0] < cx + half) & \
             (xy[arcs[:, 0], 1] > cy - half) & (xy[arcs[:, 0], 1] < cy + half)
    for a in arcs[inside][:: max(1, int(max(inside.sum(), 1) // 2000))]:
        u, v = int(a[0]), int(a[1])
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], "-", c="0.86", lw=.8, zorder=1)

    m = (trxy[:, 0] > cx - half) & (trxy[:, 0] < cx + half) & \
        (trxy[:, 1] > cy - half) & (trxy[:, 1] < cy + half)
    ax.plot(trxy[m, 0], trxy[m, 1], ".", ms=1.7, c="0.55", zorder=2)

    for a in route:
        u, v = int(arcs[a][0]), int(arcs[a][1])
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], "-", lw=1.9, c="#2b6cb0", zorder=3)

    if b["kind"] == "label skip" and b["path"]:
        P = xy[b["path"]]
        ax.plot(P[:, 0], P[:, 1], "-", lw=3.6, c="#2f855a", alpha=.95,
                solid_capstyle="round", zorder=4)
    elif b["path"] and b["plen"] < 2500:
        P = xy[b["path"]]
        ax.plot(P[:, 0], P[:, 1], ls=(0, (4, 2)), lw=1.8, c="#2f855a", alpha=.7, zorder=4)

    if b["nfix"] >= 2:
        F = trxy[b["fixes_idx"]]
        w = (F[:, 0] > cx - half) & (F[:, 0] < cx + half) & (F[:, 1] > cy - half) & (F[:, 1] < cy + half)
        if w.sum() >= 2:
            ax.plot(F[w, 0], F[w, 1], "-", lw=2.2, c="#dd6b20", zorder=5)

    ax.plot([p[0], q[0]], [p[1], q[1]], ls=(0, (1, 1.8)), lw=1.1, c="#c53030", zorder=6)
    ax.plot(*p, "o", ms=7, mfc="white", mec="#c53030", mew=1.7, zorder=7)
    ax.plot(*q, "o", ms=7, mfc="white", mec="#c53030", mew=1.7, zorder=7)

    city = LOC.get(b["inst"], {}).get("city", "") or LOC.get(b["inst"], {}).get("region", "")
    tag = "TEST " if b["test"] else ""
    if b["kind"] == "label skip":
        verdict = "TYPE A: an arc joins them, the label skipped it"
    elif b["kind"] == "network gap" and b["road_near"] == 0:
        verdict = f"TYPE B: no arc joins them, and no mapped road crosses at all"
    elif b["kind"] == "network gap":
        verdict = f"TYPE B: no arc joins them ({b['road_near']} mapped roads pass nearby)"
    else:
        verdict = "borderline: detour is longer than the gap"
    ax.set_title(f"{tag}{b['inst']} - {city[:15]}\n"
                 f"jump {b['gap']:.0f} m; graph detour {b['plen']:.0f} m ({b['hops']} roads)\n"
                 f"{verdict} ({b['nfix']} fixes cross it)", fontsize=7.4, loc="left", linespacing=1.35,
                 color=COLOUR[b["kind"]])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(COLOUR[b["kind"]])

ax = axs[n]
aa = [b for b in breaks if b["kind"] == "label skip"]
bb = [b for b in breaks if b["kind"] == "network gap"]
ax.hist([b["ratio"] for b in breaks], bins=np.logspace(-0.1, 2, 18), color="#2b6cb0")
ax.axvline(1.15, ls="--", c="#2f855a", lw=1.2)
ax.axvline(1.5, ls="--", c="#c53030", lw=1.2)
ax.set_xscale("log")
ax.set_title(f"Detour length / straight gap\n{nk['label skip']} at ~1.0 (label skips a road), "
             f"{nk['network gap']} above 1.5 (road missing)", fontsize=8.6, loc="left")
ax.set_xlabel("graph detour / straight-line gap", fontsize=8)
ax.set_ylabel("number of breakpoints", fontsize=8)
ax.grid(alpha=.25, lw=.5)

ax = axs[n + 1]
ax.axis("off")
txt = ("WHAT IS ACTUALLY MISSING\n\n"
       f"TYPE A - {nk['label skip']} of {len(breaks)} breaks\n"
       "an arc joins the two ends; `.route`\n"
       "just skipped it. Repair = insert it.\n\n"
       f"TYPE B - {nk['network gap']} of {len(breaks)} breaks\n"
       "no arc joins the ends, and the\n"
       "detour is far longer than the jump.\n"
       "The car crossed on a road that is\n"
       "not in `.arcs` at all. Inserting a\n"
       "path here would be fiction.\n\n"
       f"borderline - {nk['unclear']}\n\n"
       "The GPS crosses the jump in every\n"
       "case (2 to 4,000 fixes): the car did\n"
       "drive there, so what is missing is\n"
       "the map or the label, not the drive.")
ax.text(0, 1, txt, va="top", ha="left", fontsize=8.2, linespacing=1.5)
ax.set_title("Two different faults, two different repairs", fontsize=9, loc="left")

axs[0].legend(handles=[
    Line2D([], [], color="#2b6cb0", lw=1.9, label="answer route (labelled)"),
    Line2D([], [], color="#dd6b20", lw=2.2, label="what the car drove across the jump"),
    Line2D([], [], color="#2f855a", lw=3.6, label="TYPE A: road exists, label skipped it"),
    Line2D([], [], color="#2f855a", ls=(0, (4, 2)), lw=1.8, label="TYPE B: detour that was NOT driven"),
    Line2D([], [], color="#c53030", ls=(0, (1, 1.8)), label="the jump in the file")],
    fontsize=6.4, loc="lower left", framealpha=.94)

for a in axs[n + 2:]:
    a.axis("off")

fig.suptitle(f"The 14 jumps in the answer labels are two different faults: "
             f"{nk['label skip']} skipped a road that exists, {nk['network gap']} jump between nodes the graph never joins",
             fontsize=11.5, x=.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(os.path.join(FIGS, "12_breakpoints_detail.png"), dpi=115, bbox_inches="tight")
print("wrote notes/figs/12_breakpoints_detail.png")
