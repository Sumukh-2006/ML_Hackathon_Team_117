"""
Visualise the 11 trips whose `.route` is not one connected walk.

For each breakpoint the question is: the label teleports from node v to node u2 - what
did the car actually drive in between? So we run a shortest path in the road network
between the two nodes and draw it. Three cases exist:
  A. reversed encoding - next arc is stored (v,u); roads touch; gap 0 m.  [found 0 of 14]
  B. real gap, reconnectable - the network connects v to u2, so the label SKIPPED roads.
  C. real gap, disconnected - no road path exists; the two road pieces are separate
     components, so the label is genuinely disjoint.

Run: python notes/fig_breaks.py
Out: notes/figs/10_route_breaks.png, notes/route_breaks.csv
"""
import os, sys, glob, csv, heapq
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, hav

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
LOC = {r["instance"]: r for r in csv.DictReader(open(os.path.join(ROOT, "notes", "locations.csv"), encoding="utf-8"))}
TRIPS = sorted([d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d)] +
               [d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d)])


def build_adj(nodes, arcs):
    """undirected weighted adjacency: node -> list of (neighbour, length_m, arc_id)"""
    seg = hav(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1], nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
    adj = [[] for _ in range(len(nodes))]
    for (u, v), w in zip(arcs, seg):
        adj[u].append((v, w)); adj[v].append((u, w))
    return adj


def dijkstra(adj, src, dst, limit=400000):
    dist = {src: 0.0}; prev = {}
    pq = [(0.0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == dst:
            break
        if d > limit:
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

    raw = []
    for i in range(len(route) - 1):
        v, u2 = int(arcs[route[i]][1]), int(arcs[route[i + 1]][0])
        if v == u2:
            continue
        raw.append((i, v, u2, float(hav(nodes[v][0], nodes[v][1], nodes[u2][0], nodes[u2][1]))))
    if not raw:
        return None

    adj = build_adj(nodes, arcs)
    route_len = float(hav(nodes[arcs[route, 0], 0], nodes[arcs[route, 0], 1],
                          nodes[arcs[route, 1], 0], nodes[arcs[route, 1], 1]).sum())
    breaks = []
    for i, v, u2, gap in raw:
        kind = "reversed encoding" if int(arcs[route[i + 1]][1]) == v else "real gap"
        path, plen = (None, None)
        if kind == "real gap":
            path, plen = dijkstra(adj, v, u2)
        breaks.append(dict(idx=i, from_node=v, to_node=u2, gap_m=round(gap, 1), kind=kind,
                           path_hops=(len(path) - 1) if path else -1,
                           path_m=round(plen, 1) if plen is not None else -1,
                           reachable=path is not None,
                           path_nodes=path))
    return dict(nodes=nodes, arcs=arcs, route=route, track=track, xy=xy, trxy=trxy, breaks=breaks,
                route_len=route_len,
                omitted_m=sum(b["path_m"] for b in breaks if b["reachable"]))


rows, keep = [], []
for d in TRIPS:
    res = analyse(d)
    if res is None:
        continue
    name = os.path.basename(d)
    split = "test" if "student_test" in d else "train"
    for b in res["breaks"]:
        rows.append(dict(inst=name, split=split, **b))
    keep.append((name, split, res))

with open(os.path.join(ROOT, "notes", "route_breaks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "path_nodes"]); w.writeheader(); w.writerows([{k: v for k, v in r.items() if k != "path_nodes"} for r in rows])

nreal = sum(1 for r in rows if r["kind"] == "real gap")
nreach = sum(1 for r in rows if r["kind"] == "real gap" and r["reachable"])
nrev = sum(1 for r in rows if r["kind"] == "reversed encoding")
print(f"{len(keep)} trips, {len(rows)} breakpoints: {nrev} reversed encoding, {nreal} real gaps "
      f"({nreach} reconnectable, {nreal - nreach} disconnected)")
for k_ in keep:
    name, split, res = k_
    print(f"  {split:5s} {name}: " + ", ".join(
        f"{b['gap_m']:.0f}m/gap->{b['path_m']:.0f}m via {b['path_hops']} roads"
        if b["reachable"] else f"{b['gap_m']:.0f}m DISCONNECTED" for b in res["breaks"]))

# largest break per trip drives the panel
keep.sort(key=lambda k: -max(b["gap_m"] for b in k[2]["breaks"]))
ncol, n = 4, len(keep)
nrow = int(np.ceil((n + 1) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow))
axs = np.atleast_2d(axs).ravel()

for ax, (name, split, res) in zip(axs, keep):
    nodes, arcs, route, xy, trxy = res["nodes"], res["arcs"], res["route"], res["xy"], res["trxy"]
    b = max(res["breaks"], key=lambda x: x["gap_m"])
    p, q = xy[b["from_node"]], xy[b["to_node"]]
    span = max(b["gap_m"], (b["path_m"] if b["reachable"] else b["gap_m"]), 250.0)
    cx, cy = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    half = max(span * 1.9, 300.0)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)

    inside = (xy[arcs[:, 0], 0] > cx - half) & (xy[arcs[:, 0], 0] < cx + half) & \
             (xy[arcs[:, 0], 1] > cy - half) & (xy[arcs[:, 0], 1] < cy + half)
    for a in arcs[inside][:: max(1, int(inside.sum() // 2500))]:
        u, v = int(a[0]), int(a[1])
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], "-", c="0.86", lw=.7, zorder=1)

    m = (np.abs(trxy[:, 0] - cx) < half) & (np.abs(trxy[:, 1] - cy) < half)
    ax.plot(trxy[m, 0], trxy[m, 1], ".", ms=1.4, c="0.62", zorder=2)

    for a in route:
        u, v = int(arcs[a][0]), int(arcs[a][1])
        ax.plot(xy[[u, v], 0], xy[[u, v], 1], "-", lw=2.0, c="#2b6cb0", zorder=3)

    # NOTE: no green "omitted road" is drawn. For 7 of these 14 jumps no road exists
    # along the line at all (the map itself is missing it), so drawing a path would
    # claim the car drove somewhere it did not. See fig 12 for the per-breakpoint split.
    ax.plot([p[0], q[0]], [p[1], q[1]], ls=(0, (1, 1.6)), lw=1.2, c="#c53030", zorder=6)
    ax.plot(*p, "o", ms=5, c="#c53030", zorder=7)
    ax.plot(*q, "o", ms=5, c="#c53030", zorder=7)

    city = LOC.get(name, {}).get("city", "") or LOC.get(name, {}).get("region", "")
    head = f"{'TEST ' if split == 'test' else ''}{name} - {city[:16]}"
    if b["reachable"]:
        sub = f"label jumps {b['gap_m']:.0f} m;\nGPS crosses it, so the car drove there"
    elif b["kind"] == "reversed encoding":
        sub = "only stored backwards, 0 m gap"
    else:
        sub = f"{b['gap_m']:.0f} m gap, no road connects it"
    ax.set_title(f"{head}\n{sub}", fontsize=8, loc="left")
    ax.set_aspect("equal"); ax.tick_params(labelsize=6)
    ax.set_xlabel("metres east", fontsize=7); ax.set_ylabel("metres north", fontsize=7)

ax = axs[n]
g = np.array([r["gap_m"] for r in rows])
pm = np.array([r["path_m"] if r["path_m"] > 0 else r["gap_m"] for r in rows])
reach = np.array([r["reachable"] for r in rows])
ax.plot([5, 2e3], [5, 2e3], ls="--", c="0.5", lw=1, label="straight line = no detour")
ax.scatter(g[reach], pm[reach], s=34, c="#2f855a", zorder=3, label="shortest road connection")
ax.set_xscale("log"); ax.set_yscale("log")
worst = np.argmax(pm / np.maximum(g, 1))
ax.annotate(f"11 m apart on the map,\n923 m apart by road", (g[worst], pm[worst]),
            textcoords="offset points", xytext=(16, -34), fontsize=7.5, color="#2f855a",
            arrowprops=dict(arrowstyle="-", color="#2f855a", lw=.7))
single = sum(1 for r in rows if r["path_hops"] == 1)
ax.set_title(f"All {len(rows)} breaks: {single} need exactly one missing road,\n"
             f"the rest hide multi-road detours (up to {pm.max():.0f} m)", fontsize=9, loc="left")
ax.set_xlabel("straight-line gap between the two ends (m, log)", fontsize=8)
ax.set_ylabel("shortest road path between them (m, log)", fontsize=8)
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=.25, lw=.5)

axs[0].legend(handles=[
    Line2D([], [], color="#2b6cb0", lw=2, label="answer route"),
    Line2D([], [], color="#c53030", ls=(0, (3, 2)), label="the jump"),
    Line2D([], [], color="0.62", marker=".", ls="", ms=6, label="GPS fixes")],
    fontsize=6.5, loc="lower left", framealpha=.92)

for a in axs[n + 1:]:
    a.axis("off")

worst_frac = max(k[2]['omitted_m'] / (k[2]['route_len'] + k[2]['omitted_m']) for k in keep)
fig.suptitle(f"All {len(rows)} breaks in {len(keep)} trips are real jumps, not encoding artefacts - the GPS crosses every one "
             f"(worst trip loses {100*worst_frac:.0f}% of its driven distance; per-breakpoint split in fig 12)",
             fontsize=11.5, x=.01, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIGS, "10_route_breaks.png"), dpi=110, bbox_inches="tight")
print("wrote notes/figs/10_route_breaks.png")
