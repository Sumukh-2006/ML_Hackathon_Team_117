"""
Is the road graph OpenStreetMap-derived?

Rationale: the hackathon solution only needs the graph, but "where did it come from"
decides which failures are possible. If the graph is a raw OSM export, geometry
vertices are nodes and one-way streets are real; if it is a contracted/synthetic
graph, half the intuition from map-matching papers does not transfer. Everything
below is measured from the 4 provided files only - no internet, no external data.

Run from repo root:  python notes/osm_check.py
Writes:              notes/figs/07_osm_fingerprint.png  (+ stdout table)

Measured (not inferred):
  oneway_share      = 1 - share of arcs that have exactly a (v,u) twin arc
  deg1/2/3+_frac    = fraction of nodes by UNDIRECTED neighbour count
  arc_len_*         = haversine node-to-node length of each arc, metres
  n_components      = connected components of the undirected graph
  deg1_border_frac  = share of degree-1 nodes within 2% of the bbox edge
  straight_frac     = share of degree-2 nodes whose two arcs are within 5 deg of
                      collinear (0 deg = perfectly straight through)
  node_sp*          = modal spacing of the machine-written coordinate grid
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import ROOT, FIGS, load, hav, to_xy

try:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ------------------------------------------------------------------ helpers
def undirected_neighbours(n_nodes, u, v):
    """neighbour sets, ignoring direction and duplicate arcs"""
    nbr = [set() for _ in range(n_nodes)]
    for a, b in zip(u.tolist(), v.tolist()):
        if a != b:
            nbr[a].add(b)
            nbr[b].add(a)
    return nbr


def components(n_nodes, u, v):
    if HAVE_SCIPY:
        A = coo_matrix((np.ones(len(u)), (u, v)), shape=(n_nodes, n_nodes))
        k, lab = connected_components(A, directed=False)
        return k, np.bincount(lab) / n_nodes
    # union-find fallback
    p = list(range(n_nodes))

    def f(x):
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    for a, b in zip(u.tolist(), v.tolist()):
        ra, rb = f(a), f(b)
        if ra != rb:
            p[ra] = rb
    sz = Counter(f(i) for i in range(n_nodes))
    tot = sum(sz.values())
    return len(sz), np.array(sorted(sz.values(), reverse=True)) / tot


def decimal_places(path, nlines=20000):
    """how many digits after the '.' are actually written in the file"""
    c = Counter()
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= nlines:
                break
            for tok in line.split():
                c[len(tok.split(".")[1]) if "." in tok else 0] += 1
    return c


def chain_stats(n_nodes, u, v, seg):
    """Contract every degree-2 node: how long is one real junction-to-junction
    road stretch, and how many arcs does it contain? This is the direct test of
    'arcs are junction-to-junction edges' vs 'arcs are OSM way segments'."""
    nbr = [dict() for _ in range(n_nodes)]
    for k, (a, b) in enumerate(zip(u.tolist(), v.tolist())):
        if a != b:
            nbr[a][b] = k
            nbr[b][a] = k
    deg = np.array([len(s) for s in nbr])
    used = np.zeros(len(u), bool)
    L, NA = [], []
    for s in np.where(deg != 2)[0]:
        for nb in list(nbr[s]):
            ai = nbr[s][nb]
            if used[ai]:
                continue
            cur, nxt, tot, na = s, nb, 0.0, 0
            while True:
                ai = nbr[cur][nxt]
                if used[ai]:
                    break
                used[ai] = True
                tot += seg[ai]
                na += 1
                prev, cur = cur, nxt
                if deg[cur] != 2:
                    break
                nxts = [k for k in nbr[cur] if k != prev]
                if not nxts:
                    break
                nxt = nxts[0]
            L.append(tot)
            NA.append(na)
    return np.array(L), np.array(NA)


def measure(inst):
    nodes, arcs, route, track = load(inst)
    name = os.path.basename(inst)
    n = len(nodes)
    u, v = arcs[:, 0].astype(np.int64), arcs[:, 1].astype(np.int64)

    seg = hav(nodes[u, 0], nodes[u, 1], nodes[v, 0], nodes[v, 1])

    # ---- one-way proxy: does the reverse arc exist?
    key = u * n + v
    has_rev = np.isin(v * n + u, key)
    # ---- duplicate arcs (same (u,v) more than once)
    uk, cnt = np.unique(key, return_counts=True)
    dup = int(cnt.sum() - len(uk))

    # ---- degree
    nbr = undirected_neighbours(n, u, v)
    ndeg = np.array([len(s) for s in nbr])
    ddir = np.bincount(np.concatenate([u, v]), minlength=n)

    # ---- dangling ends vs crop border
    deg1 = np.where(ndeg == 1)[0]
    lo, hi = nodes.min(0), nodes.max(0)
    span = np.maximum(hi - lo, 1e-12)
    rel = (nodes[deg1] - lo) / span
    if len(deg1):
        at_border = np.mean((rel.min(1) < 0.02) | (rel.max(1) > 0.98))
    else:
        at_border = np.nan

    # ---- geometry: how straight are mid-road nodes?
    d2 = np.where(ndeg == 2)[0]
    rng = np.random.default_rng(0)
    d2s = rng.choice(d2, min(3000, len(d2)), replace=False) if len(d2) else d2
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    xy = np.column_stack(to_xy(nodes, lon0, lat0))
    dev = []
    for i in d2s:
        a, b = list(nbr[i])[:2]
        v1, v2 = xy[a] - xy[i], xy[b] - xy[i]
        c = v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        dev.append(180.0 - np.degrees(np.arccos(np.clip(c, -1, 1))))
    dev = np.array(dev) if len(dev) else np.array([np.nan])

    # ---- junction-to-junction spacing (contract every deg-2 node)
    chL, chNA = chain_stats(n, u, v, seg)

    ncomp, comp_share = components(n, u, v)

    # ---- angle between the arms of a 3-way junction
    deg3 = np.where(ndeg == 3)[0]
    jang = []
    for i in rng.choice(deg3, min(3000, len(deg3)), replace=False):
        a, b, c3 = list(nbr[i])[:3]
        vec = [xy[j] - xy[i] for j in (a, b, c3)]
        ang = []
        for p in range(3):
            for q in range(p + 1, 3):
                cc = vec[p] @ vec[q] / (np.linalg.norm(vec[p]) * np.linalg.norm(vec[q]) + 1e-12)
                ang.append(np.degrees(np.arccos(np.clip(cc, -1, 1))))
        jang.append(min(ang))          # the two arms of the "through" road = smallest
    jang = np.array(jang) if jang else np.array([np.nan])

    dec = decimal_places(os.path.join(inst, f"{name}.nodes"), 5000)
    dec_main = max(dec, key=dec.get)

    # ---- track speed
    if len(track) > 1:
        dist = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1]).sum()
        spd = dist / max(track[-1, 2] - track[0, 2], 1e-9) * 3.6
    else:
        spd = np.nan

    return dict(
        instance=name, split="student_test" if "student_test" in inst else "train",
        n_nodes=n, n_arcs=len(arcs),
        oneway_share=float(1.0 - has_rev.mean()),
        deg1_frac=float((ndeg == 1).mean()), deg2_frac=float((ndeg == 2).mean()),
        deg3p_frac=float((ndeg >= 3).mean()),
        dir_deg2_frac=float((ddir == 2).mean()),
        arc_med=float(np.median(seg)), arc_p95=float(np.percentile(seg, 95)),
        arc_max=float(seg.max()), arc_mean=float(seg.mean()),
        self_loops=int((u == v).sum()), dup_arcs=dup,
        zero_len=int((seg < 0.5).sum()),
        n_comp=ncomp, largest_comp=float(comp_share.max()),
        n_deg1=len(deg1), deg1_border_frac=float(at_border),
        straight_frac=float((dev < 5).mean()) if np.isfinite(dev).any() else np.nan,
        dev_med=float(np.nanmedian(dev)),
        chain_med=float(np.median(chL)) if len(chL) else np.nan,
        chain_p95=float(np.percentile(chL, 95)) if len(chL) else np.nan,
        arcs_per_chain=float(np.mean(chNA)) if len(chNA) else np.nan,
        junc_min_angle_med=float(np.nanmedian(jang)),
        junc_90_frac=float((np.abs(jang - 90) < 20).mean()) if np.isfinite(jang).any() else np.nan,
        decimals=dec_main, speed_kmh=float(spd),
        slot_len=float(np.median(seg)),
    )


# ------------------------------------------------------------------ figure
CB_BLUE, CB_ORANGE, CB_GREEN, CB_PURPLE = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
plt.rcParams.update({"font.size": 11})


def figure(rows, pooled_seg, focus, labels):
    fig, ax = plt.subplots(1, 4, figsize=(14, 4), dpi=120)
    (lab_hw, s_hw), (lab_city, s_city) = focus

    # ---- panel 1: one-way share
    ow = np.array([r["oneway_share"] for r in rows])
    ax[0].hist(ow * 100, bins=np.arange(0, 70, 5), color=CB_BLUE, edgecolor="white")
    ax[0].axvline(np.median(ow) * 100, c="k", ls="--", lw=1.4)
    ax[0].annotate(f"median {np.median(ow)*100:.0f}%", (np.median(ow) * 100, ax[0].get_ylim()[1] * 0.92),
                   xytext=(6, 0), textcoords="offset points", fontsize=10)
    mx = int(np.argmax(ow))
    ax[0].annotate(f"{labels.get(rows[mx]['instance'], rows[mx]['instance'])}:\n{ow[mx]*100:.0f}% one-way",
                   xy=(ow[mx] * 100, 2.5), xycoords="data",
                   xytext=(0.52, 0.72), textcoords="axes fraction", fontsize=9,
                   arrowprops=dict(arrowstyle="->", lw=1, color="0.3"))
    ax[0].set_title(f"One-way share runs 0-{ow.max()*100:.0f}% and is city-specific:\nreal street topology, not a uniform grid",
                    fontsize=10.5)
    ax[0].set_xlabel("one-way arcs (% with no reverse twin)")
    ax[0].set_ylabel("number of drives")

    # ---- panel 2: undirected degree mix (pooled)
    d1 = np.mean([r["deg1_frac"] for r in rows]) * 100
    d2f = np.mean([r["deg2_frac"] for r in rows]) * 100
    d3p = np.mean([r["deg3p_frac"] for r in rows]) * 100
    bars = ax[1].bar(["1", "2", "3+"], [d1, d2f, d3p],
                     color=[CB_ORANGE, CB_GREEN, CB_PURPLE], edgecolor="white", width=0.62)
    for b, val, nm in zip(bars, [d1, d2f, d3p], ["dead ends", "mid-road", "junctions"]):
        ax[1].text(b.get_x() + b.get_width() / 2, val + 2, f"{val:.0f}%\n{nm}", ha="center",
                   fontsize=9.5, va="bottom")
    ax[1].set_ylim(0, 105)
    ax[1].set_xlabel("node degree (undirected neighbours)")
    ax[1].set_ylabel("nodes (%, mean of 90 drives)")
    ax[1].set_title(f"{d2f:.0f}% of nodes are mid-road, so arcs are\nper way segment, not junction-to-junction", fontsize=10.5)
    ax[1].text(0.02, 0.98,
               f"my first pass counted directed\nedges and read {np.mean([r['dir_deg2_frac'] for r in rows])*100:.0f}% here: a two-way\nstreet then has degree 4",
               transform=ax[1].transAxes, ha="left", va="top", fontsize=8.5, color="0.2",
               bbox=dict(fc="0.96", ec="0.6", boxstyle="round,pad=0.3"))

    # ---- panel 3: arc length distribution (pooled)
    ax[2].hist(np.clip(pooled_seg, 0.5, 2000), bins=np.logspace(np.log10(0.5), np.log10(2000), 60),
               color=CB_BLUE, edgecolor="none")
    ax[2].set_xscale("log")
    med, p95 = np.median(pooled_seg), np.percentile(pooled_seg, 95)
    ch = np.median([r["chain_med"] for r in rows])
    ax[2].axvline(med, c="k", ls="--", lw=1.4)
    ax[2].axvline(ch, c=CB_ORANGE, ls="-", lw=1.6)
    ax[2].annotate(f"arc median\n{med:.0f} m", (med, ax[2].get_ylim()[1] * 0.82), xytext=(-16, 0),
                   textcoords="offset points", fontsize=9.5, ha="right")
    ax[2].annotate(f"junction-to-junction\nstretch {ch:.0f} m", (ch, ax[2].get_ylim()[1] * 0.42),
                   xytext=(8, 0), textcoords="offset points", fontsize=9.5, color="#8a5f00")
    ax[2].set_xlabel("arc length (m, log scale)")
    ax[2].set_ylabel("arcs in sample (90 drives)")
    ax[2].set_title(f"Arcs are way segments: median {med:.0f} m,\np95 {p95:.0f} m, continuous heavy tail", fontsize=10.5)

    # ---- panel 4: fast corridor vs dense city
    for (lab, s), c in (((lab_hw, s_hw), CB_ORANGE), ((lab_city, s_city), CB_BLUE)):
        xs = np.sort(np.clip(s, 0.5, 5000))
        ax[3].plot(xs, np.linspace(0, 100, len(xs)), color=c, lw=2.2, label=lab)
    ax[3].set_xscale("log")
    ax[3].set_xlim(0.6, 30000)
    ax[3].set_xlabel("arc length (m, log scale)")
    ax[3].set_ylabel("arcs shorter than x (%)")
    ax[3].set_ylim(0, 108)
    ax[3].axhline(50, c="0.6", lw=0.8, ls=":")
    ax[3].text(0.98, 0.52, "both reach 50%\nat ~20 m", transform=ax[3].transAxes, ha="right",
               va="bottom", fontsize=9, color="0.25")
    ax[3].legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1.0, 0.02), framealpha=0.95)
    ax[3].set_title("133 km/h drive vs Amsterdam: same 20 m arcs,\nso arcs are not road-class filtered", fontsize=10.5)

    fig.tight_layout()
    out = os.path.join(FIGS, "07_osm_fingerprint.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    train = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
    test = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))
    allinst = train + test

    rows = [measure(d) for d in allinst]
    keys = list(rows[0].keys())
    print(f"{'instance':>9} {'split':>5} {'nodes':>7} {'arcs':>7} {'1way':>5} {'d2':>5} "
          f"{'arcMed':>6} {'arcP95':>6} {'comp':>5} {'d1':>5}")
    for r in rows:
        print(f"{r['instance']:>9} {r['split'][:4]:>5} {r['n_nodes']:>7} {r['n_arcs']:>7} "
              f"{r['oneway_share']:.2f} {r['deg2_frac']:>5.2f} {r['arc_med']:>6.1f} "
              f"{r['arc_p95']:>6.1f} {r['n_comp']:>5} {r['deg1_frac']:>5.3f}")

    def stat(k):
        a = np.array([r[k] for r in rows], float)
        return dict(med=np.median(a), mn=a.min(), mx=a.max(), mean=a.mean())

    print("\n================ POOLED SUMMARY (90 drives) ================")
    for k in ["oneway_share", "deg1_frac", "deg2_frac", "deg3p_frac", "dir_deg2_frac",
              "arc_med", "arc_p95", "arc_max", "chain_med", "arcs_per_chain",
              "n_comp", "largest_comp",
              "deg1_border_frac", "straight_frac", "dev_med",
              "junc_min_angle_med", "junc_90_frac", "speed_kmh", "decimals"]:
        s = stat(k)
        print(f"{k:<18} median {s['med']:>8.3f}   mean {s['mean']:>8.3f}   "
              f"min {s['mn']:>8.3f}   max {s['mx']:>8.3f}")
    print("self_loops total", sum(r["self_loops"] for r in rows),
          "| dup_arcs total", sum(r["dup_arcs"] for r in rows),
          "| zero_len arcs total", sum(r["zero_len"] for r in rows))

    # tangent: does one-way share track city density?
    print("\nmost one-way drives:")
    for r in sorted(rows, key=lambda r: -r["oneway_share"])[:5]:
        print(f"  {r['instance']} {r['oneway_share']:.2f}  n_nodes={r['n_nodes']}")
    print("least one-way drives:")
    for r in sorted(rows, key=lambda r: r["oneway_share"])[:5]:
        print(f"  {r['instance']} {r['oneway_share']:.2f}  n_nodes={r['n_nodes']}")

    # pooled arc lengths (subsample for the plot)
    pooled = []
    rng = np.random.default_rng(1)
    for d in allinst:
        nodes, arcs, _, _ = load(d)
        s = hav(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1],
                nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
        pooled.append(rng.choice(s, min(4000, len(s)), replace=False))
    pooled = np.concatenate(pooled)
    print(f"\npooled arc sample n={len(pooled)}, median {np.median(pooled):.1f} m, "
          f"p95 {np.percentile(pooled,95):.1f} m")

    # panel-4 focus: fastest drive vs the largest city-like network that is not it
    fast = max(rows, key=lambda r: r["speed_kmh"])
    city = max([r for r in rows
                if r["oneway_share"] >= 0.15 and r["instance"] != fast["instance"]],
               key=lambda r: r["n_nodes"])
    print(f"panel 4: fast={fast['instance']} {fast['speed_kmh']:.0f} km/h vs "
          f"city={city['instance']} {city['n_nodes']} nodes, oneway {city['oneway_share']:.2f}")

    def segs_of(name):
        d = [p for p in allinst if os.path.basename(p) == name][0]
        nodes, arcs, _, _ = load(d)
        return hav(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1],
                   nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])

    s_hw = segs_of(fast["instance"])
    s_city = segs_of(city["instance"])
    print(f"  {fast['instance']} ({fast['speed_kmh']:.0f} km/h): median {np.median(s_hw):.1f} m  "
          f"| {city['instance']} ({city['n_nodes']} nodes): median {np.median(s_city):.1f} m")

    # human labels from the already-verified reverse geocoding
    labels = {}
    try:
        import csv as _csv
        for r in _csv.DictReader(open(os.path.join(ROOT, "notes", "locations.csv"))):
            labels[r["instance"]] = r["city"].split(",")[0]
    except Exception:
        labels = {}

    def lab(r, extra=""):
        c = labels.get(r["instance"], "").split()[0] if labels.get(r["instance"]) else ""
        return f"{r['instance'][-2:]} {c}{extra}" if c else f"{r['instance']}{extra}"

    out = figure(rows, pooled,
                 ((lab(fast, f" - {fast['speed_kmh']:.0f} km/h"), s_hw),
                  (lab(city, f" - city"), s_city)),
                 labels)
    print("\nwrote", out)
