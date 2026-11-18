"""
Report-grade EDA figures for the map-matching dataset.

Design rule: every panel title states the finding, not the variable. A reader who
looks at the picture for five seconds should learn one fact with one number in it.

Run from repo root:  python notes/eda_report.py
Writes notes/figs/00_*.png ... 04_*.png
"""
import os, sys, csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, hav, to_xy, seg_dist, route_polyline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
os.makedirs(FIGS, exist_ok=True)

TRAIN = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
TEST = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))
LOC = {r["instance"]: r for r in csv.DictReader(open(os.path.join(ROOT, "notes", "locations.csv"), encoding="utf-8"))}

C_TRAIN, C_TEST, C_BAD, C_ACC = "#2b6cb0", "#c05621", "#c53030", "#2f855a"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
                     "figure.dpi": 120, "savefig.dpi": 120})


# ------------------------------------------------------------------ helpers
def segments_of(arr, nodes):
    """metric-plane segment endpoints for a list of arc ids"""
    return nodes[arr[0]], nodes[arr[1]]


def resid_to_route(nodes, arcs, route, track, n_pts=250):
    """distance from sampled GPS points to the ground-truth route polyline.

    Route-gap chords are dropped: a discontinuous label would otherwise draw a
    straight line across the map and make off-route points look solved.
    """
    _, poly = route_polyline(nodes, arcs, route)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    P = np.column_stack(to_xy(track[:: max(1, len(track) // n_pts)], lon0, lat0))
    poly_xy = np.column_stack(to_xy(poly, lon0, lat0))
    A, B = poly_xy[:-1], poly_xy[1:]
    keep = np.hypot(*(B - A).T) < 2000.0          # drop gap chords
    if keep.sum() == 0:
        return np.zeros(1)
    return seg_dist(P, A[keep], B[keep]).min(axis=1)


def analyse(d, n_pts=250, n_cand_pts=120):
    nodes, arcs, route, track = load(d)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    seg = np.column_stack(to_xy(nodes, lon0, lat0))
    A, B = seg[arcs[:, 0]], seg[arcs[:, 1]]
    seglen = np.hypot(*(B - A).T)

    res = resid_to_route(nodes, arcs, route, track, n_pts)
    dts = np.diff(track[:, 2])
    dpos = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    good = dts > 0
    speed = dpos[good] / dts[good]

    # ambiguity: how many arcs lie within 25 m of a GPS point
    step = max(1, len(track) // n_cand_pts)
    P = np.column_stack(to_xy(track[::step], lon0, lat0))
    dd = seg_dist(P, A, B)
    ncand = (dd < 25.0).sum(axis=1)
    n25 = float(np.median(ncand))

    return dict(inst=os.path.basename(d), nodes=len(nodes), arcs=len(arcs), route=len(route),
                track=len(track), res_med=float(np.median(res)), res_p95=float(np.percentile(res, 95)),
                med_arc=float(np.median(seglen)), spd_med=float(np.median(speed)),
                spacing_med=float(np.median(dpos[good])),
                dur_h=float(track[-1, 2] / 3600), dt_max=float(dts.max()),
                cand25=n25, decoy=float(len(route) / len(arcs)))


print("analysing 90 drives ...")
STATS = []
for d in TRAIN + TEST:
    s = analyse(d)
    s["split"] = "test" if "student_test" in d else "train"
    STATS.append(s)
    print(f"  {s['inst']} res_med={s['res_med']:.0f} p95={s['res_p95']:.0f} cand25={s['cand25']:.0f}")
with open(os.path.join(ROOT, "notes", "eda_stats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(STATS[0].keys())); w.writeheader(); w.writerows(STATS)
S = {s["inst"]: s for s in STATS}


# ------------------------------------------------------------------ 00 world map
def fig_world():
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for s, col, mk, lab in (("train", C_TRAIN, "o", f"train ({len(TRAIN)} drives)"),
                            ("test", C_TEST, "^", f"test ({len(TEST)} drives)")):
        pts = [s_ for s_ in STATS if s_["split"] == s]
        ax.scatter([float(LOC[p["inst"]]["lon"]) for p in pts],
                   [float(LOC[p["inst"]]["lat"]) for p in pts],
                   s=42, c=col, marker=mk, edgecolor="white", lw=.6, label=lab, zorder=3)
    for name, ds in (("Ufa", 0.6), ("Frankfurt", -1), ("Almaty", 2), ("Townsville", 0)):
        pass
    ann = {"00000024": ("Frankfurt", 2, 3), "00000001": ("", 0, 0),
           "00000080": ("Buenos Aires", -9, -4), "T00000000": ("Ufa", 3, -4),
           "00000037": ("Toyama", -3, 4), "00000021": ("Laos", 2, -4),
           "00000098": ("Porto Alegre", -3, -4), "T00000056": ("Minneapolis", -4, 4)}
    for inst, (lab, dx, dy) in ann.items():
        if inst in LOC and lab:
            ax.annotate(lab, (float(LOC[inst]["lon"]), float(LOC[inst]["lat"])),
                        textcoords="offset points", xytext=(dx * 4, dy * 4), fontsize=8, color="0.25")
    ax.set_xlabel("longitude (degrees)"); ax.set_ylabel("latitude (degrees)")
    ax.set_title(f"Not one city: {len(STATS)} drives spread over 27 countries, so nothing city-specific transfers\n"
                 f"each drive comes with its own local road network (median 18.8k roads, 5-155k)", loc="left")
    ax.grid(alpha=.25, lw=.5); ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-130, 150); ax.set_ylim(-45, 72)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "00_world.png")); plt.close(fig)


# ------------------------------------------------------------------ 01 anatomy
def fig_anatomy(inst_dir, name, note):
    nodes, arcs, route, track = load(inst_dir)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    nx, ny = to_xy(nodes, lon0, lat0)
    chain, poly = route_polyline(nodes, arcs, route)
    px, py = to_xy(poly, lon0, lat0)
    tx, ty = to_xy(track, lon0, lat0)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    spacing = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])

    ax[0].scatter(nx, ny, s=.6, c="0.8", lw=0, label="road network (decoy)")
    ax[0].plot(px, py, "-", c=C_BAD, lw=1.6, label="answer route")
    ax[0].scatter(px[0], py[0], c=C_ACC, s=50, marker="o", zorder=5, label="start")
    ax[0].scatter(px[-1], py[-1], c=C_TRAIN, s=50, marker="s", zorder=5, label="end")
    ax[0].set_title(f"(a) The answer is {len(route)} of {len(arcs):,} roads\n"
                    f"{100*len(route)/len(arcs):.1f}% of the map, the rest is decoy", loc="left", fontsize=9.5)
    ax[0].set_xlabel("metres east"); ax[0].set_ylabel("metres north")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=7, loc="lower right", framealpha=.9)
    ax[0].tick_params(labelsize=7)

    # zoom around a point on the route, fixed 900 m window
    cx, cy = px[len(px) // 2], py[len(py) // 2]
    m = (np.abs(tx - cx) < 900) & (np.abs(ty - cy) < 900)
    zoom_res = resid_to_route(nodes, arcs, route, track[m] if m.any() else track, 400)
    ax[1].plot(px, py, "-", c=C_BAD, lw=2.2, label="answer route")
    ax[1].plot(tx[m], ty[m], ".-", c="0.15", ms=2.5, lw=.6, alpha=.9,
               label=f"raw GPS, 1 fix/s ({int(m.sum())} fixes)")
    ax[1].set_xlim(cx - 900, cx + 900); ax[1].set_ylim(cy - 900, cy + 900)
    ax[1].set_title(f"(b) Fixes wobble {np.median(zoom_res):.0f} m off the road,\n"
                    f"but only land {np.median(spacing):.0f} m apart - a turn fits in that gap",
                    loc="left", fontsize=9.5)
    ax[1].set_xlabel("metres east"); ax[1].tick_params(labelsize=7); ax[1].legend(fontsize=7.5)
    ax[1].set_aspect("equal")

    res = resid_to_route(nodes, arcs, route, track, 900)
    ax[2].hist(res, bins=np.arange(0, min(60, res.max() + 2), 1.5), color=C_TRAIN)
    med, p95 = np.median(res), np.percentile(res, 95)
    ax[2].axvline(med, c="k", ls="--", lw=1.2); ax[2].axvline(p95, c=C_BAD, ls=":", lw=1.4)
    ax[2].annotate(f"median {med:.0f} m", (med, ax[2].get_ylim()[1] * .55), fontsize=9,
                   ha="right", xytext=(-5, 0), textcoords="offset points")
    ax[2].annotate(f"p95 {p95:.0f} m", (p95, ax[2].get_ylim()[1] * .55), fontsize=9,
                   xytext=(5, 0), textcoords="offset points", color=C_BAD)
    ax[2].set_title(f"(c) Error of a fix against its own road\nmedian {med:.0f} m, p95 {p95:.0f} m",
                    loc="left", fontsize=9.5)
    ax[2].set_xlabel("distance from GPS fix to true road (m)"); ax[2].set_ylabel("number of fixes")

    dts = np.diff(track[:, 2]); dpos = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    v = dpos[dts > 0] / dts[dts > 0] * 3.6
    w = 15
    ker = np.ones(w) / w
    vs = np.convolve(v, ker, mode="same")
    tt = np.arange(len(v))
    ax[3].plot(tt, np.clip(v, 0, np.percentile(v, 99)), lw=.4, c="0.7", label="raw (GPS jitter included)")
    ax[3].plot(tt, vs, lw=1.6, c=C_TRAIN, label=f"{w} s rolling median")
    ax[3].axhline(np.median(v), ls="--", c="k", lw=1.1)
    ax[3].set_title(f"(d) Speed varies inside one drive\n{track[-1, 2]/60:.0f} min, "
                    f"median {np.median(v):.0f} km/h, p98 {np.percentile(v, 98):.0f} km/h",
                    loc="left", fontsize=9.5)
    ax[3].set_xlabel("time since first fix (seconds)")
    ax[3].set_ylabel("speed between fixes (km/h)"); ax[3].set_ylim(0, None)
    ax[3].legend(fontsize=7.5, loc="upper left")

    fig.suptitle(note, y=1.02, fontsize=11, x=.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=2.0)
    fig.savefig(os.path.join(FIGS, "01_anatomy.png"), bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ 02 error spread
def fig_error_spread():
    s = sorted(STATS, key=lambda x: x["res_med"])
    med = np.array([x["res_med"] for x in s]); p95 = np.array([x["res_p95"] for x in s])
    x = np.arange(len(s))
    bad = [i for i, v in enumerate(med) if v > 20]
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.vlines(x, med, p95, color="0.8", lw=1)
    ax.scatter(x, med, s=18, c=[C_BAD if v > 20 else C_TRAIN for v in med],
               label="median error per drive", zorder=3)
    ax.scatter(x, p95, s=10, marker="_", c="0.45", label="95th percentile")
    for k, i in enumerate(bad):
        ax.annotate(f"{s[i]['inst']}: {s[i]['res_med']:.0f} m", (i, s[i]["res_med"]),
                    textcoords="offset points", xytext=(-6, 10 + 12 * k), ha="right",
                    fontsize=8, color=C_BAD,
                    arrowprops=dict(arrowstyle="-", color=C_BAD, lw=.7))
    ax.set_yscale("log")
    ax.set_title(f"One snap radius cannot serve all drives: {len(STATS)-len(bad)}/{len(STATS)} drives have a median error of "
                 f"{np.median(med[med<=20]):.0f} m,\nbut {len(bad)} do not ({', '.join(s[i]['inst'] for i in bad)}) - "
                 "a fixed 50 m gate would silently fail on them", loc="left")
    ax.set_xlabel("drives, sorted by median error (90 drives, train and test mixed)")
    ax.set_ylabel("error vs true road (m, log scale)")
    ax.grid(alpha=.25, axis="y", lw=.5); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_xticks([])
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "02_error_spread.png")); plt.close(fig)
    return bad, s


# ------------------------------------------------------------------ 03 ambiguity
def fig_ambiguity():
    cand = np.array([s["cand25"] for s in STATS])
    arcmed = np.array([s["med_arc"] for s in STATS])
    spacing_med = np.array([s["spacing_med"] for s in STATS])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    order = np.argsort(cand)
    ax[0].bar(range(len(cand)), cand[order], color=[C_BAD if c >= 5 else C_TRAIN for c in cand[order]])
    ax[0].axhline(np.median(cand), ls="--", c="k", lw=1)
    ax[0].annotate(f"median {np.median(cand):.0f} candidate roads", (0, np.median(cand)),
                   fontsize=9, xytext=(2, 4), textcoords="offset points")
    ax[0].set_title(f"(a) A GPS fix is ambiguous: {np.median(cand):.0f} roads lie within 25 m\n"
                    f"worst drive has {cand.max():.0f}", loc="left")
    ax[0].set_xlabel("drives, sorted by ambiguity"); ax[0].set_ylabel("roads within 25 m of a fix")
    ax[0].set_xticks([])

    ax[1].hist(arcmed, bins=25, range=(0, 90), color=C_TRAIN, alpha=.85,
               label="length of one road segment")
    ax[1].hist(spacing_med, bins=25, range=(0, 90), color=C_BAD, alpha=.6,
               label="gap between two GPS fixes")
    ax[1].axvline(np.median(arcmed), ls="--", c=C_TRAIN, lw=1.4)
    ax[1].axvline(np.median(spacing_med), ls="--", c=C_BAD, lw=1.4)
    ax[1].set_title(f"(b) A fix lands every {np.median(spacing_med):.0f} m, a road is "
                    f"{np.median(arcmed):.0f} m long\nso short roads get no fix and must be inferred", loc="left")
    ax[1].set_xlabel("metres (per drive medians compared on one axis)"); ax[1].set_ylabel("number of drives")
    ax[1].legend(fontsize=8.5)

    decoy = np.array([s["decoy"] for s in STATS]) * 100
    ax[2].hist(decoy, bins=25, color=C_ACC)
    ax[2].axvline(np.median(decoy), ls="--", c="k")
    ax[2].set_title(f"(c) The map is mostly decoy\nmedian drive uses {np.median(decoy):.1f}% of its own network", loc="left")
    ax[2].set_xlabel("answer roads as % of that drive's network"); ax[2].set_ylabel("number of drives")
    ax[2].set_xscale("log")

    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "03_ambiguity.png")); plt.close(fig)


# ------------------------------------------------------------------ 04 sessions
def fig_sessions():
    spd = np.array([s["spd_med"] for s in STATS]); dur = np.array([s["dur_h"] for s in STATS])
    npts = np.array([s["track"] for s in STATS]); dtmax = np.array([s["dt_max"] for s in STATS])
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    ax[0].hist(spd, bins=25, color=C_TRAIN)
    ax[0].axvline(np.median(spd), ls="--", c="k")
    ax[0].annotate(f"median {np.median(spd):.0f} km/h", (np.median(spd), ax[0].get_ylim()[1] * .9), fontsize=9,
                   xytext=(4, 0), textcoords="offset points")
    ax[0].set_title(f"(a) Speeds span 1-133 km/h\nso a single motion prior is wrong for most drives", loc="left")
    ax[0].set_xlabel("median speed of a drive (km/h)"); ax[0].set_ylabel("number of drives")

    ax[1].hist(dur, bins=25, color=C_TRAIN)
    ax[1].set_title(f"(b) Drives run {dur.min()*60:.0f} min to {dur.max():.1f} h\nmedian {np.median(dur)*60:.0f} min", loc="left")
    ax[1].set_xlabel("drive duration (hours)"); ax[1].set_ylabel("number of drives")

    ax[2].hist(npts, bins=25, color=C_TRAIN)
    ax[2].set_title(f"(c) {npts.min()} to {npts.max():,} GPS fixes per drive\nmedian {np.median(npts):.0f}, all at 1 fix per second", loc="left")
    ax[2].set_xlabel("number of GPS fixes"); ax[2].set_ylabel("number of drives")

    big = dtmax > 30
    ax[3].hist(np.clip(dtmax, 0, 120), bins=30, color=C_TRAIN)
    ax[3].set_yscale("log")
    ax[3].set_title(f"(d) {big.sum()}/{len(STATS)} drives contain a recording gap over 30 s\n"
                    f"largest gap {dtmax.max()/3600:.1f} h - the matcher must restart there", loc="left")
    ax[3].set_xlabel("largest gap between fixes in a drive (s, clipped at 120)")
    ax[3].set_ylabel("number of drives (log)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "04_sessions.png")); plt.close(fig)


if __name__ == "__main__":
    fig_world(); print("00_world.png")
    fig_anatomy(TRAIN[2], "train 00000003", "Anatomy of one drive - what the four files actually contain")
    print("01_anatomy.png")
    bad, s = fig_error_spread(); print("02_error_spread.png")
    fig_ambiguity(); print("03_ambiguity.png")
    fig_sessions(); print("04_sessions.png")
    print("flagged drives:", [s[i]["inst"] for i in bad])
