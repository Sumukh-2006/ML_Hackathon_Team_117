"""
Candidate generation + per-observation labels (exact spatial grid version).

Why this file exists: the graded metric is observation-level Top-1 / macro-F1 / log-loss
over ROAD SEGMENTS plus "candidate miss rate". Every model sits on top of a candidate set,
so the generator's recall is a hard ceiling. This measures that ceiling exactly.

Fix vs the first attempt: a KD-tree on arc MIDPOINTS silently drops long arcs (a 700 m
motorway arc has a midpoint 350 m away from a fix that sits right on it). Here arcs are
bucketed into a uniform grid by their bounding box, so a query returns EVERY arc whose
bounding box touches the search disk - exact, no distance bias.

Labels: per-fix reference arc = nearest segment of the ground-truth route, searched only
in a forward window from the previous match (monotone), so a fix cannot snap back onto a
parallel road already passed.

Output: notes/candidates_stats.csv, notes/figs/05_candidates.png, notes/figs/06_candidates_zoom.png
Run:    python notes/candidates.py
"""
import os, sys, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, to_xy, seg_dist, route_polyline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "notes", "figs")
TRAIN = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
TEST = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))
KS = (1, 3, 5, 10, 20)
RS = (25, 50, 75)
CELL = 50.0


class ArcGrid:
    """uniform grid over arc bounding boxes -> exact 'arcs within r of p' queries"""

    def __init__(self, A, B, cell=CELL):
        self.A, self.B, self.c = A, B, cell
        x0 = np.minimum(A[:, 0], B[:, 0]); x1 = np.maximum(A[:, 0], B[:, 0])
        y0 = np.minimum(A[:, 1], B[:, 1]); y1 = np.maximum(A[:, 1], B[:, 1])
        self.i0 = np.floor(x0 / cell).astype(np.int64); self.i1 = np.floor(x1 / cell).astype(np.int64)
        self.j0 = np.floor(y0 / cell).astype(np.int64); self.j1 = np.floor(y1 / cell).astype(np.int64)
        nx, ny = self.i1 - self.i0 + 1, self.j1 - self.j0 + 1
        cnt = nx * ny
        m = len(A)
        rep = np.repeat(np.arange(m), cnt)
        start = np.repeat(np.cumsum(cnt) - cnt, cnt)
        pos = np.arange(cnt.sum()) - start
        ci = self.i0[rep] + pos // ny[rep]
        cj = self.j0[rep] + pos % ny[rep]
        self.key = ci * 1_000_003 + cj
        order = np.argsort(self.key, kind="stable")
        self.key_sorted = self.key[order]
        self.arc_sorted = rep[order]
        self.uniq, self.first = np.unique(self.key_sorted, return_index=True)
        self.last = np.append(self.first[1:], len(self.key_sorted))

    def query(self, P, r):
        """list of (arc indices) arrays, one per point, all arcs whose bbox meets the disk"""
        i0 = np.floor((P[:, 0] - r) / self.c).astype(np.int64)
        i1 = np.floor((P[:, 0] + r) / self.c).astype(np.int64)
        j0 = np.floor((P[:, 1] - r) / self.c).astype(np.int64)
        j1 = np.floor((P[:, 1] + r) / self.c).astype(np.int64)
        out = []
        for k in range(len(P)):
            ii = np.arange(i0[k], i1[k] + 1)
            jj = np.arange(j0[k], j1[k] + 1)
            kb = (ii[:, None] * 1_000_003 + jj[None, :]).ravel()
            pos = np.searchsorted(self.uniq, kb)
            pos = np.clip(pos, 0, len(self.uniq) - 1)
            ok = self.uniq[pos] == kb
            if not ok.any():
                out.append(np.empty(0, dtype=int)); continue
            lo = self.first[pos[ok]]; hi = self.last[pos[ok]]
            idx = np.concatenate([self.arc_sorted[a:b] for a, b in zip(lo, hi)])
            out.append(np.unique(idx))
        return out


def label_observations(nodes, arcs, route, track, window=80, max_off=250.0):
    _, poly = route_polyline(nodes, arcs, route)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    P = np.column_stack(to_xy(track, lon0, lat0))
    R = np.column_stack(to_xy(poly, lon0, lat0))
    A, B = R[:-1], R[1:]
    n = len(route)
    lab = np.full(len(P), -1, dtype=int)
    dist = np.full(len(P), np.inf)
    lo = 0
    for i in range(len(P)):
        hi = min(n, lo + window)
        if hi <= lo:
            lo, hi = max(0, n - window), n
        d = seg_dist(P[i:i + 1], A[lo:hi], B[lo:hi])[0]
        j = int(d.argmin())
        dist[i] = d[j]
        if d[j] < max_off:
            lab[i] = route[lo + j]
        lo = max(0, lo + j - 3)
    return lab, dist


def build_drive(d, r_max=75.0, n_sample=300, k_keep=20):
    nodes, arcs, route, track = load(d)
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    V = np.column_stack(to_xy(nodes, lon0, lat0))
    A, B = V[arcs[:, 0]], V[arcs[:, 1]]
    seglen = np.hypot(*(B - A).T)
    grid = ArcGrid(A, B)

    lab, dist = label_observations(nodes, arcs, route, track)
    step = max(1, len(track) // n_sample)
    sel = np.arange(0, len(track), step)
    P = np.column_stack(to_xy(track[sel], lon0, lat0))
    lab_s, dist_s = lab[sel], dist[sel]

    pool = grid.query(P, r_max)
    cand = []
    for i, pi in enumerate(pool):
        if len(pi) == 0:
            cand.append(np.empty((0, 2))); continue
        dd = seg_dist(P[i:i + 1], A[pi], B[pi])[0]
        keep = dd < r_max
        if not keep.any():
            cand.append(np.empty((0, 2))); continue
        dd, pi = dd[keep], pi[keep]
        o = np.argsort(dd)
        cand.append(np.column_stack([dd[o], pi[o]])[:k_keep])

    valid = lab_s >= 0
    res = {}
    for r in RS:
        for K in KS:
            hit = 0
            for i in np.where(valid)[0]:
                c = cand[i]
                if len(c) == 0:
                    continue
                c = c[c[:, 0] < r][:K]
                hit += int(np.any(c[:, 1].astype(int) == lab_s[i]))
            res[f"hit_r{r}_k{K}"] = hit / max(1, valid.sum())

    # miss rate split by whether the fix is itself close to its own road: a fix 300 m
    # from its road cannot be in a 50 m candidate set, and that is the fix's fault,
    # not the generator's. This separates generator failure from GPS failure.
    near = valid & (dist_s < 50.0)
    hit_near = 0
    for i in np.where(near)[0]:
        c = cand[i]
        if len(c) == 0:
            continue
        c = c[c[:, 0] < 50.0][:10]
        hit_near += int(np.any(c[:, 1].astype(int) == lab_s[i]))
    res["miss_r50_k10_fix_within_50m"] = 1.0 - hit_near / max(1, near.sum())
    res["fix_offroad_frac"] = float(np.mean(dist_s[valid] > 50.0))
    res["fix_dist_p99"] = float(np.percentile(dist_s[valid], 99))

    ncand = np.array([len(c) for c in cand])
    nearest_true = float(np.mean([len(c) > 0 and int(c[0, 1]) == l for c, l in zip(cand, lab_s)]))
    return dict(inst=os.path.basename(d), points=len(track), sampled=len(P),
                offroute_frac=float(1 - valid.mean()), cand_mean=float(ncand.mean()),
                empty_frac=float((ncand == 0).mean()), nearest_is_true=nearest_true,
                med_arc_len=float(np.median(seglen)), max_arc_len=float(seglen.max()),
                pts_per_arc=float(valid.sum() / max(1, len(route))), **res), (
        nodes, arcs, route, track, cand, lab_s, sel, A, B)


def main():
    rows, demo = [], None
    for d in TRAIN + TEST:
        r, extra = build_drive(d)
        rows.append(r)
        if os.path.basename(d) == "00000003":
            demo = extra
        print(f"{r['inst']} miss@50/k10={100*(1-r['hit_r50_k10']):5.2f}%  miss@50/k1={100*(1-r['hit_r50_k1']):5.2f}%"
              f"  cand={r['cand_mean']:4.1f}  pts/arc={r['pts_per_arc']:4.1f}  maxarc={r['max_arc_len']:.0f}m")

    with open(os.path.join(ROOT, "notes", "candidates_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\nMEAN over 90 drives:")
    for r in RS:
        print("  r=%2d m: " % r + "  ".join(
            f"K={K}: miss {100*np.mean([1 - x[f'hit_r{r}_k{K}'] for x in rows]):5.2f}%" for K in KS))
    print("  generator-only miss at 50 m / K=10 (fixes already within 50 m of a road): %.2f%%"
          % (100 * np.mean([x["miss_r50_k10_fix_within_50m"] for x in rows])))
    print("  fixes >50 m from their own road: %.2f%%   p99 fix-to-road distance: %.0f m"
          % (100 * np.mean([x["fix_offroad_frac"] for x in rows]),
             np.mean([x["fix_dist_p99"] for x in rows])))

    # ---------------------------------------------------------------- fig 5
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for r, col in zip(RS, ("#c05621", "#2b6cb0", "#2f855a")):
        miss = [100 * np.mean([1 - x[f"hit_r{r}_k{K}"] for x in rows]) for K in KS]
        ax[0].plot(KS, miss, "o-", c=col, label=f"radius {r} m")
    ax[0].set_xscale("log"); ax[0].set_xticks(KS); ax[0].set_xticklabels(KS)
    ax[0].set_ylim(-1, None)
    ax[0].set_title("(a) Candidate miss rate: the ceiling on every model\n"
                    f"50 m + 10 candidates loses {100*np.mean([1-x['hit_r50_k10'] for x in rows]):.2f}% of fixes, "
                    f"nearest-1 alone loses {100*np.mean([1-x['hit_r50_k1'] for x in rows]):.0f}%", loc="left")
    ax[0].set_xlabel("candidates kept per GPS fix (K)")
    ax[0].set_ylabel("fixes whose true road is missing (%)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25, lw=.5)

    npts = np.array([x["pts_per_arc"] for x in rows])
    ax[1].hist(np.clip(npts, 0, 8), bins=32, color="#2b6cb0")
    ax[1].set_title(f"(b) Label imbalance: median {np.median(npts):.1f} fixes per true road\n"
                    f"{int((npts<1).sum())}/{len(rows)} drives have more roads than fixes", loc="left")
    ax[1].set_xlabel("GPS fixes per ground-truth road segment (clipped at 8)")
    ax[1].set_ylabel("number of drives")

    off = np.array([x["offroute_frac"] for x in rows]) * 100
    med = np.array([x["med_arc_len"] for x in rows])
    ax[2].scatter(med, off, s=24, color="#c53030")
    ax[2].set_title("(c) Off-route fixes concentrate in long-road drives\n"
                    f"max {off.max():.0f}% of fixes sit >250 m from the reference route", loc="left")
    ax[2].set_xlabel("median road length in the drive's network (m)")
    ax[2].set_ylabel("fixes >250 m from reference route (%)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "05_candidates.png")); plt.close(fig)

    # ---------------------------------------------------------------- fig 6
    if demo is not None:
        nodes, arcs, route, track, cand, lab_s, sel, A, B = demo
        lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
        V = np.column_stack(to_xy(nodes, lon0, lat0))
        _, poly = route_polyline(nodes, arcs, route)
        R = np.column_stack(to_xy(poly, lon0, lat0))
        P = np.column_stack(to_xy(track[sel], lon0, lat0))
        i = int(np.argmax([len(c) for c in cand]))
        c = cand[i]; px, py = P[i]
        fig, ax = plt.subplots(figsize=(7.8, 6.4))
        ax.plot(R[:, 0], R[:, 1], "-", c="#c53030", lw=2.6, label="reference route (the answer)", zorder=2)
        for j, (dd, a) in enumerate(c[:8]):
            a = int(a)
            ax.plot(V[arcs[a]][:, 0], V[arcs[a]][:, 1], "-", lw=3.4, alpha=.9, zorder=3,
                    color=plt.cm.viridis(j / 9), label=f"candidate {j+1}: {dd:.0f} m away")
        ax.plot(px, py, "o", c="k", ms=10, zorder=6, label="one noisy GPS fix")
        truth = np.where(c[:, 1].astype(int) == lab_s[i])[0]
        if len(truth):
            j = int(truth[0]); a = int(c[j, 1])
            ax.plot(V[arcs[a]][:, 0], V[arcs[a]][:, 1], "-", lw=5.5, alpha=.35, c="#c53030", zorder=4)
        ax.set_xlim(px - 130, px + 130); ax.set_ylim(py - 130, py + 130)
        ax.set_title(f"One GPS fix, {len(c)} candidate roads inside {75} m\n"
                     f"the model must pick one; the true road is candidate "
                     f"{int(truth[0])+1 if len(truth) else 'MISSING'}", loc="left")
        ax.set_xlabel("metres east"); ax.set_ylabel("metres north")
        ax.legend(fontsize=7.5, loc="upper right"); ax.set_aspect("equal")
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "06_candidates_zoom.png")); plt.close(fig)
        print("wrote 06_candidates_zoom.png")


if __name__ == "__main__":
    main()
