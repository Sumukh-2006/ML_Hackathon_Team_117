"""Independent verification of two EDA claims, vectorised.

1. per-trip coverage: fraction of fixes whose nearest reference-route arc is close,
   and the position of the last well-covered fix (tests the "route stops covering
   the track" claim).
2. heading as a twin discriminator: among fixes whose nearest route arc has a
   geometrically identical reverse twin, how often does the track heading pick the
   true direction over the twin.

Run: python notes/verify_eda.py
"""
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(d, rid, ext):
    return np.loadtxt(f"{d}/{rid}/{rid}.{ext}")


def unit_bearing(x1, y1, x2, y2):
    dlon = np.radians(x2 - x1)
    p1 = np.radians(y1)
    p2 = np.radians(y2)
    return np.degrees(np.arctan2(np.sin(dlon) * np.cos(p2),
                                 np.cos(p1) * np.sin(p2)
                                 - np.sin(p1) * np.cos(p2) * np.cos(dlon))) % 360.0


def adiff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def nearest_arc(P, A, B, chunk=1500):
    """Distance from every point to every segment, returning (min_dist, argmin)."""
    ab = B - A
    L2 = np.maximum((ab * ab).sum(1), 1e-18)
    out_d = np.empty(len(P))
    out_i = np.empty(len(P), dtype=np.int64)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk]
        ap = p[:, None, :] - A[None, :, :]
        t = np.clip((ap * ab[None, :, :]).sum(2) / L2[None, :], 0.0, 1.0)
        proj = A[None, :, :] + t[:, :, None] * ab[None, :, :]
        dd = np.sqrt(((p[:, None, :] - proj) ** 2).sum(2))
        out_d[s:s + chunk] = dd.min(1)
        out_i[s:s + chunk] = dd.argmin(1)
    return out_d, out_i


def main():
    cov_rows = []
    head = {1: [0, 0, 0, 0], 3: [0, 0, 0, 0], 5: [0, 0, 0, 0]}  # (ok30, n30, okAll, nAll)

    for split in ("train", "student_test"):
        for rid in sorted(os.listdir(os.path.join(ROOT, split))):
            t = load(split, rid, "track")
            arcs = load(split, rid, "arcs").astype(np.int64)
            nodes = load(split, rid, "nodes")
            route = load(split, rid, "route").astype(np.int64)

            lat0 = math.radians(t[:, 1].mean())
            kx = 111320.0 * math.cos(lat0)
            ky = 110540.0
            P = np.c_[t[:, 0] * kx, t[:, 1] * ky]
            A = np.c_[nodes[arcs[route][:, 0], 0] * kx, nodes[arcs[route][:, 0], 1] * ky]
            B = np.c_[nodes[arcs[route][:, 1], 0] * kx, nodes[arcs[route][:, 1], 1] * ky]

            dist, near = nearest_arc(P, A, B)

            near = route[near]
            close = dist <= 30.0
            last_close = np.where(close)[0]
            tail = 1.0 - (last_close.max() / (len(t) - 1)) if len(last_close) and len(t) > 1 else 1.0
            cov_rows.append((rid, split, float(close.mean()), float(tail),
                             float(dist.max()), len(t)))

            # heading as twin discriminator
            lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(arcs)}
            twin = np.full(len(arcs), -1, dtype=np.int64)
            for a in route:
                u, v = arcs[a]
                twin[a] = lookup.get((int(v), int(u)), -1)

            for w in (1, 3, 5):
                h = unit_bearing(t[:-2 * w, 0], t[:-2 * w, 1], t[2 * w:, 0], t[2 * w:, 1])
                mid = near[w:-w]
                ba = unit_bearing(nodes[arcs[mid, 0], 0], nodes[arcs[mid, 0], 1],
                                  nodes[arcs[mid, 1], 0], nodes[arcs[mid, 1], 1])
                # nearest-arc assignment is unvalidated, so only score fixes whose
                # assignment is unambiguous (nearest arc is close and has a twin)
                ok = (twin[mid] >= 0) & (dist[w:-w] <= 30.0)
                good = adiff(h[ok], ba[ok]) < adiff(h[ok], (ba[ok] + 180.0) % 360.0)
                head[w][0] += int(good.sum())
                head[w][1] += int(ok.sum())
                allm = twin[mid] >= 0
                goodl = adiff(h[allm], ba[allm]) < adiff(h[allm], (ba[allm] + 180.0) % 360.0)
                head[w][2] += int(goodl.sum())
                head[w][3] += int(allm.sum())

    cov_rows.sort(key=lambda r: -r[3])
    print(f"{'trip':>10} {'split':>13} {'cov30':>7} {'tail':>7} {'maxd_m':>9} {'fixes':>7}")
    for r in cov_rows[:8]:
        print(f"{r[0]:>10} {r[1]:>13} {r[2]:7.3f} {r[3]:7.3f} {r[4]:9.1f} {r[5]:7d}")
    print()
    print(f"trips with tail > 0.10 (route stops covering track early): "
          f"{sum(1 for r in cov_rows if r[3] > 0.10)}")
    print(f"  {[r[0] for r in cov_rows if r[3] > 0.10]}")
    print()
    print(f"heading twin test (window = +/- N fixes):")
    for w, (g, n, ga, na) in head.items():
        print(f"  +/-{w} fix : within 30 m {g:7d}/{n:7d} = {100 * g / max(n, 1):5.1f}%"
              f"   all fixes {ga:7d}/{na:7d} = {100 * ga / max(na, 1):5.1f}%")


if __name__ == "__main__":
    main()
