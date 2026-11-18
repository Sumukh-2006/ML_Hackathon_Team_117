"""Deeper check: coordinate order, noise level, route coverage, anomalies."""
import glob, os, math, statistics, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lines(p):
    with open(p) as f:
        return [l.strip() for l in f if l.strip() != ""]


def load(inst_dir):
    name = os.path.basename(inst_dir)
    nodes = [tuple(map(float, l.split())) for l in lines(f"{inst_dir}/{name}.nodes")]
    arcs = [tuple(map(int, l.split())) for l in lines(f"{inst_dir}/{name}.arcs")]
    route = [int(l) for l in lines(f"{inst_dir}/{name}.route")]
    track = [tuple(map(float, l.split())) for l in lines(f"{inst_dir}/{name}.track")]
    return nodes, arcs, route, track


def hav(a, b):
    R = 6371000.0
    la1, la2 = math.radians(a[1]), math.radians(b[1])
    dphi = la2 - la1
    dl = math.radians(b[0] - a[0])
    h = math.sin(dphi / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(1.0, h)))


def seg_dist(p, a, b):
    """metres from p to segment a-b, flat-earth approx (fine at city scale)"""
    k = math.cos(math.radians(p[1]))
    px, py = p[0] * k, p[1]
    ax, ay = a[0] * k, a[1]
    bx, by = b[0] * k, b[1]
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0:
        return hav(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    # convert back to lon/lat metres
    ddx = (px - qx) * 111320.0
    ddy = (py - qy) * 111320.0
    return math.hypot(ddx, ddy)


train = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
test = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))

print("=== coordinate order check (first col as longitude vs latitude) ===")
allfirst = []
for d in train + test:
    nodes, *_ = load(d)
    allfirst += [n[0] for n in nodes]
print(f"first column: min {min(allfirst):.2f} max {max(allfirst):.2f}  -> values above 90 exist: {max(allfirst) > 90}")
allsecond = []
for d in train + test:
    nodes, *_ = load(d)
    allsecond += [n[1] for n in nodes]
print(f"second column: min {min(allsecond):.2f} max {max(allsecond):.2f} -> values above 90 exist: {max(allsecond) > 90}")
print("=> first column is LONGITUDE, second is LATITUDE (only lon can reach +-180)\n")

print("=== per-instance: route vs track vs noise ===")
hdr = f"{'id':>9} {'nodes':>7} {'arcs':>7} {'|R|':>5} {'|T|':>5} {'route_km':>8} {'drive_km':>8} {'ratio':>6} {'noise_med':>9} {'noise_p90':>9} {'cov300m':>8} {'gaps':>5}"
print(hdr)
rows = []
for tag, insts in (("TRAIN", train), ("TEST", test)):
    print(f"--- {tag} ---")
    for d in insts:
        nodes, arcs, route, track = load(d)
        rl = sum(hav(nodes[arcs[i][0]], nodes[arcs[i][1]]) for i in route)
        tl = sum(hav(track[k][:2], track[k + 1][:2]) for k in range(len(track) - 1))
        segs = [(nodes[arcs[i][0]], nodes[arcs[i][1]]) for i in route]
        # residual of each track point from the true route polyline
        devs = []
        step = max(1, len(track) // 400)
        for p in track[::step]:
            devs.append(min(seg_dist(p[:2], a, b) for a, b in segs))
        med = statistics.median(devs)
        p90 = sorted(devs)[int(0.9 * (len(devs) - 1))]
        cov = sum(1 for v in devs if v < 300) / len(devs)
        gaps = sum(1 for i in range(len(route) - 1) if arcs[route[i]][1] != arcs[route[i + 1]][0])
        ratio = rl / max(tl, 1)
        rows.append((ratio, med))
        print(f"{os.path.basename(d):>9} {len(nodes):7d} {len(arcs):7d} {len(route):5d} {len(track):5d} "
              f"{rl/1000:8.2f} {tl/1000:8.2f} {ratio:6.2f} {med:9.1f} {p90:9.1f} {cov:8.2f} {gaps:5d}")

print("\n=== summaries ===")
print("route_len/drive_len ratio: min %.2f med %.2f max %.2f" % (
    min(r[0] for r in rows), statistics.median(r[0] for r in rows), max(r[0] for r in rows)))
print("noise median (m): min %.1f med %.1f max %.1f" % (
    min(r[1] for r in rows), statistics.median(r[1] for r in rows), max(r[1] for r in rows)))
