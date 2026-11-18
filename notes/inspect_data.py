"""Structural sanity-check of the map-matching dataset formats."""
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


def hav(a, b):  # metres, a/b = (lon, lat)
    R = 6371000.0
    la1, la2 = math.radians(a[1]), math.radians(b[1])
    dphi = la2 - la1
    dl = math.radians(b[0] - a[0])
    h = math.sin(dphi / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


train = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
test = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))

print(f"train instances: {len(train)}   test instances: {len(test)}")

for tag, insts in (("TRAIN", train), ("TEST ", test)):
    lon0, lon1, lat0, lat1 = 1e9, -1e9, 1e9, -1e9
    nn, na, nt, nreg = [], [], [], 0
    for d in insts:
        nodes, arcs, route, track = load(d)
        lons = [p[0] for p in nodes]; lats = [p[1] for p in nodes]
        lon0, lon1 = min(lon0, min(lons)), max(lon1, max(lons))
        lat0, lat1 = min(lat0, min(lats)), max(lat1, max(lats))
        nn.append(len(nodes)); na.append(len(arcs)); nt.append(len(track))
    print(f"{tag} node lon [{lon0:.3f},{lon1:.3f}] lat [{lat0:.3f},{lat1:.3f}]")
    print(f"{tag} nodes: min {min(nn)} med {int(statistics.median(nn))} max {max(nn)}")
    print(f"{tag} arcs : min {min(na)} med {int(statistics.median(na))} max {max(na)}")
    print(f"{tag} track: min {min(nt)} med {int(statistics.median(nt))} max {max(nt)}")

# --- detailed invariants on a handful of instances
print("\n=== per-instance invariants ===")
for insts, tag in ((train[:3], "TRAIN"), (test[:2], "TEST")):
    for d in insts:
        nodes, arcs, route, track = load(d)
        # 1. arc ids valid
        arc_ok = all(0 <= i < len(arcs) for i in route)
        # 2. arc endpoints valid node ids
        node_ok = all(0 <= u < len(nodes) and 0 <= v < len(nodes) for u, v in arcs)
        # 3. route is a connected walk: arcs[i].v == arcs[i+1].u
        gaps = sum(1 for i in range(len(route) - 1) if arcs[route[i]][1] != arcs[route[i + 1]][0])
        # 4. undirected validity: either forward or reverse connection
        gaps_u = sum(
            1
            for i in range(len(route) - 1)
            if {arcs[route[i]]} != {arcs[route[i + 1]]}
            and arcs[route[i]][1] not in arcs[route[i + 1]]
            and arcs[route[i]][0] not in arcs[route[i + 1]]
        )
        # 5. duplicates in route?
        dup = len(route) - len(set(route))
        # 6. route length vs straight-line distance of track
        rl = sum(hav(nodes[arcs[i][0]], nodes[arcs[i][1]]) for i in route)
        tl = sum(hav(track[k][:2], track[k + 1][:2]) for k in range(len(track) - 1))
        # 7. track timestamps
        dts = [round(track[k + 1][2] - track[k][2], 3) for k in range(min(5, len(track) - 1))]
        # 8. deviation of raw GPS from nearest node
        r1 = lambda v: round(v, 1)
        devs = []
        for p in track[:400]:
            devs.append(min(hav(p[:2], n) for n in nodes))
        print(
            f"{os.path.basename(d)} nodes={len(nodes):6d} arcs={len(arcs):6d} route={len(route):4d} track={len(track):5d} "
            f"arc_ids_ok={arc_ok} node_ids_ok={node_ok} walk_gaps={gaps}/{len(route)-1} undirected_gaps={gaps_u} dup_arc={dup} "
            f"route_len={r1(rl)}m track_len={r1(tl)}m dev_med={r1(statistics.median(devs))}m dt={dts} "
            f"route_ascending={route == sorted(route)}"
        )

# --- directed vs undirected: is every arc reversed somewhere?
print("\n=== graph directionality / weights ===")
d = train[0]
nodes, arcs, route, track = load(d)
s = set(arcs)
both = sum(1 for u, v in arcs if (v, u) in s)
print(f"{os.path.basename(d)}: {len(arcs)} arcs, {both} have a reverse twin ({100*both/len(arcs):.1f}%)")
lens = [hav(nodes[u], nodes[v]) for u, v in arcs]
print("arc length m: min %.1f med %.1f p95 %.1f max %.1f" % (
    min(lens), statistics.median(lens), sorted(lens)[int(0.95 * len(lens))], max(lens)))
deg = collections.Counter()
for u, v in arcs:
    deg[u] += 1
print("out-degree: med %d max %d, isolated nodes %d" % (
    statistics.median(deg.values()), max(deg.values()), len(nodes) - len(deg)))
# arcs file ordering: is it sorted by source node?
print("arcs sorted by source node:", [a[0] for a in arcs] == sorted(a[0] for a in arcs))

# --- track sampling
print("\n=== track sampling ===")
d = train[0]
nodes, arcs, route, track = load(d)
dts = collections.Counter(round(track[i + 1][2] - track[i][2], 3) for i in range(len(track) - 1))
print("dt histogram (s):", dts.most_common(5))
steps = [hav(track[i][:2], track[i + 1][:2]) for i in range(len(track) - 1)]
print("step m: med %.1f mean %.1f max %.1f" % (
    statistics.median(steps), statistics.mean(steps), max(steps)))
print("t0=%.1f tN=%.1f  (=%d s of driving)" % (track[0][2], track[-1][2], track[-1][2] - track[0][2]))
