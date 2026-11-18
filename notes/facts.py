"""One consolidated fact sheet - every number below is measured, nothing estimated."""
import os, sys, glob, csv, math
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, hav

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = sorted(d for d in glob.glob(os.path.join(ROOT, "train", "*")) if os.path.isdir(d))
TEST = sorted(d for d in glob.glob(os.path.join(ROOT, "student_test", "*")) if os.path.isdir(d))
L = list(csv.DictReader(open(os.path.join(ROOT, "notes", "locations.csv"), encoding="utf-8")))
ST = {r["inst"]: r for r in csv.DictReader(open(os.path.join(ROOT, "notes", "eda_stats.csv")))}
CA = {r["inst"]: r for r in csv.DictReader(open(os.path.join(ROOT, "notes", "candidates_stats.csv")))}

agg = dict(pts=[], nodes=[], arcs=[], route=[], km=[], dur=[], avgspd=[], dt1=[], gaps=[], timereset=[],
           nan=[], duppts=[], revs=[], nan_route=[], route_dup=[], ts_back=[])
total_bytes = 0
for d in TRAIN + TEST:
    nodes, arcs, route, track = load(d)
    agg["pts"].append(len(track)); agg["nodes"].append(len(nodes)); agg["arcs"].append(len(arcs))
    agg["route"].append(len(route))
    dts = np.diff(track[:, 2])
    agg["dt1"].append(float(np.mean(dts == 1.0)))
    agg["gaps"].append(float(dts.max()))
    agg["ts_back"].append(int(np.sum(dts <= 0)))
    dpos = hav(track[:-1, 0], track[:-1, 1], track[1:, 0], track[1:, 1])
    km = dpos.sum() / 1000
    agg["km"].append(km); agg["dur"].append(track[-1, 2] / 3600)
    agg["avgspd"].append(km / max(track[-1, 2] / 3600, 1e-9))
    agg["nan"].append(int(np.sum(~np.isfinite(track)) + np.sum(~np.isfinite(nodes))))
    agg["duppts"].append(int(np.sum((dpos == 0) & (dts == 0))))
    s = set(map(tuple, arcs.tolist()))
    agg["revs"].append(float(np.mean([(v, u) in s for u, v in arcs])))
    agg["route_dup"].append(len(route) - len(set(route.tolist())))
    for e in ("nodes", "arcs", "route", "track"):
        p = os.path.join(d, f"{os.path.basename(d)}.{e}")
        if os.path.exists(p):
            total_bytes += os.path.getsize(p)

P = lambda a: (np.min(a), np.median(a), np.max(a))
out = []
def say(s=""):
    out.append(s); print(s)

say("=" * 78)
say("PATHSNAP DATASET - MEASURED FACTS")
say("=" * 78)
say(f"trips shipped            : {len(TRAIN)} train + {len(TEST)} student-test = {len(TRAIN)+len(TEST)}")
say(f"files                    : 4 per trip (nodes, arcs, route, track) = {(len(TRAIN)+len(TEST))*4}")
say(f"total size on disk       : {total_bytes/1e6:.1f} MB")
say(f"evaluator-testing trips  : 10, not shipped (stated in the brief)")
say()
say(f"GPS fixes                : {sum(agg['pts']):,} total, per trip min/med/max = {P(agg['pts'])}")
say(f"road nodes               : {sum(agg['nodes']):,} total, per trip min/med/max = {P(agg['nodes'])}")
say(f"road segments (arcs)     : {sum(agg['arcs']):,} total, per trip min/med/max = {P(agg['arcs'])}")
say(f"answer route segments    : {sum(agg['route']):,} total, per trip min/med/max = {P(agg['route'])}")
say(f"answer route as share of network : median {100*np.median([r_/a for r_, a in zip(agg['route'], agg['arcs'])]):.2f}%"
    f" (min {100*min(r_/a for r_, a in zip(agg['route'], agg['arcs'])):.2f}%, max {100*max(r_/a for r_, a in zip(agg['route'], agg['arcs'])):.1f}%)")
say()
say(f"drive length             : median {np.median(agg['km']):.1f} km (min {min(agg['km']):.1f}, max {max(agg['km']):.1f})")
say(f"drive duration           : median {np.median(agg['dur'])*60:.0f} min (min {min(agg['dur'])*60:.0f} min, max {max(agg['dur']):.1f} h)")
say(f"drive average speed      : median {np.median(agg['avgspd']):.1f} km/h (min {min(agg['avgspd']):.1f}, max {max(agg['avgspd']):.1f})")
say(f"samples at exactly 1 s   : median {100*np.median(agg['dt1']):.1f}% of steps per trip"
    f" (worst trip {100*min(agg['dt1']):.1f}%)")
say(f"trips with a gap >30 s   : {sum(1 for g in agg['gaps'] if g > 30)}/{len(agg['gaps'])}; "
    f"largest single gap {max(agg['gaps'])/3600:.1f} h")
say(f"non-increasing timestamps: {sum(agg['ts_back'])} rows across all trips")
say(f"NaN/inf values           : {sum(agg['nan'])} rows")
say(f"repeated (t, position)   : {sum(agg['duppts'])} rows")
say(f"trips whose route repeats a segment : {sum(1 for x in agg['route_dup'] if x > 0)}"
    f" (max {max(agg['route_dup'])} repeats)")
say()
say("--- geometry / noise ---")
res = np.array([float(ST[r]['res_med']) for r in ST]); p95 = np.array([float(ST[r]['res_p95']) for r in ST])
say(f"fix-to-its-own-road error: median per trip min/med/max = {P(res)[0]:.1f}/{P(res)[1]:.1f}/{P(res)[2]:.1f} m")
say(f"                           p95  per trip min/med/max = {P(p95)[0]:.1f}/{P(p95)[1]:.1f}/{P(p95)[2]:.1f} m")
sp = np.array([float(ST[r]['spacing_med']) for r in ST])
say(f"gap between two fixes    : median per trip = {np.median(sp):.1f} m (p10 {np.percentile(sp,10):.1f}, p90 {np.percentile(sp,90):.1f})")
ma = np.array([float(ST[r]['med_arc']) for r in ST])
say(f"road segment length      : median per trip = {np.median(ma):.1f} m (p10 {np.percentile(ma,10):.1f}, p90 {np.percentile(ma,90):.1f})")
say(f"roads within 25 m of a fix: median = {np.median([float(ST[r]['cand25']) for r in ST]):.0f}")
say(f"one-way roads (no reverse twin): median {100*np.median(agg['revs']):.0f}% of segments are paired,"
    f" so {100*(1-np.median(agg['revs'])):.0f}% are one-way")
say()
say("--- candidate sets (K nearest roads within r) ---")
for r in (25, 50, 75):
    for K in (1, 5, 10, 20):
        m = np.mean([1 - float(CA[i][f"hit_r{r}_k{K}"]) for i in CA])
        say(f"  r={r:2d} m, K={K:2d}: true road missing from candidates for {100*m:5.2f}% of fixes")
say(f"  fixes already >50 m from their own road: {100*np.mean([float(CA[i]['fix_offroad_frac']) for i in CA]):.2f}%")
say(f"  misses at r=50/K=10 among fixes that ARE within 50 m: "
    f"{100*np.mean([float(CA[i]['miss_r50_k10_fix_within_50m']) for i in CA]):.2f}%")
say(f"  fixes per true road segment: median {np.median([float(CA[i]['pts_per_arc']) for i in CA]):.1f}")
say()
say("--- where the data is ---")
cc = Counter(r["country"] for r in L)
say(f"countries: {len(cc)} -> " + ", ".join(f"{k} ({v})" for k, v in cc.most_common(8)) + ", ...")
say(f"cities (reverse-geocoded): " + "; ".join(sorted({r['city'] for r in L if r['city']})[:20]) + " ...")
lons = np.array([float(r["lon"]) for r in L]); lats = np.array([float(r["lat"]) for r in L])
say(f"longitude span {lons.min():.1f} .. {lons.max():.1f}, latitude span {lats.min():.1f} .. {lats.max():.1f}")

# near-duplicate areas (possible leakage)
pts = np.array([[float(r["lat"]), float(r["lon"])] for r in L])
ids = [r["instance"] for r in L]
close = [(ids[i], ids[j], 111*math.dist(pts[i], pts[j]))
         for i in range(len(ids)) for j in range(i + 1, len(ids))
         if 111 * math.dist(pts[i], pts[j]) < 5]
say(f"trip pairs whose network centres are <5 km apart: {len(close)}")
for a, b, dd in close[:12]:
    say(f"   {a} / {b}  {dd:.1f} km apart")

with open(os.path.join(ROOT, "notes", "FACTS.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\nwrote notes/FACTS.txt")
