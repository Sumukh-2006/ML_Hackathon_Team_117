"""Identify where each instance is + graph fingerprint (is it OSM-derived?)."""
import sys, os, glob, json, time, csv, urllib.request
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def geocode(lat, lon):
    url = ("https://api.bigdatacloud.net/data/reverse-geocode-client"
           f"?latitude={lat:.4f}&longitude={lon:.4f}&localityLanguage=en")
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            j = json.loads(r.read().decode())
        return j.get("city") or j.get("locality") or "", j.get("principalSubdivision", ""), j.get("countryName", "")
    except Exception:
        return "", "", ""


rows = []
insts = sorted(glob.glob(os.path.join(ROOT, "train", "*"))) + \
        sorted(glob.glob(os.path.join(ROOT, "student_test", "*")))
for d in insts:
    if not os.path.isdir(d):
        continue
    nodes, arcs, route, track = load(d)
    lat, lon = float(nodes[:, 1].mean()), float(nodes[:, 0].mean())
    s = set(map(tuple, arcs.tolist()))
    oneway = float(np.mean([(v, u) not in s for u, v in arcs]))
    deg = Counter(arcs[:, 0].tolist() + arcs[:, 1].tolist())
    tot = sum(deg.values())
    d1 = sum(1 for v in deg.values() if v == 1) / len(deg)
    d2 = sum(1 for v in deg.values() if v == 2) / len(deg)
    city, region, country = geocode(lat, lon)
    rows.append(dict(instance=os.path.basename(d),
                     split="test" if "student_test" in d else "train",
                     lat=round(lat, 4), lon=round(lon, 4), city=city, region=region, country=country,
                     nodes=len(nodes), arcs=len(arcs),
                     oneway_frac=round(oneway, 3), deg1_frac=round(d1, 3), deg2_frac=round(d2, 3)))
    print(f"{rows[-1]['instance']} {lat:8.3f} {lon:9.3f} {country[:18]:18s} {region[:18]:18s} {city[:20]:20s}"
          f" oneway={oneway:.2f} deg1={d1:.2f} deg2={d2:.2f}")
    time.sleep(0.15)

with open(os.path.join(ROOT, "notes", "locations.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\ncountries:", Counter(r["country"] for r in rows).most_common())
