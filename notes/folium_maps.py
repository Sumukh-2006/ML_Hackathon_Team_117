"""
Folium maps for the report - MANDATORY in the brief ("Do not submit unrelated or
manually fabricated maps"). Layers, one per thing the brief asks to compare:
  raw GPS fixes / road network / reference route / start and end / predicted route.

The network layer is restricted to arcs near the drive (a whole 165k-arc network in
Leaflet is unusably slow) - the restriction is stated in the layer name.

Run:  python notes/folium_maps.py
Out:  notes/maps/trip_<id>.html  and  notes/maps/index.html
"""
import os, sys, glob
import numpy as np
import folium
from folium import FeatureGroup, LayerControl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda import load, to_xy, seg_dist, route_polyline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notes", "maps")
os.makedirs(OUT, exist_ok=True)


def nearby_arcs(nodes, arcs, track, radius=700.0, cap=6000):
    """arcs within `radius` of any fix - keeps the Leaflet page loadable"""
    lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
    V = np.column_stack(to_xy(nodes, lon0, lat0))
    A, B = V[arcs[:, 0]], V[arcs[:, 1]]
    step = max(1, len(track) // 120)
    P = np.column_stack(to_xy(track[::step], lon0, lat0))
    keep = np.zeros(len(arcs), bool)
    for i in range(0, len(P), 25):
        d = seg_dist(P[i:i + 25], A, B).min(axis=0)
        keep |= d < radius
    idx = np.where(keep)[0]
    if len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
    return idx


def make_map(inst_dir, predicted=None):
    nodes, arcs, route, track = load(inst_dir)
    name = os.path.basename(inst_dir)
    _, poly = route_polyline(nodes, arcs, route)

    centre = [float(track[:, 1].mean()), float(track[:, 0].mean())]
    m = folium.Map(location=centre, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    # 1. road network near the drive
    near = nearby_arcs(nodes, arcs, track)
    fg = FeatureGroup(name=f"road network near drive ({len(near)} of {len(arcs):,} roads)")
    for a in near[:: max(1, len(near) // 4000)]:
        u, v = int(arcs[a][0]), int(arcs[a][1])
        folium.PolyLine([(nodes[u][1], nodes[u][0]), (nodes[v][1], nodes[v][0])],
                        color="#999999", weight=1.2, opacity=.6).add_to(fg)
    fg.add_to(m)

    # 2. reference route
    fg = FeatureGroup(name="reference route (ground truth)")
    folium.PolyLine([(p[1], p[0]) for p in poly], color="#d32f2f", weight=4, opacity=.9).add_to(fg)
    fg.add_to(m)

    # 3. raw GPS
    fg = FeatureGroup(name=f"raw GPS fixes ({len(track)})")
    for p in track:
        folium.CircleMarker((p[1], p[0]), radius=2.2, color="#1565c0", weight=0,
                            fill=True, fill_opacity=.75).add_to(fg)
    fg.add_to(m)

    # 4. predicted route, if given (list of node ids in order)
    if predicted is not None and len(predicted) > 1:
        fg = FeatureGroup(name="predicted route")
        folium.PolyLine([(nodes[n][1], nodes[n][0]) for n in predicted],
                        color="#2e7d32", weight=4, opacity=.9, dash_array="6").add_to(fg)
        fg.add_to(m)

    folium.Marker((poly[0][1], poly[0][0]), tooltip="start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((poly[-1][1], poly[-1][0]), tooltip="end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    LayerControl(collapse=False).add_to(m)

    out = os.path.join(OUT, f"trip_{name}.html")
    m.save(out)
    return out, name, len(track), len(route), len(arcs)


if __name__ == "__main__":
    trips = [os.path.join(ROOT, "train", "00000003"),
             os.path.join(ROOT, "train", "00000001"),
             os.path.join(ROOT, "student_test", "00000000")]
    rows = []
    for t in trips:
        out, name, nt, nr, na = make_map(t)
        rows.append((name, nt, nr, na, os.path.basename(out)))
        print(f"wrote {out}  ({nt} fixes, {nr} route roads, {na:,} roads in network)")

    html = ["<html><head><meta charset='utf-8'><title>PathSnap EDA maps</title></head><body>",
            "<h2>PathSnap - interactive trip maps</h2><ul>"]
    for name, nt, nr, na, fn in rows:
        html.append(f"<li><a href='{fn}'>trip {name}</a> - {nt} GPS fixes, "
                    f"{nr} true road segments out of {na:,} in the local network</li>")
    html.append("</ul></body></html>")
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print("wrote", os.path.join(OUT, "index.html"))
