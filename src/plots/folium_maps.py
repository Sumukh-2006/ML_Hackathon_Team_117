"""Folium maps of a corpus trip: road network, reference route and GPS fixes.

Why this file exists: the report needs geographical maps that nobody drew by hand, and the
submission needs the code that produced them. One page per trip, plus an index.

Leaflet is SVG by default, which makes a page with a few thousand polylines unpleasant to
pan and zoom. Three choices keep it responsive:
  - `prefer_canvas`, so the vector layers paint to one canvas instead of to the DOM;
  - the road network goes in as a single MultiLineString, not one polyline per road;
  - fixes are drawn as one path plus a decimated dot layer, not one marker per fix.
Coordinates are rounded to 5 decimals, about 1.1 m, which is finer than the GPS noise.

Run: uv run python -m plots.folium_maps <corpus root> <output dir> [options]
     uv run python -m plots.folium_maps data/corpus docs/figures/maps
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import folium
from folium import FeatureGroup, LayerControl

from dataset import Corpus, Folder, RoadGraph, Route, Track

RADIUS_M = 700.0     # OWN CHOICE: how far from the drive the drawn network reaches
NET_CAP = 12000      # OWN CHOICE: road segments sent to the browser
DOT_CAP = 1500       # OWN CHOICE: fixes drawn as dots; the polyline carries the path
SAMPLES = 120        # OWN CHOICE: fixes used to find the nearby network
DP = 5               # decimals of a degree; 1e-5 is about 1.1 m


def segment_distance(px, py, ax, ay, bx, by) -> np.ndarray:
    """Distance from points to segments, all arrays, broadcasting on a shared shape."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denominator = abx * abx + aby * aby
    t = np.clip((apx * abx + apy * aby) / np.maximum(denominator, 1e-18), 0.0, 1.0)
    return np.hypot(apx - t * abx, apy - t * aby)


def nearby_arcs(graph: RoadGraph, track: Track, radius: float = RADIUS_M,
                cap: int = NET_CAP) -> np.ndarray:
    """Arc ids within `radius` of the drive, capped for page size."""
    ax = graph.node_x[graph.arc_from]
    ay = graph.node_y[graph.arc_from]
    bx = graph.node_x[graph.arc_to]
    by = graph.node_y[graph.arc_to]

    step = max(1, track.x.size // SAMPLES)
    px, py = track.x[::step], track.y[::step]

    keep = np.zeros(ax.size, dtype=bool)
    # 4 fixes at a time: the temporary is 4 x arc count, which stays small on a
    # 300 000 arc network while still being vectorised.
    for start in range(0, px.size, 4):
        d = segment_distance(px[start:start + 4, None], py[start:start + 4, None],
                             ax[None, :], ay[None, :], bx[None, :], by[None, :])
        keep |= (d < radius).any(axis=0)

    index = np.flatnonzero(keep)
    if index.size > cap:
        index = index[np.linspace(0, index.size - 1, cap).astype(int)]
    return index


def route_parts(graph: RoadGraph, route: Route) -> list[list[list[float]]]:
    """The reference route as [lon, lat] parts, split where the arcs do not connect."""
    parts: list[list[list[float]]] = []
    previous_end = -1
    for arc in route.arcs.tolist():
        start, end = int(graph.arc_from[arc]), int(graph.arc_to[arc])
        if start != previous_end:
            parts.append([[float(graph.node_lon[start]), float(graph.node_lat[start])]])
        parts[-1].append([float(graph.node_lon[end]), float(graph.node_lat[end])])
        previous_end = end
    return parts


def rounded(points) -> list[list[float]]:
    return [[round(float(lon), DP), round(float(lat), DP)] for lon, lat in points]


def build(name: str, folder_path: Path, out: Path, radius: float = RADIUS_M) -> dict | None:
    graph: RoadGraph = RoadGraph.from_folder(folder_path)
    folder: Folder = Folder(split=folder_path.parent.name, path=folder_path)
    track: Track = folder.track(graph)

    centre = [float(np.mean(track.lat)), float(np.mean(track.lon))]
    page = folium.Map(location=centre, zoom_start=13, control_scale=True,
                      tiles="OpenStreetMap", prefer_canvas=True)

    arcs = nearby_arcs(graph, track, radius)
    network = {
        "type": "Feature",
        "properties": {"kind": "road"},
        "geometry": {
            "type": "MultiLineString",
            "coordinates": [
                [[round(float(graph.node_lon[graph.arc_from[a]]), DP),
                  round(float(graph.node_lat[graph.arc_from[a]]), DP)],
                 [round(float(graph.node_lon[graph.arc_to[a]]), DP),
                  round(float(graph.node_lat[graph.arc_to[a]]), DP)]]
                for a in arcs.tolist()],
        },
    }
    group = FeatureGroup(name=f"road network near the drive ({arcs.size:,} of "
                              f"{graph.arc_count():,} segments)")
    folium.GeoJson(network, smooth_factor=2.0,
                   style_function=lambda _: {"color": "#8d8d8d", "weight": 1.1,
                                             "opacity": 0.55}).add_to(group)
    group.add_to(page)

    route_arcs = 0
    if folder.has_route():
        route = folder.route(graph)
        route_arcs = int(route.arcs.size)
        group = FeatureGroup(name=f"reference route ({route_arcs} segments)")
        folium.PolyLine([(lat, lon) for part in route_parts(graph, route)
                         for lon, lat in part],
                        color="#d32f2f", weight=4, opacity=0.9).add_to(group)
        group.add_to(page)

    group = FeatureGroup(name=f"GPS fixes ({track.lon.size:,})")
    folium.PolyLine(rounded(zip(track.lon, track.lat)), color="#1565c0",
                    weight=1.4, opacity=0.55).add_to(group)
    group.add_to(page)

    stride = max(1, int(np.ceil(track.lon.size / DOT_CAP)))
    shown = np.arange(0, track.lon.size, stride)
    group = FeatureGroup(name=f"GPS fixes as dots ({shown.size:,} of {track.lon.size:,})")
    folium.GeoJson(
        {"type": "Feature", "properties": {},
         "geometry": {"type": "MultiPoint",
                      "coordinates": rounded(zip(track.lon[shown], track.lat[shown]))}},
        smooth_factor=2.0,
        marker=folium.CircleMarker(radius=2.2, color="#1565c0", weight=0, fill=True,
                                   fill_opacity=0.8),
    ).add_to(group)
    group.add_to(page)

    folium.CircleMarker((round(float(track.lat[0]), DP), round(float(track.lon[0]), DP)),
                        radius=7, color="#2e7d32", weight=2, fill=True, fill_opacity=0.9,
                        tooltip="start").add_to(page)
    folium.CircleMarker((round(float(track.lat[-1]), DP), round(float(track.lon[-1]), DP)),
                        radius=7, color="#c62828", weight=2, fill=True, fill_opacity=0.9,
                        tooltip="end").add_to(page)
    LayerControl(collapse=False).add_to(page)

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.html"
    page.save(str(path))
    return {"name": name, "path": path, "fixes": int(track.lon.size),
            "route_arcs": route_arcs, "arcs_total": graph.arc_count(),
            "arcs_drawn": int(arcs.size), "kb": round(path.stat().st_size / 1024)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path, help="corpus root: <split>/<id>/<id>.*")
    parser.add_argument("out", type=Path, help="output directory for the pages")
    parser.add_argument("--split", default=None, help="only this split, e.g. student_test")
    parser.add_argument("--trips", type=int, default=3, help="how many trips (0 = all)")
    parser.add_argument("--radius", type=float, default=RADIUS_M,
                        help="metres around the drive to draw")
    args = parser.parse_args(argv)

    corpus = Corpus.from_root(args.corpus)
    chosen: list[Folder] = []
    for split, folders in corpus.splits.items():
        if args.split and split != args.split:
            continue
        chosen.extend(folders)
    if not chosen:
        raise SystemExit(f"no folders in {args.corpus}"
                         f"{' for split ' + args.split if args.split else ''}")
    if args.trips:
        chosen = chosen[:args.trips]

    rows = []
    for folder in chosen:
        row = build(folder.name(), folder.path, args.out, args.radius)
        if row:
            rows.append(row)
            print(f"{row['name']}  {row['fixes']:,} fixes  "
                  f"{row['arcs_drawn']:,}/{row['arcs_total']:,} segments  "
                  f"{row['kb']} KB", flush=True)

    index = args.out / "index.html"
    links = "\n".join(
        f"<li><a href='{row['path'].name}'>{row['name']}</a>: {row['fixes']:,} fixes, "
        f"{row['route_arcs']} reference segments, {row['arcs_drawn']:,} of "
        f"{row['arcs_total']:,} road segments drawn</li>"
        for row in rows)
    index.write_text(
        "<!doctype html><meta charset='utf-8'><title>PathSnap maps</title>"
        f"<h1>PathSnap maps</h1><ul>{links}</ul>", encoding="utf-8")
    print(f"wrote {index}")


if __name__ == "__main__":
    main()
