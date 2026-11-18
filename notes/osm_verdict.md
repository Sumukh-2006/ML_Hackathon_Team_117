# Are these road graphs OpenStreetMap-derived?

Measured on all 90 drives (2,024,174 nodes / 4,061,945 arcs) from the four provided files
only. Script: `notes/osm_check.py`. Figure: `notes/figs/07_osm_fingerprint.png`.
Nothing below uses the internet, OSM tags, or `locations.csv` except for the city names.

## Measured

- **One-way share** (share of arcs with no reverse twin arc): median **4.9%**, mean 7.8%,
  quartiles 1.6% / 4.9% / 10.9%, range **0.0% - 64.9%**. Top: `00000080` Buenos Aires 64.9%,
  `00000035` Rome 32.8%, `00000074` 25.5%, `00000054` Amsterdam 23.5%. Bottom: 10 of 90 drives
  below 0.5%, two of them exactly 0.0% (`00000070`, `00000085`).
- **Degree (undirected neighbours)** — mean over drives: degree 1 **6.5%**, degree 2
  **72.9%** (median 73.5%, range 30.9% - 94.4%), degree 3+ **20.5%**.
  Counting *directed* degree instead gives deg-2 = **16.0%**, which is where the earlier
  "0.05-0.38" came from: a two-way street's mid-point has in-degree 2 + out-degree 2 = 4.
- **Arc length**: pooled median **26.4 m**, p95 **148.9 m**. Per-instance medians 6.7 - 84.9 m,
  p95 55 - 640 m, longest single arc 9.3 km. Continuous, heavy-tailed, no preferred spacing.
- **Junction-to-junction stretch** (contract every degree-2 node): median **79.5 m**, built
  from **3.1 arcs** on average. So a "block" is ~80 m of road chopped into ~3 arcs.
- **Geometry of mid-road nodes**: 65% of degree-2 nodes turn more than 5 deg from straight
  (median deviation 8.9 deg). These are real bends, not redundant collinear points.
- **Junction angles**: at degree-3 nodes the sharpest pair of arms has median **85.1 deg**;
  **76%** of them are within 90 +/- 20 deg - i.e. T-junctions and grid corners.
- **Connected components**: median **19** per drive (range 1 - 421), largest component
  holds **98%** of nodes. Small detached fragments are present everywhere.
- **Dead ends**: 6.5% of nodes have degree 1, and only **17.7%** of those lie within 2% of
  the bounding-box edge - so they are interior cul-de-sacs, not arcs cut by the crop.
- **Degenerate arcs**: 24 self-loops (`u == v`) and 1,903 duplicate arcs out of 4.06 M, plus
  1,397 arcs shorter than 0.5 m. Rare but non-zero - consistent with un-cleaned source data.
- **Coordinates**: exactly **6 decimal places** in all 90 `.nodes` files (and in `.track`).
  1e-6 deg ~ 0.11 m, so every file was written by an exporter that rounds.
- **No road-class filtering by length**: `00000001` is driven at 133 km/h and has median arc
  **20.2 m**; Amsterdam `00000054` has **20.4 m**. Same distribution, so arcs are not
  motorway-vs-street filtered.

## Verdict

**(a) Consistent with OSM: yes, strongly.** Five independent fingerprints agree:
per-city one-way rates spanning 0-65% (a generator would not know that Buenos Aires is a
one-way grid and rural Brandenburg is not); heavy-tailed arc lengths with a ~26 m median,
the resolution OSM stores way geometry at; 76% right-angled T-junctions; dozens of small
disconnected fragments; and interior cul-de-sacs. A synthetic generator reproduces none of
these simultaneously - it would give uniform arc lengths, near-constant one-way share, and
no detached fragments.

**(b) Which kind: OSM-derived, geometry preserved, then re-exported.** Specifically:
- *Not* a raw OSM export verbatim: no tags/ids survive and coordinates are rounded to 6 dp.
  A verbatim export of every OSM way would also be dominated by footway/driveway/building
  vertices and sit well above 73% degree-2 (`unverified:` magnitude, needs an OSM extract).
- *Not* junction-contracted: 73% of nodes are mid-road, a contracted graph would be near 0%.
  So each OSM way is stored as its original polyline, one arc per segment.
- The Buenos Aires drive (65% one-way, only 31% degree-2, 84 m median arc, 113 m stretches)
  is the one instance that *looks* contracted - which is what a one-way grid looks like after
  geometry simplification, not evidence of a different pipeline.
- Conclusion: an OSM extract of drivable roads, lightly simplified and rounded, keeping way
  polylines. `unverified:` whether a `highway=*` class filter was applied and which classes.

**(c) What would settle it with internet access.** Take one small drive - `00000045`
Treuenbrietzen, 2,034 nodes, 4,335 arcs, bbox = the file's own lat/lon min-max - and run:
1. **Node match rate**: download the Geofabrik Germany extract, clip to that bbox, and for
   each dataset node find the nearest OSM node. If >90% of dataset nodes land within 0.2 m of
   an OSM node, the graph is a direct derivative; if only junctions match, it was simplified.
   Do the same after contracting degree-2 chains, to separate "rounded" from "rebuilt".
2. **One-way share check**: compute `one-way highway length / total highway length` for the
   same bbox from OSM. Our 4.9% median and 64.9% Buenos Aires figure should reproduce within
   a few percent. A mismatch of >5 points would falsify the OSM hypothesis.
3. **Arc-length equality test**: for the matched ways, compare the OSM segment lengths to ours
   per arc. Identical to ~0.1 m = the polylines were copied, not resampled.
4. **Road-class census**: count OSM km per `highway=*` value in the bbox and compare with the
   graph's total length; this is the only way to confirm or refute the class filter.

## Measured vs inferred

| Claim | Status |
|---|---|
| one-way share, degree mix, arc lengths, components, junction angles, precision | **measured** (files only) |
| arcs are OSM way segments | **inferred** from degree-2 fraction + arc length + bend angles |
| source is OSM rather than another road dataset (TomTom/HERE/OSRM-derived) | **inferred**, not proven; the specific test is (c) |
| city names next to instance ids | inherited from `notes/locations.csv` (earlier reverse geocoding), not re-checked here |
| road-class filter (motorway/primary/... only) | **unverified** |

## One correction to earlier numbers

The earlier "deg-2 fraction 0.05-0.38, i.e. NOT raw OSM" was an artefact of counting
*directed* degree: a mid-road node on a two-way street has 4 directed arc-ends, so it lands
in the degree-4 bin. Counted as undirected neighbours the same nodes are **73% degree-2**,
which is exactly what "OSM way polylines kept as-is" produces. The conclusion flips:
the graphs are per-way-segment, not contracted.
