# Quality & leakage audit — 90 map-matching drives

Script: `notes/quality_audit.py` (re-run: `python notes/quality_audit.py`).
Outputs: `notes/quality_audit.csv` (one row per drive, 35 columns), `notes/figs/09_leakage_quality.png`.
Nothing else was modified.

**Short version.**
Leakage: no test drive shares a single road node with any train drive, and no drive appears twice —
so `student_test` is a clean holdout. Inside train, 13 of 80 drives have a near-twin, so a *random*
training/validation split over-states accuracy; split by network.
Labels: 11 of 90 drives have a disconnected ground-truth route and 3 label under 90% of their GPS
points at 50 m, but 2 of those 3 are a parked car with GPS jitter, not a wrong road. Only
`00000002` is a genuine label error.
Tracks: every drive is a continuous 1 Hz session, but 30 have a >30 s time gap and 23 contain implied
speeds above 200 km/h.

---

## 1. Leakage between train and test

### 1.1 Method

Two drives share a road network if their **node coordinate sets** coincide. Node ids cannot be used
(they are line numbers, renumbered per file), so all 90×89/2 pairs were compared on coordinates
rounded to 1e-5° (≈1.1 m), giving an exact shared-node count per pair.

For the pairs that share nodes I also measured **route overlap**: the share of the shorter labelled
route that lies within 10 m of the other drive's route. That is the number that matters for leakage —
two drives can use the same network and different roads.

### 1.2 What I found

| group | drives | shared nodes | node Jaccard | extracted | route overlap |
|---|---|---|---|---|---|
| 1 | 00000010, 00000016, 00000029 | 8182 (10–16), 5488 (10–29), 5251 (16–29) | 0.96 / 0.40 / 0.38 | Krasnogorsk (Moscow Oblast) | 57% / 9% / 26% |
| 2 | 00000032, 00000045 | 2034 (= all of 45) | containment 1.00 | Treuenbrietzen (Brandenburg) | 0% |
| 3 | 00000053, 00000093 | 43 819 | 0.55 | Medio Cudeyo (Cantabria) | 43% |
| 4 | 00000062, 00000086 | 25 399 | 0.21 | Yekaterinburg | 1% |
| 5 | 00000075, 00000097 | 1022 (= all of 75) | containment 1.00 | Horsens / Helsinge (Denmark) | 0% |
| — | 00000043, 00000054 | 9 400 | 0.06 | Utrecht / Amsterdam, 28.6 km apart | 20% |

* `00000010` and `00000016` — the pair spotted by eye — **are the same network**: 96% of the node
  sets coincide, 8182 nodes are identical to the metre, and every node of 16 has a partner in 10.
  They are *not* the same file: the node numbering is different (1% of indices agree) and the node
  counts differ (8502 vs 8205), i.e. two separate OSM extractions of one area, plus a third
  (`00000029`) 1.8 km away.
* They are **not the same drive**. Route arrays differ (210 vs 204 arcs), no route geometry hash
  matches anywhere in the 90 files, durations differ (2711 s vs 1305 s) and the longest common
  contiguous stretch is 52 nodes ≈ 25% of the route. They overlap for 57% of their length because
  both use the same suburban roads out of Krasnogorsk.
* Groups 2 and 5 are **nested extracts**: the smaller drive's network is a strict subset of the
  larger one, but the two drives use completely different roads, so they carry no label leakage.
* **No duplicate drive exists anywhere.** No two files share an arc sequence, no two share identical
  route geometry, no two share a track signature (start/end coordinates + sample count), and all 90
  `.nodes` and `.arcs` files are byte-unique.

### 1.3 Test vs train

| test drive | region | nearest train drive | shared nodes |
|---|---|---|---|
| 00000040 | Chelyabinsk Oblast (RU) | 00000028 (same oblast, 28 km) | **0** |
| 00000065 | Samara Oblast (RU) | 00000066 (same oblast, 28 km) | **0** |
| the other 8 | — | — | **0** |

Every one of the 10 test drives shares **0 nodes with all 80 train drives summed**; the maximum node
Jaccard over all 800 train–test pairs is 0.0000 and the maximum route overlap is 0.0000.
So the two test drives whose *administrative region* also occurs in train are testing a different
extraction of it — a familiar region, not a familiar network. That is the honest generalisation test.

⚠️ will be asked: *"Isn't a random split fine then?"* — It is fine for the *test* set, which is
already disjoint. It is not fine for a train/validation split: 13 of the 80 train drives have a
near-twin in train, so a random 80/20 split will place roughly 3 of them on the wrong side and
inflate validation by an unmeasured amount (up to those 3 drives' weight in a 16-drive validation set).

### 1.4 Verdict

> A random train/validation split is **not** safe: no test drive shares any road with train (0 shared
> nodes in all 800 pairs), but 13 of the 80 train drives reuse another train drive's network — up to
> 96% of nodes and 57% of the route — so validation must be grouped by network/region, and
> `student_test` can be trusted as the untouched holdout.

Practical rule: cluster by shared-network (the 5 groups + the Utrecht/Amsterdam pair), put whole
clusters on one side of the split, and report the metric per cluster as well as overall.

---

## 2. Label quality

### 2.1 Route gaps — 11 of 90 drives

A gap is a place where `arcs[route[i]].to != arcs[route[i+1]].from`, i.e. the label teleports.

| drive | gaps | largest jump (m) |
|---|---|---|
| 00000019 | 3 | 596 |
| 00000002 | 1 | 325 |
| 00000080 | 1 | 108 |
| 00000086 | 1 | 98 |
| 00000033 | 1 | 92 |
| 00000075 | 2 | 73 |
| 00000041 | 1 | 57 |
| 00000069 | 1 | 41 |
| 00000042 | 1 | 31 |
| 00000089 | 1 | 29 |
| 00000076 (test) | 1 | 15 |

All 11 are short hops (15–596 m) at junctions where the route steps across a node the extraction
split differently. They do not move the route to another street, but any code that treats the route
as one polyline will draw a false chord across them.

Cross-check: `notes/route_breaks.csv` (produced by a separate analysis in this repo) finds the same
14 gaps in the same 11 trips with the same distances, and reports that every gap is reconnectable
inside the graph — i.e. the label steps over a junction node, it does not lose a road segment.

### 2.2 Label coverage — and a correction to the 2 km rule

Coverage = share of GPS points within 50 m / 200 m of the labelled route, with the **gap chords
removed** (segment *i+1* after a gap at *i*).

⚠️ assumption: the earlier rule — *"exclude any polyline segment longer than 2 km"* — does not do
what it was meant to do on this dataset. All 11 gap chords are 15–596 m long, so the 2 km rule
deletes **none** of them, and instead deletes genuine long highway arcs:

| drive | chord-excluded (correct) | 2 km rule | why they differ |
|---|---|---|---|
| 00000020 | 0.985 | 0.825 | route has 2 arcs of 3–4.7 km carrying 16% of the points |
| 00000066 | 0.957 | 0.865 | same, 2 arcs > 2 km |
| 00000002 | 0.878 | 0.922 | the 2 km rule leaves a 325 m gap chord in, which falsely covers 4% of points |
| 00000000 | 1.000 | 0.937 | long arc deleted |
| 00000005 | 1.000 | 0.941 | long arc deleted |
| 00000080 | 0.976 | 1.000 | gap chord removed (108 m), which is correct |

Both columns are in the CSV (`cov50` and `cov50_2km`). Under the correct rule **3 drives** fall below
0.90, not 5.

### 2.3 Only one of the three is a real label error

⚠️ will be asked: *"Is a point 200 m from the label a labelling error or a parked car?"*
I separated the two by asking where the off-label points sit: if >50% of them fall in a single
100 m cell, it is one parked vehicle, not a drive along a different road.

| drive | cov50 | cov50 on moving points | off-label points in one 100 m cell | verdict |
|---|---|---|---|---|
| 00000054 | 0.128 | 0.877 | 83% | **parked, not mislabelled** |
| 00000033 | 0.412 | 0.944 | 85% | **parked, not mislabelled** |
| 00000002 | 0.878 | 0.885 | 21% | **real label error** |

* `00000054` — 19 313 samples over 5.4 h. 87% of them lie inside a 400 m disc for **4.7 h**: a car
  parked in a lot, its median fix 86 m from the nearest mapped road and 120 m from the labelled
  route, with jitter reaching 250 m. The label covers the 13% that actually drove (those sit 4 m
  from the route, median). Useless for training as-is; do not "fix" the label.
* `00000033` — same story, 1.2 h of a 1.7 h session parked, 6.9 m from the nearest mapped road.
* `00000002` — genuine truncation: the last **552 samples (637 s) of the drive, up to 570 m away**,
  are not labelled at all. The route ends while the track continues; the missing part is 91 m from
  any mapped road, so the label simply stops early. Also one 325 m route gap.

Six drives have a parked blob (00000020, 00000033, 00000054, 00000066, 00000080, 00000096) but only
those three lose coverage, because in the other four the park happens within 50 m of the label.

⚠️ will be asked: *"Why not just drop the parked samples?"* — Because the parked samples are where
the GPS error is largest (up to 250 m), so a model trained on them learns the wrong noise level.
Drop them, and report coverage on moving points only.

---

## 3. Track-session anomalies

| defect | drives affected | worst case |
|---|---|---|
| time gap > 30 s | **30** / 90 | 153 064 s (42.5 h) in `00000090`; 52 810 s in `00000078`; 2 658 s in `00000069` |
| position jump > 200 m between consecutive fixes | **11** / 90 | 888 m `00000055`, 638 m `00000043`, 541 m `00000070`, 510 m `00000054` |
| implied speed > 200 km/h | **23** / 90 | 648 km/h `00000079`, 511 `00000033`, 510 `00000013` |
| duplicate consecutive points | **5** / 90 | `00000084` (2), `00000002`, `00000046`, `00000062`, `00000085` (1 each) |
| route gaps | **11** / 90 | see §2.1 |
| label coverage < 90% | **3** / 90 | see §2.3 |

Also checked and clean in all 90 files: no duplicated timestamps, no NaNs, no out-of-range
coordinates, `t` always strictly increasing. Sessions start at `t = 1 s` or an arbitrary offset
(`00000006` at 693 s, `00000022` at 90 s) — treat `t` as relative, not as wall-clock.

⚠️ will be asked: *"Are those 23 speed spikes real?"* — Mostly not. At 1 Hz, 200 km/h is 55 m
between fixes, which a noisy receiver reaches on a single sample. Only `00000032` is sustained:
610 consecutive samples above 200 km/h, median implied speed 147 km/h on a rural road, which points
at a coarse or drifting position fix rather than at traffic. Count those as data-quality flags, not
as ground-truth errors.

---

## 4. Do the 10 test `.route` files look like real answer keys?

Yes. Five independent properties hold:

1. **They are valid connected walks** — 9 of 10 have zero discontinuity; `00000076` has one 15 m
   hop. That is what a *label produced by map-matching* looks like. (It is also what a trivially
   "connectivity-repaired" fake would look like, so this alone proves nothing.)
2. **They follow the GPS in time.** Spearman correlation between route progress and timestamp is
   0.998–1.000 for 9 drives (`00000056` = 0.754 because it revisits the same roads), and 96.5–100%
   of consecutive points move forward along the route. A synthetic or shuffled key would not.
3. **They explain the GPS.** Coverage at 50 m is 0.993–1.000 for all 10 test drives (worse than
   the train median only for `00000056`, 0.993).
4. **They start and end where the recording does** — 3–94 m at the start, 9–294 m at the end.
5. **The speeds are plausible** — 14.7 to 73.7 km/h average over the labelled length.

The defect profile of the test routes also matches train (3 drives with a >200 km/h spike, 2 with a
>30 s time gap, 1 route gap), which is what you expect if all 90 files came from one generation
process. So: genuine keys, produced the same way as train, including the same imperfections.

---

## 5. Defect list (what to act on)

| action | drives |
|---|---|
| **drop or repair** (label truncated) | `00000002` |
| **drop the parked tail** (keep the moving part) | `00000054`, `00000033` |
| keep, but break the polyline at the gap | `00000019`, `00000080`, `00000086`, `00000033`, `00000075`, `00000041`, `00000069`, `00000042`, `00000089`, `00000002`, `00000076`(test) |
| split across validation, never same side | {`00000010`,`00000016`,`00000029`}, {`00000053`,`00000093`}, {`00000062`,`00000086`}, {`00000032`,`00000045`}, {`00000075`,`00000097`}, {`00000043`,`00000054`} |
| treat as slow/coarse GPS, check before trusting speeds | `00000032`, `00000013` |

Everything else is usable as-is: 70 of 90 drives (61 of 80 train, 9 of 10 test) have no route gap,
no coverage loss below 90% and no parking blob.

---

## 6. Figure

`notes/figs/09_leakage_quality.png` (14×4.6 in, 120 dpi, 4 panels, fonts ≥ 5.6 pt inside the map
key and ≥ 9.8 pt for titles):

1. World map of the 90 centroids, train vs test, with a zoom on Europe/Urals — the only place where
   networks are shared. The zoom rings every drive whose network coincides with another's and names
   the groups; no test triangle is ringed, including `00000052` (Tver), which lies 122 km from the
   nearest Moscow drive with 0 shared nodes.
2. Sorted label coverage for all 90 drives, worst at the bottom, with the three <90% drives named.
3. Counts of drives affected by each defect, printed at the bar ends.
4. `00000054` drawn at 1:1 — labelled route, GPS below 10 km/h (grey) and GPS moving (blue) — with
   the 4.7 h parked cloud visible beside the road it never reached.

Font sizes: 10 pt for panel titles, 9.5 pt for axis labels, 7.5–9.5 pt for ticks and legends,
7–8 pt for the two dense map keys. The 10 pt floor holds everywhere except inside the world map,
where 90 points and 5 labelled groups in a 3-inch panel do not fit at 10 pt; those keys are 7 pt and
were checked at 4× zoom in the PNG.

The PNG was read back and inspected four times; earlier versions had clipped titles, a legend over
the data and a key box over points, all fixed. Final state: no clipped text, no annotation over a
data point, every axis carries units, and the colour palette is Okabe-Ito (blue/vermillion), which is
safe for the common colour-vision deficiencies.

## 7. What this audit does not cover

* Arc *direction* consistency: I checked that the route is a connected walk, not that each arc is
  traversed in its declared direction (the `arcs` file is directed, so this matters for a
  turn-aware model).
* Whether the GPS track's *order* is consistent with the label when the route revisits a road
  (`00000056`, `00000043`, `00000054`) — progress is monotone only up to the revisits.
* Per-drive duplicates *within* a network at looser than 1 m tolerance; 1 m is the resolution the
  coordinate strings support.
