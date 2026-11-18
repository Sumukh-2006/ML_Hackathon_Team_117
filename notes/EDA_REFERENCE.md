# EDA Reference — PathSnap

**Authority:** `docs/Dataset Exploration.pdf` (the brief). Everything below is either
**[BRIEF]** quoted from it, **[MEASURED]** re-derived from the shipped files, or
**[GAP]** something the brief asks for that we have not yet evidenced.

Generated 2026-11-18. Re-measure with the scripts in `notes/`.

---

## 0. The one structural fact everything else depends on

The four files do **not** share row indices. The brief states this explicitly
(§3: *"Do not assume that a row in a GPS file corresponds directly to a row in a
road-network file"*), and it is the most common way to get the whole pipeline wrong.

| file | rows are | row index means | join key |
|---|---|---|---|
| `.track` | GPS fixes, chronological | nothing — order is time | position only (no ids) |
| `.nodes` | road-network nodes | **`node_id`** (0-based) | `node_id` |
| `.arcs` | **directed** road segments | **`arc_id`** (0-based) | `arc_id`, plus `from_node_id`/`to_node_id` |
| `.route` | the ground-truth path | order of travel | values are `arc_id` |

⚠️ will be asked: *"How does a GPS fix become an arc?"* — Fix → candidate arcs by
distance → arc ids are indices into `.arcs` → arc endpoints index into `.nodes`.
No index is shared across files except through those two id spaces.

---

## 1. Brief-stated facts, re-verified

| Brief claim (§3, §7) | Measured on the 90 shipped trips | Verdict |
|---|---|---|
| `.track`: 3 whitespace cols, lon, lat, timestamp in seconds, no header | exactly 3 cols in all 90 files | ✅ |
| `.nodes`: 2 cols, lon, lat, 0-based row index | 2 cols, 0-based | ✅ |
| `.arcs`: 2 int cols, `from_node_id`, `to_node_id`, row index = `arc_id` | 2 int cols; row count = max route id + 1 in every trip | ✅ |
| `.route`: 1 int col, `arc_id` sequence | 1 col; all ids < len(arcs) | ✅ |
| "nominally 1 Hz: about 97% of steps are exactly 1 s apart" | **96.68%** pooled (219,934 / 227,487 steps) | ✅ brief is right |
| Partitions: 80 training / 10 student testing / 10 evaluator testing | 80 + 10 shipped; 10 evaluator not shipped | ✅ |
| No trip straddles partitions | 0 shared fixes; max cross-partition Jaccard 0.0 | ✅ |

**On the 97%:** pooled it is 96.68%, but per-trip the median is 99.1% and the worst trip
is 53.1%. The distribution is skewed by a few trips with long gaps, so the pooled number
is the honest one to quote. The two numbers are not in conflict — they answer different
questions. Say which one you mean.

---

## 2. Section 4 — required EDA scope (12 items)

The brief lists exactly what the analysis "should identify". This is the grading checklist.

| # | Brief asks for | Measured | Evidence | Status |
|---|---|---|---|---|
| 1 | Trips per partition | 80 / 10 / 10 | `FACTS.txt` | ✅ |
| 2 | GPS observations per trip | 227,577 total; min 262, median 1,496.5, max 19,313 | `FACTS.txt`, `eda_stats.csv` | ✅ |
| 3 | File availability and sizes | 4 files × 90 trips = 360; 94.1 MB | `FACTS.txt` | ✅ |
| 4 | Data types and column names | all numeric, no headers; 3/2/2/1 cols | §1 above | ✅ |
| 5 | Missing or invalid GPS observations | 0 NaN/inf; 0 duplicate (t, lon, lat) rows | `FACTS.txt` | ✅ |
| 6 | Timestamp ordering and irregular gaps | 0 non-increasing; 30/90 trips have a gap >30 s; largest 42.5 h | `FACTS.txt`, `quality_audit.csv` | ✅ |
| 7 | Geographical extent per trajectory | lon −100.3…137.2, lat −34.6…61.7; 27 countries; 25 UTM zone/hemisphere combos | `FACTS.txt`, `locations.csv` | ✅ |
| 8 | Nodes and segments per network | 2,024,174 nodes / 4,061,945 arcs; per trip 299–155,003 nodes | `FACTS.txt` | ✅ |
| 9 | **Road-network connectivity** | `network_cluster` populated for only **11/90** trips | `quality_audit.csv` | ⚠️ **GAP** |
| 10 | **Isolated nodes and disconnected components** | not measured per trip | — | ⚠️ **GAP** |
| 11 | Trips with unusual gaps or anomalies | flags on 55/90 trips; 3 trips with labels that stop covering the track | `quality_audit.csv`, `breakpoints_detail.csv` | ✅ |
| 12 | Availability and structure of reference route | present for all 90; 34,813 arcs; median 1.39% of the network (0.20–18.3%) | `FACTS.txt` | ✅ |

**Items 9 and 10 are the only two the brief names that we have not answered.** They are
cheap to close (one connected-components pass per trip over `arcs`) and they are named
explicitly in the brief, so a panel will notice their absence.

---

## 3. Section 5 — geographic requirements

| Brief demands | What we do | Status |
|---|---|---|
| Longitude/latitude are angular | handled; never treat degrees as metres | ✅ |
| Euclidean distance in degrees is not physical distance | haversine for all point-to-point distances | ✅ |
| Choose **and justify** a projected CRS or geodesic calc | one UTM zone per trip (25 distinct), never one global projection | ✅ |
| Check coordinate order carefully | verified col 0 = lon, col 1 = lat on all files (|lat| ≤ 61.7 fits both, so check per trip) | ✅ |
| Inspect plausibility before spatial work | node lon/lat ranges vs trip bbox | ✅ |
| Consider GPS noise, low sampling, gaps, irregular timestamps | residual to own road: median 4.7 m, p95 14.5 m; gap between fixes median 13.2 m | ✅ |

**Justification to have ready:** a single global projection is invalid here — the trips span
25 UTM zone/hemisphere combinations and 137° of longitude, so any one projection imposes
metre-level scale error somewhere. One UTM zone per trip keeps scale error under 0.1%
within each trip's own bbox. Alternative that lost: single Web Mercator — metre error grows
with latitude (up to ~2× at 60°N, and the dataset reaches 61.7°N).

---

## 4. Section 6 — reference routes are not verified truth

The brief's exact instruction: *"Do not assume that it represents independently verified,
error-free physical ground truth."* Evidence we can cite:

- 3 train trips (`00000002`, `00000033`, `00000054`) where the route stops covering the
  track partway through — the label degrades to a prefix of the drive.
- 10 trips whose route revisits the same arc (max 255 repeats).
- Route length vs drive length ratio is not 1.0 on every trip.
- GPS fixes sit a median 4.7 m off their own reference arc, p95 14.5 m — some mismatch is
  measurement noise, not label error, but the two cannot be separated.

⚠️ will be asked: *"If the labels can be wrong, how do you know your accuracy number is
real?"* — Answer: report the 3 suspect trips separately and state that they cap achievable
score; score the other 87. Do not silently drop them.

---

## 5. Section 7 — partitions and leakage rules

| Rule (verbatim from brief) | How we comply | Status |
|---|---|---|
| No trip split across partitions | trip = atomic unit; 0 shared fixes | ✅ |
| Evaluator data not used for training | not shipped, not present locally | ✅ |
| Evaluator labels not used for feature selection | never opened | ✅ |
| Evaluator results not used to choose algorithms/hyperparameters | all model choice on the 80 train/split | ✅ |
| No manual altering of predictions from known evaluation routes | — | ✅ |
| No hardcoded expected answers | — | ✅ |
| Learned preprocessing fit on training data only | ⚠️ must confirm for any fitted scaler/projection learned from labels | ⚠️ **CHECK** |
| Keep evaluator ground truth out of the prediction pipeline | separate path, never imported | ✅ |

### The leak that actually exists

The given tracks are the public Zenodo tracks plus i.i.d. Gaussian noise (σ ≈ 5.1 m per
axis, zero bias, same point count, same timestamps). That public archive also contains the
**reference route for all 10 evaluator trips** (IDs `15, 17, 23, 30, 49, 59, 67, 73, 87, 94`).

The brief forbids using them. Treat `data/non-given-dataset/` as usable only for
understanding the noise model and building training pairs — never as a lookup for a
scoring trip. If asked "why is that file in your repo?", the answer is "to characterise
the perturbation", and it must not be imported anywhere in the scoring path.

---

## 6. Section 8 — six required outputs

| # | Brief asks for | Artifact | Status |
|---|---|---|---|
| 1 | Dataset inventory: trips, files, observations | `FACTS.txt` | ✅ |
| 2 | Summary stats: trajectory length, sampling intervals, missing values | `eda_stats.csv`, `quality_audit.csv` | ✅ |
| 3 | Plots: coordinate distributions and trajectory patterns | `figs/` (24 PNGs) | ✅ |
| 4 | Folium maps of trajectories + network | `maps/index.html` + per-trip | ✅ |
| 5 | Tables/visualisations of network size and connectivity | size ✅, **connectivity ⚠️** | ⚠️ partial |
| 6 | Short record of data-quality issues, assumptions, decisions | `preprocessing_log.md` — **still a blank template** | ⚠️ **GAP** |

Item 6 is due and is the one the brief says to fill *as you work*. It currently has
`___` placeholders for rows, features, split rationale.

---

## 7. Section 9 — exit criteria (5 questions you must be able to answer)

1. **How is the raw data organised?** Per trip: one `.track`, one `.nodes`, one `.arcs`, one `.route`. 4 files × 90 trips. Each trip has its *own* road network — they are not shared.
2. **What does each file represent?** See §0 table.
3. **How do the components relate?** `.route` values index `.arcs`; `.arcs` columns index `.nodes`; `.track` connects to the network only through matching, never through an id.
4. **Main data-quality concerns?** 5 m GPS noise vs 25 m median segment length; 30/90 trips with >30 s gaps; 3 trips with labels that stop early; 62.1% of reference arcs have a geometrically identical reverse twin.
5. **Which assumptions drive the approach?** Per-trip UTM; heading needed to break twin ties; route labels are reference, not truth.

---

## 8. Numbers stated in different places that need one agreed value

Quoting two different numbers for the same thing is the fastest way to lose credibility in
a viva. These three need a single stated denominator:

| Claim | Value A | Value B | Resolution needed |
|---|---|---|---|
| Candidate recall, 50 m / K=10 | 99.3% ("clean trips", chat summary) | 96.19% ("all fixes", `FACTS.txt`: 3.81% miss) | state: all 227,577 fixes or clean-trip subset; strict or twin-counted |
| Reverse-twin share | ~70% (median trip) | 62.1% pooled route arcs; 92.3% of all graph arcs | say which population: reference arcs vs whole graph |
| 1 Hz sampling | ~97% (brief, pooled) | 99.1% (median per trip) | pooled is the defensible one |

Also unresolved: `FACTS.txt` says median 1.39% of the network is covered by the route, but
one trip reaches 18.3% — before quoting a "route is a tiny fraction" argument, check whether
that wide trip is a short network rather than a long route.

---

## 9. Remaining work, in priority order

1. Connected components + isolated-node counts per trip → fills brief items 9 and 10. (~30 min)
2. Fill `preprocessing_log.md` — it is a graded deliverable and is currently empty.
3. Resolve the three number conflicts in §8 to one value each.
4. Confirm no fitted preprocessing touches labels (leakage rule 7).
5. Re-run the viva skill before declaring EDA done.
