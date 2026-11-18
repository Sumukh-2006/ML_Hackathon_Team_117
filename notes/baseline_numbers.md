# Baseline numbers - independent nearest-arc snapping

Built by `notes/baseline_check.py`; figure `notes/figs/08_baseline.png`. Each GPS point is snapped to the geometrically nearest road segment on its own: no previous point, no network connectivity, no smoothing, no candidate set. This is the floor every real map-matcher has to clear.

Coverage: 90 drives, 227,577 GPS points (every point, no subsampling), 4,061,945 road segments, 34,491 distinct true-route segments. Runtime 40 s.

## Two ways to count a hit - read this before any number below

92% of the arcs in these graphs have a reverse twin: the same geometry stored as `(u,v)` and `(v,u)`. Both sit at exactly the same distance from a GPS point, so an argmin picks one by arc id - a coin flip. Every score below is therefore given twice:

- **strict**: the snapped arc must equal the route arc, direction included. This is the literal definition and the metric a submission is graded on.
- **direction ignored**: the snapped arc or its reverse twin counts. This separates 'snapped to the wrong street' from 'snapped to the right street the wrong way round'.
- Exact ties are broken by lowest arc id, reproducibly. Because the twins are geometrically identical, the strict score still depends on the tie rule: two arbitrary-but-consistent rules gave 57% and 63% strict recall on these 90 drives. Treat strict recall as 57% +/- 6 points and never compare two methods whose tie rules differ; the direction-ignored number is stable.

## 1. Point-level accuracy (the generous number - quote it only with this warning)

- Strict: **61.8%** of GPS points land on an arc that is in the true route set. Direction ignored: **89.5%** (10th-90th pct 75-98%).
- The 28-point difference between those two numbers is pure direction luck, not a modelling failure - the geometry is identical in both cases.
- Even the generous version is an **upper bound, not an achievement**. The true route set is hundreds of arcs long; a hit means the point landed on *any* of them, which forgives being several arcs away from the right one. A matcher can score high here and still output a sequence that turns the wrong way.
- The metric cannot see the real failure, which is coverage: it averages over points, and points are exactly what the baseline is allowed to use.

## 2. Arc-level recall (the honest number)

- Strict: **57.0%** of true route arcs have at least one GPS point snapped to them. Direction ignored: **88.7%**. Strict arc F1 54.3% (precision 52.0%).
- Spread: strict recall runs 34% (worst drive) to 100% (best), 10th-90th percentile 43-82%. Direction ignored: 56-100%.
- **36 route arcs per median drive get no point at all** (median route length 383 arcs, so about 1 in 11 of the road it drove), even with direction ignored. Those arcs are never the nearest thing to any sample.
- Direction costs a further 32 points of recall. Both failures need the same fix: information from the *sequence* of points, not from the individual point.
- Mechanism (a) **sampling**: GPS spacing is 13 m on the median drive against a median arc length of 25 m; on 11 of 90 drives the spacing exceeds the median arc, so short arcs are stepped over without a point landing on them (correlation of spacing/arc-length with recall: r = -0.54). Mechanism (b) **ambiguity**: the point is sometimes closer to a parallel street, and with no history there is nothing to prefer the true one. (a) is cured by interpolating between two connected samples, (b) by a prior over paths.

## 3. Spread across the 90 drives

Panel 1 of the figure: one marker per drive, sorted by arc recall. The spread is the finding - a single headline number hides drives at 34%.

| metric (arc baseline) | p10 | median | p90 | mean | worst | best |
|---|---|---|---|---|---|---|
| point accuracy, strict | 43.6% | 61.8% | 80.8% | 62.0% | 7.4% | 98.4% |
| point accuracy, direction ignored | 75.3% | 89.5% | 97.7% | 86.5% | 9.6% | 99.7% |
| arc recall, strict | 42.6% | 57.0% | 82.4% | 59.9% | 34.3% | 100.0% |
| arc recall, direction ignored | 72.5% | 88.7% | 96.8% | 86.0% | 55.9% | 100.0% |
| arc precision, strict | 40.2% | 52.0% | 70.2% | 54.2% | 34.5% | 89.7% |
| arc F1, strict | 41.8% | 54.3% | 72.0% | 56.4% | 34.8% | 94.6% |

## 4. Difficulty threshold: accuracy vs distance to the nearest road

Accuracy is direction-ignored here, so this measures the geometric question only: did the snap land on the road the car is on?

| dist to nearest road | GPS points | drives with >=50 pts | accuracy |
|---|---|---|---|
| 0-5 m | 114,632 | 90 | 88.4% |
| 5-10 m | 62,449 | 90 | 88.5% |
| 10-15 m | 20,910 | 76 | 89.0% |
| 15-20 m | 5,827 | 35 | 86.1% |
| 20-25 m | 2,136 | 12 | 85.8% |
| 25-30 m | 1,231 | 8 | 85.2% |
| 30-35 m | 924 | 4 | 82.4% |
| 35-40 m | 614 | 4 | 79.0% |
| 40-45 m | 346 | 2 | 59.0% |
| 45-50 m | 276 | 1 | 47.5% |
| 50+ m | 18,232 | 6 | 7.2% |

- Median point-to-road distance is 4.2 m; 50% of points sit within 5 m of a road and 90% within 20 m. The collapse below hits a minority of samples - but exactly the minority a matcher has to survive.
- Why even the on-road bin is 88% and not 100%: at 0-5 m the nearest arc is often a parallel carriageway, a service road, or the cross street at an intersection. Geometry alone cannot resolve those; the sequence can.
- **The threshold is 35-40 m**: 88% on the road, 82% at 30-35 m, then 59% at 40-45 m. Past 35 m the nearest road is decided by noise rather than by the drive, so independent snapping is guessing.
- Sample caution: only 346 points (on 2 drive(s)) land in the 40-45 m bin and 276 in the next, so read the crossing point as 'about 35 m', not as a sharp edge. The trend, not the exact bin, is the number to quote.
- Practical reading: **35 m is the edge of validity for a no-history snap**, not a conservative special case. That is the number to quote for 'how far off the road can a point be before this stops working'.
- Caveat on the last bin: 91% of the 50+ m points belong to a single drive whose whole track sits ~84 m from its graph (see risks), which is why it reads 7% instead of the 45-50 m level.

## 5. Nearest-arc vs nearest-node (second trivial baseline)

Both scored on the same road-set basis (predicted arcs in either direction vs the route's arcs in either direction), so the comparison is about the snap, not about the direction tie-break.

| baseline | point accuracy | arc recall | arc F1 |
|---|---|---|---|
| snap to nearest **arc** | 89.5% | 88.7% | 81.1% |
| snap to nearest **node** | 82.0% | 98.0% | 66.9% |

**Nearest arc wins, by 14.2 points of arc F1** and 7.4 points of point accuracy - even though nearest node scores *higher* on recall (98% vs 89%). That gap is the reason recall alone is a bad way to compare baselines: a snapped node 'covers' the 2-4 arcs incident to it, so it is credited with arcs it never localised, and its precision collapses. On F1, the metric that punishes that, nearest arc wins clearly. Nearest arc is the correct floor to beat.

## The single number to quote

**Arc-level recall, strict = 57.0% (median over 90 drives; arc F1 54.3%).** One sentence with it: *'Snapping each point to its nearest road independently puts 89% of points on the right road and touches 89% of the route's roads, but only 57% of segments in the right direction.'* A method at or below that has not added path inference.

Which of the two recall numbers to put in a table depends on the reader's question: **strict** if the deliverable is an arc sequence in travel order (the actual task), **direction-ignored** if the question is only 'did it find the road'. Quote the pair, never one alone. Point accuracy should not be quoted as a result at all; if a slide needs it, label it '89% - upper bound, direction ignored'.

## Assumptions and risks

- Nearest-arc search uses a 100 m spatial hash: any point whose reported distance is below 200 m has been compared with every arc that could possibly be closer, and every point further out is re-solved by an exhaustive scan over all candidate arcs (`--validate` checks this against brute force).
- Arcs further than 1000 m outside the track bounding box are dropped for speed; they cannot be the argmin for any point we score (max observed point-to-road distance 241 m).
- Recall denominator is **distinct** route arcs (routes repeat 322 arcs in total across all drives, e.g. out-and-back legs); counting duplicates would credit a revisit twice.
- Distances use a local metric plane (equirectangular at the track centroid): error <0.1% at city scale, far below the 5 m bins.
- ⚠️ assumption: the graph covers the drive. One drive sits a steady ~84 m from every road in its graph and ~139 m from its own route - a data-alignment problem, not snapping error. It also carries 19k of the 228k points, so it distorts pooled averages; check it before reading the 50+ m bin.
- ⚠️ will be asked: 'is the 57% recall not just a direction trick?' No - direction-ignored recall is 89%, and 36 arcs per drive still receive no point at all.
