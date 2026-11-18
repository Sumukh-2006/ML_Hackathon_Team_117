# PathSnap demo app

A FastAPI backend and a single-page evaluator interface for the map-matching
baseline: each GPS fix is snapped to its nearest road segment on its own, then
scored against the reference route that ships with the trip.

The interface is not a dashboard. It is a chart sheet: whole-drive geometry at
fit, a 500 m magnifier around the current fix, and a time strip that marks only
the fixes where the snap disagrees with the reference.

## Run

```bash
pip install -r app/requirements.txt
python -m app.build_payloads        # once: writes app/web/data/*.json (~90 trips)
python -m uvicorn app.server:app --port 8077
# open http://127.0.0.1:8077/
```

The raw `train/`, `student_test/` and `data/` folders are not in the repository.
`/api/trips` works from the committed payloads; `build_payloads` and
`POST /api/match` need those folders. Without them the match endpoint answers
503 with that explanation.

`--host 0.0.0.0` if the evaluator's laptop has to reach it over the room network.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | status, trip count, method |
| GET | `/api/trips` | index: one summary row per drive |
| GET | `/api/trips/{id}` | full payload for the map (track, network, route, prediction, metrics) |
| POST | `/api/match` | runs the matcher live on `{trip_id, track?}` and returns arcs, offsets, scores, wall-clock ms |
| GET | `/api/methodology` | scoring definitions, what is N/A, and the assumptions |

`POST /api/match` with `{"trip_id": "00000001"}` reproduces the numbers on the
sheet; pass your own `track: [[lon, lat], ...]` to match an arbitrary trace
against that drive's road network.

## What the numbers mean

- **Route recovered** (hero): share of reference-route arcs that received at
  least one fix, direction included. This is the literal task.
- **Any direction**: the same, treating an arc and its reverse twin as one road.
  92% of these graphs store both directions at identical geometry, so the strict
  number depends on the tie rule; this view separates "found the road" from
  "faced the right way".
- **Missed arcs**: route arcs no fix reaches, in either direction.
- **Breaks**: consecutive fixes whose snapped arcs share no node.
- **N/A**: macro F1 and log loss are classifier metrics; this matcher emits no
  per-class probabilities, so quoting them would be invented.

## Verify

```bash
python -m app.test_smoke      # scoring invariants + payload freshness (a few seconds)
python tools/check_ui.py      # drives the page: keys, scrub, layers, map click, API
python tools/perf.py          # timings on the heaviest drive (19,313 fixes)
```

The smoke test exists because two scoring bugs shipped and were caught here: a
direction-ignored recall that double-counted both directions in the denominator,
and a per-fix direction flag that never fired. Each assertion is a claim the
sheet makes on screen.

## Honest limits

⚠️ assumption: this is a baseline, not a model. There is no classifier, no HMM,
no tuning. It exists to show the floor and where sequence information would pay.

⚠️ assumption: the drawn network is a 650 m corridor around the drive; matching
runs against every arc within 1 km of the trace.

⚠️ will be asked: *why open on this drive?* The picker lists all 90 shipped trips
and the default is the first training drive. The worst drives are one selection
away, and nothing is filtered out.

## Files

```
app/
  config.py          paths + matcher knobs (shared by server and builder)
  match.py           loading, spatial hash, nearest-arc match, scoring
  build_payloads.py  precomputes app/web/data/trip_*.json and index.json
  server.py          FastAPI app + static mount
  web/index.html     the sheet
  web/styles.css     chart paper palette, titles, tape
  web/app.js         renderer, interactions, tape, dialogs
  web/data/          generated payloads (committed)
```

Screenshots used in the report: `notes/ui/` (`python tools/shoot.py`,
`python tools/interactions.py`).

The generated payloads are ~48 MB on disk. They are committed so the demo runs
without a rebuild; if repository size is a concern, drop `app/web/data/*.json`
and run `python -m app.build_payloads` after cloning.
