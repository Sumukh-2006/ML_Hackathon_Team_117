# Map-matching model server: streaming API and website guide

The website sends GPS fixes **one at a time** (a live phone, or a replay of
a `.track`) and shows the route as it grows. Two programs, two processes:

- **Model server** (`src/serve/`, Flask, `127.0.0.1:8100` only): owns the
  matchers, the parsed road graphs and the per-session decoder state.
- **Website** (Flask, `web/`): pages, uploads, drawing. It never imports
  `dataset`, `hmm`, `learned` or torch, only HTTP.

Values marked OWN CHOICE are picked for this design, not measured or
sourced. **Open decision** marks something the user has not decided.

## 0. Running the minimal implementation

```
uv run python -m serve            # model server, 127.0.0.1:8100
uv run python web/app.py          # website, http://127.0.0.1:5000
uv run pytest tests/test_hmm_online.py tests/test_serve.py
```

| file | what it is |
|------|------------|
| `src/hmm/online.py` | streaming HMM; gives the batch route exactly (checked on all 90 folders, fixes one at a time and in chunks of 7) |
| `src/serve/app.py` | the HTTP API below |
| `src/serve/workers.py` | sticky streaming workers + process pool for at-finish models |
| `src/serve/models.py` | model registry, batch runner, GeoJSON, scoring |
| `web/model_client.py` | the only place the website talks to the server |
| `web/app.py` | website: page, proxy, upload |
| `web/static/app.js`, `web/templates/index.html` | replay page (Leaflet) |

**Built:** everything in §3 except what is listed next. **Not built yet**
(spec only): warm-up before `ready`, `GET /maps/{id}/arcs`, the off-map
check, live GPS page, reconnect-after-reload in the page, an online `mcm`.
`learned` is `unavailable` until `src/learned/serving.py` exists with
`match(graph, track) -> (Route, counts)`.

**Known issue (open decision): exact ties stall committing.** On train/54
the car stands at node 1937 for ~50 min. Its two candidates there, the
start of arc 3972 and the end of its reverse arc 3991, are the same point,
so their Viterbi scores are exactly equal and their paths never merge;
nothing commits until the car moves (median 4 730 pending fixes on that
track, against ~180 elsewhere). The batch decoder breaks the tie by
taking the first index. Collapsing candidates at the same point is a rule
nobody has supplied yet, so it is not done.

---

## 1. What changes when fixes arrive one at a time

**Re-matching the whole track after every fix does not scale.** The
batch matchers take the whole track. Calling `match()` on the prefix after
each fix costs a sum over all prefixes, roughly n/2 full runs. Folder 13
(2 580 fixes): HMM runs once in 1.89 s, so that is ~40 min per replay;
`segment` runs once in 37.6 s, so ~13 h. A streaming API needs matchers
that keep state and extend it by one fix.

**The route has a settled part and a part that can still change.** An
online Viterbi keeps every surviving candidate path. Once all survivors
share the same ancestor at some fix, everything before that fix can never
change: it is **committed**. The part after it is **tentative**: the best
guess now, which the next fix may rewrite. This is the "variable sliding
window" decoding of Goh et al. (ITSC 2012, our note
`literature/papers/07-goh-ohmm.md`). It is exact: at `finish`, the route
equals the batch route on the same fixes. So the API returns, per fix:
newly committed arcs (append only) plus the whole tentative tail (replace).
The client never re-downloads the route, and bandwidth per fix stays
small however long the drive is.

**Which matchers can do this, from their code:**

| model | streams? | why |
|-------|----------|-----|
| `hmm` | yes, exact | thinning is causal (`hmm/match.py:thin` keeps a fix if it is far enough from the last kept one); Viterbi runs forward. Online Viterbi per Goh et al. |
| `mcm` | yes, per the paper | the score matrix is filled one fix column at a time (`mcm/score_matrix.py`), and Li et al. (2023) give an online algorithm next to the offline one. Our notes skipped its pages 7–9, so they need reading before it is built. |
| `segment` | no | key points are Douglas-Peucker over the whole track (`segment/simplify.py`): the last fix is always a key, so a new fix can change earlier keys. |
| `mdp` | no | Douglas-Peucker in `mdp/clean.py`, and value iteration swept backwards from the last point. |
| `learned` | not as built | same Douglas-Peucker (`learned/track.py:decision_points`). Its decode is a DP over a layered graph, which can run forward like Viterbi, and its speed fit only looks `half_window` seconds ahead (a bounded delay). Douglas-Peucker is the only blocker. |

**Open decision: the three non-streaming models.** Options:

- **A. Only at finish.** The session collects fixes, and `finish` runs the
  batch `match()` once. No live route for these models, and finish can
  take a while (37.6 s `segment` on folder 13; 207 s for one `learned`
  lattice on folder 33). Results are identical to the benchmark.
- **B. Re-run on a timer.** Re-run the batch match on the prefix every k
  fixes and show it as tentative only, with nothing committed until
  finish. Works for `hmm`-speed models. Not viable for `segment`
  (37.6 s per run), and k is a made-up value.
- **C. Make `learned` causal**, while the other session is still building
  it: swap Douglas-Peucker for a causal thinning such as HMM's distance
  rule, then decode online like HMM. This changes approach 5's method, and
  the measured ~20 m tolerance was fitted to Douglas-Peucker point
  density, so it would need re-measuring.

My recommendation: A for `segment` and `mdp` (they are baselines), and
decide on C now, because building a causal design in costs less than
retrofitting one. The API below supports all three: a model reports its
`mode` and the website just draws what comes back.

**No GPU in the streaming path.** `learned/evaluate.py` already decodes
on the CPU. The per-fix work is about a millisecond (HMM: 1.89 s / 2 580
fixes on folder 13, graph loading included), far below the cost of a
CUDA kernel launch plus copy.

---

## 2. Formats

| thing | format |
|-------|--------|
| nodes / arcs | corpus text: `lon<TAB>lat` / `from<TAB>to`; id = 0-based line |
| fix | JSON `[lon, lat, t]`, t in seconds, strictly increasing within a session |
| route | JSON `[arc_id, ...]`; the `.route` text on download endpoints |
| geometry | GeoJSON, `[lon, lat]`. Leaflet wants `[lat, lon]`: swap in the browser |

**`map_id`** = first 16 hex chars of `sha256(sha256(nodes) || sha256(arcs))`
over the exact file bytes, so the website can compute it itself.

---

## 3. Endpoints (base `http://127.0.0.1:8100/v1`)

### 3.1 Service

`GET /health`
```json
{"ready": true,
 "models": {"hmm": {"state": "ready", "mode": "online"},
            "mcm": {"state": "ready", "mode": "at_finish"},
            "segment": {"state": "ready", "mode": "at_finish"},
            "mdp": {"state": "ready", "mode": "at_finish"},
            "learned": {"state": "unavailable", "mode": "at_finish"}},
 "workers": {"streaming": 8, "batch": 4}, "sessions": 3}
```
`ready` stays false until every worker has run one warm-up match, so
Numba has compiled before the first real fix. `mode` is `online` (live
committed + tentative), `rerun` (option B: tentative only) or `at_finish`
(option A).

`GET /models`: `id`, `name`, `mode`, `available`, `reason` (why not),
and `version`, the hash of `results/<model>/parameters.json` and
`src/<model>/hyperparameters.py`.

### 3.2 Maps (uploaded once, reused by every session)

| call | does |
|------|------|
| `HEAD /maps/{map_id}` | 200 if stored, 404 if not; call before uploading |
| `POST /maps` multipart `nodes`, `arcs` | stores it (idempotent) → `{"map_id", "nodes", "arcs", "bbox"}`; 64 MB limit (OWN CHOICE; largest corpus map 6.8 MB) |
| `GET /maps/{map_id}` | the same metadata |

Maps are stored in corpus layout under `results/serve/maps/<map_id>/`,
so `dataset.RoadGraph.from_folder` reads them unchanged and its
`file:line:` errors come back as `400 invalid_map`.

### 3.3 Corpus (for replay demos)

| call | does |
|------|------|
| `GET /corpus` | `[{"split", "id", "map_id", "fixes", "has_route", "tags"}]`. Tags are the Kubicka hand labels from `data/corpus_fixed`. |
| `GET /corpus/{split}/{id}/track` | `{"fixes": [[lon, lat, t], ...]}`: the website replays these one by one |

### 3.4 Sessions: the streaming core

#### `POST /sessions`
```json
{"map_id": "9c1e0b7d2a4f6e31",
 "models": ["hmm", "mcm", "learned"],
 "options": {"geometry": true,
             "truth": {"corpus": {"split": "train", "id": "00000001"}}}}
```
`201`:
```json
{"session_id": "s_4fQ9",
 "models": {"hmm": {"mode": "online", "version": "3f2a91c0"},
            "mcm": {"mode": "online", "version": "11b0c2d4"},
            "learned": {"mode": "at_finish", "version": "a07c55e1"}},
 "seq_next": 0}
```
One session carries several models, so one request per fix feeds all of
them; the server fans the fix out in parallel. `truth` is optional
(a corpus route, or `[arc_id, ...]`) and only used for the score at
finish.

#### `POST /sessions/{sid}/fixes`
```json
{"seq": 120, "fixes": [[8.595203, 49.844053, 121.0]]}
```
`seq` is the index of the first fix in this request. Send one fix, or
several when replaying faster than real time.

`200`:
```json
{"seq_next": 121,
 "updates": {
   "hmm": {
     "kept": 1,
     "committed": {"at": 57, "arcs": [64973, 133948]},
     "tentative": [133949, 2210],
     "pending_fixes": 9,
     "break": false,
     "geometry": {"committed": {"type": "MultiLineString", "coordinates": [...]},
                  "tentative": {"type": "LineString", "coordinates": [...]}}
   },
   "mcm": {"kept": null},
   "learned": {"kept": null}
 }}
```

How the client applies an update, per model:
- `committed.arcs` are **appended** at position `committed.at` of that
  model's committed route. They never change again.
- `tentative` **replaces** the previous tentative tail completely.
- The route shown = committed + tentative.
- `kept` = how many of this request's fixes the matcher's own thinning
  kept (HMM keeps 611 of 5 844 on folder 33). 0 is not an error.
- `pending_fixes` = how many fixes since the last commit. Shown to the
  user, it says how far behind the settled route is.
- `break: true` = the matcher started a new trip (HMM's gap rule).
  Everything before it is committed at once, and geometry gets a new
  line part.
- `at_finish` models answer `{"kept": null}` until finish.

**Ordering and retries.** A session takes fixes in `seq` order only:
- `seq` < `seq_next` with the same fixes is a retry: the server returns
  the stored response for that request and does not apply it twice.
- `seq` > `seq_next` is a gap: `409 seq_gap` with `{"expected": seq_next}`.
- t not after the previous fix's t: `400 invalid_fix`, which names the
  offending fix. The session is unchanged and the next valid fix still
  works.

#### `POST /sessions/{sid}/finish?wait=25`
Flushes every model. `online` models commit their tentative tail, and
`at_finish` models run their batch match on the collected fixes. The
response has the same per-model shape plus the final result:
```json
{"status": "running",
 "results": {
   "hmm": {"status": "done", "route": [2641, 64973, ...], "length_m": 18342.7,
           "breaks": [], "stats": {"fixes": 1315, "kept": 412, "trips": 1},
           "score": {"correct_fraction": 0.991, "overlap_m": 18201.3,
                     "predicted_length_m": 18342.7, "true_length_m": 18290.0}},
   "learned": {"status": "running", "stage": "lattice"}}}
```
`wait` holds the request up to that many seconds (max 30) for slow
`at_finish` models. Poll with `GET /sessions/{sid}?wait=25` until every
model is `done` or `failed`. `score` is `benchmark.metrics.score_route`,
the same numbers as the benchmark tables. After finish, the session
accepts no more fixes.

#### Other session calls
| call | does |
|------|------|
| `GET /sessions/{sid}?wait=` | full state: `seq_next`, `live` (each online model's whole committed route + tentative, arc ids), `results` (per model status / final result). For reconnecting after a page reload, and for polling finish. |
| `GET /sessions/{sid}/route?model=hmm` | `.route` text (committed + tentative; final after finish) |
| `DELETE /sessions/{sid}` | ends it and frees its worker state |

Idle sessions expire after 15 min (OWN CHOICE). Finished sessions are
kept 24 h (OWN CHOICE).

### 3.5 Errors

`{"error": {"code": "...", "message": "..."}}`

| HTTP | code |
|------|------|
| 400 | `invalid_map`, `invalid_fix`, `bad_request` |
| 404 | `unknown_map`, `unknown_session`, `unknown_model` |
| 409 | `seq_gap` (body has `expected`), `session_finished` |
| 413 | `too_large`: upload over 64 MB, more than 1 000 fixes in one request, or more than 100 000 per session (OWN CHOICE; longest corpus track 19 313) |
| 503 | `not_ready`, `model_unavailable`, `too_many_sessions` (> 64, OWN CHOICE), with `Retry-After` |
| 500 | `match_failed`: the matcher raised. That model's status becomes `failed`; the session's other models carry on. |

---

## 4. Inside the server (`src/serve/`)

```
  HTTP process (Flask, 1 process, threads)
    session table: sid -> {model -> worker}, seq, stored last responses
    validation (map, seq, time order)
          │ one Pipe per worker; a session's messages go to fixed workers
  worker 1 … worker N (spawned processes)
    parsed graphs + per-model indices (LRU by map_id)
    decoder state per (session, model)
```

1. **One HTTP process with threads.** Session state lives in memory, so
   several gunicorn workers would split it; use `-w 1 -k gthread`.
2. **Sticky workers, not a pool.** A session's decoder state for a model
   lives in one worker process, and every fix for it goes there in order.
   `ProcessPoolExecutor` cannot pin tasks to a process, so use N
   `multiprocessing` processes with a `Pipe` each. N = 8 of 16 threads
   (OWN CHOICE).
3. **Placement:** put a new (session, model) on a worker that already has
   that `map_id` parsed, else the least-loaded one. A cache miss costs up
   to 0.25 s of parsing (folder 13) plus the model's index build, paid
   once per session, not per fix.
4. **Warm-up** in each worker at start: one short corpus replay per model,
   so the first fix doesn't pay Numba's compile.
5. **The streaming contract** each `online` model implements, next to its
   batch `match`:
   ```python
   class Online(Protocol):
       def __init__(self, graph: RoadGraph, parameters: dict) -> None: ...
       def push(self, lon: float, lat: float, t: float) -> Update: ...
       def finish(self) -> tuple[Route, dict]: ...

   @dataclass
   class Update:
       kept: bool
       committed: list[int]   # new arcs only
       tentative: list[int]   # whole tail
       pending_fixes: int
       new_trip: bool
   ```
   The required test, per model: on every corpus folder, pushing the fixes
   one by one and calling `finish` gives exactly the route the batch
   `match` writes to `results/<model>/routes/`.
   - **HMM:** `thin` already works fix by fix. `hmm/viterbi.decode`
     becomes a forward step plus a convergence check over the survivors'
     back-pointers.
   - **`at_finish` models:** wrap the batch `match` and buffer the fixes.
6. **What the approach-5 session needs to provide**: a no-truth
   `match(graph, track) -> (Route, counts)`, built from
   `Builder(graph).lattice(track, parameters, None)` →
   `evaluate.labelling` → `decode.drive`, plus the weights path. With
   option C, an `Online` implementation as well.

---

## 5. Writing the website

### 5.1 Order of work
1. Check the server with curl before any page:
   ```
   curl -s 127.0.0.1:8100/v1/health
   S=$(curl -s -X POST 127.0.0.1:8100/v1/sessions -H 'Content-Type: application/json' \
       -d '{"map_id":"<id from /v1/corpus>","models":["hmm"]}' | jq -r .session_id)
   curl -s -X POST 127.0.0.1:8100/v1/sessions/$S/fixes -H 'Content-Type: application/json' \
       -d '{"seq":0,"fixes":[[8.595203,49.844053,1.0]]}'
   ```
2. Write `web/model_client.py` (below). Every model-server call goes
   through it.
3. **Replay page** first: pick a corpus folder, pick models, set the
   speed (1×, 10×, 50×), play/pause, watch the routes grow.
4. **Upload page**: nodes + arcs + track. Same replay view afterwards.
5. **Live page** last: `navigator.geolocation.watchPosition` in the
   browser. It needs HTTPS, and the map must cover where the phone is.
   The corpus maps are fixed regions, so for live use the user uploads
   (or the site fetches) a map of their own area.

### 5.2 Client module
```python
# web/model_client.py
import hashlib, requests

API = "http://127.0.0.1:8100/v1"     # from config
http = requests.Session()             # keep-alive: one TCP connection for all fixes

def map_id(nodes: bytes, arcs: bytes) -> str:
    inner = hashlib.sha256(nodes).digest() + hashlib.sha256(arcs).digest()
    return hashlib.sha256(inner).hexdigest()[:16]

def ensure_map(nodes: bytes, arcs: bytes) -> str:
    mid = map_id(nodes, arcs)
    if http.head(f"{API}/maps/{mid}").status_code == 404:
        http.post(f"{API}/maps", files={"nodes": nodes, "arcs": arcs}).raise_for_status()
    return mid

def open_session(map_id, models, truth=None) -> dict:
    body = {"map_id": map_id, "models": models, "options": {"geometry": True}}
    if truth: body["options"]["truth"] = truth
    r = http.post(f"{API}/sessions", json=body); r.raise_for_status(); return r.json()

def push(sid, seq, fixes) -> dict:          # fixes: [[lon, lat, t], ...]
    r = http.post(f"{API}/sessions/{sid}/fixes", json={"seq": seq, "fixes": fixes}, timeout=10)
    r.raise_for_status(); return r.json()

def finish(sid, wait=25) -> dict:
    r = http.post(f"{API}/sessions/{sid}/finish", params={"wait": wait}, timeout=wait + 10)
    r.raise_for_status(); return r.json()

def state(sid, wait=0) -> dict:
    r = http.get(f"{API}/sessions/{sid}", params={"wait": wait}, timeout=wait + 10)
    r.raise_for_status(); return r.json()
```

### 5.3 Where the fixes come from, and the request path

- **Replay:** the browser holds the fix list (from
  `GET /corpus/.../track` via the website) and a timer. Every tick it
  POSTs the next fixes to the website's `/api/sessions/<sid>/fixes`, which
  forwards them with `push`. At 1× that is one fix per request. Above
  ~10×, send every fix due since the last tick in one request. A fix
  costs ~1 ms to match, so at 50× per-request overhead would dominate.
- **Live:** each `watchPosition` callback POSTs one fix the same way.
  Use the callback's timestamp in seconds as t.
- **Browser-side state, per model:** `committed[]`, `tentative[]` and two
  Leaflet layers: committed solid, tentative dashed. On each response,
  append to `committed`, replace `tentative` and redraw only the
  tentative layer plus the new committed piece. Swap `[lon, lat]` to
  `[lat, lon]`.
- **One driver per session.** The tab that opened it sends the fixes and
  owns `seq`. On a network error, resend the same request with the same
  `seq`; the server returns the stored response.
- **Reload / reconnect:** keep `sid` in the Flask session. On page load,
  call `GET /sessions/{sid}` to rebuild the routes and continue from
  `seq_next`.
- **End of drive:** the Stop button calls `finish`. Draw the
  `at_finish` models when they arrive (poll `state(sid, wait=25)`). Show
  the scores and offer each `.route` download.

### 5.4 Done when
- [ ] Replay train/00000001 at 1× with `hmm` and `mcm`: committed (solid)
      routes grow, and tentative tails (dashed) flicker and settle.
- [ ] After finish, `hmm`'s downloaded route is byte-identical to
      `results/hmm/routes/train/00000001.route`. This proves streaming
      equals batch.
- [ ] Replay folder 33 (hives, 5 844 fixes) at 50×: the page stays
      responsive, and most HMM updates say `kept: false`.
- [ ] Reload the tab mid-replay: the routes come back and the replay
      continues from `seq_next`.
- [ ] Kill the network for one request: the retry does not duplicate arcs.
- [ ] Send a fix with a repeated timestamp: the error is shown and the
      next fix is accepted.
- [ ] `segment`/`mdp`/`learned` appear after Stop, with "running" shown
      while they work. `learned` without weights shows as unavailable,
      and the others are unaffected.
