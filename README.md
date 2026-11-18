# PathSnap

Map matching on the Kubicka et al. GPS dataset. Given a noisy GPS trajectory and the
road network it was driven on, recover the sequence of road segments the vehicle
actually took.

Team 117, UE24CS352A Machine Learning, PES University. Hackathon submission,
18 November 2026.

Five matchers are implemented and scored on the same splits with the same metric, then
served through one streaming HTTP API. A demo page drives that API one GPS fix at a time
and draws the route as it is decoded.

---

## Technology stack

| Layer | What is used |
| --- | --- |
| Language | Python 3.13 or newer |
| Environment | `uv` with `pyproject.toml` and a locked `uv.lock` |
| Numerics | numpy 2.5, numba 0.68 (compiled lattice and routing kernels), pandas 3.1 |
| Learning | torch 2.15 for the approach 5 hinge and its GPU decoder |
| API | Flask 3.1 (model server), FastAPI 0.163 and uvicorn 0.54 (demo page) |
| HTTP client | requests 2.34, one keep alive session per process |
| Plots | matplotlib 3.11, seaborn 0.13 |
| Tests | pytest 9.2 |

The streaming path runs on the CPU. A GPU is optional and is used only for trainings.

---

## Architecture

Four stages, each a package under `src/`:

```
raw files  ->  dataset  ->  candidates  ->  decoder  ->  route
               (graph)      (per fix)      (Viterbi,    (arc ids,
                                            DP, hinge)   geometry)
```

1. **`dataset`** reads the four files of a trip, validates them, and projects
   coordinates per trip. Every trip carries its own road network, so nothing is shared
   between trips.
2. Each matcher builds its own candidate set (nearest road segments within a radius),
   then decodes a path. The HMM and MCM use Viterbi, the segment and MDP matchers use a
   dynamic program over decision points, and approach 5 decodes a structured hinge over
   a lattice.
3. **`benchmark`** scores a predicted route against the reference route by length
   weighted overlap, and groups the results by the dataset's hand tags.

Serving splits the same code into two processes:

```
browser  --HTTP-->  web/minimal.py (FastAPI)  --HTTP-->  serve (Flask, 127.0.0.1:8100)
                                                              |
                                                     one worker process per
                                                     streaming decoder, plus a
                                                     process pool for at finish
                                                     models
```

The page never imports a matcher. It proxies only `corpus`, `models` and `sessions`, so a
page shared on the network cannot be used to upload maps or reach anything else.

---

## Getting started

Prerequisites: Python 3.13 or newer, `uv`, and the dataset (see below).

```bash
uv sync
```

### The dataset

The corpus is **not in this repository**: it is about 300 MB and it is course material.
`data/*` is gitignored except `data/hand_labels/`.

A corpus root is a directory whose immediate children are split folders, and each split
folder holds one folder per trip:

```
data/corpus/
  train/00000001/00000001.track      lon, lat, seconds
  train/00000001/00000001.nodes      lon, lat
  train/00000001/00000001.arcs       from_node_id, to_node_id
  train/00000001/00000001.route      reference arc ids
  student_test/00000000/...
```

`Corpus.from_root` treats every immediate child as a split, so point it at a directory
built for the purpose, never at the repository root.

| Root | Used by | Contents |
| --- | --- | --- |
| `data/corpus` | default for `python -m serve` | the trips as provided |
| `data/corpus_fixed` | benchmark grouping | the same trips plus `.tags` files, so results group by hand label |
| `data/corpus_original` | `python -m learned measure` | clean original tracks, used only to measure the operating point |
| `data/corpus/executor_test` | `python -m serve.executor_test` | the 10 held out folders, fetched clean by `python -m originals.held_out` |

The upstream dataset is published under CC BY-SA 4.0. The hand labels derived from its
metadata ship in `data/hand_labels/`.

---

## Project structure

| Path | What is in it |
| --- | --- |
| `src/dataset/` | Reads and validates the raw files, projection, road graph, corpus |
| `src/fixes/` | Cleaning and tag handling for the raw corpus |
| `src/originals/` | Recovers clean original tracks, and fetches the 10 held out folders |
| `src/hmm/` | Newson and Krumm HMM, batch and online decoders |
| `src/mcm/` | Multiple candidate matching (Li et al. 2023) |
| `src/segment/` | Feature based matcher (Yin et al. 2018) |
| `src/mdp/` | Port of Woeltche's Markov decision process matcher |
| `src/learned/` | Approach 5: measured operating point, lattices, structured hinge, decoder, serving |
| `src/benchmark/` | Scores predicted routes against reference routes |
| `src/serve/` | Streaming HTTP API, worker processes, the executor test |
| `web/` | Demo pages: `minimal.py` (streaming) and `app.py` (proxy and upload) |
| `tests/` | Test suite |
| `results/` | Measured parameters, predicted routes, benchmark tables, approach 5 record |
| `docs/` | `api.md` (the HTTP contract) and the derivation notes |
| `data/` | Corpus and hand labels, not committed |

---

## Key features

* **Five matchers, one metric.** Every model is scored by the same length weighted
  correct fraction, on the same splits, so the comparison table is like for like.
* **Streaming decode.** The HMM commits a settled prefix and keeps a tentative tail,
  using the variable sliding window of Goh et al. Fed one fix at a time it returns
  exactly the batch route.
* **One API, several models.** A session carries several models, so one request per fix
  feeds all of them.
* **Measured operating point.** Thinning tolerance, speed window and the noise model are
  measured from the clean originals rather than guessed.
* **Reproducible results.** Every table in `results/` is produced by the commands in this
  file, and the checks behind approach 5 are in `results/learned/record.md`.

---

## Running

### Model server

```bash
uv run python -m serve
# http://127.0.0.1:8100/v1
```

Options: `--corpus data/corpus`, `--results results`, `--port 8100`,
`--streaming 8`, `--batch 4`.

One process, many threads: sessions live in this process's memory. It binds
`127.0.0.1` only, because the browser talks to the page, not to the model server.

### Demo page

```bash
uv run python web/minimal.py
# http://<this host>:5001
```

Pick a corpus folder and a model, press Match. Fixes are sent one at a time and the route
is drawn as it is decoded, solid once committed and dashed while still tentative.

The page binds `0.0.0.0` by default so it can be opened from another machine; `HOST=127.0.0.1`
keeps it local. It proxies only `corpus`, `models` and `sessions`, and only `sessions`
accepts POST, so the model server stays private.

To point the page at a model server elsewhere:

```bash
MODEL_API=http://<host>:8100/v1 uv run python web/minimal.py
```

### Tests

```bash
uv run pytest
```

---

## Deployment

Two processes, one machine:

1. Start the model server, then check it: `curl http://127.0.0.1:8100/v1/health` returns
   `{"ready": true, ...}`. `ready` stays false until every worker has run one warm up
   match, so Numba compiles before the first real fix.
2. Start the demo page and open it from whichever machine needs it.

The model server has no authentication and must stay on `127.0.0.1`. If the page must be
reachable from another machine, the page's default `0.0.0.0` binding is enough, and the
port needs to be allowed through the firewall. Expose the page, never the model server.

---

## API

The full contract is in [`docs/api.md`](docs/api.md). The endpoints the demo uses:

| Call | Does |
| --- | --- |
| `GET /v1/models` | model ids, `mode` (`online` or `at_finish`), availability, version |
| `HEAD /v1/maps/{map_id}` | is this road network already stored |
| `POST /v1/maps` | upload `nodes` and `arcs` once, reused by every session |
| `POST /v1/sessions` | open a session for one map and one or more models |
| `POST /v1/sessions/{sid}/fixes` | send `{"seq", "fixes"}` and get per model updates |
| `POST /v1/sessions/{sid}/finish` | flush: online models commit, the rest run their batch match |
| `GET /v1/sessions/{sid}` | full state, for polling or for reconnecting |

An online model returns, per fix, the arcs it has **committed** (append only, never
revised) and its whole **tentative** tail (replaced each fix). The route shown is the two
concatenated. Only the HMM streams; the other four answer at finish.

---

## Models

| Model | Family | Mode |
| --- | --- | --- |
| `hmm` | Newson and Krumm hidden Markov model, Viterbi | online |
| `mcm` | Multiple candidate matching, score matrix | at finish |
| `segment` | Yin et al. feature based matcher | at finish |
| `mdp` | Woeltche's Markov decision process | at finish |
| `learned` | Approach 5, structured hinge over a lattice | at finish |

`learned` is served through `src/learned/serving.py`, which builds the lattice in memory
and uses the frozen prototype in `results/learned/`.

---

## Results

Weighted correct fraction on the corpus, from `results/benchmark/`:

| Split | MDP | Yin | HMM | MCM | Approach 5 |
| --- | --- | --- | --- | --- | --- |
| train, all 80 | 97.04 | 94.35 | 91.98 | 79.18 | **98.45** |
| student_test, all 10 | **99.34** | 97.10 | 97.06 | 76.90 | 99.26 |

On the 10 held out folders (clean Kubicka originals, `results/executor_test/`):

| Model | Weighted correct fraction |
| --- | --- |
| HMM | **99.82** |
| Segment | 99.69 |
| MDP | 99.69 |
| Approach 5 | 99.60 |
| MCM | 89.46 |

These held out folders are the clean published tracks. The organiser's noised copies are
a different corpus and are not scored here.

---

## Reproducing the results

```bash
# the baseline matchers, then score them
uv run python -m hmm       data/corpus_fixed results
uv run python -m mcm       data/corpus_fixed results
uv run python -m segment   data/corpus_fixed results
uv run python -m mdp       data/corpus_fixed results
uv run python -m benchmark data/corpus_fixed results hmm

# approach 5, in order
uv run python -m learned measure   data/corpus_fixed data/corpus_original results
uv run python -m learned lattices  data/corpus_fixed data/corpus_original results
uv run python -m learned prepare   data/corpus_fixed results
uv run python -m learned freeze    results
uv run python -m learned match     data/corpus_fixed results
uv run python -m benchmark data/corpus_fixed results learned

# the 10 held out folders
uv run python -m originals.held_out data/corpus data/corpus/executor_test
uv run python -m serve.executor_test <root holding only executor_test>
```

`freeze` writes the starting weights to `results/learned/weights.json`. The `train`
stage would overwrite that file with trained weights.

---

## Development workflow

* `main` is the integration branch.
* Work happens on branches, and the streaming work landed on `server` before being merged
  into `main`.
* The early canvas interface is preserved on `archive/old-ui`; `main` carries the
  streaming interface only.
* Commit messages state what changed and why, and record measured numbers where they
  decide a design.

---

## Coding standards

* **Every non obvious decision carries its reason.** Comments explain why a choice was
  made, not what the line does.
* **`OWN CHOICE` marks a value that neither the literature nor a measurement gives.**
  There are 46 such markers in `src/`, each with the reason beside it. A reviewer can
  find every invented number with `grep -rn "OWN CHOICE" src/`.
* **Module docstrings state the contract**, including what the module guarantees against
  the alternative implementation, for example that the online HMM returns exactly the
  batch route.
* **Names are spelled out** (`fixes`, `lattice`, `candidates`), and units appear in the
  name or the docstring.
* **No silent fallbacks.** A missing input raises with the file and line, rather than
  degrading to a default.

---

## Testing

`tests/` holds the suite, run with `uv run pytest`.

| File | Covers |
| --- | --- |
| `test_dataset.py` | The reader, validation and projection |
| `test_metrics.py` | The length weighted score and its edge cases |
| `test_hmm_online.py` | Online HMM against the batch HMM, fix by fix |
| `test_learned_decode.py` | The learned decoder against brute force |
| `test_learned_batch.py`, `test_learned_routes.py` | Lattice and route search |
| `test_learned_truth.py` | Placing training targets on the true route |
| `test_serve.py` | The streaming API end to end |
| `test_fixes.py` | Tag handling |

The required check for any new streaming model: on every corpus folder, pushing the fixes
one at a time and calling finish must give exactly the route the batch matcher produces.

---

## Contributing

This is a course submission, so the workflow is short:

1. Branch from `main`.
2. Keep the module docstring and the `OWN CHOICE` discipline described above.
3. Add or extend a test that would fail without the change.
4. Run `uv run pytest` before opening a pull request.
5. Do not commit the corpus, credentials or generated lattices. `.gitignore` already
   covers all three.

---

## License

No license file is present. The code is coursework and is not licensed for reuse.

The upstream dataset is published under CC BY-SA 4.0, and its share alike term applies to
derived data.

---

## Notes and limitations

* **Reference routes are reference, not verified truth.** The dataset's own routes have
  known errors, and trip ends cannot be matched exactly. Both cap any method below 100.
* **Approach 5 carries starting weights.** `results/learned/weights.json` holds
  Woeltche's weights and the five new features are weighted zero. Training converges but
  currently scores slightly below the starting weights in sample; the measured cause is
  in `results/learned/record.md`.
* **Direction ambiguity.** 92 percent of the arcs in these graphs have a reverse twin with
  identical geometry, so a distance only score cannot separate them. Every matcher here
  uses heading as well.
* **Sample rate.** Sampling is nominally 1 Hz, but 15 of the 90 trips carry most of the
  irregular steps, so time is handled per step rather than per trip.
