# Approach 5 (`src/learned/`): prototype record

Frozen on 2026-11-18. The routes in `results/learned/routes/` and the scores in
`results/benchmark/learned/` come from this version.

## What runs

1. **Measure** (`python -m learned measure`), using train only: the clean
   originals give three values.
   - Thinning tolerance **16.57 m**: the Douglas–Peucker tolerance at which
     the noisy train tracks keep as many decision points as the originals keep
     at Wöltche's 5 m (11 435 points). It is found by bisection.
   - Speed window: a half window of **5 s**. It gives a median step-speed
     error of 0.29 m/s, against 1.08 m/s at 1 s.
   - Impossible speed above **60.7 m/s**: the 99.9th percentile of true step
     speeds.
   - Noise model, fitted per axis: Gaussian core σ 5.02 m, of which a share
     of 0.36 drifts with correlation 0.846 per second; outliers with chance
     0.0057 and σ 14.0 m. Regenerated noise reproduces the measured noise:
     std 5.13 against 5.13 m, kurtosis 3.95 against 3.93, and the
     autocorrelation at lags 1/2/3/5/10 s within 0.003.
2. **Lattices** (`python -m learned lattices`): the decision points,
   candidates (16 nearest roads, adoption), and a compiled route search
   (Numba) for every candidate pair. Stored per pair: 11 step features
   (Wöltche's geometry, coverage both ways, speed consistency, impossible
   speed, turn force), the skip mask and the step error. One lattice per
   folder, plus 8 noisy copies per train original: 730 lattices, 531 M pairs,
   323 s on 16 cores.
3. **Weights** (`results/learned/weights.json`): the starting weights w0,
   which are not trained.
   - Wöltche's weights for his terms: distance 0.8, last distance 0.8,
     length 0.3, heading 0.5, turn 0.5, direction change 0.6.
   - First distance 2.4 and restart distance 0.08 (his factors 3 and 0.1,
     times 0.8).
   - Skip 235 713: the median of his fail cost over the train points.
   - 0 for the five new features.
4. **Match** (`python -m learned match`): exact decoding with no discount
   (Theorem 1), then driving the chosen routes. All 90 folders take 2.7 s
   once the lattices exist.

Cost per folder's own track: lattice build median 2.2 s, worst 205 s (track
33, a "hives" track whose points have a median of 492 candidates). The 90
own lattices take 520 s of CPU time in total.

## Results (weighted correct fraction, %)

| split / group | MDP (Wöltche) | Yin | HMM | MCM | **approach 5** |
|---|---|---|---|---|---|
| train, all 80 | 97.04 | 94.35 | 91.98 | 79.18 | **98.45** |
| train, clean | 92.38 | 96.31 | 94.50 | 78.06 | **98.75** |
| train, gaps | 98.05 | 93.53 | 91.29 | 80.13 | **98.28** |
| train, hives | **96.22** | 69.52 | 71.57 | 70.11 | 89.17 |
| train, u-turns | 98.24 | 96.61 | 91.95 | 66.92 | **98.65** |
| train, loops | 97.83 | 94.07 | 90.70 | 73.69 | **98.93** |
| train, congruence issues | 95.28 | 87.19 | 84.00 | 61.29 | **95.77** |
| student_test, all 10 | **99.34** | 97.10 | 97.06 | 76.90 | 99.26 |

Train is a fair comparison here because nothing was fitted to train scores:
only the three measured values above came from train, and only from the clean
originals. On train, approach 5 is better than the MDP on 21 tracks and worse
on 30, but the tracks it gains on are long ones where the MDP failed badly, so
the length-weighted score rises. The MDP's failures on the noisy tracks (the
candidate explosion on 33, the skip loophole on 82) came from thinning at 5 m.
Thinning at the measured tolerance is the main change.

Per track on student_test (MDP → approach 5):

| track | tags | MDP | approach 5 |
|---|---|---|---|
| 00000000 | gaps | 99.90 | 99.90 |
| 00000007 | clean | 99.30 | 99.30 |
| 00000009 | gaps | 97.94 | 99.73 |
| 00000040 | clean | 100.00 | 99.92 |
| 00000052 | gaps | 99.69 | 99.69 |
| 00000056 | gaps, loops, u-turns, congruence issues | 98.56 | 99.70 |
| 00000065 | gaps, u-turns | 100.00 | **92.95** |
| 00000076 | gaps, congruence issues | 98.61 | 96.98 |
| 00000077 | loops, u-turns | 100.00 | 99.33 |
| 00000091 | clean | 99.56 | 97.12 |

Track 00000065 accounts for most of the gap to the MDP on student_test. It has
not been investigated yet.

## The ceiling

On train, the oracle is the labelling of least step error that these lattices
allow. It scores **99.75%**. The target is 99.5%, so the representation has
room for it; the weights are what is missing. The oracle's worst tracks fail
on the data, not the matcher:
- 00000066: one decision point is 1.9 km from any road in the map, and two
  short steps have 734 m of true route that the GPS does not show.
- 00000075 and 00000095: gaps, loops and u-turns.

## Checks (26 tests pass)

- The compiled router, with Wöltche's settings, gives `mdp/`'s candidates,
  route lengths, direction changes and paths, pair for pair, on 3 tracks.
- With Wöltche's weights, fail cost and γ = 0.99, the new cleaning, lattice
  and decoder give byte-identical routes to `mdp/` on 3 tracks.
- The batched GPU decoder reaches the reference decoder's optimum, plain,
  loss-augmented and as the oracle, on 4 lattices.
- The decoder agrees with brute force on 20 small random lattices.
- Theorem 2 on real lattices: the summed step errors of random labellings are
  never below their routes' real error.
- The noise fit recovers a known model.

## Not done yet

- **Training does not yet beat the starting weights.** The structured hinge
  trains and converges: the proximal bundle method takes 15–30 passes, and
  oracle ties are broken by cost (CCCP). But in-sample it scores 98.37%
  (λ = 1e-2) against 98.45% for w0. The cause is measured. The hinge comes
  to about 3.4 per metre of route while the real error is 0.047, and it is
  dominated by parallel roads: routes 11 m from the fixes that are hundreds
  of metres wrong, which the margin asks to separate by their full error.
  This is the next thing to work on.
- Cross-validation, the ablations and the paired bootstrap against the MDP
  have not been run (the code for the CV and ablation runs is in
  `src/learned/cv.py` and the `cv` stage).
- student_test was scored once here, for this record.

## Own choices in this version

- float32 feature storage instead of the planned float16. The length feature
  passes float16's limit of 65 504 on long detours. On the GPU, features are
  float16 after dividing by a per-feature scale.
- Coverage is estimated from at most 16 evenly spaced fixes per step.
- The skip's starting weight is the median of Wöltche's fail cost.
- Points are placed on the true route only on arcs they have a candidate on.
  This changes only the training targets, not the matching.
- Track 33's noisy copies are excluded from training, for memory. This does
  not affect this record, which is untrained.
- The measure-stage grids and quantiles are listed in
  `src/learned/hyperparameters.py`, each marked OWN CHOICE.

## Reproduce

```
uv run python -m learned measure  data/corpus_fixed data/corpus_original results
uv run python -m learned lattices data/corpus_fixed data/corpus_original results
uv run python -m learned prepare  data/corpus_fixed results
uv run python -m learned freeze   results
uv run python -m learned match    data/corpus_fixed results
uv run python -m benchmark data/corpus_fixed results learned
```

`freeze` writes `parameters.json`'s `start_weights` to `weights.json`. The
`train` stage would overwrite that file with trained weights.
