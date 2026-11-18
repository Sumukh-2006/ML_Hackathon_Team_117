# Findings: SoTA on the Kubicka map-matching dataset, and five approaches

Written 2026-11-18, after the Cleave pass (project 32, log in `cleave/`) and the
web pass (notes in `papers/01..05`). The review log is `review.md`.

## The task and the score

Each folder gives a road graph (`.nodes`, `.arcs`), a 1 Hz GPS track (`.track`)
and the true route (`.route`, a chain of arc ids). We must output the arc chain.

Papers on this dataset use two scores, and they don't convert into each other.

- **RMF**, the route mismatch fraction (Newson & Krumm): wrongly added length
  plus wrongly missed length, divided by true length. Lower is better. Used by
  Yin et al. and by GSMM.
- **Correct fraction**: overlap divided by the longer route, length-weighted.
  Higher is better. Used by Wöltche (map_matching_2).
- MCM uses a third, point accuracy: the share of GPS points put on the right
  segment.

We'll compute RMF and the correct fraction both, so we can stand next to either.

## What surprised me

Cleave surfaced about 15 learned map matchers: DiffMM, TRMMA/MMA, RNTrajRec,
RLOMM, SceneGTMM, the transfer-learning transformer, and others. I grepped the
full text of every one of them for this dataset. **None evaluates on it.** They
cite Kubicka's 2018 survey, but they train and test on Porto, Beijing, Chengdu
or Tokyo taxi data, at sparse sampling, with hundreds of thousands of
trajectories.

The papers that do report on this dataset are all non-neural. The only learned
method with a number here is the IRL-weighted HMM in Yin et al.'s table, and it
does *worse* than the plain HMM at 60 s.

One team (arXiv 2402.11866) tried GNN and CNN pipelines around this dataset and
gave up.

The second surprise is the SoTA itself. Wöltche's "MDP with reinforcement
learning" is not trained on anything. It is a deterministic sequential decision
problem, solved exactly by value iteration (a backward dynamic program, the same
family as Viterbi), with five hand-weighted geometric penalties and a "skip"
action. Its edge over the HMM comes from the reward design and the skip, not from
learning.

## The SoTA on this dataset

**map_matching_2 (Wöltche, Transactions in GIS 2023; PhD thesis 2024)** scores
98.73% weighted correct fraction on all 100 Worldwide tracks at native 1 Hz. In
"export edges" mode, which outputs whole edges just as our `.route` does, it
scores 98.93%. The source is the repository's benchmark table; the note is
`papers/01`. The weights are hand-set defaults, so this is not a held-out score.
The comparison table against other matchers is in the paper and thesis, which I
couldn't get (see "Unread").

## Five approaches, each deciding the path a different way

| # | Approach | How it decides the path | Evidence on this dataset |
|---|---|---|---|
| 1 | **HMM + Viterbi** (Newson & Krumm 2009) | Each point has candidate roads. Emission is Gaussian in distance. Transition is exponential in \|straight-line − route distance\|. Viterbi finds the most likely chain. | RMF 0.0657 at 60 s (Yin). ~93% point accuracy at 1 s (MCM paper). |
| 2 | **MDP + value iteration** (Wöltche 2023) | The state is the previous point's choice. The reward is minus a weighted sum of distance, length mismatch × detour factor, azimuth, turn and route wiggle. "Skip" drops outliers at a fixed cost. Preprocessing: warp removal, Douglas–Peucker at 5 m, median merge of hives, candidate adoption. | **98.73% / 98.93%** correct fraction at 1 Hz, all 100 tracks. |
| 3 | **Segment-based feature matching** (Yin et al., ACM TSAS 2018) | Simplify the track to key points. Between them, take the shortest path in an "action graph" whose cost is distance-to-segment × (length + ω·turn). Viterbi over key points only. | Best RMF at every rate from 60 s to 300 s: 0.0578 at 60 s against the HMM's 0.0657. |
| 4 | **Multiple-candidate similarity DP** (MCM, TCS 2023) | No probabilities. A score matrix over (edge, point) keeps every "alive" route and backtracks offline, recursing over breaks. | ~92% at 1 s (HMM ~93%), ~95% at 10 s, and zero matching breaks against the HMM's 5–15. |
| 5 | **Ours: a route decoder whose scoring is learned from the annotated routes** (design below) | The same exact dynamic program as 2, over pairs of consecutive arcs. Each step is scored by a weighted feature vector that combines the cues from 1–4, and the weights are trained so that decoding reproduces the 80 annotated routes, with route-length error as the loss. | Target: >99.5%. To be measured. |

Replaced from the earlier list: a neural candidate scorer pretrained on
synthetic tracks (MMA/TRMMA, arXiv 2508.10460; the transfer-learning
transformer, arXiv 2108.00439; DiffMM, arXiv 2601.08482). No neural model
reports on this dataset.

Also seen and ruled out: GSMM (Fiedler et al.), a single global Dijkstra with
trace-similarity costs. It can't handle loops, and 24 of the 100 tracks have
them. The Fréchet-distance matchers (Cleave's graph blob) are geometric theory,
with no benchmark on this data.

## Approach 5, ours: what it takes from each of the others

The bet: the ~1% Wöltche still loses is mostly in places where the right answer
depends on how the annotator decided ambiguous cases. Those are gaps, hives, U-turns,
parallel roads, and track/map disagreements. Hand-set weights can't learn that.
Weights trained on the annotated routes can. The IRL result in Yin et al. shows
how not to do it: fitting each GPS point made learning *worse* on noisy tracks.
So our loss is measured on the route itself, in the same units as the score.

1. **Clean the track** (from Wöltche). Drop warps, meaning points that imply an
   impossible speed. Merge hives into one point (a median within a radius).
   Remember where the time gaps are.
2. **Candidates** (from Wöltche and MCM). For each kept point, the arcs within an
   adaptive radius, up to k of them. Both directions of a road are separate
   arcs, so direction is part of the choice. "Candidate adoption" also gives a
   point its neighbours' candidates, so an outlier still has the true arc on
   its list.
3. **Routes between candidates** (from the HMM). Bounded Dijkstra on the arc
   graph from each candidate to the next point's candidates. Working on arcs,
   not nodes, makes U-turns (arc → its reverse) and turn angles explicit.
4. **Features of one step**, from candidate a at point i to candidate b at
   point i+1:
   - distance from each point to its arc (HMM emission);
   - |straight-line − route length| and the detour ratio (HMM, Wöltche);
   - the azimuth difference and the turn difference between track and route
     (Wöltche);
   - the sum of turn angles along the route and the U-turn count (Yin, Osogami
     & Raymond);
   - backtracking over the previous step (Wöltche's zig-zag);
   - **coverage**, new here. The length-weighted distance from the route to
     the stretch of track between the two points, measured both ways. It
     catches a parallel road or a spur that the endpoint features can't see.
     It's the Fréchet line's idea in a form a dynamic program can use.
   - a **skip** action with its own weight, so one bad point never breaks the
     route (Wöltche's skip, MCM's "never break").
4b. **Physics features** (added 2026-11-18, after a second Cleave pass,
   project 33, and a web pass; notes `papers/06`–`09`). The car's speed along
   the road is part of what is scored. Precedents: Goh et al. 2012 (distance
   discrepancy and momentum change, fused by a learned SVM); Stas et al. 2025
   (transitions from predicted position p + v·Δt); rail tracking (state = arc,
   distance along it, speed along it). None was applied to this dataset or
   combined with an exact offline decoder.

   The measured constraint: one-second speeds are mostly noise (the median
   change between consecutive seconds is 4.8 m/s), while 5-second windows give
   real accelerations (median 0.29, p99 1.7 m/s²). So the physics is scored
   over steps of several seconds, which the Douglas–Peucker cleaning already
   produces. The features are:
   - **speed consistency**: |route length / Δt − the speed measured from the
     fixes around that step|. This is Goh's dead-reckoning discrepancy, with
     the speed taken from positions instead of a speedometer. It is decisive
     across gaps: a detour through a 30 s dropout implies a speed the car had
     neither before nor after.
   - **longitudinal acceleration** between consecutive steps, from the two
     along-route speeds. The state already holds the previous step, so this
     costs nothing extra in the decoder.
   - **lateral acceleration** at the route's turns: speed² × the curvature of
     the turn. A ramp taken at 30 m/s, or a junction turn taken without
     slowing, is physically implausible. This is the cue that elastic pathing
     relies on ("slow before a turn").
   - **stopped**: a speed near zero favours staying on the same arc (hives).

   The scales of all four (speeds, accelerations) are measured on the 80
   annotated routes. They are not chosen.

5. **Decoder.** Exact dynamic programming (Viterbi, which on these
   deterministic steps is what Wöltche's value iteration computes). The state
   is the previous arc and the current arc, so turn and backtracking features
   can be scored.
6. **Learning.** The score is w·φ. w is trained with a structured large-margin
   loss (structured perceptron or structured SVM, starting from Wöltche's
   defaults). The loss is the length of wrongly added plus wrongly missed arcs
   against the annotated route. That loss splits over arcs, so "the most
   violating route" comes from the same decoder. Cross-validation goes over
   the 80 train folders, and student_test is scored once at the end.
7. **Check.** Every output arc chains to the next, and each track gets an error
   breakdown by tag (gap, hive, U-turn, loop, incongruent). Anything left over
   gets studied, not tuned blindly.

**Target (decided 2026-11-18):** more than 99.5% on Wöltche's measure, the
length overlap divided by the longer route, length-weighted over tracks. It is
directly comparable to his 98.73/98.93%. RMF and arc F1 are reported alongside.

What could stop us short of 99.5%: annotation errors in the routes themselves.
Wöltche calls these unavoidable. The per-track error breakdown after the
baseline tells us how much of the remaining error is label noise, before we
spend effort on the last half percent.

## Recommendation (before approach 5 replaced the neural scorer)

Build **approach 2, the MDP**, as the architecture, and make its five reward
weights the learned part. Fit them on the 80 train tracks against RMF, by
coordinate search first. Keep **approach 1, the HMM**, as the baseline, since it
shares candidates, routing and scoring with the MDP and costs little extra.

Why not lead with the neural model (5)? There is no number for it on this data,
the IRL result in Yin et al. shows that learned weights fitted to every point can
lose to the HMM on noisy tracks, and our tracks are dense 1 Hz, which is the
regime where geometry already gets ~99%. If the hackathon needs a neural
component, the place for it is inside the MDP: an MMA-style scorer replacing the
hand-made distance term, pretrained on synthetic tracks drawn from our graphs.
That's an extension we'd test against the tuned MDP, not a replacement.

The decision is yours.

## Unread, and why

- Wöltche 2023 (Transactions in GIS) and the thesis (doi 10.20378/irb-116612):
  Wiley blocks scripted downloads, and the thesis repository is campus-only.
  These hold the benchmark against FMM, Barefoot, GraphHopper and others on this
  set. If you can open them from the university network, their tables would
  complete the leaderboard.
- Kubicka et al. 2015 (the dataset) and 2018 (the comparative study): HAL served
  HTML, not the PDF.
- The trend-HMM paper (RMF up to 17.58% better than HMM on this set):
  ResearchGate returned 403.
- The neural papers (DiffMM, TRMMA, RNTrajRec, RLOMM, SceneGTMM, the
  transfer-learning transformer): downloaded to `pdfs/`, but only the abstracts
  and dataset sections were read. None evaluates here. They need a full read
  only if you pick approach 5.
