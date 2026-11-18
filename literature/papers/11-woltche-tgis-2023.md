# Open source map matching with Markov decision processes: a new method and a detailed benchmark with existing approaches

Adrian Wöltche, Transactions in GIS 27(7):1959-1991, 2023, doi 10.1111/tgis.13107. The user supplied the PDF on 2026-11-18. Also used: his PhD thesis "Map matching methods for assigning vehicle positions to road networks" (Bamberg, doi 10.20378/irb-116612), from which I read sections 3.1.1-3.1.7 (the method) and 5.4.4 (our dataset).

What I read on 2026-11-18: the paper in full, all 33 pages. Tables 3 and 8 come from the layout text.

## The construction

1. **Track simplification** (paper 2.1, thesis 3.1.1), in this order:
   - remove spatial duplicates;
   - warp detection: drop a point needing more than 1500 km/h;
   - Douglas-Peucker, removing reverse movements within the tolerance;
   - median merge: collapse runs of directly connected points within 10 m into their median point. The radius is enlarged "by up to half the distance to the next road" for off-road clouds. Thesis rationale: a car under 5 m long in a lane under 4 m wide, times two.
2. **Candidate search** (thesis Def. 3.2): combined k-nearest (k = 16) and circular search. The adaptive radius is the smaller of the distances to the previous and next track point, but at least the minimum radius; it is doubled up to an upper limit when nothing is found. Each candidate is the closest point of each directed edge, so a two-way road gives two candidates at the same spot.
3. **Candidate adoption** (thesis 3.1.3):
   - "siblings": adopt the candidate sets of the previous and next points;
   - "nearby": adopt every other set's candidates within this set's range.
   Both run in sequence, adopting each candidate at most once.
4. **MDP** (thesis 3.1.6, Fig. 3.11):
   - a state is s = (p_n, c_n^m, p_{n+1}): the current point, its chosen candidate, and the next point;
   - an action picks a candidate c_{n+1};
   - the reward (Eq. 3.3) is r = -[w_d d_c + w_l |l_t - l_r| + w_a |δ(a_t, a_r)| + w_adc adc_r + w_t |δ(t_t, t_r)|]. Here d_c is the distance from candidate to point; l_t and l_r are the track-segment and route lengths; a_t and a_r are their azimuths (the route's is the overall start-to-end azimuth); adc_r is the sum of absolute direction changes along the route; t_t is the track turn, and t_r the heuristic turn from the previous track segment's azimuth to the route.
   - The weights (Table 3) are distance 0.8, length 0.3, azimuth 0.5, direction changes 0.6, turn 0.5. They were tuned by Bayesian optimisation (gp_minimize) on the Hof dataset only.
   - "Skip": when no route exists between adjacent points, the agent can jump to a later candidate within the same model (thesis 3.1.6, "graceful recovery"). Neither text gives the cost of a skip.
   - Solver: value iteration (in place, beneficially preordered, reverse sweeps); Q-learning as a demonstration.
5. **Metrics are purely geometric.** Time is not used (thesis 3.1.6-3.1.7).

## Evidence on our dataset (original tracks, all 100)

- Paper Table 8, mean correct fraction:

  | configuration | mean CF |
  |---|---|
  | MDP + TS + CA | 98.48% |
  | MDP + TS | 98.40% |
  | MDP + CA | 97.35% |
  | MDP alone | 97.82% |
  | HMM + TS + CA | 97.00% |
  | HMM alone | 85.72% |
  | FMM (HMM) | 82.96% |

- Thesis Table 5.15, weighted CF, newer software:

  | configuration | weighted CF |
  |---|---|
  | MDP | 98.750% |
  | MDP without CA | 98.630% |
  | HMM (same metrics) | 97.051% |
  | Barefoot | 96.750% |
  | Valhalla | 64.18% |

- Track 54 could not be matched at all without TS and CA (it needed more than 1 TB of memory).
- The weights were not tuned on this dataset, so these are held-out numbers.

## Limits (stated)

The method makes wrong U-turns where a detour was right, picks the wrong one of several close parallel roads, and its weights may be biased toward the Hof data. "Ground truth" here is hand-corrected, and errors in it are acknowledged.

## Where it sits in our matcher

It is approach 2, the SoTA to beat: 98.75% on the clean originals; our tracks carry about 5 m of extra noise. The code (single.hpp) refines Eq. 3.3:
- the length term is multiplied by l_r / l_t;
- the first point's distance counts ×3, and the last point's gets an extra ×2;
- skips carry a fail cost of 3 × (track length + total track direction change) plus a local Gaussian-weighted term;
- a zig-zag term is used only with the larger state size.

The paper text does not have these. A faithful port must choose between them.
