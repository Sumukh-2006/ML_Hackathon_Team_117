# A robust map matching method by considering memorized multiple matching candidates (MCM)

Wanting Li, Yongcai Wang, Deying Li, Xiaojia Xu. Theoretical Computer Science 941 (2023) 104–120, doi 10.1016/j.tcs.2022.10.032. Code at github.com/lindalee-inlab/MCM.

What I read on 2026-11-18: pages 1–6 and 10–16 of the PDF (the introduction, related work, problem model, online and offline algorithms, experiments, conclusion). I skipped pages 7–9 (the score-matrix update details and proofs of the online variant).

## The construction

- Point-to-road similarity is S(p,e) = ε − dist(p,e) if dist < ε, else 0, with ε = 30 m.
- A score matrix M has rows for road edges and columns for points. M(e, t) is the best summed similarity of a route ending on e at point t.
- A column is filled only from "alive" routes. The last (edge, point) pair of each live route sits in a queue, and an edge can follow only if it is reachable within (half the gap between points + an error margin). That reachability is a Dijkstra distance.
- Offline: take the maximum of the last column and backtrack. If the match broke somewhere, zero out the matched block and recurse on the unmatched sub-trajectories.
- There are no probabilities and nothing is trained.

## Evidence

It uses the Kubicka set under the name "Global dataset": 100 tracks, 73 with gaps, 25 with U-turns, 24 with loops, 13 with hives, 20 with severe incongruence, and 19 clean. The point-to-route median absolute deviation is about 1–3 m and the maximum is up to ~100 m (Fig. 10).

Results are point accuracy (the % of points on the correct segment), read from Fig. 11b:
- 1 s: MCM ~92.2%, offline HMM (Newson–Krumm) ~93.2%;
- 10 s: ~95.2% against ~92.5%;
- 240 s: ~85% against ~75%.
MCM has zero matching breaks at every rate, where the HMM has 5 to 15. Runtime is 54.7 s against the HMM's 171.3 s for the whole set.

## Limits

The metric is point accuracy, not RMF, so the numbers can't be compared with Yin's or Wöltche's. At 1 Hz it is slightly worse than the HMM. Its gains show up when the track is subsampled.

## Where it sits in our matcher

It is an alternative to the HMM/MDP chain: a multiple-hypothesis similarity DP. The robustness idea (never break; recurse on gaps) is covered in the MDP by the skip action.
