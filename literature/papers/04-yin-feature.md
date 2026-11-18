# Feature-based Map Matching for Low-Sampling-Rate GPS Trajectories

Yifang Yin, Rajiv Ratn Shah, Guanfeng Wang, Roger Zimmermann. ACM Transactions on Spatial Algorithms and Systems (2018).

What I read on 2026-11-18: everything (sections 1–6). The tables come from the layout text, because the paragraph-joined text scrambled them.

## The construction

1. Douglas–Peucker simplification (ε = 0.001°) gives key points. The trajectory between them is split into segments.
2. Each key point gets its 10 nearest road segments as candidates. The emission is Gaussian on distance (σ = 10 m).
3. An action graph: the nodes are road segments, and the edges are turns between connected segments. An action's cost is C = min(dist(e, segment of the track), 100 m) × (length(e) + ω·turn_cost(angle)), with ω = 100. The turn cost follows Osogami & Raymond. The product (not a sum) keeps the trajectory cost from scaling with the number of road pieces.
4. The transition between candidates of neighbouring key points is the shortest path in the action graph. Its probability is exp(−N_seg/n · C*/dist(key points)).
5. Viterbi over the key points.

## Evidence

On all 100 Kubicka tracks, subsampled to 60 to 300 s. Average RMF (Table 3):
| interval | HMM | IRL (Osogami & Raymond, learned weights) | feature-based |
| 60 s | 0.0657 | 0.0854 | 0.0578 |
| 120 s | 0.1189 | 0.1157 | 0.0975 |
| 180 s | 0.1773 | 0.1621 | 0.1406 |
| 240 s | 0.2610 | 0.2085 | 0.1917 |
| 300 s | 0.2856 | 0.2147 | 0.2009 |
F1 at 60 s is 0.968 for HMM, 0.958 for IRL and 0.972 for feature-based. The IRL method is worse than the HMM at 60 s and much worse on hives, because it fits every point.

## Limits

Nothing at 1 Hz is reported. Loops hurt it in online mode, because simplification loses the loop's shape.

## Where it sits in our matcher

It is the main evidence that turn and length costs plus simplification beat the plain HMM on this dataset, and the only learned-weights number on it (IRL). It supports giving the MDP reward turn and length terms. It also warns that learning weights that fit every point can hurt on noisy tracks.
