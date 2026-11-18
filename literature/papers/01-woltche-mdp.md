# map_matching_2: map matching with Markov decision processes (Wöltche)

Adrian Wöltche. "Open source map matching with Markov decision processes: A new method and a detailed benchmark with existing approaches", Transactions in GIS 27 (2023), doi 10.1111/tgis.13107. Also his PhD thesis, doi 10.20378/irb-116612. Code: github.com/addy90/map-matching-2 (AGPL-3.0, C++).

What I read on 2026-11-18: the repository README in full; `data/kubicka-et-al/README.md`; the MDP environment `src/library/include/environment/environments/single.hpp` in full (581 lines); the default settings in `matching/settings.hpp`. I did not read the paper or the thesis. Wiley refuses scripted downloads, and the thesis repository (fis.uni-bamberg.de) is reachable only from the campus network. The benchmark against other matchers is in those two, so I have not seen it.

## The construction

The track is preprocessed first. Duplicate points are removed, and so are "warps": jumps that would need more than 1500 km/h. A Douglas–Peucker simplification with 5 m tolerance follows, then a "median merge" that collapses point clouds within 10 m (adaptive), which handles stops and hives.

Each remaining point gets candidates: up to k=16 nearest road positions inside an adaptive radius of at least 200 m. "Candidate adoption" then lets a point also use the candidates of its neighbours (siblings) and of anything close by. This is how a point that is an outlier still gets the right road among its options.

The MDP is defined as follows.

- A state is (index of the current point, chosen candidate at the current point, index of the next point). With state_size=3 it also keeps the previous pair.
- An action picks one candidate edge for the next point. There are two special actions, "no action" and "skip".
- The reward for going from candidate c at point i to candidate c' at point j is minus the sum of:
  - 0.8 × (distance of c' from its point + distance of c from its point);
  - 0.3 × |track-segment length − network route length| × (route length / track-segment length). Detours are punished more than shortcuts;
  - 0.5 × |track azimuth − route azimuth| in degrees;
  - 0.5 × |track turn angle − route turn angle|;
  - 0.6 × the sum of absolute direction changes along the route (wiggly routes cost more).
  - With state_size 3, two more terms: a turn-consistency term over three candidates, and 2 × the length of any "return line" where the route doubles back on itself (zig-zag).
- The first point's distance is weighted ×3, and so is the last point's ×2 extra, so the match starts and ends near the track.
- If no route exists between two candidates, or the route is more than 2 × 5 times longer than the straight line, the transition is replaced by "skip". The skip drops the next point and costs a fail penalty: 3 × track length + 3 × the total direction change of the track, plus a Gaussian-weighted local term. Skipping is expensive, but it never breaks the match.
- The routes between candidates come from an upper-bounded Dijkstra (single source, many targets), cached.
- It is solved by in-place value iteration with discount 0.99. Q-learning and policy iteration are options, and Q-learning is reported as less accurate.

My reading: the transitions are deterministic and the horizon is finite, so value iteration here is a backward dynamic program over a layered graph. It is the same kind of computation as Viterbi, over a state that remembers the previous choice. Despite the "reinforcement learning" label, nothing is learned from data. The five weights are hand-set defaults. The same tool's HMM mode uses the same five features with different weights (0.8, 0.1, 0.2, 0.8, 0.6) and does worse. So the gain comes from the reward design, the skip action and the second-order state, not from a learning step.

## Evidence

On the Worldwide (Kubicka) set, all 100 tracks at native 1 Hz, the weighted mean correct fraction is 98.73%, and 98.93% in "export edges" mode. That mode outputs whole edges, which is what our `.route` asks for. The metric is the overlap with the hand-corrected route, divided by the longer of the two, weighted by length. It is not RMF, though the same compare tool also computes an error fraction of added plus missed length.

Other sets: GIS Cup 98.59%, floating car data 99.65%, Seattle (Newson–Krumm) 99.89%. Matching all 100 tracks takes 8.2 s, multithreaded.

## Limits

The author notes that the hand-corrected ground truth has errors of its own, and that track ends cannot be matched perfectly. Those two limit any method to below 100%.

Seen by me: the weights are hand-tuned on data the author had in view, so the 98.73% is not a held-out number. The reward is linear in five features, which makes it a clean target for learning weights from our 80 train tracks.

## Where it sits in our matcher

This is the SoTA on this dataset, and the candidate architecture. Every stage maps onto ours: preprocessing, candidates, reward, DP, route stitching. The weights are the natural "learned" part. See [04-yin-feature] (simplification plus turn costs, the same idea at low sampling) and [02-mcm] (keeping alternatives alive, which the skip action plays the role of here).
