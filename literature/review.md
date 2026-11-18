# Literature review log: map matching on the Kubicka dataset

## Cleave pass, 2026-11-18 (project 32)

Raw log: `cleave/session-mapmatch.txt`. The embedder was local: llama-server
with Qwen3-Embedding-8B-Q8_0 on 127.0.0.1:8096. The tunnel port 8095 was down,
so Cleave fell back to 8096. The cover was 8000 papers, and the goal was sent
once:
"map matching of noisy GPS trajectories onto a given road network graph,
recovering the exact sequence of road segments driven".
The root has 10 blobs.

### Where the papers were

**7 Trajectory Data Publication → 7 Survey of Map-Matching Algorithm** (33
papers, a final blob). This is the core of the field. With
`prefer! evaluated on a public map-matching benchmark dataset with ground-truth routes, reporting route mismatch accuracy`
the order was:
Route Based Map Matching via a Structured Codebook and Token Sequence Decoding;
A Survey on Map-Matching Algorithms; Map Matching Algorithm for Large-scale
Datasets (GSMM); Map matching when the map is wrong; Driving with Context; A
Comparison and Evaluation of Map Construction Algorithms; The path inference
filter; A Force-Directed Approach for Offline GPS Trajectory Map Matching;
semMatch; Multi-track Map Matching; Offline Map Matching Based on Localization
Error Distribution Modeling; Deformable Map Matching for Uncertain Loop-Less
Maps; Continuous Map Matching to Paths under Travel Time Constraints; An
effective Time-Aware Map Matching process for low sampling GPS data; RLOMM;
DiffMM; Efficient Methods for Accurate Sparse Trajectory Recovery and Map
Matching; Transformer-based Map Matching Model with Limited Ground-Truth Data;
Surrogate Modeling of Trajectory Map-matching (transformer seq2seq); A
Map-matching Algorithm with Extraction of Multi-group Information; Using
iterated local alignment; Map matching queries under the Fréchet distance;
SceneGTMM; RSS-based map matching; Kharita; Two Online Map Matching Algorithms
(AHP, fuzzy); Hierarchical Spatial-Temporal Graph-Enhanced Model for
Map-Matching; UM3; Fast 2D Map Matching Based on Area Graphs; Enhancing
Interactive Voting-Based Map Matching; Zero-Shot Cellular Trajectory Map
Matching; Road Graph Generator; Reconstructing Movement from Sparse Samples.
Taken: all of the GPS-to-road ones. Not taken: Kharita, Road Graph Generator
and the map-construction comparison (map inference), UM3 (map-to-map),
Area Graphs (robot maps), RSS and cellular (not GPS).

**7 → 2 Survey on the Pre-training of Trajectory Embeddings** (79, final).
With `prefer! recovering the road segment sequence from GPS points, map matching or trajectory recovery on a road network`
the top was: RNTrajRec; Micro-Macro Spatial-Temporal Graph-based Encoder-Decoder
for Map-Constrained Trajectory Recovery (MM-STGED); TRACE; COLTRANE (map
inference, not taken); Dual-Decoding Framework; Time-Aware Transformer for
Trajectory Recovery; Critical Perceptual Pre-trained Model; PTR;
Diff-RNTraj; BERT4Traj.
This is the learned trajectory recovery line, which outputs a road segment and
a position ratio per timestep, from sparse input. Taken: RNTrajRec, MM-STGED,
TRACE, Diff-RNTraj. The rest of the blob is representation learning,
similarity and generation, which is context only.

**1 Generic algorithm for approximation [matching, graphs]** (3085). The title
is a lexical false match: graph matching, TSP, shortest paths. I ran
`must!! matching a GPS trajectory or polyline to a path in a road network graph`
here (1685 dropped, 1400 left), and `ls` brought up the Fréchet line:
Intermittent Map Matching with the Discrete Fréchet Distance; Map-Matching
Queries under Fréchet Distance on Low-Density Spanners; Map Matching with
Simplicity Constraints; Locally Correct Fréchet Matchings. This is a separate
geometric theory family, and it is taken. Child 1.1 (noisy network inference)
turned out to be curve and shape metrics, a dead end. The filter was removed
afterwards.

### Dead ends (looked at, nothing taken)

- 2 Localization and Mapping (1514). Under the `must!!` GPS-to-road filter (794
  left), the top was visual localization, HD maps and SLAM.
- 3 Path planning and object tracking (681). Under the same filter (371 left),
  the top was multi-target tracking, PHD filters and roadmaps.
- 4 Autonomous driving priors: trajectory prediction, lanes, road extraction
  from satellite images (that last is map inference).
- 5 Localization error, GNSS/Kalman, indoor, radio maps.
- 6 Traffic estimation and prediction.
- 9 VANETs, ride hailing, graph construction.
- 10 Graph2Route: routing, VRP, language navigation. One possibly useful item
  for transition priors: Massively Scalable Inverse Reinforcement Learning in
  Google Maps.

## Web pass

(continues below)
Fetched the Cleave papers from arXiv into `pdfs/` (20 papers) and grepped every
full text for the dataset ("Kubi", "zenodo 57731", "global dataset", "RMF").
None of the arXiv learned matchers evaluate on it. They cite only the Kubicka
2018 survey.

Web searches for the dataset's citers turned up:
- Wöltche, Transactions in GIS 2023 (an MDP matcher, benchmarked on the
  "Worldwide" set). The paper and thesis were unreachable (Wiley bot block;
  thesis campus-only). Read instead: the README, the Kubicka benchmark notes,
  and the MDP source of github.com/addy90/map-matching-2. → `papers/01`.
- MCM, TCS 2023 (uses it as the "Global dataset"). → `papers/02`.
- GSMM, arXiv 1910.05312 (uses the tracks without cycles). → `papers/03`.
- Yin et al., ACM TSAS 2018 (all 100 tracks, RMF tables). → `papers/04`.
- Lin et al., arXiv 2402.11866 (qualitative only; failed deep learning). →
  `papers/05`.
Not reachable: the Kubicka 2015 and 2018 PDFs (HAL served HTML), and trend-HMM
(ResearchGate 403).

The result is in `findings.md`.

## Physics pass, 2026-11-18 (Cleave project 33)

Raw log: `cleave/session-physics.txt`. Local embedder; cover 8000.
Goal: "tracking a vehicle on a road network with a motion model of its position
and speed along the roads, constrained to the road graph, from noisy GPS
positions". Root: 10 blobs.

- **3 Bayesian estimation with particle filters** (1150). I applied
  `must!! a vehicle or target whose motion is constrained to a road network map, tracked with a motion model`
  (549 dropped, 601 left). The `ls` top was mostly generic multi-target
  tracking (PHD, multi-Bernoulli), which is a dead end. Child 3 (smoothing)
  gave context only: Student-t robust Kalman smoothers and Kalman smoothing as
  optimisation.
- **6 Localization** (746). With
  `must!! map-matching or road-network constraints in vehicle position estimation with a motion or dynamics model`
  (400 left), I took: Neural Augmented Kalman Filters for Road Network assisted
  GNSS positioning; Path-Constrained State Estimation for Rail Vehicles;
  Arc-Length-Based Map Matching. Seen but not taken: route-constrained
  MEMS/GNSS fusion and tram positioning (they need inertial sensors).
- **1 Dynamic geospatial networks** (1786). With
  `must!! map matching GPS trajectories to roads using vehicle speed, travel time or kinematic constraints`
  (934 left), I took Elastic Pathing (routes from speed alone). The rest was
  time-dependent routing, a dead end.
- Not dived: 2 trajectory prediction, 4 motion planning, 5 generative CAVs,
  7 SLAM, 8 MOT, 9 radar, 10 LiDAR. These are prediction, planning and
  perception, not estimation on a given map.

Web: Gustafsson et al. 2002 (the road-constrained particle filter; the liu.se
server doesn't respond, so unread); Goh et al. ITSC 2012 (read); Stas et al.
arXiv 2505.04763 (read). Notes are `papers/06`–`09`.

Measured on train: the median change of 1-s speed between consecutive seconds
is 4.8 m/s (p99 22.9). The median 5-s-window acceleration is 0.29 m/s²
(p90 0.85, p99 1.68).
