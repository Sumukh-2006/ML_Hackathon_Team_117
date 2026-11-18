"""Given values for approach 5. Values from Woeltche's map_matching_2 keep
his defaults; values that are measured from train are not here (they are
saved by the measure stage in results/learned/parameters.json).
OWN CHOICE marks a value neither the literature nor a measurement gives.
"""

# Candidate search (map_matching_2 defaults).
K_NEAREST: int = 16            # nearest road edges per decision point
RADIUS: float = 200.0          # metres; also the adaptive radius' lower limit
RADIUS_UPPER: float = 10000.0  # metres; the radius doubles up to this

# Route search stops at this factor times the longest straight distance from
# the start node to the goal nodes (map_matching_2 default).
ROUTING_MAX_DISTANCE_FACTOR: float = 5.0

# A step whose route is more than this many times the track segment is a
# skip (map_matching_2: 2 x the routing factor).
SKIP_ROUTE_FACTOR: float = 2 * ROUTING_MAX_DISTANCE_FACTOR

# Trajectory cleaning (map_matching_2 defaults; thesis 3.1.1). The
# Douglas-Peucker tolerance is not here: it is measured (derivation §3.1).
WARP_SPEED: float = 1500.0 / 3.6      # m/s
MEDIAN_MERGE_TOLERANCE: float = 10.0  # metres
MEDIAN_MERGE_ROAD_SHARE: float = 0.5  # adaptive: half the distance to a road

# The reference point density: the clean originals thinned at Woeltche's
# tolerance. The measured tolerance makes the noisy tracks match it.
REFERENCE_TOLERANCE: float = 5.0  # metres

# The measure stage (derivation §3.1, §5.3, §7).
# OWN CHOICE: the tolerance is searched by bisection over this range.
TOLERANCE_RANGE: tuple[float, float] = (5.0, 60.0)  # metres
BISECTION_STEPS: int = 10
# OWN CHOICE: the half windows (seconds) tried for the speed fit.
HALF_WINDOWS: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
# OWN CHOICE: true step speed is the original's path sampled every 5 s
# (finer sampling would add the originals' own GPS jitter to the travel).
TRUE_SPEED_SAMPLE: int = 5  # seconds
# OWN CHOICE: "impossible speed" starts above this percentile of true
# step speeds on train.
TOO_FAST_QUANTILE: float = 99.9
# OWN CHOICE: lags (seconds) over which the drift's autocorrelation is fitted.
FIT_LAGS: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NOISE_SEED: int = 0
# Noisy copies of every train original (plan: 8 to start).
COPIES: int = 8

# OWN CHOICE: coverage is a mean distance, estimated from at most this many
# evenly spaced raw fixes of a step (a long step can span 100+ fixes, and
# every one costs a pass over every route of the step).
COVERAGE_FIXES: int = 16

# OWN CHOICE: the spatial index's cell size; it changes speed, not results.
GRID_CELL: float = 200.0  # metres

# OWN CHOICE: tracks processed in parallel (this machine has 16 threads).
WORKERS: int = 16

# Training (derivation §6).
# OWN CHOICE: pairs sampled per lattice to set the float16 storage scale.
SCALE_SAMPLE: int = 200_000
# OWN CHOICE: pairs per GPU batch; the batch's tensors take about 50 bytes
# a pair on the 8 GB card.
BATCH_PAIRS: int = 40_000_000
# OWN CHOICE: the bundle method stops when the decrease its model predicts
# is within this share of J, or after this many decoding passes.
CUTTING_TOLERANCE: float = 1e-3
CUTTING_ROUNDS: int = 60
# OWN CHOICE: the oracle's tie-break may trade at most this many metres of
# error per track for a cheaper labelling; and at most this many CCCP
# rounds of choosing oracles.
ORACLE_TIE: float = 1e-3
ORACLE_ROUNDS: int = 6
# OWN CHOICE: projected-gradient iterations for the small dual over the
# planes.
DUAL_ITERATIONS: int = 20000

# Cross-validation (derivation §8).
FOLDS: int = 5
FOLD_SEED: int = 0
# OWN CHOICE: track 33's copies are left out of training. Its points have a
# median of 492 candidates (hand-tagged hives), so its 8 copies hold about
# 220M pairs, 40% of all training pairs, more than the host memory holds
# with the rest. Its own track stays in.
EXCLUDED_COPIES: set[str] = {"00000033"}
# OWN CHOICE: the regulariser strengths tried (lambda, per squared cost
# per metre).
STRENGTHS: list[float] = [1e-4, 1e-3, 1e-2, 1e-1]
