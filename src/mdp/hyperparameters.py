"""Given values for Woeltche's MDP matcher. Source: the defaults of
map_matching_2 (src/library/include/matching/settings.hpp) and its MDP
environment (environment/environments/single.hpp), which the user chose
over the paper's Eq. 3.3 (Transactions in GIS 27(7), 2023). Values marked
OWN READING or OWN CHOICE are not in either.
"""

# Trajectory simplification (thesis 3.1.1).
WARP_SPEED: float = 1500.0 / 3.6     # m/s: a point needing more is dropped
SIMPLIFY_TOLERANCE: float = 5.0      # m: Douglas-Peucker
MEDIAN_MERGE_TOLERANCE: float = 10.0  # m: point clouds merge to their median
# Adaptive median merge: the radius grows to half the distance to the
# nearest road when that is larger.
MEDIAN_MERGE_ROAD_SHARE: float = 0.5

# Candidate search, "combined" (thesis Def. 3.2).
K_NEAREST: int = 16          # nearest road edges per point
RADIUS: float = 200.0        # m; also the lower limit of the adaptive radius
RADIUS_UPPER: float = 10000.0  # m; the radius doubles up to this when empty

# Route search: Dijkstra stops at this factor times the longest straight
# distance from the start node to the goal nodes.
# OWN READING: the bound is applied as a cap on the node-to-node distance.
ROUTING_MAX_DISTANCE_FACTOR: float = 5.0

# Reward weights (paper Table 3; settings.hpp mdp_*_factor).
DISTANCE_WEIGHT: float = 0.8
LENGTH_WEIGHT: float = 0.3
AZIMUTH_WEIGHT: float = 0.5
DIRECTION_WEIGHT: float = 0.6
TURN_WEIGHT: float = 0.5

# single.hpp: the first point's distance counts 3 times, the last point's
# distance 2 more times, and a point right after a skip 0.1 times.
FIRST_DISTANCE: float = 3.0
LAST_EXTRA_DISTANCE: float = 2.0
AFTER_SKIP_DISTANCE: float = 0.1

# single.hpp: a route longer than this many times the track segment, or no
# route at all, makes the agent skip to a fresh start at the next point.
SKIP_ROUTE_FACTOR: float = 2 * ROUTING_MAX_DISTANCE_FACTOR

# single.hpp _fail_error: 3 x (track length + total absolute turning), plus
# for every point a Gaussian-weighted (min 2, max 5, sigma 3 points) share
# of its segment length and turn.
FAIL_BASE: float = 3.0
FAIL_WEIGHT_MIN: float = 2.0
FAIL_WEIGHT_MAX: float = 5.0
FAIL_WEIGHT_SIGMA: float = 3.0

# Value iteration's discount (settings.hpp).
DISCOUNT: float = 0.99

# OWN CHOICE: the spatial index's cell size; it changes speed, not results.
GRID_CELL: float = 200.0  # m

# OWN CHOICE: tracks matched in parallel (this machine has 16 threads).
WORKERS: int = 16
