"""Given values for the HMM matcher of Newson & Krumm, "Hidden Markov Map
Matching Through Noise and Sparseness" (ACM GIS 2009). Every value is the
paper's, with its section, unless marked OWN CHOICE.

sigma and beta are not here: they are measured from the training routes by
`hmm.fit` (paper section 5.2) and saved with the results.
"""

# Section 3.1: a road segment farther than this from a fix is no candidate.
SEARCH_RADIUS: float = 200.0  # metres

# Section 4.1: a fix within this many sigma of the last kept fix is dropped.
PREPROCESS_SIGMAS: float = 2.0

# Section 5.2: median absolute deviation -> Gaussian standard deviation.
MAD_TO_SIGMA: float = 1.4826

# Section 4.2: a route longer than the great circle by this much is
# impossible (the route search stops there).
MAX_DETOUR: float = 2000.0  # metres

# Section 4.2: a route that needs a faster average speed is impossible.
MAX_SPEED: float = 50.0  # metres per second

# Section 4.2: points around a break are removed until it heals, or until
# the break spans more than this; then the track is split into trips.
MAX_BREAK: float = 180.0  # seconds

# OWN CHOICE: the spatial index's cell size; it changes speed, not results.
GRID_CELL: float = 200.0  # metres

# OWN CHOICE: tracks matched in parallel (this machine has 16 threads).
WORKERS: int = 16
