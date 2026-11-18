"""Given values for Yin, Shah, Wang & Zimmermann, "Feature-based Map Matching
for Low-Sampling-Rate GPS Trajectories" (ACM TSAS, 2018). Every value is
the paper's, with its section, unless marked OWN READING or OWN CHOICE.
"""

import math

# Section 5.1: Douglas-Peucker tolerance for key points, "0.001".
# OWN READING: in degrees of the lon/lat coordinates (about 70-111 m);
# metres would keep almost every point, which the paper's Fig. 9 rules out.
KEY_TOLERANCE: float = 0.001  # degrees

# Section 4.2.2: the nearest ten road segments of a key point.
CANDIDATES: int = 10

# Section 5.3: the standard deviation of GPS error in the emission.
SIGMA: float = 10.0  # metres

# Section 4.1.2: the trajectory cost of a road is capped at 100 m.
MAX_TRAJECTORY_COST: float = 100.0  # metres

# Sections 5.2-5.3: the weight of turns against road length.
OMEGA: float = 100.0

# Section 4.1.2, Eq. 4: turn cost 0 below 45 degrees, 1 below 135, else 2.
GENTLE_TURN: float = math.pi / 4
SHARP_TURN: float = 3 * math.pi / 4

# OWN CHOICE: the first search radius for the nearest arcs; it doubles until
# ten arcs are found, so it changes speed, not results.
FIRST_RADIUS: float = 100.0  # metres

# OWN CHOICE: the spatial index's cell size; it changes speed, not results.
GRID_CELL: float = 100.0  # metres

# OWN CHOICE: tracks matched in parallel (this machine has 16 threads).
WORKERS: int = 16
