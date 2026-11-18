"""Given values for MCM: Li, Wang, Li & Xu, "A robust map matching method by
considering memorized multiple matching candidates", Theoretical Computer
Science 941 (2023). Every value is the paper's, with its section, unless
marked OWN CHOICE or taken from the authors' code (github.com/lindalee-
inlab/MCM, lcsg_q.m).
"""

# Section 8.2.1: the point-to-road similarity is EPSILON - distance within
# EPSILON of a fix, else 0. (The authors' test.m uses 10; the paper's
# experiments on this dataset use 30.)
EPSILON: float = 30.0  # metres

# Definition 10: an arc may follow another when the driving distance
# between them is below EPSILON_TOPO. The paper gives no value; the authors'
# code compares that distance with the same threshold as EPSILON.
EPSILON_TOPO: float = 30.0  # metres

# OWN CHOICE: the spatial index's cell size; it changes speed, not results.
GRID_CELL: float = 100.0  # metres

# OWN CHOICE: tracks matched in parallel (this machine has 16 threads).
WORKERS: int = 16
