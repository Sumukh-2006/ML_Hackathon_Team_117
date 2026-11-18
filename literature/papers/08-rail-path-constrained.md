# Path-Constrained State Estimation for Rail Vehicles

C. von Einem, A. Cramariuc, R. Siegwart, C. Cadena, F. Tschopp. arXiv 2308.12082 (2023).

What I read on 2026-11-18: sections I–III-D (introduction, related work, map structure, path-constrained EKF, multi-hypothesis tracking). I skipped the experiments section except the abstract numbers.

## The construction

The state is ⟨track id, s, ṡ⟩: which track, the distance along it, and the speed along it. This is literally the vehicle moving on the network. A constant-velocity EKF propagates s and ṡ. GNSS fixes are projected onto the track, and the perpendicular error ε drives an acceptance gate: rejected fixes inflate the covariance instead of updating. At every switch a new EKF is spawned for each branch; hypotheses that disagree with the measurements are pruned.

## Evidence

Zurich trams, with VIO plus standard GNSS against RTK. RMSE 4.78 m, and track selectivity up to 94%.

## Limits

Online, with odometry from visual-inertial sensing. It is multi-hypothesis by pruning, so not exact. It is tested on rail, where speed changes slowly.

## Where it sits in our matcher

It shows the full physical state in its cleanest form. For us an exact offline decoder dominates an online pruned one. What we take is the state idea, distance and speed along the arc, and the acceptance gate, which maps onto our skip.
