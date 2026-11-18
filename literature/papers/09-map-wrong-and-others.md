# Context papers on motion models (short notes)

Read on 2026-11-18: abstracts plus method skims.

- **Map matching when the map is wrong** (arXiv 1809.09755). Sample-based (HMM or particle) map matching extended with an off-road mode, an extended Kalman filter for free-space motion. Tracking switches between on-road and off-road when the map can't explain the motion. For us, it is a way to treat the "severe incongruence" tracks, where the annotated route and the track disagree.
- **Elastic pathing** (arXiv 1710.06932). It reconstructs routes from speed alone, given the start: the speed profile is fitted against the map, with stops at turns and slowdowns before curves. Destinations are within 250 m for 15–17% of traces. This shows a speed profile alone carries information about which turns were taken, the "slow before a turn" physics.
- **Neural Augmented Kalman Filters for Road Network assisted GNSS positioning** (arXiv 2507.00654). A temporal GNN picks the road segment and its uncertainty, which then serve as a Kalman measurement update. Online positioning, not route output.
- **Arc-length-based map matching** (arXiv 2410.12208). Dead reckoning from steering and wheel speed, corrected by matching arc length to the map. Needs vehicle sensors we don't have.
- **Gustafsson et al., Particle filters for positioning, navigation and tracking** (IEEE TSP 2002). The classic road-constrained particle filter. Not read: the server at liu.se didn't respond.
