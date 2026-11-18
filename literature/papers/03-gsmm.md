# Map Matching Algorithm for Large-scale Datasets (GSMM)

David Fiedler, Michal Čáp, Jan Nykl, Pavol Žilecký. arXiv 1910.05312 (VEHITS 2022).

What I read on 2026-11-18: the full text through ar5iv (sections I–VII). I did not read the figure values, because the results exist only as plots.

## The construction

There is no per-point decoding. It is one Dijkstra-like search from a start edge (the one minimising distance to the first point plus distance to the trace) to an end edge. The cost of a neighbour is (c1 + c2)·l_e/α + c3. Here c1 is the node-to-trace distance, with chronological projection. c2 is the edge-midpoint-to-trace distance. c3 is |edge length − along-trace length between the projections|. A forward heuristic −β·(progress along the trace) speeds the search up.

## Evidence

On the Kubicka set it uses only the tracks without cycles, subsampled from 1 s to 5 min. RMF appears only in plots. "Comparable to HMM up to 30 s", about 10× faster. The incremental baseline fails.

## Limits

It cannot match tracks with loops. That excludes the 24 loop tracks, which our test may contain.

## Where it sits in our matcher

Context only: a global geometric search family. It is not a candidate for us, because of the cycle restriction.
