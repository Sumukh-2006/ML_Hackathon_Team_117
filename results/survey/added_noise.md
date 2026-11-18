# Noise the organisers added to every .track (measured 2026-11-18)

Comparison of data/corpus tracks with data/corpus_original (Kubicka et al.,
Zenodo 57731), fix by fix, on each folder's plane projection. Per-folder
numbers are in original_vs_ours_tracks.csv.

- The same fixes and the same timestamps in all 90 folders; every position moved.
- Added offset per axis: mean east -0.01 m, north 0.01 m; std east 5.14 m,
  north 5.12 m; east/north correlation -0.001 (isotropic).
- Kurtosis 3.97 east, 3.88 north (Gaussian = 3), so the tails are a little heavy.
- Radial offset: median 5.91, p90 10.83, p99 16.10, max 47.8 m. A Gaussian
  with std 5.13 would give median 6.06 and p99 15.61.
- Autocorrelation of the east offset by lag: 1 s 0.29, 2 s 0.25, 3 s 0.21,
  5 s 0.15, 10 s 0.07, 30 s -0.01. This is not AR(1) (AR(1) at 0.29 would
  give 0.08 at 2 s): a white part plus a slowly varying part.
