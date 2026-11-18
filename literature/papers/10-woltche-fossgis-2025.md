# Artificial Ground Truth Data Generation for Map Matching with Open Source Software

Adrian Wöltche, FOSSGIS 2025, doi 10.5281/zenodo.14774143. Tools under AGPL-3.0 at figshare, doi 10.6084/m9.figshare.27952545. The user supplied the PDF (`pdfs/woltche-mdp/fossgis_2025_artificial_ground_truth_Woeltche_fin.pdf`).

What I read on 2026-11-18: all 17 pages, including the tables and the bibliography.

## What it is

It is not the MDP method paper; that one is [32], Transactions in GIS 27(7) 2023, still unread. This paper is a tool chain for making labelled training data:

1. SUMO simulates random trips on an OSM network and emits exact positions every second. The route is known by construction, so there is no annotation error.
2. `apply_error.py` adds artificial GNSS noise. It offers three noise models:
   - Gaussian (independent per fix; the paper argues this is unrealistic at high sampling rates);
   - ARMA (autocorrelated, but it drifts away from zero);
   - Ornstein-Uhlenbeck, X_t = X_{t-1} + θ(μ - X_{t-1})Δt + σ W √Δt, which is autocorrelated and mean-reverting. This is the paper's choice.

   On top of the noise model it applies two more steps: a moving-average smoothing of the noise (window w), and outliers. Outliers are individual points moved by a separate Gaussian (σ_out), chosen with a probability p and a "closeness" parameter.
3. The noise parameters can be fitted to real data with Optuna, by matching the distribution of fix-to-matched-road distances. On real FCD the fit gave σ 0.8, w 2, outlier p 0.022, outlier σ 43.81 m, at Δt 5 s.

## Evidence

On 1000 simulated tracks in Hof, the weighted mean correct fraction (Table 3) is:

| scenario | Map Matching 2 | Barefoot | FMM (HMM) |
|---|---|---|---|
| hypertuned noise, 5 s (H5) | 99.53 | 97.37 | 74.16 |
| no noise, 1 s (B) | 99.68 | 99.60 | 99.55 |
| low / mid / high noise, 1 s | 99.60 / 99.31 / 98.23 | 99.50 / 97.79 / 92.75 | 56.4 / 37.0 / 28.4 |
| outliers low / mid / high, 1 s | 99.57 / 98.43 / 95.25 | 97.06 / 90.18 / 80.06 | 47.5 / 29.6 / 21.2 |

So the MDP stays near 98-99.6% under noise at 1 s, while HMM-based tools collapse when noise is dense and autocorrelated. The paper also restates map_matching_2's real-data accuracies: 98.73% on Kubicka (our dataset, original tracks).

## Limits

- Outliers are modelled simply.
- OU with smoothing lacks the fine jitter of real GNSS.
- The benchmark tools treat the road network differently (turn restrictions, removed roads).
- The weighted measure is not comparable with the unweighted one of [32].

## Where it sits in our matcher

1. **Our noise.** The organisers' added noise (results/survey/added_noise.md) looks like this kind of generator. It is isotropic, with σ about 5.1 m per axis and kurtosis about 3.9 (a few outliers). Its autocorrelation of 0.29, 0.25, 0.21, 0.15, 0.07 at 1, 2, 3, 5 and 10 s fits a white part (about 66% of the variance) plus a mean-reverting AR(1)/OU part (about 34%, decaying by about 0.86 per second).
   My reading, unverified: I have not run `apply_error.py`.
2. **Training data.** This is a documented, published way to boost training. Apply the same kind of noise, with fresh random seeds, to our 90 original tracks (data/corpus_original), and every original yields many noisy copies with an exact route. SUMO trips on our own graphs would add new routes too.
3. **MDP robustness.** It is evidence that Wöltche's MDP is the method to beat on noisy 1 Hz data. Our NK HMM falling to about 92% on train is consistent with the FMM (HMM) collapse seen here.
