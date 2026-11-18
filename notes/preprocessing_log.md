# Preprocessing log — Team 117

Fill this in **as you work**, not at the end. One row per decision.
If a row is blank when the panel asks, that step earns nothing.

## Data at a glance

- Rows: ___ Features: ___ Target: ___ Classes: ___
- Train / val / test sizes: ___ / ___ / ___
- Split method: ☐ stratified ☐ random ☐ time-based — why: ___________

## Decisions

| # | Decision | What you did | Why | Evidence | Score before → after |
|---|---|---|---|---|---|
| 1 | Missing values |  |  |  |  |
| 2 | Duplicates / IDs |  |  |  |  |
| 3 | Encoding |  |  |  |  |
| 4 | Scaling |  |  |  |  |
| 5 | Outliers |  |  |  |  |
| 6 | Engineered features |  |  |  |  |
| 7 | Feature selection |  |  |  |  |
| 8 |  |  |  |  |  |

## Feature selection — the four numbers

- Features started with: ___
- Features kept: ___
- Method: ☐ correlation threshold ☐ mutual information ☐ RFE / RFECV
  ☐ L1 / Lasso ☐ tree importance ☐ domain reasoning ☐ variance threshold
- Score with all features: ___
- Score with selected features: ___

> This block alone is a large part of the preprocessing argument. Fill it even if the model is unfinished.

## Leakage check

- [ ] Scalers / imputers / selectors fitted on **train only**, applied to val and test
- [ ] No target-derived column in the feature set
- [ ] No duplicate rows split across train and test
- [ ] Test set untouched until the final freeze
- [ ] Time-based data split by time, not randomly

## One sentence to say in the demo

> "We started with ___ features, removed ___ using ___, kept ___, and that changed the
> score from ___ to ___. Scaling and imputation were fitted on the training split only."
