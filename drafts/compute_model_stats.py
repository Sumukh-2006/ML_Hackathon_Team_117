"""Numbers for the models part of report.typ: weighted CF of every model on
every test set, and paired bootstrap intervals for (ours - MDP).

The bootstrap resamples tracks (derivation §8). On the noisy held-out set
the 8 draws of a track are summed into one row first, so a resample keeps
a track's draws together. Run from the project root:

    .venv/bin/python drafts/compute_model_stats.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[1]
RESULTS: Path = ROOT / "results"
MODELS: list[str] = ["hmm", "segment", "mcm", "mdp", "learned", "learned_trained"]
RESAMPLES: int = 10_000
SEED: int = 0

SETS: dict[str, Path] = {
    "train": RESULTS / "benchmark",
    "student_test": RESULTS / "benchmark",
    "held_out_noisy": RESULTS / "held_out_noisy" / "benchmark",
    "held_out_clean": RESULTS / "executor_test" / "benchmark",
}
NOT_FOR_TRAINED: set[str] = {"train", "held_out_clean"}  # in-sample / not run


def rows(model: str, name: str) -> pd.DataFrame:
    df = pd.read_csv(SETS[name] / model / "per_track.csv", dtype={"id": str})
    if name in ("train", "student_test"):
        df = df[df.split == name]
    df = df.assign(longer=np.maximum(df.predicted_length, df.true_length))
    return df.groupby("id")[["overlap", "longer", "true_length"]].sum()


def weighted(df: pd.DataFrame) -> float:
    return float(df.overlap.sum() / df.longer.sum())


def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    random = np.random.default_rng(SEED)
    pick = random.integers(0, len(ids), (RESAMPLES, len(ids)))
    da = a.overlap.values[pick].sum(1) / a.longer.values[pick].sum(1)
    db = b.overlap.values[pick].sum(1) / b.longer.values[pick].sum(1)
    diff = 100 * (db - da)
    return {"difference": 100 * (weighted(b) - weighted(a)),
            "low": float(np.percentile(diff, 2.5)), "high": float(np.percentile(diff, 97.5)),
            "share_better": float((diff > 0).mean()), "tracks": int(len(ids))}


def training() -> dict:
    """Cross-validated held-out CF of every way of setting the weights,
    read from the CV logs (pooled = the last pooled line of a setting)."""
    import re
    out: dict = {}
    for log, setting, name in [("cv.log", "start", "hand"),
                               ("cv.log", "search_woeltche_terms", "grid_own"),
                               ("cv_copies.log", "copies_woeltche_terms", "grid_copies"),
                               ("cv_exact.log", "exact_woeltche_terms", "exact_copies")]:
        text = (RESULTS / "learned" / log).read_text()
        folds: dict[int, float] = {}
        for k, v in re.findall(setting + r" fold (\d): held-out CF ([\d.]+)", text):
            folds[int(k)] = 100 * float(v)
        pooled = re.findall(r"== " + setting + r": pooled held-out CF ([\d.]+)", text)
        out[name] = {"folds": [folds.get(k) for k in range(5)],
                     "pooled": 100 * float(pooled[-1]) if pooled else None}
    return out


def main() -> None:
    out: dict = {"cf": {}, "ours_minus_mdp": {}, "training": training()}
    for name in SETS:
        out["cf"][name] = {}
        for m in MODELS:
            if m == "learned_trained" and name in NOT_FOR_TRAINED:
                continue
            try:
                out["cf"][name][m] = 100 * weighted(rows(m, name))
            except FileNotFoundError:
                pass
        if "mdp" in out["cf"][name] and "learned" in out["cf"][name]:
            out["ours_minus_mdp"][name] = paired(rows("mdp", name), rows("learned", name))
        if "learned_trained" in out["cf"][name]:
            out.setdefault("trained_minus_mdp", {})[name] = paired(rows("mdp", name), rows("learned_trained", name))
            out.setdefault("trained_minus_hand", {})[name] = paired(rows("learned", name), rows("learned_trained", name))
    # Cross-validated: held-out rows of the exact search against the hand
    # weights' routes on the same tracks (the train benchmark of `learned`).
    cv = pd.read_csv(RESULTS / "learned" / "cv_exact_rows.csv", dtype={"id": str})
    cv = cv.assign(longer=np.maximum(cv.predicted_length, cv.true_length)).set_index("id")
    out["cv_exact_minus_hand"] = paired(rows("learned", "train"), cv[["overlap", "longer", "true_length"]])
    out["cv_exact_minus_mdp"] = paired(rows("mdp", "train"), cv[["overlap", "longer", "true_length"]])
    path = ROOT / "drafts" / "stats" / "models.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
