"""Per-track score rows, and their length-weighted totals per tag group."""

import csv
from pathlib import Path

import numpy as np

from benchmark.metrics import RouteScore, f1

# The original dataset's hand tags (feature=...), plus "all" and "clean".
GROUPS: list[str] = ["all", "clean", "gaps", "hives", "u-turns", "loops",
                     "congruence-issues"]

TRACK_COLUMNS: list[str] = [
    "split", "id", "tags", "missing", "predicted_breaks", "correct_fraction",
    "mismatch_fraction", "precision", "recall", "overlap", "added", "missed",
    "predicted_length", "true_length"]

SUMMARY_COLUMNS: list[str] = [
    "split", "group", "tracks", "missing", "weighted_correct_fraction",
    "mean_correct_fraction", "mismatch_fraction", "precision", "recall", "f1"]


def track_row(split: str, name: str, tags: str, score: RouteScore,
              missing: bool, breaks: int) -> dict:
    """One CSV row: the track's tags, its lengths and its measures."""
    return {
        "split": split, "id": name, "tags": tags, "missing": int(missing),
        "predicted_breaks": breaks,
        "correct_fraction": score.correct_fraction(),
        "mismatch_fraction": score.mismatch_fraction(),
        "precision": score.precision(), "recall": score.recall(),
        "overlap": score.overlap, "added": score.added,
        "missed": score.missed, "predicted_length": score.predicted_length,
        "true_length": score.true_length}


def tag_text(tags: list[str] | None) -> str:
    """`gaps+loops` from hand tags; `clean` for none; `untagged` when the
    corpus carries no tags (only a fixed corpus does)."""
    if tags is None:
        return "untagged"
    values: list[str] = []
    for pair in tags:
        values.append(pair.split("=", 1)[1])
    return "+".join(values) or "clean"


def summary_rows(rows: list[dict]) -> list[dict]:
    """Totals for every split, over all tracks and over each tag group."""
    summary: list[dict] = []
    for split in sorted({row["split"] for row in rows}):
        in_split: list[dict] = [r for r in rows if r["split"] == split]
        for group in GROUPS:
            members: list[dict] = group_members(in_split, group)
            if len(members) == 0:
                continue
            summary.append(group_totals(split, group, members))
    return summary


def group_members(rows: list[dict], group: str) -> list[dict]:
    if group == "all":
        return rows
    members: list[dict] = []
    for row in rows:
        if group in row["tags"].split("+"):
            members.append(row)
    return members


def group_totals(split: str, group: str, members: list[dict]) -> dict:
    """Weighted measures add lengths over tracks before dividing, so a long
    track counts in proportion to its length; the mean does not."""
    overlap: float = sum(r["overlap"] for r in members)
    longer: float = sum(max(r["predicted_length"], r["true_length"])
                        for r in members)
    wrong: float = sum(r["added"] + r["missed"] for r in members)
    predicted: float = sum(r["predicted_length"] for r in members)
    true: float = sum(r["true_length"] for r in members)

    precision: float = overlap / predicted if predicted > 0 else 0.0
    recall: float = overlap / true
    mean_cf: float = float(np.mean([r["correct_fraction"] for r in members]))
    return {
        "split": split, "group": group, "tracks": len(members),
        "missing": sum(r["missing"] for r in members),
        "weighted_correct_fraction": overlap / longer,
        "mean_correct_fraction": mean_cf, "mismatch_fraction": wrong / true,
        "precision": precision, "recall": recall,
        "f1": f1(precision, recall)}


def write_csv(rows: list[dict], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer: csv.DictWriter = csv.DictWriter(handle, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary: list[dict]) -> None:
    print(f"{'split':<13}{'group':<18}{'tracks':>7}{'missing':>8}"
          f"{'wCF %':>9}{'meanCF %':>10}{'RMF %':>8}{'F1 %':>8}")
    for row in summary:
        print(f"{row['split']:<13}{row['group']:<18}{row['tracks']:>7}"
              f"{row['missing']:>8}"
              f"{100 * row['weighted_correct_fraction']:>9.2f}"
              f"{100 * row['mean_correct_fraction']:>10.2f}"
              f"{100 * row['mismatch_fraction']:>8.2f}"
              f"{100 * row['f1']:>8.2f}")
