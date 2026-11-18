"""The original dataset's hand tags, kept for the corpus' own folders.

The source is metadata.xml of Kubicka et al.'s dataset (Zenodo 57731),
fetched alone and reduced to our 90 folders: data/hand_labels/
kubicka_metadata_90.csv. Its `tags` column holds `key=value` pairs joined
by `;`, such as `feature=gaps;feature=loops`, or nothing for a clean track.
"""

import csv
from pathlib import Path

from dataset import DatasetError


def read_hand_tags(path: Path) -> dict[str, list[str]]:
    """Folder id -> its hand tags, for every folder in the file."""
    tags: dict[str, list[str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pairs: list[str] = []
            for pair in row["tags"].split(";"):
                if pair != "":
                    pairs.append(pair)
            tags[row["id"]] = pairs
    if len(tags) == 0:
        raise DatasetError(f"{path}: no folders listed")
    return tags


def write_tags(tags: list[str], path: Path) -> None:
    """One `key=value` per line; a clean track gets an empty file."""
    path.write_text("".join(f"{pair}\n" for pair in tags))
