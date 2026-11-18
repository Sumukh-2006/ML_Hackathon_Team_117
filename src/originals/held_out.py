"""Fetch the folders held out from our corpus: the "executor test".

Usage: uv run python -m originals.held_out <corpus root> <target>

The original dataset has ids 0..99; our corpus gives 90 of them (train and
student_test). The other 10 are fetched from the same Zenodo archive, clean
as Kubicka published them (the organisers' added noise is not applied), and
written to <target>/<id>/ (normally <corpus root>/executor_test).
"""

import sys
from pathlib import Path

from dataset import Folder
from originals.__main__ import (ARCHIVE_URL, TAIL_SIZE, fetch_folder,
                                member_ends)
from originals.remote_zip import table_of_contents

IDS: range = range(100)
GIVEN_SPLITS: list[str] = ["train", "student_test"]


def held_out_ids(corpus_root: Path) -> list[str]:
    given: set[str] = {path.name for split in GIVEN_SPLITS
                       for path in (corpus_root / split).iterdir()}
    return [f"{i:08d}" for i in IDS if f"{i:08d}" not in given]


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python -m originals.held_out <corpus root> <target>")
    corpus_root, target = Path(sys.argv[1]), Path(sys.argv[2])
    if target.exists():
        sys.exit(f"{target} already exists; remove it first to refetch")
    members, directory_start = table_of_contents(ARCHIVE_URL, TAIL_SIZE)
    ends = member_ends(members, directory_start)
    names: list[str] = held_out_ids(corpus_root)
    for name in names:
        fetch_folder(Folder(target.name, Path(name)), members, ends, target / name)
        print(name, flush=True)
    print(f"fetched {len(names)} held-out folders into {target}")


if __name__ == "__main__":
    main()
