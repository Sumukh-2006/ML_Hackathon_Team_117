"""Fetch the original dataset's files for the folders our corpus has.

Usage: uv run python -m originals <corpus root> <new root>

The original is Kubicka et al.'s dataset on Zenodo (record 57731), one zip of
100 folders. Only the folders that exist in <corpus root> are requested;
the others (held out from us) are never downloaded. Each folder's files are
written to <new root>/<split>/<id>/ with the split our corpus gives it, after
their checksums are verified.
"""

import sys
from pathlib import Path

from dataset import Corpus, DatasetError, Folder
from originals.remote_zip import (Member, fetch_range, table_of_contents,
                                  unpack)

ARCHIVE_URL: str = ("https://zenodo.org/records/57731/files/"
                    "map-matching-dataset.zip?download=1")

# NOTE: the archive's table of contents (503 entries) fits in its last
# 256 KiB; `table_of_contents` refuses to continue if it does not.
TAIL_SIZE: int = 256 * 1024

SUFFIXES: list[str] = ["nodes", "arcs", "track", "route"]

# A download that drops mid-transfer is tried this many times in all.
ATTEMPTS: int = 3


def main() -> None:
    corpus_root, target = arguments(sys.argv[1:])
    corpus: Corpus = Corpus.from_root(corpus_root)
    members, directory_start = table_of_contents(ARCHIVE_URL, TAIL_SIZE)
    ends: dict[int, int] = member_ends(members, directory_start)

    count: int = 0
    for split in corpus.splits:
        for folder in corpus.folders(split):
            fetch_folder(folder, members, ends, target / split / folder.name())
            count += 1
    print(f"fetched the original files of {count} folders into {target}")


def arguments(words: list[str]) -> tuple[Path, Path]:
    if len(words) != 2:
        sys.exit("usage: python -m originals <corpus root> <new root>")
    target: Path = Path(words[1])
    if target.exists():
        sys.exit(f"{target} already exists; remove it first to refetch")
    return Path(words[0]), target


def member_ends(members: dict[str, Member],
                directory_start: int) -> dict[int, int]:
    """For each member's header offset, the last byte before the next
    member begins; the last member ends where the directory starts."""
    offsets: list[int] = sorted(m.header_offset for m in members.values())
    offsets.append(directory_start)
    ends: dict[int, int] = {}
    for index in range(len(offsets) - 1):
        ends[offsets[index]] = offsets[index + 1] - 1
    return ends


def fetch_folder(folder: Folder, members: dict[str, Member],
                 ends: dict[int, int], target: Path) -> None:
    """Download one folder's files in a single range and write them."""
    wanted: list[Member] = []
    for suffix in SUFFIXES:
        name: str = f"{folder.name()}/{folder.name()}.{suffix}"
        if name not in members:
            raise DatasetError(f"{ARCHIVE_URL}: no member {name}")
        wanted.append(members[name])

    first: int = min(m.header_offset for m in wanted)
    last: int = max(ends[m.header_offset] for m in wanted)
    block: bytes = fetch_range(ARCHIVE_URL, first, last, ATTEMPTS)

    target.mkdir(parents=True)
    for member in wanted:
        content: bytes = unpack(member, block, first)
        (target / member.name.split("/")[1]).write_bytes(content)


if __name__ == "__main__":
    main()
