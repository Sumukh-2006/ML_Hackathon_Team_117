"""Copy the corpus and add the original dataset's hand tags.

Usage: uv run python -m fixes <corpus root> <hand tags csv> <new root>

Every original file is copied byte for byte. Next to the copies, each
folder the hand tags file lists gets `<id>.tags`: its hand tags, one
`key=value` per line, and an empty file for a clean track.
"""

import shutil
import sys
from pathlib import Path

from dataset import Corpus, Folder
from fixes.hand_tags import read_hand_tags, write_tags


def main() -> None:
    source, tags_path, target = arguments(sys.argv[1:])
    corpus: Corpus = Corpus.from_root(source)
    hand_tags: dict[str, list[str]] = read_hand_tags(tags_path)

    tagged: int = 0
    for split in corpus.splits:
        for folder in corpus.folders(split):
            tagged += copy_folder(folder, hand_tags,
                                  target / split / folder.name())

    print(f"copied {source} to {target}; hand tags added to {tagged} folders")


def arguments(words: list[str]) -> tuple[Path, Path, Path]:
    """The source corpus, the hand tags, and a target that must not exist."""
    if len(words) != 3:
        sys.exit("usage: python -m fixes <corpus root> <hand tags csv> "
                 "<new root>")
    target: Path = Path(words[2])
    if target.exists():
        sys.exit(f"{target} already exists; remove it first to rebuild")
    return Path(words[0]), Path(words[1]), target


def copy_folder(folder: Folder, hand_tags: dict[str, list[str]],
                target: Path) -> int:
    """Copy one folder; add its hand tags if it has any. Returns 1 when
    tags were added, 0 when the hand tags file does not list the folder."""
    target.mkdir(parents=True)
    for path in sorted(folder.path.iterdir()):
        shutil.copyfile(path, target / path.name)

    if folder.name() not in hand_tags:
        return 0
    write_tags(hand_tags[folder.name()], target / f"{folder.name()}.tags")
    return 1


if __name__ == "__main__":
    main()
