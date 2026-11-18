"""The hand tags file is read into folder id -> its tags, clean ones empty."""

from pathlib import Path

from fixes.hand_tags import read_hand_tags


def test_hand_tags_by_folder(tmp_path: Path) -> None:
    path: Path = tmp_path / "tags.csv"
    path.write_text("split,id,tags\n"
                    "train,00000001,feature=gaps;feature=loops\n"
                    "train,00000002,\n")
    assert read_hand_tags(path) == {
        "00000001": ["feature=gaps", "feature=loops"],
        "00000002": []}
