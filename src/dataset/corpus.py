"""The corpus on disk: splits of folders, each one track on one map."""

from dataclasses import dataclass
from pathlib import Path

from dataset.graph import RoadGraph
from dataset.route import Route
from dataset.table import DatasetError
from dataset.track import Track


@dataclass(frozen=True)
class Folder:
    """One corpus folder `<split>/<id>/` holding `<id>.{nodes,arcs,track}`
    and, unless it is held out, `<id>.route`."""
    split: str
    path: Path

    def name(self) -> str:
        return self.path.name

    def graph(self) -> RoadGraph:
        return RoadGraph.from_folder(self.path)

    def track(self, graph: RoadGraph) -> Track:
        return Track.from_file(self.path / f"{self.name()}.track",
                               graph.projection)

    def has_route(self) -> bool:
        return (self.path / f"{self.name()}.route").exists()

    def route(self, graph: RoadGraph) -> Route:
        return Route.from_file(self.path / f"{self.name()}.route", graph)

    def has_tags(self) -> bool:
        """Only a fixed corpus (`python -m fixes`) carries hand tags."""
        return (self.path / f"{self.name()}.tags").exists()

    def tags(self) -> list[str]:
        """The original dataset's hand tags, e.g. `feature=gaps`; none for
        a clean track."""
        path: Path = self.path / f"{self.name()}.tags"
        tags: list[str] = []
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "=" not in line:
                raise DatasetError(f"{path}:{number}: expected key=value, "
                                   f"found {line!r}")
            tags.append(line)
        return tags


@dataclass(frozen=True)
class Corpus:
    """Every folder under the corpus root, grouped by split."""
    root: Path
    splits: dict[str, list[Folder]]

    @classmethod
    def from_root(cls, root: Path) -> "Corpus":
        """Each subdirectory of `root` is a split; each of its subdirectories
        is a folder, which must hold the three input files."""
        if not root.is_dir():
            raise DatasetError(f"{root}: corpus root is not a directory")

        splits: dict[str, list[Folder]] = {}
        for split_path in sorted(root.iterdir()):
            if not split_path.is_dir():
                continue
            folders: list[Folder] = []
            for path in sorted(split_path.iterdir()):
                check_inputs_exist(path)
                folders.append(Folder(split_path.name, path))
            splits[split_path.name] = folders
        return cls(root, splits)

    def folders(self, split: str) -> list[Folder]:
        if split not in self.splits:
            raise DatasetError(
                f"{self.root}: no split {split!r}; "
                f"there are {sorted(self.splits)}")
        return self.splits[split]


def check_inputs_exist(path: Path) -> None:
    """A folder must hold its nodes, arcs and track files."""
    missing: list[str] = []
    for suffix in ["nodes", "arcs", "track"]:
        if not (path / f"{path.name}.{suffix}").exists():
            missing.append(f"{path.name}.{suffix}")
    if len(missing) > 0:
        raise DatasetError(f"{path}: missing {', '.join(missing)}")
