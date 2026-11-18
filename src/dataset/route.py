"""A route: the arcs a vehicle drove, in order."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.graph import RoadGraph
from dataset.table import DatasetError, read_table


@dataclass(frozen=True)
class Route:
    """A sequence of arc ids.

    A route usually chains (each arc ends where the next starts), but 11
    true routes in the corpus jump 10 to 600 m, at 14 places in all, where
    the annotated route leaves the road graph. `breaks` finds those places instead of
    assuming they cannot happen.
    """
    arcs: np.ndarray

    @classmethod
    def from_file(cls, path: Path, graph: RoadGraph) -> "Route":
        """Read a `.route` file of arc ids, one per line."""
        (arcs,) = read_table(path, [int])
        bad: np.ndarray = np.nonzero((arcs < 0)
                                     | (arcs >= graph.arc_count()))[0]
        if len(bad) > 0:
            raise DatasetError(
                f"{path}:{int(bad[0]) + 1}: arc {arcs[bad[0]]} does not "
                f"exist; the graph has {graph.arc_count()} arcs")
        return cls(arcs)

    def write(self, path: Path) -> None:
        """Write one arc id per line, the same format the corpus uses."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [str(int(arc)) for arc in self.arcs]
        path.write_text("\n".join(lines) + "\n")

    def length(self, graph: RoadGraph) -> float:
        """Total length in metres."""
        return float(graph.arc_length[self.arcs].sum())

    def breaks(self, graph: RoadGraph) -> np.ndarray:
        """Positions i where arc i does not end where arc i + 1 starts."""
        ends: np.ndarray = graph.arc_to[self.arcs[:-1]]
        starts: np.ndarray = graph.arc_from[self.arcs[1:]]
        return np.nonzero(ends != starts)[0]
