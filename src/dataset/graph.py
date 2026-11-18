"""The road network of one corpus folder: nodes and directed straight arcs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.projection import Projection, great_circle
from dataset.table import DatasetError, read_table


@dataclass(frozen=True)
class RoadGraph:
    """Nodes with positions and directed arcs between them.

    NOTE: a node's id is its 0-based line in `.nodes` and an arc's id is its
    0-based line in `.arcs`. Every arc is one straight segment, and a
    two-way road is two arcs. Some folders hold several arcs between the
    same two nodes ("twins"); they are the same road, so `canonical_arc`
    maps each arc to the lowest id among its twins.
    """
    projection: Projection
    node_lon: np.ndarray
    node_lat: np.ndarray
    node_x: np.ndarray
    node_y: np.ndarray
    arc_from: np.ndarray
    arc_to: np.ndarray
    arc_length: np.ndarray
    canonical_arc: np.ndarray
    reverse_arc: np.ndarray
    out_start: np.ndarray
    out_arc: np.ndarray

    @classmethod
    def from_folder(cls, folder: Path) -> "RoadGraph":
        """Read `<id>.nodes` and `<id>.arcs` from a corpus folder."""
        name: str = folder.name
        lon, lat = read_table(folder / f"{name}.nodes", [float, float])
        arc_from, arc_to = read_table(folder / f"{name}.arcs", [int, int])
        check_arc_nodes(folder / f"{name}.arcs", arc_from, arc_to, len(lon))

        projection: Projection = Projection.around(lon, lat)
        x, y = projection.to_xy(lon, lat)
        length: np.ndarray = great_circle(lon[arc_from], lat[arc_from],
                                          lon[arc_to], lat[arc_to])
        canonical, reverse = twins_and_reverses(arc_from, arc_to, len(lon))
        out_start, out_arc = outgoing(arc_from, len(lon))
        return cls(projection, lon, lat, x, y, arc_from, arc_to, length,
                   canonical, reverse, out_start, out_arc)

    def node_count(self) -> int:
        return len(self.node_lon)

    def arc_count(self) -> int:
        return len(self.arc_from)

    def out_arcs(self, node: int) -> np.ndarray:
        """The arcs that leave `node`."""
        return self.out_arc[self.out_start[node]:self.out_start[node + 1]]


def check_arc_nodes(path: Path, arc_from: np.ndarray, arc_to: np.ndarray,
                    node_count: int) -> None:
    """Every arc must join two nodes that exist."""
    bad: np.ndarray = np.nonzero(
        (arc_from < 0) | (arc_from >= node_count)
        | (arc_to < 0) | (arc_to >= node_count))[0]
    if len(bad) > 0:
        line: int = int(bad[0]) + 1
        raise DatasetError(
            f"{path}:{line}: arc joins node {arc_from[bad[0]]} -> "
            f"{arc_to[bad[0]]}, but there are {node_count} nodes "
            f"({len(bad)} such arcs in the file)")


def twins_and_reverses(arc_from: np.ndarray, arc_to: np.ndarray,
                       node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """For every arc: its lowest-id twin, and its reverse arc (or -1)."""
    key: np.ndarray = arc_from * node_count + arc_to
    unique_keys, first, inverse = np.unique(key, return_index=True,
                                            return_inverse=True)
    canonical: np.ndarray = first[inverse]

    reverse_key: np.ndarray = arc_to * node_count + arc_from
    position: np.ndarray = np.searchsorted(unique_keys, reverse_key)
    position = np.minimum(position, len(unique_keys) - 1)
    found: np.ndarray = unique_keys[position] == reverse_key
    reverse: np.ndarray = np.where(found, first[position], -1)
    return canonical, reverse


def outgoing(arc_from: np.ndarray,
             node_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Compressed rows: arcs leaving node n are out_arc[start[n]:start[n+1]]."""
    out_arc: np.ndarray = np.argsort(arc_from, kind="stable")
    counts: np.ndarray = np.bincount(arc_from, minlength=node_count)
    out_start: np.ndarray = np.zeros(node_count + 1, dtype=np.int64)
    out_start[1:] = np.cumsum(counts)
    return out_start, out_arc
