"""Many lattices decoded at once on the GPU (derivation Theorem 1, batched).

Tracks have different lengths, so they are aligned at their last point:
sweep s handles, for every track at once, the step that ends s points
before its last point. Within a sweep the minimum over a row's pairs is a
scatter-min; the first pair (or candidate) attaining it is found by a
second scatter-min over indices, which keeps np.argmin's tie-breaking.

A step's pairs are a K x K' block in row-major order, so a pair's source
and target candidates are not stored: each block keeps its first pair, K',
and the ids of its first source and first target candidate, and the GPU
rebuilds the rest. Features sit in float16 divided by a per-feature scale
(far below float16's limit of 65504); costs are summed in float32. That is
27 bytes a pair.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FLOAT16_LIMIT: float = 60000.0


@dataclass
class Batch:
    tracks: int
    points: torch.Tensor        # per track: point count
    point_base: torch.Tensor    # per track: global id of its first point
    true_length: torch.Tensor   # per track, metres
    cand_point: torch.Tensor    # per candidate: global point
    cand_distance: torch.Tensor
    cand_first: torch.Tensor    # bool: on the first point of its track
    fresh_error: torch.Tensor   # start error, or the error of a skip into it
    cand_slices: list[tuple[int, int]]   # candidates with r = s, per s
    block_start: torch.Tensor   # per block: its first pair
    block_width: torch.Tensor   # per block: K'
    block_row: torch.Tensor     # per block: id of its first source candidate
    block_col: torch.Tensor     # per block: id of its first target candidate
    block_slices: list[tuple[int, int]]  # blocks of sweep s
    features: torch.Tensor      # (pairs, F) float16, scaled
    skip: torch.Tensor
    error: torch.Tensor
    scale: torch.Tensor         # (F,) float32

    def to(self, device: str) -> "Batch":
        """The same batch with every tensor on `device`."""
        moved: dict = {}
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value = value.to(device, non_blocking=True)
            moved[name] = value
        return Batch(**moved)

    def ends(self, pairs: torch.Tensor, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Source and target candidate of each pair, given its block."""
        local: torch.Tensor = pairs - self.block_start[blocks]
        width: torch.Tensor = self.block_width[blocks]
        return (self.block_row[blocks] + local // width,
                self.block_col[blocks] + local % width)

    def block_of(self, pairs: torch.Tensor) -> torch.Tensor:
        return torch.searchsorted(self.block_start, pairs, right=True) - 1


def read_flat(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name] for name in data.files}


def build(paths: list[Path], scale: np.ndarray, device: str) -> Batch:
    """Lay the lattices out sweep by sweep, on `device` ("pinned" keeps
    them in page-locked host memory, to be moved to the GPU per use).
    Within a point the candidates, and within a step the pairs, keep their
    order."""
    flats: list[dict[str, np.ndarray]] = [read_flat(p) for p in paths]
    sizes: list[np.ndarray] = [f["sizes"].astype(np.int64) for f in flats]
    counts: np.ndarray = np.array([len(s) for s in sizes])
    sweeps: int = int(counts.max())
    tracks: int = len(flats)
    cand_offset: list[np.ndarray] = [np.concatenate([[0], np.cumsum(s)]) for s in sizes]
    pair_offset: list[np.ndarray] = [np.concatenate([[0], np.cumsum(s[:-1] * s[1:])]) for s in sizes]
    track_cands: np.ndarray = np.concatenate([[0], np.cumsum([s.sum() for s in sizes])])
    track_pairs: np.ndarray = np.concatenate([[0], np.cumsum([p[-1] for p in pair_offset])])
    point_base: np.ndarray = np.concatenate([[0], np.cumsum(counts)])

    # Candidates, by r (points before the last), then track, then local.
    first_id: list[np.ndarray] = [np.zeros(len(s), dtype=np.int64) for s in sizes]
    order: list[np.ndarray] = []
    cand_slices: list[tuple[int, int]] = []
    placed: int = 0
    for r in range(sweeps):
        start: int = placed
        for t in range(tracks):
            j: int = int(counts[t]) - 1 - r
            if j < 0:
                continue
            first_id[t][j] = placed
            order.append(track_cands[t] + np.arange(cand_offset[t][j], cand_offset[t][j + 1]))
            placed += int(sizes[t][j])
        cand_slices.append((start, placed))
    cand_order: np.ndarray = np.concatenate(order)

    # Pairs: blocks by sweep, then track.
    pair_order: list[np.ndarray] = []
    blocks: list[tuple[int, int, int, int]] = []
    block_slices: list[tuple[int, int]] = []
    placed = 0
    for s in range(sweeps):
        start = len(blocks)
        for t in range(tracks):
            j = int(counts[t]) - 2 - s
            if j < 0:
                continue
            first: int = int(track_pairs[t] + pair_offset[t][j])
            count: int = int(sizes[t][j] * sizes[t][j + 1])
            blocks.append((placed, int(sizes[t][j + 1]), int(first_id[t][j]),
                           int(first_id[t][j + 1])))
            pair_order.append(np.arange(first, first + count))
            placed += count
        block_slices.append((start, len(blocks)))
    pairs: np.ndarray = np.concatenate(pair_order)
    table: np.ndarray = np.array(blocks, dtype=np.int64)

    point_of: np.ndarray = np.concatenate([
        point_base[t] + np.repeat(np.arange(counts[t]), sizes[t]) for t in range(tracks)])
    first: np.ndarray = np.concatenate([
        np.arange(int(sizes[t].sum())) < sizes[t][0] for t in range(tracks)])
    fresh: np.ndarray = np.concatenate([
        np.concatenate([f["start_errors"], f["skip_errors"]]) for f in flats])
    distance: np.ndarray = np.concatenate([f["point_distance"] for f in flats])
    features: np.ndarray = np.concatenate([f["features"] for f in flats])[pairs]
    features = np.clip(features / scale, -FLOAT16_LIMIT, FLOAT16_LIMIT)

    def put(values: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(values)).to(dtype=dtype)
        if device == "pinned":
            return tensor.pin_memory()
        return tensor.to(device)

    return Batch(
        tracks=tracks, points=put(counts, torch.int64),
        point_base=put(point_base[:-1], torch.int64),
        true_length=put(np.array([float(f["true_length"][0]) for f in flats]), torch.float64),
        cand_point=put(point_of[cand_order], torch.int64),
        cand_distance=put(distance[cand_order], torch.float32),
        cand_first=put(first[cand_order], torch.bool),
        fresh_error=put(fresh[cand_order], torch.float32),
        cand_slices=cand_slices,
        block_start=put(table[:, 0], torch.int64), block_width=put(table[:, 1], torch.int64),
        block_row=put(table[:, 2], torch.int64), block_col=put(table[:, 3], torch.int64),
        block_slices=block_slices,
        features=put(features, torch.float16),
        skip=put(np.concatenate([f["skips"] for f in flats])[pairs], torch.bool),
        error=put(np.concatenate([f["errors"] for f in flats])[pairs], torch.float32),
        scale=put(scale, torch.float32))


@dataclass
class Decoded:
    """Per track: the labelling's summed weight features (F + 3: step
    features, skips, first distance, restart distances) and its error."""
    features: torch.Tensor   # (tracks, F + 3) float64
    error: torch.Tensor      # (tracks,) float64


def decode(batch: Batch, weights: torch.Tensor, error_weight: float,
           cost_weight: float, dtype: torch.dtype = torch.float32) -> Decoded:
    """Minimise cost_weight x C_w(y) - error_weight x Delta(y) for every
    track. (1, 0) decodes; (1, 1) is loss-augmented; (epsilon, -1) is the
    oracle with ties broken by cost, which needs float64 so that epsilon
    survives next to sums of metres."""
    device = batch.features.device
    w: torch.Tensor = weights.to(device=device, dtype=dtype)
    step_w: torch.Tensor = w[:-3] * batch.scale.to(dtype)
    skip_w, first_w, restart_w = w[-3], w[-2], w[-1]
    fresh_cost: torch.Tensor = cost_weight * torch.where(
        batch.cand_first, first_w, restart_w) * batch.cand_distance.to(dtype) \
        - error_weight * batch.fresh_error.to(dtype)

    candidates: int = len(batch.cand_point)
    points: int = int(batch.points.sum())
    value: torch.Tensor = torch.zeros(candidates, device=device, dtype=dtype)
    best_pair: torch.Tensor = torch.zeros(candidates, dtype=torch.int64, device=device)
    fresh: torch.Tensor = torch.zeros(points, device=device, dtype=dtype)
    fresh_best: torch.Tensor = torch.zeros(points, dtype=torch.int64, device=device)
    big: int = 1 << 62
    pair_count: int = len(batch.skip)
    for s in range(len(batch.cand_slices)):
        a, b = batch.cand_slices[s]
        here: torch.Tensor = fresh_cost[a:b] + value[a:b]
        where: torch.Tensor = batch.cand_point[a:b]
        fresh.scatter_reduce_(0, where, here, "amin", include_self=False)
        ids: torch.Tensor = torch.arange(a, b, device=device)
        hit: torch.Tensor = torch.where(here == fresh[where], ids, big)
        fresh_best.scatter_reduce_(0, where, hit, "amin", include_self=False)

        first_block, last_block = batch.block_slices[s]
        if first_block == last_block:
            continue
        p: int = int(batch.block_start[first_block])
        q: int = pair_count if last_block == len(batch.block_start) \
            else int(batch.block_start[last_block])
        pair_ids: torch.Tensor = torch.arange(p, q, device=device)
        lengths: torch.Tensor = torch.diff(torch.cat([
            batch.block_start[first_block:last_block],
            torch.tensor([q], device=device)]))
        blocks: torch.Tensor = torch.repeat_interleave(
            torch.arange(first_block, last_block, device=device), lengths)
        row, col = batch.ends(pair_ids, blocks)
        skip: torch.Tensor = batch.skip[p:q]
        cost: torch.Tensor = cost_weight * torch.where(
            skip, skip_w, batch.features[p:q].to(dtype) @ step_w) \
            - error_weight * torch.where(skip, 0.0, batch.error[p:q].to(dtype))
        onward: torch.Tensor = torch.where(skip, fresh[batch.cand_point[col]], value[col])
        total: torch.Tensor = cost + onward
        value.scatter_reduce_(0, row, total, "amin", include_self=False)
        hit = torch.where(total == value[row], pair_ids, big)
        best_pair.scatter_reduce_(0, row, hit, "amin", include_self=False)
    return follow(batch, best_pair, fresh_best)


def follow(batch: Batch, best_pair: torch.Tensor,
           fresh_best: torch.Tensor) -> Decoded:
    """Walk every track's minimising choices forward and sum what the
    labelling uses."""
    device = batch.features.device
    width: int = batch.features.shape[1]
    sums: torch.Tensor = torch.zeros(batch.tracks, width + 3, dtype=torch.float64, device=device)
    error: torch.Tensor = torch.zeros(batch.tracks, dtype=torch.float64, device=device)
    tracks: torch.Tensor = torch.arange(batch.tracks, device=device)
    c: torch.Tensor = fresh_best[batch.point_base]
    sums[:, width + 1] += batch.cand_distance[c].double()
    error += batch.fresh_error[c].double()
    scale: torch.Tensor = batch.scale.double()
    for j in range(int(batch.points.max()) - 1):
        active: torch.Tensor = j < batch.points - 1
        live: torch.Tensor = tracks[active]
        pair: torch.Tensor = best_pair[c[active]]
        _, target = batch.ends(pair, batch.block_of(pair))
        skipped: torch.Tensor = batch.skip[pair]
        after: torch.Tensor = fresh_best[batch.point_base[live] + j + 1]
        following: torch.Tensor = torch.where(skipped, after, target)
        step: torch.Tensor = batch.features[pair].double() * scale
        sums[live, :width] += torch.where(skipped[:, None], 0.0, step)
        sums[live, width] += skipped.double()
        sums[live, width + 2] += torch.where(skipped, batch.cand_distance[following].double(), 0.0)
        error[live] += torch.where(skipped, batch.fresh_error[following].double(),
                                   batch.error[pair].double())
        c = c.clone()
        c[active] = following
    return Decoded(sums, error)
