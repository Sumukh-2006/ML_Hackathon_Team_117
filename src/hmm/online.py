"""Newson & Krumm's HMM fed one GPS fix at a time.

The batch decoder (`hmm.viterbi.decode`) is rerun here as a loop that can
stop and resume: it advances over the kept fixes it has, and pauses when a
break repair (`heal`) needs a fix that has not arrived yet. Fed every fix
and finished, it returns exactly the route `hmm.match.match` returns.

Committing (the variable sliding window of Goh et al., ITSC 2012): a
column's candidate is final once every path that can still win runs through
it. Two things can still change the path:
  1. Viterbi: the best path depends on fixes to come. Every path that can
     still win runs through a live candidate of some recent column.
  2. Break repair: `heal` pops columns off the stack, but only ones later
     than the column it reconnects to. That column must lie within
     MAX_BREAK seconds of the fix it reconnects with, which is a kept fix
     after the next one to decode: either one already here (a paused heal)
     or one still to come, later than every fix received. So no column
     earlier than (that fix's t - MAX_BREAK) is ever popped.
The anchor is therefore the earliest column within MAX_BREAK seconds of
that fix. Its live candidates are traced back until they meet in one
candidate; that candidate and everything before it are committed. The cost
of exactness: the committed route trails the latest fix by at least
MAX_BREAK (180 s), and the rest is shown as tentative.
"""

from dataclasses import dataclass, field

import numpy as np

from dataset import RoadGraph, Route, great_circle
from hmm.candidates import Candidates, GridIndex, candidates
from hmm.hyperparameters import (GRID_CELL, MAX_BREAK, MAX_DETOUR, MAX_SPEED,
                                 PREPROCESS_SIGMAS, SEARCH_RADIUS)
from hmm.routing import Router
from hmm.segments import arc_segments
from hmm.viterbi import Column, Model, connect, emission

NEED_MORE: int = -2   # heal outcome: the next kept fix has not arrived


@dataclass
class GrowingTrack:
    """The fixes received so far, laid out like `dataset.Track` so that
    `hmm.viterbi.connect` can index it."""
    lon: list[float] = field(default_factory=list)
    lat: list[float] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    t: list[float] = field(default_factory=list)

    def fix_count(self) -> int:
        return len(self.t)


@dataclass
class Update:
    """What one call to `push` changed."""
    kept: int                # fixes of this call the thinning kept
    committed: list[int]     # arcs newly final, to append to the route
    tentative: list[int]     # the whole tail after the committed arcs
    pending_fixes: int       # fixes received after the last committed one
    new_trip: bool           # a break that could not be repaired ended a trip


class OnlineHMM:
    def __init__(self, graph: RoadGraph, sigma: float, beta: float) -> None:
        self.graph: RoadGraph = graph
        self.spacing: float = PREPROCESS_SIGMAS * sigma
        self.segment: np.ndarray = arc_segments(graph)
        self.index: GridIndex = GridIndex(graph, GRID_CELL)
        self.router: Router = Router(graph)
        self.model: Model = Model(sigma, beta, MAX_DETOUR, MAX_SPEED, MAX_BREAK)
        self.track: GrowingTrack = GrowingTrack()
        self.fixes: list[int] = []            # kept fixes, as track indices
        self.near: list[Candidates] = []      # their candidates
        self.i: int = 0                       # next kept fix to decode
        self.stack: list[Column] = []         # the current trip's columns
        self.trips: int = 0
        # Committed state: the route's final arcs, and the current trip's
        # committed columns with their chosen (arc, along).
        self.arcs: list[int] = []
        self.path: list[tuple[int, float]] = []
        self.matched: int = 0
        self.new_trip: bool = False

    # ---- feeding fixes ---------------------------------------------------

    def push(self, fixes: list[tuple[float, float, float]]) -> Update:
        """Add fixes (lon, lat, t) in time order and decode what can be."""
        before: int = len(self.arcs)
        kept: int = 0
        for lon, lat, t in fixes:
            kept += self.receive(lon, lat, t)
        self.new_trip = False
        self.advance(final=False)
        self.commit()
        return Update(kept, self.arcs[before:], self.tentative(),
                      self.pending(), self.new_trip)

    def finish(self) -> tuple[Route, dict]:
        """Decode to the end; the same route and counts as `hmm.match`."""
        self.advance(final=True)
        if len(self.stack) > 0:
            self.end_trip()
        counts: dict = {"fixes": self.track.fix_count(),
                        "kept": len(self.fixes), "trips": self.trips,
                        "matched": self.matched}
        return Route(np.array(self.arcs, dtype=np.int64)), counts

    def receive(self, lon: float, lat: float, t: float) -> int:
        """Store one fix; thin it as `hmm.match.thin` does. 1 if kept."""
        x, y = self.graph.projection.to_xy(np.array([lon]), np.array([lat]))
        track: GrowingTrack = self.track
        if track.fix_count() > 0 and t <= track.t[-1]:
            raise ValueError(f"time {t} does not come after {track.t[-1]}")
        track.lon.append(lon); track.lat.append(lat); track.t.append(t)
        track.x.append(float(x[0])); track.y.append(float(y[0]))
        fix: int = track.fix_count() - 1
        if len(self.fixes) > 0:
            last: int = self.fixes[-1]
            apart: float = float(great_circle(track.lon[last], track.lat[last],
                                              lon, lat))
            if not apart > self.spacing:
                return 0
        self.fixes.append(fix)
        self.near.append(candidates(self.index, self.segment, lon, lat,
                                    track.x[fix], track.y[fix], SEARCH_RADIUS))
        return 1

    # ---- the batch decoder's loop, resumable -----------------------------

    def advance(self, final: bool) -> None:
        """`hmm.viterbi.decode`'s loop, stopping where it needs a fix that
        has not arrived (unless `final`, when the track has ended)."""
        while self.i < len(self.fixes):
            i: int = self.i
            if len(self.stack) == 0:
                if self.near[i].count() > 0:
                    self.stack.append(Column(
                        self.fixes[i], self.near[i],
                        emission(self.near[i], self.model.sigma),
                        np.full(self.near[i].count(), -1)))
                self.i += 1
                continue
            column: Column | None = connect(self.stack[-1], self.fixes[i],
                                            self.near[i], self.track,
                                            self.router, self.model)
            if column is not None:
                self.stack.append(column)
                self.i += 1
                continue
            resume: int = self.heal(i, final)
            if resume == NEED_MORE:
                return
            if resume >= 0:
                self.i = resume
                continue
            self.end_trip()     # i stays: the next trip starts at the break

    def heal(self, broken: int, final: bool) -> int:
        """`hmm.viterbi.heal`, which may also answer NEED_MORE."""
        stack: list[Column] = self.stack
        removed: list[Column] = []
        right: int = broken
        outcome: int = -1
        while True:
            if len(stack) <= 1:
                break
            if right + 1 >= len(self.fixes):
                outcome = -1 if final else NEED_MORE
                break
            removed.append(stack.pop())
            right += 1
            gap: float = float(self.track.t[self.fixes[right]]
                               - self.track.t[stack[-1].fix])
            if gap > self.model.max_break:
                break
            column: Column | None = connect(stack[-1], self.fixes[right],
                                            self.near[right], self.track,
                                            self.router, self.model)
            if column is not None:
                stack.append(column)
                if len(stack) < len(self.path) + 1:
                    raise AssertionError("heal popped a committed column")
                return right + 1
        for column in reversed(removed):
            stack.append(column)
        return outcome

    # ---- committing ------------------------------------------------------

    def end_trip(self) -> None:
        """The trip is over: its best path is final."""
        j: int = int(np.argmax(self.stack[-1].score))
        self.extend_to(len(self.stack) - 1, j)
        self.matched += len(self.stack)
        self.stack, self.path = [], []
        self.trips += 1
        self.new_trip = True

    def commit(self) -> None:
        """Commit every column all still-winnable paths run through."""
        if len(self.stack) == 0:
            return
        # The earliest fix a future `heal` can reconnect from: the kept fix
        # after the next one to decode if it has arrived (a paused heal
        # resumes over fixes already here), else any fix still to come.
        upcoming: int = self.i + 1
        earliest: float = (self.track.t[self.fixes[upcoming]]
                           if upcoming < len(self.fixes) else self.track.t[-1])
        horizon: float = earliest - self.model.max_break
        anchor: int = len(self.stack) - 1
        while anchor > 0 and self.track.t[self.stack[anchor - 1].fix] >= horizon:
            anchor -= 1
        alive: set[int] = set(np.nonzero(
            self.stack[anchor].score > -np.inf)[0].tolist())
        k: int = anchor
        done: int = len(self.path)            # columns 0 .. done-1 committed
        while len(alive) > 1 and k >= done:
            alive = {int(self.stack[k].back[j]) for j in alive}
            k -= 1
        if len(alive) == 1 and k >= done:
            self.extend_to(k, alive.pop())

    def extend_to(self, k: int, j: int) -> None:
        """Commit columns len(path) .. k, ending on candidate j of column k."""
        steps: list[tuple[int, float]] = self.walk_back(k, j, len(self.path))
        for arc, along in steps:
            self.drive_to(self.arcs, self.path[-1] if self.path else None,
                          arc, along)
            self.path.append((arc, along))

    def walk_back(self, k: int, j: int, stop: int) -> list[tuple[int, float]]:
        """(arc, along) of columns stop .. k on the path ending at (k, j)."""
        steps: list[tuple[int, float]] = []
        for index in range(k, stop - 1, -1):
            column: Column = self.stack[index]
            steps.append((int(column.candidates.arc[j]),
                          float(column.candidates.along[j])))
            j = int(column.back[j])
        steps.reverse()
        return steps

    def drive_to(self, arcs: list[int], previous: tuple[int, float] | None,
                 arc: int, along: float) -> None:
        """`hmm.match.assemble` for one step: append the arcs driven."""
        if previous is None:
            driven: list[int] = [arc]
        else:
            driven = self.router.path(previous[0], previous[1], arc, along)
        for each in driven:
            if len(arcs) == 0 or arcs[-1] != each:
                arcs.append(each)

    # ---- reporting -------------------------------------------------------

    def tentative(self) -> list[int]:
        """Arcs after the committed ones on the path that is best now."""
        if len(self.stack) <= len(self.path):
            return []
        last: int = len(self.stack) - 1
        steps = self.walk_back(last, int(np.argmax(self.stack[last].score)),
                               len(self.path))
        arcs: list[int] = self.arcs[-1:]      # for the back-to-back rule
        previous = self.path[-1] if self.path else None
        for arc, along in steps:
            self.drive_to(arcs, previous, arc, along)
            previous = (arc, along)
        return arcs[1:] if self.arcs else arcs

    def pending(self) -> int:
        if len(self.path) == 0:
            return self.track.fix_count()
        last_fix: int = self.stack[len(self.path) - 1].fix
        return self.track.fix_count() - 1 - last_fix
