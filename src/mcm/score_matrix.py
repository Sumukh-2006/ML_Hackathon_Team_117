"""The score matrix M (section 4, Eq. 8) and the offline method (section 6).

M[arc, fix] is the best summed similarity of a candidate route whose last
matching pair is (arc, fix). A column is filled from the routes alive so
far: each one ending on an arc s that is near() arc l continues into l; an
arc no alive route reaches starts a new route. Routes never die; a route
ending on s at fix k stays alive with M[s, k] until a later fix extends it.
Only cells with similarity above zero are stored.
"""

from dataclasses import dataclass

from mcm.routing import Predecessors


@dataclass(frozen=True)
class Cell:
    value: float
    back_arc: int  # -1 when the cell starts a new route
    back_fix: int


def fill(similarity: list[dict[int, float]], first: int, last: int,
         predecessors: Predecessors) -> list[dict[int, Cell]]:
    """Columns first..last of M, computed on that stretch of fixes alone."""
    columns: list[dict[int, Cell]] = []
    alive: dict[int, int] = {}
    ending: dict[int, list[int]] = {}
    for fix in range(first, last + 1):
        column: dict[int, Cell] = {}
        for arc, value in similarity[fix].items():
            column[arc] = best_cell(arc, value, alive, ending, columns, first,
                                    predecessors)
        columns.append(column)

        for arc in column:
            if arc not in alive:
                end: int = predecessors.arc_to[arc]
                ending.setdefault(end, []).append(arc)
            alive[arc] = fix
    return columns


def best_cell(arc: int, value: float, alive: dict[int, int],
              ending: dict[int, list[int]], columns: list[dict[int, Cell]],
              first: int, predecessors: Predecessors) -> Cell:
    """Eq. 8: extend the best alive route that is near() this arc, else
    start a new route with this arc's similarity alone."""
    sources: list[int] = []
    if arc in alive:
        sources.append(arc)
    reach, _ = predecessors.into(predecessors.arc_from[arc])
    for node in reach:
        for source in ending.get(node, []):
            sources.append(source)

    best: Cell = Cell(value, -1, -1)
    for source in sources:
        fix: int = alive[source]
        total: float = value + columns[fix - first][source].value
        if total > best.value:
            best = Cell(total, source, fix)
    return best


def last_pair(columns: list[dict[int, Cell]],
              first: int) -> tuple[int, int] | None:
    """The best cell of the last column that has any (section 6.1, step 1:
    max over arcs of M[arc, p_n])."""
    for offset in range(len(columns) - 1, -1, -1):
        column: dict[int, Cell] = columns[offset]
        if len(column) == 0:
            continue
        best_arc: int = -1
        for arc, cell in column.items():
            if best_arc < 0 or cell.value > column[best_arc].value:
                best_arc = arc
        return first + offset, best_arc
    return None


def backtrack(columns: list[dict[int, Cell]], first: int, fix: int,
              arc: int) -> list[tuple[int, int]]:
    """The matching pairs (fix, arc) of the route ending at (fix, arc)."""
    pairs: list[tuple[int, int]] = []
    while arc >= 0:
        pairs.append((fix, arc))
        cell: Cell = columns[fix - first][arc]
        fix, arc = cell.back_fix, cell.back_arc
    pairs.reverse()
    return pairs


def offline(similarity: list[dict[int, float]],
            predecessors: Predecessors) -> list[tuple[int, int]]:
    """Section 6.1: match the whole track; while a stretch before or after
    the matched pairs is left, match that stretch on its own. Stretches
    with no arc near any fix stay unmatched."""
    pairs: list[tuple[int, int]] = []
    stretches: list[tuple[int, int]] = [(0, len(similarity) - 1)]
    while len(stretches) > 0:
        first, last = stretches.pop()
        if first > last:
            continue
        columns: list[dict[int, Cell]] = fill(similarity, first, last,
                                              predecessors)
        end: tuple[int, int] | None = last_pair(columns, first)
        if end is None:
            continue
        found: list[tuple[int, int]] = backtrack(columns, first, end[0],
                                                 end[1])
        pairs.extend(found)
        stretches.append((first, found[0][0] - 1))
        stretches.append((found[-1][0] + 1, last))
    pairs.sort()
    return pairs
