"""Reading the corpus' whitespace-separated text tables."""

from pathlib import Path

import numpy as np


class DatasetError(Exception):
    """An input file that does not have the format the corpus promises."""


def read_table(path: Path, column_types: list[type]) -> list[np.ndarray]:
    """Read a file whose every line holds len(column_types) fields.

    Returns one array per column. A line with the wrong number of fields or
    a field that does not parse is an error naming the file and line, so a
    damaged file can never pass as a shorter or different one.
    """
    width: int = len(column_types)
    columns: list[list] = [[] for _ in range(width)]
    lines: list[str] = path.read_text().splitlines()

    for number, line in enumerate(lines, start=1):
        fields: list[str] = line.split()
        if len(fields) != width:
            raise DatasetError(
                f"{path}:{number}: expected {width} fields, "
                f"found {len(fields)}: {line!r}")

        for column in range(width):
            kind: type = column_types[column]
            try:
                columns[column].append(kind(fields[column]))
            except ValueError:
                raise DatasetError(
                    f"{path}:{number}: field {column + 1} is not "
                    f"{kind.__name__}: {fields[column]!r}") from None

    arrays: list[np.ndarray] = []
    for column in range(width):
        arrays.append(np.array(columns[column], dtype=column_types[column]))
    return arrays
