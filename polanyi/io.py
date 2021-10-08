"""Input and output."""
from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import Optional, Union

import numpy as np

from polanyi.data import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM
from polanyi.typing import (
    Array1D,
    Array2D,
    Array3D,
    ArrayLike1D,
    ArrayLike2D,
    ArrayLike3D,
)
from polanyi.utils import convert_elements


def read_xyz(file: Union[str, PathLike]) -> tuple[Array1D, Union[Array2D, Array3D]]:
    """Reads xyz file.

    Returns elements as written (atomic numbers or symbols) and coordinates.

    Args:
        file: Filename or Path object

    Returns:
        elements: Elements as atomic symbols or numbers
        coordinates: Coordinates (Å)
    """
    # Read file and split lines
    with open(file) as f:
        lines = f.readlines()

    # Loop over lines and store elements and coordinates
    elements: list[Union[int, str]] = []
    coordinates: list[list[float]] = []
    n_atoms = int(lines[0].strip())
    line_chunks = zip(*[iter(lines)] * (n_atoms + 2))
    for line_chunk in line_chunks:
        for line in line_chunk[2:]:
            strip_line = line.strip().split()
            atom = strip_line[0]
            if atom.isdigit():
                atom = int(atom)
            elements.append(atom)
            coordinates.append(
                [float(strip_line[1]), float(strip_line[2]), float(strip_line[3])]
            )
    elements = np.array(elements)[:n_atoms]
    coordinates = np.array(coordinates).reshape(-1, n_atoms, 3)
    if coordinates.shape[0] == 1:
        coordinates = coordinates[0]

    return elements, coordinates


def get_xyz_string(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    comment: str = "",
) -> str:
    """Return xyz string.

    Args:
        elements: Elements as atomic symbols or numbers
        coordinates: Coordinates (Å)
        comment: Comment

    Returns:
        string: XYZ string
    """
    symbols = convert_elements(elements, output="symbols")
    coordinates = np.asarray(coordinates)
    lines = [
        f"{s:10s}{c[0]:10.5f}{c[1]:10.5f}{c[2]:10.5f}\n"
        for s, c in zip(symbols, coordinates)
    ]
    string = f"{len(lines)}\n"
    string += f"{comment}\n"
    for line in lines:
        string += line

    return string


def write_xyz(
    file: Union[str, PathLike],
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: Union[ArrayLike2D, ArrayLike3D],
    comments: Optional[Iterable[str]] = None,
) -> None:
    """Writes xyz file from elements and coordinates.

    Args:
        file: xyz file or path object
        elements: Elements as atomic symbols or numbers
        coordinates: Coordinates (Å)
        comments: Comments
    """
    # Convert elements to symbols
    symbols = convert_elements(elements, output="symbols")
    coordinates = np.array(coordinates).reshape(-1, len(symbols), 3)
    if comments is None:
        comments = [""] * len(coordinates)

    # Write the xyz file
    with open(file, "w") as f:
        for coord, comment in zip(coordinates, comments):
            xyz_string = get_xyz_string(symbols, coord, comment=comment)
            f.write(xyz_string)
            f.write("\n")


def get_coord_string(
    elements: Union[Iterable[int], Iterable[str]], coordinates: ArrayLike2D
) -> str:
    """Returns Turbomole coord string.

    Args:
        elements: Elements as atomic numbers or symbols
        coordinates: Coordinates (Å)

    Returns:
        string: String in Turbomole coord format.
    """
    symbols = convert_elements(elements, output="symbols")
    coordinates = np.asarray(coordinates)
    coordinates: Array2D = coordinates * ANGSTROM_TO_BOHR
    string = "$coord\n"
    for symbol, coordinate in zip(symbols, coordinates):
        string += (
            f"{coordinate[0]:24.10f}{coordinate[1]:24.10f}"
            f"{coordinate[2]:24.10f}{symbol:>4s}\n"
        )
    string += "$end\n"

    return string


def write_coord(
    file: Union[str, PathLike],
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
) -> None:
    """Write Turbomole coord file."""
    coord_string = get_coord_string(elements, coordinates)
    with open(file, "w") as f:
        f.write(coord_string)


def read_coord(file: Union[str, PathLike]) -> tuple[Array1D, Array2D]:
    """Read Turbomole coord file and return elements and coordinates.

    Args:
        file: Turbomole coord file

    Returns:
        elements: Elements as symbols
        coordinates: Coordinates (Å)
    """

    def read_atoms(iterlines: Iterable[str]) -> tuple[Array1D, Array2D]:
        """Read atoms helper function."""
        elements = []
        coordinates = []
        for line in iterlines:
            if "$end" in line:
                break
            strip_line = line.strip().split()
            elements.append(strip_line[3])
            coordinates.append([float(i) for i in strip_line[:3]])
        elements = np.array(elements)
        coordinates = np.array(coordinates)
        return elements, coordinates

    with open(file) as f:
        lines = f.readlines()

    iterlines = iter(lines)
    for line in iterlines:
        if "$coord" in line:
            elements, coordinates = read_atoms(iterlines)
    coordinates *= BOHR_TO_ANGSTROM
    return elements, coordinates


def write_gradient(
    file: Union[str, PathLike],
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    energy: float,
    gradient: ArrayLike1D,
) -> None:
    """Write gradient in Turbomole format.

    Args:
        file: Output file
        elements: Elements as atomic numbers or symbols
        coordinates: Coordinates (Å)
        energy: Energy (a.u.)
        gradient: Gradient (a.u.)
    """
    gradient = np.asarray(gradient)
    grad_rms = np.sqrt(np.mean(gradient ** 2))
    coord_string = get_coord_string(elements, coordinates)
    string = "$gradient\n"
    string += (
        f"cycle =      1    SCF energy ={energy:18.11f}   |dE/dxyz| ={grad_rms:10.6f}\n"
    )
    string += coord_string[7:-5]

    for grad in gradient:
        string += "".join([f"{g:22.13E}" for g in grad])
        string += "\n"
    string += "$end\n"
    with open(file, "w") as f:
        f.write(string)
