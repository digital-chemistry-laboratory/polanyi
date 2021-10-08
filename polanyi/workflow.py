"""Workflows."""
from __future__ import annotations

from collections.abc import Collection, MutableMapping
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import numpy as np

from polanyi.geometry import two_frags_from_bo
from polanyi.typing import Array1D, Array2D, ArrayLike2D
from polanyi.xtb import opt_xtb, parse_energy, run_xtb, wbo_xtb, XTBCalculator


def setup_gfnff_calculators(
    elements: Union[Collection[int], Collection[str]],
    coordinates: Collection[ArrayLike2D],
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    paths: Optional[Collection[Union[str, PathLike]]] = None,
) -> list[bytes]:
    """Sets up force fields for GFNFF calculation."""
    if paths is None:
        temp_dirs = [TemporaryDirectory() for i in range(len(coordinates))]
        xtb_paths = [Path(temp_dir.name) for temp_dir in temp_dirs]
    else:
        xtb_paths = [Path(path) for path in paths]

    topologies = []
    for coordinates_, xtb_path in zip(coordinates, xtb_paths):
        run_xtb(
            elements,
            coordinates_,
            path=xtb_path,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
        )
        with open(xtb_path / "gfnff_topo", "rb") as f:
            topology = f.read()
        topologies.append(topology)

    if paths is None:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    return topologies


def setup_gfnff_calculators_python(
    elements: Union[Collection[int], Collection[str]],
    coordinates: Collection[ArrayLike2D],
    charge: int = 0,
    solvent: Optional[str] = None,
) -> list[XTBCalculator]:
    """Sets up force fields for GFNFF calculation."""
    calculators = []
    for coordinates_ in coordinates:
        calculator = XTBCalculator(elements, coordinates_, charge=charge)
        _ = calculator.sp(return_gradient=False)
        calculators.append(calculator)
    return calculators


def opt_frags_from_complex(
    elements: Union[Collection[int], Collection[str]],
    coordinates: ArrayLike2D,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
) -> list[tuple[Array1D, Array2D]]:
    """Optimize two fragments from complex.

    Args:
        elements: Elements as symbols or numbers
        coordinates: Coordinates (Å)
        keywords: xtb command line keywords
        xcontrol_keywords: xtb xcontrol keywords

    Returns:
        fragments: Fragment elements and coordinates
    """
    elements = np.array(elements)
    coordinates = np.asarray(coordinates)
    bo_matrix = wbo_xtb(elements, coordinates, keywords=keywords)
    frag_indices = two_frags_from_bo(bo_matrix)
    fragments = []
    for indices in frag_indices:
        frag_elements = elements[indices]
        frag_coordinates = coordinates[indices]
        opt_coordinates = opt_xtb(
            frag_elements,
            frag_coordinates,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
        )
        fragments.append((frag_elements, opt_coordinates))

    return fragments


def opt_constrained_complex(
    elements: Union[Collection[int], Collection[str]],
    coordinates: ArrayLike2D,
    distance_constraints: Optional[MutableMapping[tuple[int, int], float]] = None,
    atom_constraints: Optional[Collection[int]] = None,
    fix_atoms: Optional[Collection[int]] = None,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    fc: Optional[float] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Optimize constrained complex."""
    if distance_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_constraints = xcontrol_keywords.setdefault("constrain", [])
        if fc is not None:
            xcontrol_constraints.append(f"force constant={fc}")
        for (i, j), distance in distance_constraints.items():
            string = f"distance: {i}, {j}, {distance}"
            xcontrol_constraints.append(string)
        xcontrol_keywords["constrain"] = xcontrol_constraints
    if atom_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_atom_constraints = xcontrol_keywords.setdefault("constrain", [])
        fix_string = "atoms: " + ",".join([str(i) for i in atom_constraints])
        xcontrol_atom_constraints.append(fix_string)
    if fix_atoms is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_fix_atoms = xcontrol_keywords.setdefault("fix", [])
        fix_string = "atoms: " + ",".join([str(i) for i in fix_atoms])
        xcontrol_fix_atoms.append(fix_string)

    opt_coordinates = opt_xtb(
        elements,
        coordinates,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        path=path,
    )

    return opt_coordinates


def calculate_e_shift_xtb(
    elements: Union[Collection[int], Collection[str]],
    coordinates: Collection[ArrayLike2D],
    topologies: Collection[bytes],
    keywords_ff: Optional[list[str]] = None,
    keywords_sp: Optional[list[str]] = None,
    xcontrol_keywords_ff: Optional[MutableMapping[str, list[str]]] = None,
    xcontrol_keywords_sp: Optional[MutableMapping[str, list[str]]] = None,
    paths: Optional[Collection[Union[str, PathLike]]] = None,
) -> tuple[float, float, float]:
    """Calculate energy shift between geometries."""
    if paths is None:
        temp_dirs = [TemporaryDirectory() for i in range(len(coordinates))]
        xtb_paths = [Path(temp_dir.name) for temp_dir in temp_dirs]
    else:
        xtb_paths = [Path(path) for path in paths]

    energies_ff = []
    energies_sp = []
    for coordinates_, topology, xtb_path in zip(coordinates, topologies, xtb_paths):
        xtb_path.mkdir(exist_ok=True)
        with open(xtb_path / "gfnff_topo", "wb") as f:
            f.write(topology)
        run_xtb(
            elements,
            coordinates_,
            path=xtb_path,
            keywords=keywords_ff,
            xcontrol_keywords=xcontrol_keywords_ff,
        )
        energy = parse_energy(xtb_path / "xtb.out")
        energies_ff.append(energy)

        run_xtb(
            elements,
            coordinates_,
            path=xtb_path,
            keywords=keywords_sp,
            xcontrol_keywords=xcontrol_keywords_sp,
        )
        energy = parse_energy(xtb_path / "xtb.out")
        energies_sp.append(energy)

    if paths is None:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    e_diff_gfn = energies_sp[-1] - energies_sp[0]
    e_diff_ff = energies_ff[-1] - energies_ff[0]
    e_shift = e_diff_gfn - e_diff_ff

    return e_shift, e_diff_gfn, e_diff_ff


def calculate_e_shift_xtb_python(
    calculators: Collection[XTBCalculator], method: str = ("GFN2-xTB")
) -> tuple[float, float, float]:
    """Calculate energy shift between geometries."""
    energies_gfn = []
    energies_ff = []
    for calculator in calculators:
        energy_ff = calculator.sp(return_gradient=False)
        calculator_sp = XTBCalculator(
            calculator.elements,
            calculator.coordinates,
            method=method,
            charge=calculator.charge,
            solvent=calculator.solvent,
        )
        energy_gfn = calculator_sp.sp(return_gradient=False)
        energies_gfn.append(energy_gfn)
        energies_ff.append(energy_ff)
    e_diff_gfn = energies_gfn[-1] - energies_gfn[0]
    e_diff_ff = energies_ff[-1] - energies_ff[0]
    e_shift = e_diff_gfn - e_diff_ff

    return e_shift, e_diff_gfn, e_diff_ff
