"""xtb interface."""
from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from itertools import islice
import json
import os
from os import PathLike
from pathlib import Path
import shutil
import subprocess
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from typing import Literal, Optional, overload, Union

from loguru import logger
from morfeus.conformer import ConformerEnsemble
import numpy as np
from wurlitzer import pipes
from xtb.interface import Calculator
from xtb.utils import get_method, get_solvent

from polanyi import config
from polanyi.data import ANGSTROM_TO_BOHR
from polanyi.io import read_xyz, write_coord, write_xyz
from polanyi.typing import Array1D, Array2D, ArrayLike2D
from polanyi.utils import convert_elements


class XTBCalculator:
    """xTB calculator class.

    Args:
        elements: Elements as atomic numbers
        coordnates: Coordinates (Å)

    Attributes:
        calculator: xtb-python calculator
    """

    calculator: Calculator

    def __init__(
        self,
        elements: Union[Iterable[int], Iterable[str]],
        coordinates: ArrayLike2D,
        method: str = "GFNFF",
        charge: int = 0,
        solvent: Optional[str] = None,
    ) -> None:
        elements = np.array(convert_elements(elements, output="numbers"))
        coordinates = np.ascontiguousarray(coordinates)
        coordinates_au: Array2D = coordinates * ANGSTROM_TO_BOHR
        calc_method = get_method(method)
        if calc_method is None:
            raise ValueError("Calculation method not valid")
        with pipes() as _:
            calculator = Calculator(
                calc_method, elements, coordinates_au, charge=charge
            )
        if solvent is not None:
            xtb_solvent = get_solvent(solvent)
            if xtb_solvent is None:
                raise ValueError(f"{solvent} is not supported.")
            calculator.set_solvent(xtb_solvent)
        self.calculator = calculator
        self._solvent = solvent
        self._charge = charge
        self._elements = elements
        self._coordinates = coordinates
        self._method = method

    @property
    def elements(self) -> Array1D:
        """Elements."""
        return self._elements

    @property
    def solvent(self) -> Optional[str]:
        """Solvent."""
        return self._solvent

    @property
    def charge(self) -> int:
        """Charge."""
        return self._charge

    @property
    def coordinates(self) -> Array2D:
        """Coordinates (Å)."""
        return self._coordinates

    @coordinates.setter
    def coordinates(self, coordinates: Array2D) -> None:
        self.calculator.update(coordinates * ANGSTROM_TO_BOHR)
        self._coordinates = coordinates

    @property
    def method(self) -> str:
        """Method."""
        return self._method

    @overload
    def sp(self, return_gradient: Literal[True]) -> tuple[float, Array2D]:
        ...

    @overload
    def sp(self, return_gradient: Literal[False]) -> float:
        ...

    def sp(self, return_gradient: bool = True) -> Union[float, tuple[float, Array2D]]:
        """Do single point calculation and return result."""
        with pipes() as _:
            results = self.calculator.singlepoint()
        energy: float = results.get_energy()

        if return_gradient is True:
            gradient = results.get_gradient()
            return energy, gradient
        else:
            return energy


def run_xtb(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    path: Optional[Union[str, PathLike]] = None,
    keywords: Optional[Iterable[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
) -> CompletedProcess:
    """Run standalone xtb in from command line."""
    if keywords is None:
        keywords = []
    if path is not None:
        path = Path(path)
    else:
        path = Path.cwd()
    path.mkdir(exist_ok=True)

    write_xyz(path / "xtb.xyz", elements, coordinates)
    command = "xtb xtb.xyz " + " ".join(f"{keyword}" for keyword in keywords)
    if xcontrol_keywords is not None:
        write_xcontrol(path / "xcontrol", xcontrol_keywords)
        command += " -I xcontrol"
    with open(path / "xtb.out", "w") as stdout, open(path / "xtb.err", "w") as stderr:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = f"{config.OMP_NUM_THREADS},1"
        env["MKL_NUM_THREADS"] = f"{config.OMP_NUM_THREADS}"
        env["OMP_STACKSIZE"] = config.OMP_STACKSIZE
        env["OMP_MAX_ACTIVE_LEVELS"] = str(config.OMP_MAX_ACTIVE_LEVELS)
        process = subprocess.run(
            command.split(),
            cwd=path,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )

    return process


def run_crest(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    path: Optional[Union[str, PathLike]] = None,
    keywords: Optional[Iterable[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
) -> CompletedProcess:
    """Run standalone xtb in from command line."""
    if keywords is None:
        keywords = []
    if path is not None:
        path = Path(path)
    else:
        path = Path.cwd()
    path.mkdir(exist_ok=True)

    write_xyz(path / "crest.xyz", elements, coordinates)
    keywords.update("-T", float(config.OMP_NUM_THREADS))
    command = "crest crest.xyz " + " ".join(f"{keyword}" for keyword in keywords)
    if xcontrol_keywords is not None:
        write_xcontrol(path / ".xcontrol", xcontrol_keywords)
    with open(path / "crest.out", "w") as stdout, open(
        path / "crest.err", "w"
    ) as stderr:
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = f"{config.OMP_NUM_THREADS},1"
        env["MKL_NUM_THREADS"] = f"{config.OMP_NUM_THREADS}"
        env["OMP_STACKSIZE"] = config.OMP_STACKSIZE
        env["OMP_MAX_ACTIVE_LEVELS"] = str(config.OMP_MAX_ACTIVE_LEVELS)
        process = subprocess.run(
            command.split(),
            cwd=path,
            stdout=stdout,
            stderr=stderr,
            env=env,
        )

    return process


def write_xcontrol(
    file: Union[str, PathLike],
    keywords: MutableMapping[str, list[str]],
) -> None:
    """Write keywords to xcontrol file."""
    string = ""
    for header, lines in keywords.items():
        string += f"${header}\n"
        for line in lines:
            string += f"   {line}\n"
    string += "$end\n"
    with open(file, "w") as f:
        f.write(string)


def opt_xtb(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    keywords: Optional[Iterable[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Returns xtb-optimized geometry."""
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    keywords.add("--opt")

    if path is None:
        temp_dir = TemporaryDirectory(dir=config.TMP_DIR)
        xtb_path = Path(temp_dir.name)
    else:
        xtb_path = Path(path)

    run_xtb(
        elements,
        coordinates,
        path=xtb_path,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
    )
    _, opt_coordinates = read_xyz(xtb_path / "xtbopt.xyz")
    if path is None:
        temp_dir.cleanup()

    return opt_coordinates


def opt_crest(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    keywords: Optional[Iterable[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> ConformerEnsemble:
    """Returns xtb-optimized geometry."""
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])

    if path is None:
        temp_dir = TemporaryDirectory(dir=config.TMP_DIR)
        crest_path = Path(temp_dir.name)
    else:
        crest_path = Path(path)

    run_crest(
        elements,
        coordinates,
        path=crest_path,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
    )
    conformer_ensemble = ConformerEnsemble.from_crest(crest_path)
    if path is None:
        temp_dir.cleanup()

    return conformer_ensemble


def wbo_xtb(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Returns wbo bond order matrix from xtb."""
    if path is None:
        temp_dir = TemporaryDirectory(dir=config.TMP_DIR)
        xtb_path = Path(temp_dir.name)
    else:
        xtb_path = Path(path)

    run_xtb(
        elements,
        coordinates,
        path=xtb_path,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
    )
    bo_matrix = parse_wbo(xtb_path / "wbo")

    if path is None:
        temp_dir.cleanup()

    return bo_matrix


def ts_from_gfnff_xtb(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    topologies: tuple[bytes, bytes],
    e_shift: float = 0,
    coupling: float = 0,
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Optimize TS with GFNFF."""
    if path is None:
        temp_dir = TemporaryDirectory(dir=config.TMP_DIR)
        path = Path(temp_dir.name)
        cleanup = True
    else:
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        cleanup = False
    if keywords is None:
        keywords = []

    (path / "traj.xyz").unlink(missing_ok=True)
    (path / "energies").unlink(missing_ok=True)
    (path / "gradients").unlink(missing_ok=True)

    logger.remove()
    logger.add(
        path / "polanyi.log",
        format="{message}",
        filter="polanyi.pyscf",
        level="INFO",
        mode="w",
    )

    # Set up directories and files needed for optimization
    path_r = path / "reactant_ff"
    if path_r.exists():
        shutil.rmtree(path_r)
    path_r.mkdir()
    with open(path_r / "gfnff_topo", "wb") as f:
        f.write(topologies[0])

    path_p = path / "product_ff"
    if path_p.exists():
        shutil.rmtree(path_p)
    path_p.mkdir()
    with open(path_p / "gfnff_topo", "wb") as f:
        f.write(topologies[1])

    with open(path / "coupling", "w") as f:
        f.write(str(coupling))
    with open(path / "e_shift", "w") as f:
        f.write(str(e_shift))
    write_coord(path / "coord", elements, coordinates)

    with open(path / "keywords", "w") as f:
        for keyword in keywords:
            f.write(keyword + "\n")

    if xcontrol_keywords is not None:
        write_xcontrol(path / "xcontrol", xcontrol_keywords)

    # Run optimization with xtb as optimizer
    logger.remove()
    logger.add(
        path / "polanyi.log",
        format="{message}",
        level="INFO",
        mode="w",
    )
    logger.info("Beginning TS optimization.")

    keywords = set([keyword.strip().lower() for keyword in keywords])
    keywords.add("--tm")
    opt_coordinates = opt_xtb(
        elements,
        coordinates,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        path=path,
    )

    logger.info("TS optimization done.")
    logger.remove()

    if cleanup is True:
        temp_dir.cleanup()

    return opt_coordinates


def parse_wbo(file: Union[str, PathLike], n_atoms: Optional[int] = None) -> Array2D:
    """Returns bond order matrix from xtb wbo file.

    The number of atoms will be guessed from the largest atom index in the file. This
    sometimes fails if there is no bond to that atom. In these cases, the number of
    atoms can be passed explicitly.

    Args:
        file: xtb wbo file
        n_atoms: Number of atoms

    Returns:
        bo_matrix: Bond order matrix
    """
    with open(file) as f:
        lines = f.readlines()

    # Read bond orders into dictionary
    bond_orders = {}
    if n_atoms is None:
        n_atoms = 0
    for line in lines:
        strip_line = line.strip().split()
        i, j = [int(i) for i in strip_line[:2]]
        n_atoms = max([n_atoms, i, j])
        bo = float(strip_line[2])
        bond_orders[(i, j)] = bo

    # Create bond order matrix
    bo_matrix = np.zeros((n_atoms, n_atoms))
    for (i, j), bo in bond_orders.items():
        bo_matrix[i - 1, j - 1] = bo_matrix[j - 1, i - 1] = bo

    return bo_matrix


def parse_engrad(file: Union[str, PathLike]) -> tuple[float, Array2D]:  # noqa: C901
    """Parse xtb engrad file to return energy and gradient."""

    def read_atoms(iterlines: Iterable[str]) -> int:
        """Read atoms."""
        for line in islice(iterlines, 1, None):
            if "#" in line:
                break
            n_atoms = int(line.strip().split()[0])
        return n_atoms

    def read_energy(iterlines: Iterable[str]) -> float:
        """Read energy."""
        for line in islice(iterlines, 1, None):
            if "#" in line:
                break
            energy = float(line.strip().split()[0])
        return energy

    def read_gradient(iterlines: Iterable[str]) -> Array2D:
        """Read gradient."""
        gradient = []
        for line in islice(iterlines, 1, None):
            if "#" in line:
                break
            gradient.append(float(line.strip().split()[0]))
        gradient = np.array(gradient)
        return gradient

    with open(file) as f:
        lines = f.readlines()
    iterlines = iter(lines)
    for line in iterlines:
        if "Number of atoms" in line:
            n_atoms = read_atoms(iterlines)
        if "The current total energy in Eh" in line:
            energy = read_energy(iterlines)
        if "# The current gradient in Eh/bohr" in line:
            gradient = read_gradient(iterlines)
    gradient = np.array(gradient).reshape(n_atoms, 3)
    return energy, gradient


def parse_energy_json(file: Union[str, PathLike]) -> float:
    """Parse energy from xtb JSON output."""
    with open(file) as f:
        data = json.load(f)
    energy: float = data["total energy"]
    return energy


def parse_hessian(file: Union[str, PathLike]) -> Array2D:
    """Parse hessian for xtb.

    Args:
        file: Hessian file

    Returns:
        hessian: Hessian
    """
    # Read hessian file
    with open(file) as f:
        lines = f.readlines()

    # Parse file
    hessian = []
    for line in lines[1:]:
        hessian.extend([float(value) for value in line.strip().split()])

    # Set up force constant matrix
    dimension = int(np.sqrt(len(hessian)))
    hessian = np.array(hessian).reshape(dimension, dimension)
    return hessian


def parse_energy(file: Union[str, PathLike]) -> float:
    """Parse energy from xtb log file.

    Args:
        file: xtb log file

    Returns:
        energy: Energy (a.u.)
    """
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        if "TOTAL ENERGY" in line:
            energy = float(line.strip().split()[3])
    return energy
