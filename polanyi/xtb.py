"""xtb interface."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import subprocess
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory

from loguru import logger
from morfeus.conformer import ConformerEnsemble
import numpy as np

from polanyi import config
from polanyi.io import read_xyz, write_coord, write_xyz
from polanyi.typing import Array2D, ArrayLike2D


def run_xtb(  # noqa: C901
    elements: Iterable[int] | Iterable[str],
    coordinates: ArrayLike2D,
    path: str | Path | None = None,
    keywords: Iterable[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    fragment_charges: list[int] | None = None,
) -> CompletedProcess:
    """Run standalone xtb from command line."""
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
    if fragment_charges is not None:
        for keyword in keywords:
            if keyword.startswith(("--chrg", "-c")):
                if int(keyword.split()[-1]) != int(sum(fragment_charges)):
                    raise ValueError(
                        f"The sum of the given fragment charges ({fragment_charges}) does not match the given total charge ({keyword.split()[-1]})."
                    )
        write_chrg(path / ".CHRG", fragment_charges)
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

    # If SCC did not converge, rerun xtb with increased electronic temperature and then restart with normal temperature
    if (
        "-1- scf: Self consistent charge iterator did not converge"
        in (path / "xtb.out").read_text()
    ):
        for file in path.iterdir():
            if file.name != "xtb.xyz":
                file.unlink()
        command = command + " --etemp 1000.0 && " + command + " --restart"
        with open(path / "xtb.out", "w") as stdout, open(
            path / "xtb.err", "w"
        ) as stderr:
            process = subprocess.run(
                command.split(),
                cwd=path,
                stdout=stdout,
                stderr=stderr,
                env=env,
            )

    return process


def run_crest(
    elements: Iterable[int] | Iterable[str],
    coordinates: ArrayLike2D,
    path: str | Path | None = None,
    keywords: Iterable[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
) -> CompletedProcess:
    """Run standalone xtb in from command line."""
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    keywords.add(f"-T {int(config.OMP_NUM_THREADS)}")

    if path is not None:
        path = Path(path)
    else:
        path = Path.cwd()
    path.mkdir(exist_ok=True)

    write_xyz(path / "crest.xyz", elements, coordinates)
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
    file: str | Path,
    keywords: MutableMapping[str, list[str]],
) -> None:
    """Write input instructions to xTB xcontrol file.
    Args:
        file: path to the xcontrol file to create
        keywords: xTB input instructions to write in the xcontrol file
    Returns:
        None, write xcontrol file
    """
    string = ""
    for header, lines in keywords.items():
        string += f"${header}\n"
        for line in lines:
            string += f"   {line}\n"
    string += "$end\n"
    with open(file, "w") as f:
        f.write(string)


def write_chrg(
    file: str | Path,
    fragment_charges: list[int],
) -> None:
    """Write fragment charges in xtb .CHRG file
    Args:
        file: path to the .CHRG file to create
        fragment_charges: charge of each non-covalently bound (NCI) fragment
    Returns:
        None, write .CHRG file
    """
    with open(file, "w") as f:
        # First line must be the total charge of the system
        f.write(f"{sum(fragment_charges)}\n")
        # Second line must contain the charges of the NCI fragments
        for charge in fragment_charges:
            f.write(f"{charge} ")


def opt_xtb(
    elements: Iterable[int] | Iterable[str],
    coordinates: ArrayLike2D,
    keywords: Iterable[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    fragment_charges: list[int] | None = None,
    path: str | Path | None = None,
) -> tuple[Array2D, float]:
    """Calculate xtb-optimized geometry.
    Args:
        elements: elements as symbols or numbers
        coordinates: coordinates [Å]
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        fragment_charges: charge of each non-covalently bound (NCI) fragment
        path to run the xtb optimisation
    Returns:
        optimized coordinates [Å] and energy [Eh]
    """
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
        fragment_charges=fragment_charges,
    )
    _, opt_coordinates = read_xyz(xtb_path / "xtbopt.xyz")
    opt_energy = parse_energy(xtb_path / "xtb.out")
    if path is None:
        temp_dir.cleanup()

    return opt_coordinates, opt_energy


def opt_crest(
    elements: Iterable[int] | Iterable[str],
    coordinates: ArrayLike2D,
    keywords: Iterable[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    path: str | Path | None = None,
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
    elements: Iterable[int] | Iterable[str],
    coordinates: ArrayLike2D,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    path: str | Path | None = None,
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
    elements: Iterable[int] | Iterable[str],
    coordinates: ArrayLike2D,
    topologies: tuple[bytes, bytes],
    e_shift: float = 0,
    coupling: float = 0,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    path: str | Path | None = None,
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
    opt_coordinates, _ = opt_xtb(
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


def parse_wbo(file: str | Path, n_atoms: int | None = None) -> Array2D:
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


def parse_engrad(file: str | Path) -> tuple[float, Array2D]:  # noqa: C901
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


def parse_energy_json(file: str | Path) -> float:
    """Parse energy from xtb JSON output."""
    with open(file) as f:
        data = json.load(f)
    energy: float = data["total energy"]
    return energy


def parse_hessian(file: str | Path) -> Array2D:
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


def parse_energy(file: str | Path) -> float:
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
