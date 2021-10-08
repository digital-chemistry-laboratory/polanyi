"""PyBerny geometry optimization interface."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import functools
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

from berny import Berny, Geometry
from berny.berny import OptPoint
from berny.coords import Bond, get_clusters
from loguru import logger
import numpy as np

from polanyi.data import ANGSTROM_TO_BOHR
from polanyi.evb import evb_eigenvalues
from polanyi.io import get_xyz_string
from polanyi.typing import Array2D, ArrayLike2D
from polanyi.utils import convert_elements
from polanyi.xtb import XTBCalculator


def e_g_function_python(
    elements: Iterable[str],
    coordinates: Array2D,
    calculators: Iterable[XTBCalculator],
    e_shift: float = 0,
    coupling: float = 0,
    path: Optional[Union[str, PathLike]] = None,
) -> tuple[float, Array2D]:
    """Find TS with GFN-FF."""
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    # Get coordinates
    energies = []
    gradients = []
    coordinates = np.array(coordinates)
    for calculator in calculators:
        calculator.calculator.update(coordinates * ANGSTROM_TO_BOHR)
        energy, gradient = calculator.sp(return_gradient=True)
        energies.append(energy)
        gradients.append(gradient)

    energies[-1] += e_shift

    # Solve EVB
    energies_ad, gradients_ad, indices = evb_eigenvalues(
        energies, gradients=gradients, coupling=coupling
    )
    gradient_rms = np.sqrt(np.mean(gradients_ad[1] ** 2))
    logger.info(
        f"Idx: {indices[1]} Energies: {energies_ad[0]:10.6f} {energies_ad[1]:10.6f} "
        f"Gradient RMS: {gradient_rms:10.6f}"
    )

    with open(path / "energies", "a") as f:
        f.write(str(energies_ad[1]) + "\n")
    with open(path / "gradients", "a") as f:
        f.write(str(gradient_rms) + "\n")
    xyz_string = get_xyz_string(elements, coordinates, comment=str(energy))
    with open(path / "traj.xyz", "a") as f:
        f.write(xyz_string)

    return energies_ad[1], gradients_ad[1]


def ts_from_gfnff_python(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    calculators: Iterable[XTBCalculator],
    e_shift: float = 0,
    coupling: float = 0,
    maxsteps: int = 100,
    params: Optional[dict[str, Any]] = None,
    active_bonds: Optional[Sequence[tuple[int]]] = None,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Optimize TS with GFNFF."""
    if params is None:
        params = {}
    if active_bonds is None:
        active_bonds = []
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
        path.mkdir(exist_ok=True)

    e_g_partial = functools.partial(
        e_g_function_python,
        calculators=calculators,
        e_shift=e_shift,
        coupling=coupling,
        path=path,
    )

    (path / "traj.xyz").unlink(missing_ok=True)
    (path / "energies").unlink(missing_ok=True)
    (path / "gradients").unlink(missing_ok=True)

    logger.remove()
    logger.add(
        path / "polanyi.log",
        format="{message}",
        filter="polanyi.pyberny",
        level="INFO",
        mode="w",
    )
    logger.info("Beginning TS optimization.")

    # Set up optimizer
    symbols = convert_elements(elements, output="symbols")
    geometry = Geometry(symbols, coordinates)
    optimizer = Berny(geometry, maxsteps=maxsteps, **params)

    # Make sure that the active bonds are added
    bond_sets = [frozenset([bond.i, bond.j]) for bond in optimizer._state.coords.bonds]
    _, C = get_clusters(geometry.bondmatrix())
    for indices in active_bonds:
        i, j = [i - 1 for i in indices]
        bond_set = frozenset([i, j])
        if bond_set not in bond_sets:
            bond_new = Bond(i, j, C=C.copy())
            optimizer._state.coords.append(bond_new)

    # Reset optimizer with new coordinates
    if len(bond_sets) > 0:
        optimizer._state.H = optimizer._state.coords.hessian_guess(
            optimizer._state.geom
        )
        optimizer._state.weights = optimizer._state.coords.weights(
            optimizer._state.geom
        )
        optimizer._state.future = OptPoint(
            optimizer._state.coords.eval_geom(optimizer._state.geom), None, None
        )
        optimizer._state.first = True
        for line in str(optimizer._state.coords).split("\n"):
            optimizer._log.info(line)

    # Run optimization
    for geom in optimizer:
        energy, gradients = e_g_partial(elements, geom.coords)
        optimizer.send((energy, gradients))

    logger.info("TS optimization done.")

    opt_coordinates: np.ndarray = np.ascontiguousarray(geom.coords)

    return opt_coordinates
