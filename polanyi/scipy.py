"""SciPy optimizer interface."""
from __future__ import annotations

from collections.abc import Iterable, MutableMapping
import functools
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

from loguru import logger
import numpy as np
from scipy.optimize import minimize

from polanyi import config
from polanyi.data import BOHR_TO_ANGSTROM
from polanyi.evb import evb_eigenvalues
from polanyi.io import get_xyz_string
from polanyi.typing import Array1D, Array2D, ArrayLike2D
from polanyi.xtb import parse_energy, parse_engrad, parse_hessian, run_xtb


def e_g_function(  # noqa: C901
    coordinates_1D: Array1D,
    elements: Union[Iterable[str], Iterable[int]],
    topologies: Iterable[bytes],
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    e_shift: float = 0,
    coupling: float = 0,
    path: Optional[Union[str, PathLike]] = None,
) -> Union[float, Array1D, Array2D]:
    """Find TS with GFN-FF."""
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    do_energy = False
    do_gradient = False
    do_hessian = False
    if "--grad" in keywords:
        do_gradient = True
    elif "--hess" in keywords:
        do_hessian = True
    else:
        do_energy = True

    coordinates = coordinates_1D.reshape(-1, 3)

    # Get coordinates
    topologies = list(topologies)
    if path is None:
        path = Path.cwd()
        temp_dirs = [
            TemporaryDirectory(dir=config.TMP_DIR) for i in range(len(topologies))
        ]
        xtb_paths = [path / temp_dir.name for temp_dir in temp_dirs]
        cleanup = True
    else:
        path = Path(path)
        xtb_paths = [path / str(i) for i in range(len(topologies))]
        cleanup = False

    energies = []
    gradients = []
    hessians = []
    for topology, xtb_path in zip(topologies, xtb_paths):
        xtb_path.mkdir(exist_ok=True)
        if not (xtb_path / "gfnff_topo").exists():
            with open(xtb_path / "gfnff_topo", "wb") as f:
                f.write(topology)
        run_xtb(
            elements,
            coordinates,
            path=xtb_path,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
        )
        if do_energy is True:
            energy = parse_energy(xtb_path / "xtb.out")
        elif do_gradient is True:
            energy, gradient = parse_engrad(xtb_path / "xtb.engrad")
            gradients.append(gradient)
        elif do_hessian is True:
            energy = parse_energy(xtb_path / "xtb.out")
            hessian = parse_hessian(xtb_path / "hessian")
            hessians.append(hessian)
        energies.append(energy)

    energies[-1] += e_shift

    # Clean up temporary directories
    if cleanup is True:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    # Solve EVB
    if do_energy:
        energies_ad, indices = evb_eigenvalues(energies, coupling=coupling)
        return float(energies_ad[1])
    elif do_gradient:
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
        gradient_ad: Array1D = gradients_ad[1].flatten()
        return gradient_ad
    elif do_hessian:
        energies_ad, hessians_ad, indices = evb_eigenvalues(
            energies, hessians=hessians, coupling=coupling
        )
        hessian_ad: Array2D = hessians_ad[1]
        return hessian_ad
    else:
        raise ValueError("Couldn't identify type of xtb calculation.")


def ts_from_gfnff(
    elements: Union[Iterable[int], Iterable[str]],
    coordinates: ArrayLike2D,
    topologies: Iterable[bytes],
    keywords: Optional[list[str]] = None,
    xcontrol_keywords: Optional[MutableMapping[str, list[str]]] = None,
    e_shift: float = 0,
    coupling: float = 0,
    maxsteps: int = 100,
    tol: float = 1e-6,
    path: Optional[Union[str, PathLike]] = None,
) -> Array2D:
    """Optimize TS with GFNFF."""
    coordinates = np.asarray(coordinates)

    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
        path.mkdir(exist_ok=True)

    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    for keyword in ["--grad", "--hess"]:
        keywords.discard(keyword)

    keywords_sp = keywords.copy()
    keywords_sp.add("--sp")
    keywords_grad = keywords.copy()
    keywords_grad.add("--grad")
    keywords_hess = keywords.copy()
    keywords_hess.add("--hess")

    e_g_partial_sp = functools.partial(
        e_g_function,
        elements=elements,
        topologies=topologies,
        keywords=keywords_sp,
        xcontrol_keywords=xcontrol_keywords,
        e_shift=e_shift,
        coupling=coupling,
        path=path,
    )
    e_g_partial_grad = functools.partial(
        e_g_function,
        elements=elements,
        topologies=topologies,
        keywords=keywords_grad,
        xcontrol_keywords=xcontrol_keywords,
        e_shift=e_shift,
        coupling=coupling,
        path=path,
    )
    e_g_partial_hess = functools.partial(
        e_g_function,
        elements=elements,
        topologies=topologies,
        keywords=keywords_hess,
        xcontrol_keywords=xcontrol_keywords,
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
        filter="polanyi.scipy",
        level="INFO",
        mode="w",
    )
    logger.info("Beginning TS optimization.")

    result = minimize(
        e_g_partial_sp,
        coordinates.flatten(),
        jac=e_g_partial_grad,
        hess=e_g_partial_hess,
        method="trust-ncg",
        tol=tol,
        options={
            "maxiter": maxsteps,
            "initial_trust_radius": 0.8 * BOHR_TO_ANGSTROM,
            "max_trust_radius": 1.0 * BOHR_TO_ANGSTROM,
            "gtol": 1e-4 * BOHR_TO_ANGSTROM,
        },
    )
    opt_coordinates: Array2D = result.x.reshape(-1, 3)

    return opt_coordinates
