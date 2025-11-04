"""PySCF geometry optimization interface."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import functools
from io import StringIO
import os
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
import shutil
from typing import Any

import geometric
from geometric.engine import ConicalIntersection
import numpy as np
from pyscf import __config__, lib
from pyscf.geomopt import as_pyscf_method, berny_solver, geometric_solver
from pyscf.grad.rhf import GradientsMixin
from pyscf.gto import Mole

from polanyi import config
from polanyi.evb import evb_eigenvalues
from polanyi.typing import Array2D, ArrayLike2D
from polanyi.utils import convert_elements
from polanyi.xtb import parse_engrad, run_xtb


@dataclass
class OptResults:
    """Results of PySCF geometry optimization."""

    coordinates: list[Array2D] = field(default_factory=list)
    energies_diabatic: list[list[float]] = field(default_factory=list)
    energies_adiabatic: list[list[float]] = field(default_factory=list)
    gradients_diabatic: list[list[Array2D]] = field(default_factory=list)
    gradients_adiabatic: list[list[Array2D]] = field(default_factory=list)
    indices: list[list[int]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""


class EnginesWrapper:
    """Wrapper (with list-like behaviours) for multiple engines for ConicalIntersection from geomeTRIC 1.0.1."""

    def __init__(self, engine_list):
        self.engines = engine_list

    def __getitem__(self, key):
        return self.engines[key]

    def __len__(self):
        return len(self.engines)

    def __deepcopy__(self, memo):
        # Create a new wrapper instance, sharing the same engine instances
        # to avoid errors raised by deep copying of non-pickleable parts.
        new_wrapper = EnginesWrapper(self.engines)
        memo[id(self)] = new_wrapper
        return new_wrapper


def e_g_function(
    mol: "Mole",
    topologies: Sequence[bytes],
    results: OptResults,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    e_shift: float = 0,
    coupling: float = 0,
    path: str | Path | None = None,
) -> tuple[float, Array2D]:
    """Calculate energy and gradient from GFN-FF and then solving eigenvalues of EVB.
    Args:
        mol: PySCF molecule
        topologies: sequence of GFN-FF topologies for each ground state
        results: OptResults object to store optimization results
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energy
        coupling: coupling constant between the ground states force fields
        path: path where to run calculations
    Returns:
        tuple of adiabatic energy and gradient
    """
    topologies = list(topologies)
    elements = mol.atom_charges()
    coordinates = mol.atom_coords(unit="ANG")

    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    keywords.add("--grad")

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
        energy, gradient = parse_engrad(xtb_path / "xtb.engrad")
        energies.append(energy)
        gradients.append(gradient)

    energies[-1] += e_shift

    # Solve EVB
    energies_ad, gradients_ad, indices = evb_eigenvalues(
        energies, gradients=gradients, coupling=coupling
    )

    # Clean up temporary directory
    if cleanup is True:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    # Store results
    results.coordinates.append(coordinates)
    results.energies_diabatic.append(energies)
    results.energies_adiabatic.append(energies_ad)
    results.gradients_diabatic.append(gradients)
    results.gradients_adiabatic.append(gradients_ad)
    results.indices.append(indices)

    return energies_ad[1], gradients_ad[1]


def e_g_function_ci(
    mol: "Mole",
    topology: bytes,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    e_shift: float = 0,
    path: str | Path | None = None,
) -> tuple[float, Array2D]:
    """Get energy and gradient from GFN-FF to use in conical intersection optimisation.
    Args:
        mol: PySCF molecule
        topology: GFN-FF topology for a ground state
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energy
        path: path where to run calculations
    Returns:
        tuple of energy and gradient
    """
    elements = mol.atom_charges()
    coordinates = mol.atom_coords(unit="ANG")

    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    keywords.add("--grad")

    if path is None:
        path = Path.cwd()
        temp_dir = TemporaryDirectory(dir=config.TMP_DIR)
        xtb_path = path / temp_dir.name
        cleanup = True
    else:
        xtb_path = Path(path)
        cleanup = False

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
    energy, gradient = parse_engrad(xtb_path / "xtb.engrad")
    energy += e_shift

    # Clean up temporary directory
    if cleanup is True:
        temp_dir.cleanup()

    return energy, gradient


def e_g_function_path(
    mol: "Mole",
    topologies: Sequence[bytes],
    results: OptResults,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    e_shift: float = 0,
    lam: float = 0,
    path: str | Path | None = None,
) -> tuple[float, Array2D]:
    """Calculate energy and gradient from GFN-FF and then weighted average of ground state force fields.
    Args:
        mol: PySCF molecule
        topologies: sequence of GFN-FF topologies for each ground state
        results: OptResults object to store optimization results
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        lam: lambda incrementing variable from reactant (0) to product (1)
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energy
        path: path where to run calculations
    Returns:
        tuple of weighted energy and gradient
    """
    topologies = list(topologies)
    if len(topologies) != 2:
        raise ValueError("Needs two ground state topologies to use this function.")
    elements = mol.atom_charges()
    coordinates = mol.atom_coords(unit="ANG")

    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    keywords.add("--grad")

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
        energy, gradient = parse_engrad(xtb_path / "xtb.engrad")
        energies.append(energy)
        gradients.append(gradient)

    energies[-1] += e_shift

    # Equation for EVB model potential
    energy_weighted = (1 - lam) * energies[0] + lam * energies[-1]
    gradient_weighted = (1 - lam) * gradients[0] + lam * gradients[-1]

    # Clean up temporary directory
    if cleanup is True:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    # Store results
    results.coordinates.append(coordinates)
    results.energies_diabatic.append(energies)
    results.gradients_diabatic.append(gradients)
    results.energies_adiabatic.append(energy_weighted)
    results.gradients_adiabatic.append(gradient_weighted)

    return energy_weighted, gradient_weighted


def ts_from_gfnff(
    elements: Sequence[int] | Sequence[str],
    coordinates: ArrayLike2D,
    topologies: Sequence[bytes],
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    e_shift: float = 0,
    coupling: float = 0.001,
    maxsteps: int = 100,
    callback: Callable[[dict[str, Any]], None] | None = None,
    conv_params: dict[str, Any] | None = None,
    solver: str = "geometric",
    path: str | Path | None = None,
) -> OptResults:
    """Optimize TS with GFN-FF and EVB.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: coordinates of guess TS structure [Å]
        topologies: sequence of GFN-FF topologies for each ground state
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energy
        coupling: coupling constant between the ground states force fields
        maxsteps: maximum number of optimization steps
        callback: function to call after each optimization step
        conv_params: convergence parameters for PySCF optimization
        solver: PySCF optimization solver (geometric or pyberny)
        path: path where to run calculations
    Returns:
        results of TS optimization
    """
    if conv_params is None:
        conv_params = {}
    if path:
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    if "--gfnff" not in keywords:
        keywords.add("--gfnff")
    results = OptResults()

    mole = get_pyscf_mole(elements, coordinates)
    e_g_partial = functools.partial(
        e_g_function,
        topologies=topologies,
        results=results,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        e_shift=e_shift,
        coupling=coupling,
        path=path,
    )

    if solver == "pyberny":
        pyscf_solver = berny_solver
    elif solver == "geometric":
        pyscf_solver = geometric_solver
    with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
        pyscf_solver.optimize(
            as_pyscf_method(mole, e_g_partial),
            maxsteps=maxsteps,
            callback=callback,
            **conv_params,
        )
    results.stdout = stdout.getvalue()
    results.stderr = stderr.getvalue()

    return results


def ts_from_gfnff_ci(
    elements: Sequence[int] | Sequence[str],
    coordinates: ArrayLike2D,
    topologies: Sequence[bytes],
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    e_shift: float = 0,
    maxsteps: int = 100,
    alpha: float = 0.025,
    sigma: float = 3.5,
    callback: Callable[[dict[str, Any]], None] | None = None,
    conv_params: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> tuple[Array2D, float]:
    """Optimize TS with GFN-FF and conical intersection.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: coordinates of guess TS structure [Å]
        topologies: sequence of GFN-FF topology for each ground state
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energy
        maxsteps: maximum number of optimization steps
        alpha: width parameter for penalty function in conical interesection optimization
        sigma: scaling parameter for penalty function in conical interesection optimization
        callback: function to call after each optimization step
        conv_params: convergence parameters for PySCF optimization
        path: path where to run calculations
    Returns:
        optimized TS coordinates [Å] and energy [Eh]
    """
    if conv_params is None:
        conv_params = {}
    if path:
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        path_1 = path / "0"
        path_2 = path / "1"
        if (path_1 / "gfnff_topo").exists() or (path_2 / "gfnff_topo").exists():
            raise FileExistsError(
                f"Paths {path_1} or {path_2} already contain 'gfnff_topo' files. Remove before new xtb calculations."
                f"\nIf other files are present, they will be overwritten."
            )
    else:
        path_1 = None
        path_2 = None
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    if "--gfnff" not in keywords:
        keywords.add("--gfnff")

    mole = get_pyscf_mole(elements, coordinates)
    topologies = list(topologies)

    e_g_partial_1 = functools.partial(
        e_g_function_ci,
        topology=topologies[0],
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        e_shift=0,
        path=path_1,
    )

    e_g_partial_2 = functools.partial(
        e_g_function_ci,
        topology=topologies[1],
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        e_shift=e_shift,
        path=path_2,
    )

    final_energy: float | None = None

    def store_e_cb(info: dict):
        nonlocal final_energy
        final_energy = info["energy"]
        if callback is not None:
            callback(info)

    _, opt_mole = optimize_ci(
        [as_pyscf_method(mole, e_g_partial_1), as_pyscf_method(mole, e_g_partial_2)],
        maxsteps=maxsteps,
        alpha=alpha,
        sigma=sigma,
        callback=store_e_cb,
        **conv_params,
    )

    opt_coordinates: Array2D = np.ascontiguousarray(opt_mole.atom_coords(unit="ANG"))

    return opt_coordinates, final_energy


def rxn_path_from_gfnff(
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[Array2D],
    topologies: Sequence[bytes],
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    e_shift: float = 0,
    maxsteps: int = 100,
    callback: Callable[[dict[str, Any]], None] | None = None,
    conv_params: dict[str, Any] | None = None,
    solver: str = "geometric",
    path: str | Path | None = None,
) -> list[OptResults]:
    """Optimize structures along reaction path with GFN-FF and EVB.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: guess coordinates of each point on the reaction path [Å]
        topologies: sequence of GFN-FF topologies for each ground state
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energy
        maxsteps: maximum number of optimization steps
        callback: function to call after each optimization step
        conv_params: convergence parameters for PySCF optimization
        solver: PySCF optimization solver (geometric or pyberny)
        path: path where to run calculations
    Returns:
        results for each structure optimization
    """
    if conv_params is None:
        conv_params = {}
    if path:
        path = Path(path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    if "--gfnff" not in keywords:
        keywords.add("--gfnff")

    results_path = []
    for coords, lam in zip(coordinates, np.linspace(0, 1, len(coordinates))):
        results = OptResults()
        mole = get_pyscf_mole(elements, coords)
        e_g_partial = functools.partial(
            e_g_function_path,
            topologies=topologies,
            results=results,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
            e_shift=e_shift,
            lam=lam,
            path=path,
        )

        if solver == "pyberny":
            pyscf_solver = berny_solver
        elif solver == "geometric":
            pyscf_solver = geometric_solver
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(
            StringIO()
        ) as stderr:
            pyscf_solver.optimize(
                as_pyscf_method(mole, e_g_partial),
                maxsteps=maxsteps,
                callback=callback,
                **conv_params,
            )
        results.stdout = stdout.getvalue()
        results.stderr = stderr.getvalue()
        results_path.append(results)

    return results_path


def get_pyscf_mole(
    elements: Sequence[int] | Sequence[str],
    coordinates: ArrayLike2D,
) -> "Mole":
    """Return PySCF atom list."""
    elements = convert_elements(elements, output="symbols")
    coordinates = np.array(coordinates)
    atoms = []
    for element, coord in zip(elements, coordinates):
        atoms.append((element, tuple(coord)))

    numbers = convert_elements(elements, output="numbers")
    n_electrons = sum(numbers)
    mole = Mole(verbose=0, basis="def2svp")
    mole.spin = n_electrons % 2
    mole.atom = atoms
    mole.build()

    return mole


INCLUDE_GHOST: bool = getattr(
    __config__, "geomopt_berny_solver_optimize_include_ghost", True
)
ASSERT_CONV: bool = getattr(
    __config__, "geomopt_berny_solver_optimize_assert_convergence", True
)


def optimize_ci(  # noqa: C901
    methods: list[Any],
    assert_convergence: bool = ASSERT_CONV,
    include_ghost: bool = INCLUDE_GHOST,
    constraints: Any = None,
    callback: Any = None,
    maxsteps: int = 100,
    alpha: float = 0.025,
    sigma: float = 3.5,
    **kwargs,
) -> tuple[bool, "Mole"]:
    """Modified PySCF code to run geomeTRIC with CI optimization."""
    g_scanners = []
    for method in methods:
        if isinstance(method, lib.GradScanner):
            g_scanner = method
        elif isinstance(method, GradientsMixin):
            g_scanner = method.as_scanner()
        elif getattr(method, "nuc_grad_method", None):
            g_scanner = method.nuc_grad_method().as_scanner()
        else:
            raise NotImplementedError("Nuclear gradients of %s not available" % method)
        if not include_ghost:
            g_scanner.atmlst = np.where(method.mol.atom_charges() != 0)[0]
        g_scanners.append(g_scanner)

    tmpf = tempfile.mktemp(dir=lib.param.TMPDIR)
    engine_1 = geometric_solver.PySCFEngine(g_scanners[0])
    engine_2 = geometric_solver.PySCFEngine(g_scanners[1])
    M = engine_1.M
    meci_sigma = sigma
    meci_alpha = alpha
    engine_1.callback = callback
    engines_wrapper = EnginesWrapper([engine_1, engine_2])
    engine = ConicalIntersection(M, engines_wrapper, meci_sigma, meci_alpha)
    engine.maxsteps = maxsteps
    # To avoid overwritting method.mol
    engine.mol = g_scanners[0].mol.copy()

    # When symmetry is enabled, the molecule may be shifted or rotated to make
    # the z-axis be the main axis. The transformation can cause inconsistency
    # between the optimization steps. The transformation is muted by setting
    # an explict point group to the keyword mol.symmetry (see symmetry
    # detection code in Mole.build function).
    if engine.mol.symmetry:
        engine.mol.symmetry = engine.mol.topgroup

    # Provide config file log.ini for geomeTRIC optimisation
    kwargs["logIni"] = os.path.abspath(
        os.path.abspath(os.path.join(__file__, "..", "log.ini"))
    )

    engine.assert_convergence = assert_convergence
    try:
        geometric.optimize.run_optimizer(
            customengine=engine, input=tmpf, constraints=constraints, **kwargs
        )
        conv = True
        # method.mol.set_geom_(m.xyzs[-1], unit='Angstrom')
    except geometric_solver.NotConvergedError as e:
        lib.logger.note(method, str(e))
        conv = False

    # Remove the temporary files created by geomeTRIC
    if os.path.exists(f"{tmpf}_optim.xyz"):
        os.remove(f"{tmpf}_optim.xyz")
    if os.path.exists(f"{tmpf}.tmp"):
        shutil.rmtree(f"{tmpf}.tmp")

    return conv, engine_1.mol
