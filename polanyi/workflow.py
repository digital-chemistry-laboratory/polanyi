"""Workflows."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from scipy.optimize import minimize
import shutil
from tempfile import TemporaryDirectory
import textwrap
from typing import Mapping

from morfeus.conformer import ConformerEnsemble
import numpy as np
import os

from polanyi import config
from polanyi.evb import evb_eigenvalues
from polanyi.geometry import two_frags_from_bo
from polanyi.interpolation import interpolate_geodesic
from polanyi.pyscf import (
    OptResults,
    rxn_path_from_gfnff,
    ts_from_gfnff,
    ts_from_gfnff_ci,
)
from polanyi.typing import Array1D, Array2D, ArrayLike2D
from polanyi.xtb import (
    opt_crest,
    opt_xtb,
    parse_energy,
    parse_energy_gxtb,
    parse_solv_energy,
    run_gxtb,
    run_xtb,
    wbo_xtb,
)
from polanyi.io import get_xyz_string
from polanyi.utils import is_min_xtb_version


@dataclass
class Results:
    """Results of TS optimization."""

    coordinates_opt: Array2D
    energy_opt: float
    opt_results: OptResults | None = None
    shift_results: ShiftResults | None = None


@dataclass
class ShiftResults:
    """Results of energy shift calculation."""

    energy_shift: float
    energy_diff_gfn: float
    energy_diff_ff: float
    energies_gfn: list[float]
    energies_ff: list[float]


def opt_ts_ci(
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[Array2D],
    coordinates_guess: Array2D | None = None,
    e_shift: float | None = None,
    kw_topo: Mapping | None = None,
    kw_shift: Mapping | None = None,
    kw_opt: Mapping | None = None,
    kw_interpolation: Mapping | None = None,
) -> Results:
    """Optimize transition state with xtb and PySCF using conical intersection.
    Args:
        elements: elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground states [Å]
        coordinates_guess: initial guess for the transition state [Å]
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        kw_topo: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_opt: parameters for optimization
        kw_interpolation: parameters for the TS interpolation
    Returns:
        results: coordinates [Å] and energies [Eh] of the TS optimization
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_topo is None:
        kw_topo = {}
    if kw_interpolation is None:
        kw_interpolation = {}

    topologies = setup_gfnff_topologies(elements, coordinates, **kw_topo)

    shift_results: tuple[float, float, float] | None
    if e_shift is None:
        shift_results = calculate_e_shift_xtb(
            elements,
            coordinates,
            topologies,
            **kw_shift,
        )
        e_shift = shift_results[0]
    else:
        shift_results = None

    if coordinates_guess is None:
        n_images = kw_interpolation.get("n_images")
        if n_images is None:
            n_images = signature(interpolate_geodesic).parameters["n_images"].default
        rxn_path = interpolate_geodesic(elements, coordinates, **kw_interpolation)
        coordinates_guess = rxn_path[n_images // 2]

    # Save the optimisation steps if path for optimisation is given
    if "path" in kw_opt and kw_opt["path"] is not None:
        run_path = Path(kw_opt["path"])
        xyz_file = run_path / "opt_steps.xyz"
        os.makedirs(run_path, exist_ok=True)
        step_counter = [0]  # Using list for mutable counter in callback

        def get_opt_steps_from_ci(envs):
            pyscf_mol = envs["g_scanner"].mol
            pyscf_elements = pyscf_mol.atom_charges()
            pyscf_coordinates = np.array(pyscf_mol.atom_coords(unit="ANG"), order="C")
            xyz_string = get_xyz_string(
                pyscf_elements,
                pyscf_coordinates,
                comment=f"step {step_counter[0]}",
                decimals=10,
            )
            with open(xyz_file, "a") as f:
                f.write(xyz_string)
            step_counter[0] += 1

        kw_opt["callback"] = get_opt_steps_from_ci

    coordinates_opt, energy_opt = ts_from_gfnff_ci(
        elements, coordinates_guess, topologies, e_shift=e_shift, **kw_opt
    )

    results = Results(
        coordinates_opt=coordinates_opt,
        energy_opt=energy_opt,
        shift_results=shift_results,
    )

    return results


def opt_ts(  # noqa: C901
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[Array2D],
    coordinates_guess: Array2D | None = None,
    e_shift: float | None = None,
    fit_coupling: bool = False,
    kw_topo: Mapping | None = None,
    kw_shift: Mapping | None = None,
    kw_interpolation: Mapping | None = None,
    kw_coupling: Mapping | None = None,
    kw_opt: Mapping | None = None,
) -> Results:
    """Optimize transition state with xtb and PySCF.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground state [Å]
        coordinates_guess: initial guess for the transition state [Å]
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        fit_coupling: whether to optimise EVB coupling constant
        kw_topo: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_interpolation: parameters for the TS interpolation
        kw_coupling: parameters for coupling constant fitting
        kw_opt: parameters for optimization
    Returns:
        results: coordinates [Å] and energies [Eh] of the TS optimization
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_topo is None:
        kw_topo = {}
    if kw_interpolation is None:
        kw_interpolation = {}
    if kw_coupling is None:
        kw_coupling = {}
    topologies = setup_gfnff_topologies(elements, coordinates, **kw_topo)
    shift_results: tuple[float, float, float] | None
    if e_shift is None:
        shift_results = calculate_e_shift_xtb(
            elements, coordinates, topologies, **kw_shift
        )
        e_shift = shift_results[0]
    else:
        shift_results = None
    if coordinates_guess is None or fit_coupling:
        n_images = kw_interpolation.get("n_images")
        if n_images is None:
            if fit_coupling:
                n_images = signature(fit_coupling_const).parameters["n_images"].default
            else:
                n_images = (
                    signature(interpolate_geodesic).parameters["n_images"].default
                )
            kw_interpolation["n_images"] = n_images
        rxn_path = interpolate_geodesic(elements, coordinates, **kw_interpolation)
        coordinates_guess = rxn_path[n_images // 2]

    if fit_coupling:
        coupling = fit_coupling_const(
            elements,
            coordinates,
            topologies,
            e_shift=e_shift,
            rxn_path=rxn_path,
            **kw_coupling,
        )
        kw_opt["coupling"] = coupling

    opt_results = ts_from_gfnff(
        elements, coordinates_guess, topologies, e_shift=e_shift, **kw_opt
    )

    # Save the optimisation steps if path for optimisation is given
    if "path" in kw_opt and kw_opt["path"] is not None:
        run_path = Path(kw_opt["path"])
        os.makedirs(run_path, exist_ok=True)
        xyz_file = run_path / "opt_steps.xyz"
        xyz_guess = get_xyz_string(
            elements, coordinates_guess, comment="initial guess", decimals=10
        )
        with open(xyz_file, "w") as f:
            f.write(xyz_guess)
        for i, coord in enumerate(opt_results.coordinates, start=1):
            xyz_step = get_xyz_string(elements, coord, comment=f"step {i}", decimals=10)
            with open(xyz_file, "a") as f:
                f.write(xyz_step)

    results = Results(
        coordinates_opt=opt_results.coordinates[-1],
        energy_opt=opt_results.energies_diabatic[-1][0],
        opt_results=opt_results,
        shift_results=shift_results,
    )

    return results


def setup_gfnff_topologies(  # noqa: C901
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[ArrayLike2D],
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    fragment_charges: Sequence[list[int] | None] | None = None,
    adjacency_matrices: Sequence[Array2D] | None = None,
    paths: Sequence[str | Path] | None = None,
) -> list[bytes]:
    """Set up topologies for GFN-FF calculation.
    Args:
        elements: elements as symbols or numbers
        coordinates: coordinates of each ground state [Å]
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        fragment_charges: charge of each non-covalently bound fragment, for each ground state
        adjacency_matrices: connectivity matrices of each ground state
        paths: folders to save the xtb runs
    Returns:
        topology of each ground state
    """

    # Set the xtb keywords for the GFN-FF calculations
    if keywords is None:
        keywords = []
    keywords = set([keyword.strip().lower() for keyword in keywords])
    # Give the --gfnff keyword (--gfn2 by default) if not already present
    if "--gfnff" not in keywords:
        keywords.add("--gfnff")

    if paths is None:
        temp_dirs = [
            TemporaryDirectory(dir=config.TMP_DIR) for i in range(len(coordinates))
        ]
        xtb_paths = [Path(temp_dir.name) for temp_dir in temp_dirs]
    else:
        if (Path(paths[0]) / "gfnff_topo").exists() or (
            Path(paths[1]) / "gfnff_topo"
        ).exists():
            raise FileExistsError(
                f"Paths {paths[0]} or {paths[1]} already contain 'gfnff_topo' files. Remove before new xtb calculations."
                f"\nIf other files are present, they will be overwritten."
            )
        xtb_paths = [Path(path) for path in paths]

    topologies = []
    for i, (coordinates_, xtb_path) in enumerate(zip(coordinates, xtb_paths)):

        # Write topology in xtb xcontrol file if adjacency matrices are given
        if adjacency_matrices is not None:

            # TODO: Update this requirement when xtb >6.7.1 is released
            if not is_min_xtb_version("bleed"):
                raise RuntimeError(
                    "Use bleeding edge version of xtb to give adjacency matrices as input."
                )

            ffnb_lines = []
            for atom_idx, row in enumerate(adjacency_matrices[i], 1):
                neighbours = (
                    np.nonzero(row)[0] + 1
                )  # Atoms must be 1-indexed in xcontrol file
                if len(neighbours) > 0:
                    ffnb_lines.append(
                        f"nb = {atom_idx}: {', '.join(map(str, neighbours))}"
                    )
                else:
                    ffnb_lines.append(f"nb = {atom_idx}: 0")
            if xcontrol_keywords is not None:
                xcontrol_keywords = {**xcontrol_keywords, "ffnb": ffnb_lines}
            else:
                xcontrol_keywords = {"ffnb": ffnb_lines}

        run_xtb(
            elements,
            coordinates_,
            path=xtb_path,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
            fragment_charges=fragment_charges[i] if fragment_charges else None,
        )

        with open(xtb_path / "gfnff_topo", "rb") as f:
            topology = f.read()
        topologies.append(topology)

    if paths is None:
        for temp_dir in temp_dirs:
            temp_dir.cleanup()

    return topologies


def opt_frags_from_complex(
    elements: Sequence[int] | Sequence[str],
    coordinates: ArrayLike2D,
    keywords: list[str] | None = None,
    wbo_keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
) -> list[tuple[Array1D, Array2D]]:
    """Optimize two fragments from complex.
    Args:
        elements: elements as symbols or numbers
        coordinates: coordinates [Å]
        keywords: xtb command line keywords for optimization
        wbo_keywords: xtb command line keywords for wbo calculation
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
    Returns:
        fragments: Fragment elements and coordinates
    """
    elements = np.array(elements)
    coordinates = np.asarray(coordinates)
    bo_matrix = wbo_xtb(elements, coordinates, keywords=wbo_keywords)
    frag_indices = two_frags_from_bo(bo_matrix)
    fragments = []
    for indices in frag_indices:
        frag_elements = elements[indices]
        frag_coordinates = coordinates[indices]
        opt_coordinates, _ = opt_xtb(
            frag_elements,
            frag_coordinates,
            keywords=keywords,
            xcontrol_keywords=xcontrol_keywords,
        )
        fragments.append((frag_elements, opt_coordinates))

    return fragments


def opt_constrained_complex(  # noqa: C901
    elements: Sequence[int] | Sequence[str],
    coordinates: ArrayLike2D,
    distance_constraints: MutableMapping[tuple[int, int], float] | None = None,
    atom_constraints: Sequence[int] | None = None,
    fix_atoms: Sequence[int] | None = None,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    fragment_charges: list[int] | None = None,
    fc: float | None = None,
    path: str | Path | None = None,
) -> Results:
    """Optimise constrained complex with xtb
    Args:
        elements: elements as symbols or numbers
        coordinates: coordinates [Å]
        distance_constraints: distances to constrain
        atom_constraints: atoms to constrain
        fix_atoms: atoms to fix
        keywords: xtb command line keywords
        xcontrol_keywords: input instructions to write in the xtb xcontrol file
        fragment_charges: charge of each non-covalently bound (NCI) fragment
        fc: force constant for constraints
        path: path to run the xtb optimisation
    Returns:
        results: coordinates [Å] and energies [Eh] of the TS optimization
    """
    rmsd_atoms = set(range(1, len(elements) + 1))
    if distance_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_constraints = xcontrol_keywords.setdefault("constrain", [])
        if fc is not None:
            xcontrol_constraints.append(f"force constant={fc}")
        for (i, j), distance in distance_constraints.items():
            string = f"distance: {i}, {j}, {distance}"
            xcontrol_constraints.append(string)
            rmsd_atoms.difference_update({i, j})
        xcontrol_keywords["constrain"] = xcontrol_constraints
    if atom_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_atom_constraints = xcontrol_keywords.setdefault("constrain", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, atom_constraints)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_atom_constraints.append(fix_string)
        rmsd_atoms.difference_update(atom_constraints)
    if fix_atoms is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_fix_atoms = xcontrol_keywords.setdefault("fix", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, fix_atoms)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_fix_atoms.append(fix_string)
        rmsd_atoms.difference_update(fix_atoms)

    opt_coordinates, opt_energy = opt_xtb(
        elements,
        coordinates,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        fragment_charges=fragment_charges,
        path=path,
    )

    results = Results(
        coordinates_opt=opt_coordinates,
        energy_opt=opt_energy,
    )

    return results


def crest_constrained(  # noqa: C901
    elements: Sequence[int] | Sequence[str],
    coordinates: ArrayLike2D,
    distance_constraints: MutableMapping[tuple[int, int], float] | None = None,
    atom_constraints: Sequence[int] | None = None,
    fix_atoms: Sequence[int] | None = None,
    keywords: list[str] | None = None,
    xcontrol_keywords: MutableMapping[str, list[str]] | None = None,
    fc: float | None = None,
    path: str | Path | None = None,
) -> ConformerEnsemble:
    """Run constrained CREST calculation."""
    rmsd_atoms = set(range(1, len(elements) + 1))
    if distance_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_constraints = xcontrol_keywords.setdefault("constrain", [])
        if fc is not None:
            xcontrol_constraints.append(f"force constant={fc}")
        for (i, j), distance in distance_constraints.items():
            string = f"distance: {i}, {j}, {distance}"
            xcontrol_constraints.append(string)
            rmsd_atoms.difference_update({i, j})
        xcontrol_keywords["constrain"] = xcontrol_constraints
    if atom_constraints is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_atom_constraints = xcontrol_keywords.setdefault("constrain", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, atom_constraints)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_atom_constraints.append(fix_string)
        rmsd_atoms.difference_update(atom_constraints)
    if fix_atoms is not None:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_fix_atoms = xcontrol_keywords.setdefault("fix", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, fix_atoms)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_fix_atoms.append(fix_string)
        rmsd_atoms.difference_update(fix_atoms)
    if len(rmsd_atoms) > 0:
        if xcontrol_keywords is None:
            xcontrol_keywords = {}
        xcontrol_rmsd_atoms = xcontrol_keywords.setdefault("metadyn", [])
        atom_lines = textwrap.wrap(
            ", ".join(map(str, rmsd_atoms)), break_long_words=False
        )
        for line in atom_lines:
            fix_string = "atoms: " + line
            xcontrol_rmsd_atoms.append(fix_string)

    conformer_ensemble = opt_crest(
        elements,
        coordinates,
        keywords=keywords,
        xcontrol_keywords=xcontrol_keywords,
        path=path,
    )

    return conformer_ensemble


def calculate_e_shift_xtb(  # noqa: C901
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[ArrayLike2D],
    topologies: Sequence[bytes],
    e_diff_ref: float | None = None,
    keywords_ff: list[str] | None = None,
    keywords_sp: list[str] | None = None,
    xcontrol_keywords_ff: MutableMapping[str, list[str]] | None = None,
    xcontrol_keywords_sp: MutableMapping[str, list[str]] | None = None,
    paths: Sequence[str | Path] | None = None,
) -> tuple[float, float, float]:
    """Calculate energy shift between reference (default: GFN2-xTB) and GFN-FF reaction energies.
    Args:
        elements: elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground state [Å]
        topologies: sequence of GFN-FF topologies for each ground state
        e_diff_ref: reference reaction energy [Eh]. If provided, it is used instead of the GFN2-xTB calculation
        keywords_ff: xtb command line keywords for GFN-FF calculation
        keywords_sp: xtb command line keywords for GFN2-xTB calculation
        xcontrol_keywords_ff: input instructions to write in the xtb xcontrol file for GFN-FF calculation
        xcontrol_keywords_sp: input instructions to write in the xtb xcontrol file for GFN2-xTB calculation
        paths: list of folders to save the xtb runs
    Returns:
        e_shift: difference between the GFN2-xTB and GFN-FF reaction energies
        e_diff_ref: reaction energy calculated with GFN2-xTB or given as argument
        e_diff_ff: reaction energy calculated with GFN-FF
    """

    # Set the xtb keywords for the GFN-FF calculations
    if keywords_ff is None:
        keywords_ff = []
    keywords_ff = set(keyword.strip().lower() for keyword in keywords_ff)
    # Give the --gfnff keyword (--gfn2 by default) if not already present
    if "--gfnff" not in keywords_ff:
        keywords_ff.add("--gfnff")

    if paths is None:
        temp_dirs = [
            TemporaryDirectory(dir=config.TMP_DIR) for i in range(len(coordinates))
        ]
        xtb_paths = [Path(temp_dir.name) for temp_dir in temp_dirs]
    else:
        xtb_paths = [Path(path) for path in paths]

    energies_ff = []
    if not e_diff_ref:
        energies_sp = []
    for coordinates_, topology, xtb_path in zip(coordinates, topologies, xtb_paths):
        xtb_path.mkdir(exist_ok=True)
        if (xtb_path / "gfnff_topo").exists():
            raise FileExistsError(
                f"Path {xtb_path} already contains a 'gfnff_topo' file. Remove before new xtb calculations."
                f"\nIf other files are present, they will be overwritten."
            )
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

        if not e_diff_ref:
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

    if not e_diff_ref:
        e_diff_ref = energies_sp[-1] - energies_sp[0]
    e_diff_ff = energies_ff[-1] - energies_ff[0]
    e_shift = e_diff_ref - e_diff_ff

    return e_shift, e_diff_ref, e_diff_ff


def fit_coupling_const(  # noqa: C901
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[Array2D],
    topologies: Sequence[bytes],
    e_shift: float,
    rxn_path: list[Array2D] | None = None,
    n_images: int = 9,
    keywords_ff: list[str] | None = None,
    keywords_sp: list[str] | None = None,
    xcontrol_keywords_ff: MutableMapping[str, list[str]] | None = None,
    xcontrol_keywords_sp: MutableMapping[str, list[str]] | None = None,
    path: str | Path | None = None,
) -> float:
    """Optimise EVB coupling term to fit GFN2-xTB reaction path.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground state [Å]
        topologies: sequence of GFN-FF topologies for each ground state
        rxn_path: coordinates [Å] along the reaction path
        n_images: number of structures to generate on the reaction path
        e_shift: energy shift between GFN2-xTB and GFN-FF reaction energies
        keywords_ff: xtb command line keywords for GFN-FF calculation
        keywords_sp: xtb command line keywords for GFN2-xTB calculation
        xcontrol_keywords_ff: input instructions to write in the xtb xcontrol file for GFN-FF calculation
        xcontrol_keywords_sp: input instructions to write in the xtb xcontrol file for GFN2-xTB calculation
        path: folder to save the xtb runs
    Returns:
        EVB coupling constant
    """
    if path is None:
        folder = TemporaryDirectory(dir=config.TMP_DIR)
        xtb_path = Path(folder.name)
    else:
        xtb_path = Path(path)
        if xtb_path.exists():
            shutil.rmtree(xtb_path)
        xtb_path.mkdir(parents=True, exist_ok=True)
    if keywords_ff is None:
        keywords_ff = []
    keywords_ff = set(keyword.strip().lower() for keyword in keywords_ff)
    if "--gfnff" not in keywords_ff:
        keywords_ff.add("--gfnff")

    if rxn_path is None:
        rxn_path = interpolate_geodesic(elements, coordinates, n_images=n_images)

    ref_energies = []
    gfnff_energies = []
    for i, coords in enumerate(rxn_path):
        run_path = xtb_path / "gfn2" / f"image{i+1}"
        run_xtb(
            elements,
            coords,
            path=run_path,
            keywords=keywords_sp,
            xcontrol_keywords=xcontrol_keywords_sp,
        )
        e_solv = parse_solv_energy(run_path / "xtb.out")
        run_path = xtb_path / "gxtb" / f"image{i+1}"
        if keywords_sp is not None:
            for keyword in keywords_sp:
                if keyword.startswith(("--chrg", "-c")):
                    charge = int(keyword.split()[-1])
        run_gxtb(
            elements,
            coords,
            path=run_path,
            charge=charge,
        )
        ref_energies.append(e_solv + parse_energy_gxtb(run_path / "energy"))

        energies = []
        for j, topo in enumerate(topologies):
            run_path = xtb_path / "gfnff" / f"image{i+1}" / f"{j}"
            run_path.mkdir(parents=True, exist_ok=True)
            with open(run_path / "gfnff_topo", "wb") as f:
                f.write(topo)
            run_xtb(
                elements,
                coords,
                path=run_path,
                keywords=keywords_ff,
                xcontrol_keywords=xcontrol_keywords_ff,
            )
            energies.append(parse_energy(run_path / "xtb.out"))
        energies[-1] += e_shift
        gfnff_energies.append(energies)

    # Optimise coupling constant to minimise distance between EVB and GFN2 energies
    def objective(
        coupling: float, energies_ff: Array2D, energies_ref: Array1D
    ) -> float:
        evb_min_energies = []
        for e_ff in energies_ff:
            # Solve EVB
            energies_ad, _ = evb_eigenvalues([e_ff[0], e_ff[1]], coupling=coupling)
            evb_min_energies.append(energies_ad[0])
        # Normalise
        energies_ref_norm = np.array(energies_ref) - energies_ref[0]
        energies_evb_norm = np.array(evb_min_energies) - evb_min_energies[0]
        # Calculate MSE
        residuals = energies_evb_norm - energies_ref_norm
        return float(np.mean(np.array(residuals) ** 2))

    res = minimize(
        fun=lambda x: objective(x[0], gfnff_energies, ref_energies),
        x0=np.array([1e-3], float),
        bounds=[(0.0, None)],
    )
    opt_coupling = float(res.x[0])

    return opt_coupling


def interpolate_rxn_path(
    elements: Sequence[int] | Sequence[str],
    coordinates: Sequence[Array2D],
    n_images: int = 10,
    e_shift: float | None = None,
    kw_topo: Mapping | None = None,
    kw_shift: Mapping | None = None,
    kw_opt: Mapping | None = None,
    kw_interpolation: Mapping | None = None,
) -> list[Array2D]:
    """Interpolate reaction path between geometries based on EVB.
    Args:
        elements: TS elements as symbols or numbers
        coordinates: sequence containing the coordinates of each ground state [Å]
        n_images: number of structures to generate on the reaction path
        e_shift: energy shift between reference (GFN2-xTB by default) and GFN-FF reaction energies
        kw_topo: parameters for topologies calculation
        kw_shift: parameters for energy shift calculation
        kw_opt: parameters for optimization
        kw_interpolation: parameters for the geodesicinterpolation
    Returns:
        coordinates [Å] along the reaction path
    """
    if kw_opt is None:
        kw_opt = {}
    if kw_shift is None:
        kw_shift = {}
    if kw_topo is None:
        kw_topo = {}
    if kw_interpolation is None:
        kw_interpolation = {}
    topologies = setup_gfnff_topologies(elements, coordinates, **kw_topo)
    shift_results: tuple[float, float, float] | None
    if e_shift is None:
        shift_results = calculate_e_shift_xtb(
            elements, coordinates, topologies, **kw_shift
        )
        e_shift = shift_results[0]
    else:
        shift_results = None

    if kw_interpolation.get("n_images") is not None:
        if kw_interpolation.get("n_images") != n_images:
            raise ValueError(
                "Given number of images must be the same for this function and the geodesic interpolation."
            )
    else:
        kw_interpolation["n_images"] = n_images
    guess_rxn_path = interpolate_geodesic(elements, coordinates, **kw_interpolation)

    results_path = rxn_path_from_gfnff(
        elements,
        guess_rxn_path,
        topologies,
        e_shift=e_shift,
        **kw_opt,
    )
    evb_rxn_path = [results_path[i].coordinates[-1] for i in range(len(results_path))]

    return evb_rxn_path
