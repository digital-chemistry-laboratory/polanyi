"""Interface for driving optimizations through xtb."""
from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np

from polanyi.evb import evb_eigenvalues
from polanyi.io import get_xyz_string, read_coord, write_gradient
from polanyi.xtb import parse_engrad, run_xtb


def main() -> None:
    """Run EVB calculation and write files for xtb."""
    path = Path.cwd()

    # Read coordinates
    elements, coordinates = read_coord(path / "coord")

    # Read keywords and settings
    with open("keywords") as f:
        keywords_ = f.readlines()
    keywords = set(keywords_)
    keywords.update(["--gfnff", "--grad"])

    with open("coupling") as f:
        coupling = float(f.read())

    with open("e_shift") as f:
        e_shift = float(f.read())

    (path / "control").touch()

    # Run reactant FF
    path_r = path / "reactant_ff"
    run_xtb(
        elements,
        coordinates,
        path=path_r,
        keywords=keywords,
    )

    e_1, g_1 = parse_engrad(path_r / "xtb.engrad")

    # Run product FF
    path_p = path / "product_ff"
    run_xtb(
        elements,
        coordinates,
        path=path_p,
        keywords=keywords,
    )
    e_2, g_2 = parse_engrad(path_p / "xtb.engrad")

    e_2 += e_shift

    # Solve EVB
    energies_ad, gradients_ad, indices = evb_eigenvalues(
        [e_1, e_2], gradients=[g_1, g_2], coupling=coupling
    )

    logger.add(
        "polanyi.log",
        format="{message}",
        level="INFO",
        mode="a",
    )

    grad_rms = np.sqrt(np.mean(gradients_ad[1] ** 2))
    logger.info(
        f"Idx: {indices[1]} Energies: {energies_ad[0]:10.6f} {energies_ad[1]:10.6f} "
        f"Gradient RMS: {grad_rms:10.6f}"
    )

    write_gradient(
        path / "gradient", elements, coordinates, energies_ad[1], gradients_ad[1]
    )

    with open(path / "energies", "a") as f:
        f.write(str(energies_ad[1]) + "\n")
    with open(path / "gradients", "a") as f:
        f.write(str(grad_rms) + "\n")
    xyz_string = get_xyz_string(elements, coordinates, comment=str(energies_ad[1]))
    with open(path / "traj.xyz", "a") as f:
        f.write(xyz_string)


if __name__ == "__main__":
    main()
