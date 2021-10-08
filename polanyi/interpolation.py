"""Reaction path interpolation tools."""

from __future__ import annotations

from typing import Collection, Optional, Union

from geodesic_interpolate import Geodesic, redistribute
import numpy as np

from polanyi.typing import Array3D, ArrayLike3D
from polanyi.utils import convert_elements


def interpolate_geodesic(
    elements: Union[Collection[int], Collection[str]],
    coordinates: ArrayLike3D,
    n_images: int = 3,
    sweep: Optional[bool] = None,
    tol: float = 0.002,
    maxiter: int = 15,
    microiter: int = 20,
    scaling: float = 1.7,
    dist_cutoff: float = 3.0,
    friction: float = 0.02,
) -> Array3D:
    """Geodesic interpolation between geometries.

    Args:
        elements: Elements as symbols or numbers.
        coordinates: Coordinates for endpoints of interpolation
        n_images: Total number of structures in the interpolated path
        sweep: Whether to minimize path length by sweeping
        tol: Covergence tolerance for path minimization
        maxiter: Maximum number of iterations in path minimization
        microiter: Number of microiterations when sweeping
        scaling: Expontential parameter for Morse potential, or explicit scaling
            function
        dist_cutoff: Cut-off value for the distance between a pair of atoms to be
            included in the coordinate system
        friction: Size of friction term used to prevent very large change of geometry

    Returns:
        path: Coordinates of interpolated path
    """
    # Convert input to right format
    elements = convert_elements(elements, output="symbols")
    coordinates = np.asarray(coordinates)

    # Create initial path
    raw = redistribute(elements, coordinates, n_images, tol=5 * tol)

    # Smooth path
    smoother = Geodesic(
        elements, raw, scaling, threshold=dist_cutoff, friction=friction
    )
    if sweep is None:
        sweep = len(elements) > 35
    if sweep is True:
        smoother.sweep(tol=tol, max_iter=maxiter, micro_iter=microiter)
    else:
        smoother.smooth(tol=tol, max_iter=maxiter)

    path: Array3D = smoother.path

    return path
