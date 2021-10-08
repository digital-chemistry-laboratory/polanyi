"""Code related to EVB."""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, overload, Tuple, Union

import numpy as np

from polanyi.typing import Array1D, Array3D, ArrayLike1D, ArrayLike2D


@overload
def evb_eigenvalues(
    energies: ArrayLike1D,
    *,
    coupling: Union[float, Mapping[Tuple[int, int], float]] = 0.0,
) -> Tuple[Array1D, Array1D]:
    ...


@overload
def evb_eigenvalues(
    energies: ArrayLike1D,
    *,
    gradients: Iterable[ArrayLike2D],
    coupling: Union[float, Mapping[Tuple[int, int], float]] = 0.0,
) -> Tuple[Array1D, Array3D, Array1D]:
    ...


@overload
def evb_eigenvalues(
    energies: ArrayLike1D,
    *,
    hessians: Iterable[ArrayLike2D],
    coupling: Union[float, Mapping[Tuple[int, int], float]] = 0.0,
) -> Tuple[Array1D, Array3D, Array1D]:
    ...


@overload
def evb_eigenvalues(
    energies: ArrayLike1D,
    *,
    gradients: Iterable[ArrayLike2D],
    hessians: Iterable[ArrayLike2D],
    coupling: Union[float, Mapping[Tuple[int, int], float]] = 0.0,
) -> Tuple[Array1D, Array3D, Array3D, Array1D]:
    ...


def evb_eigenvalues(
    energies: ArrayLike1D,
    *,
    gradients: Optional[Iterable[ArrayLike2D]] = None,
    hessians: Optional[Iterable[ArrayLike2D]] = None,
    coupling: Union[float, Mapping[Tuple[int, int], float]] = 0.0,
) -> Union[
    Tuple[Array1D, Array1D],
    Tuple[Array1D, Array3D, Array1D],
    Tuple[Array1D, Array3D, Array3D, Array1D],
]:
    """Returns EVB eigenvalues for energies and gradients.

    Args:
        energies: Energies of diabatic states (a.u.)
        gradients: Gradients of diabatic states (a.u.)
        hessians: Hessians of diabatic states (a.u.)
        coupling: Coupling term (a.u.)

    Returns:
        energies_ad: Energies of adiabatic states (a.u.)
        gradients_ad: Gradients of adiabatic states (a.u.), if gradients are provided
        hessians_as: Hessians of adiabatic states (a.u.), if hessians are provided
        indices: Diabatic state index most closely matching adiabatic sates

    Raises:
        ValueError: When energies and coupling are not of compatible shapes.
    """
    # Form Hamiltonian and diagonalize to get adiabatic energies
    h = np.diag(energies)
    if isinstance(coupling, Mapping):
        for key, value in coupling.items():
            h[key[0] - 1, key[1] - 1] = h[key[1] - 1, key[0] - 1] = value
    elif h.shape[0] == 2:
        for i, j in zip(*np.diag_indices_from(h)):
            if i > 0 and j > 0:
                h[i - 1, j] = h[i, j - 1] = coupling
    else:
        raise ValueError("Energies and coupling are not of compatible shapes.")
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    energies_ad = eigenvalues

    # Get diabatic state indices with highest weight for adiabatic states
    indices = np.argsort(np.abs(eigenvectors), axis=0)[:, ::-1][0]

    # Form matrix of gradients and diagonalize to get adiabatic gradients
    if gradients is not None:
        gradients = [np.asarray(gradient) for gradient in gradients]
        d_h = np.zeros(h.shape, dtype=object)
        for i, gradient in enumerate(gradients):
            d_h[i, i] = gradient
        G = eigenvectors.T @ d_h @ eigenvectors
        gradients_ad = G.diagonal()
        if hessians is None:
            return energies_ad, gradients_ad, indices

    # Form matrix of hessians and diagonalize to get adiabatic gradients
    if hessians is not None:
        hessians = [np.asarray(hessian) for hessian in hessians]
        dd_h = np.zeros(h.shape, dtype=object)
        for i, hessian in enumerate(hessians):
            dd_h[i, i] = hessian
        H = eigenvectors.T @ dd_h @ eigenvectors
        hessians_ad = H.diagonal()
        if gradients is not None:
            return energies_ad, gradients_ad, hessians_ad, indices
        else:
            return energies_ad, hessians_ad, indices

    return energies_ad, indices
