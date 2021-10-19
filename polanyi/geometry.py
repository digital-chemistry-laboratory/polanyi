"""Geometry tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from polanyi.typing import Array1D, Array2D, ArrayLike1D, ArrayLike2D


def connectivity_from_bo(bo_matrix: Array2D, thres: float = 0.3) -> Array2D:
    """Returns connectivity matrix from bond order matrix.

    Args:
        bo_matrix: Bond order matrix
        thres: Threshold for bond detection

    Returns:
        connectivity_matrix: Connectivity matrixs
    """
    connectivity_matrix: np.ndarray = np.greater(bo_matrix, thres).astype(int)

    return connectivity_matrix


def combine_frags_distant(
    fragments: Iterable[tuple[Array1D, Array2D]],
    distances: Optional[Iterable[float]] = None,
    indices: Optional[Iterable[Array1D]] = None,
) -> tuple[Array1D, Array2D]:
    """Combine fragments at distance to one structure."""
    fragments = list(fragments)
    n_fragments = len(fragments)
    n_atoms_fragments = [len(fragment[0]) for fragment in fragments]
    n_atoms = sum(n_atoms_fragments)

    if distances is None:
        distances = [i * 50 for i in range(n_fragments)]

    if indices is None:
        indices = np.split(np.arange(n_atoms), np.cumsum(n_atoms_fragments))

    elements = np.empty(n_atoms, dtype=str)
    coordinates = np.zeros((n_atoms, 3))
    for fragment, distance, indices_frag in zip(fragments, distances, indices):
        frag_elements = fragment[0]
        frag_coordinates = fragment[1]
        frag_coordinates: Array2D = frag_coordinates + distance
        elements[indices_frag] = frag_elements
        coordinates[indices_frag] = frag_coordinates

    return elements, coordinates


def frags_from_indices(
    elements: ArrayLike1D,
    coordinates: ArrayLike2D,
    indices: Iterable[Array1D],
    infer_missing: bool = True,
) -> tuple[list[tuple[Array1D, Array2D]], list[Array1D]]:
    """Return fragments from indices."""
    elements = np.asarray(elements)
    coordinates = np.asarray(coordinates)
    fragments: list[tuple[Array1D, Array2D]] = []
    indices = list(indices)
    if infer_missing is True:
        indices_all = np.unique(np.concatenate(indices))
        indices_mol = np.arange(len(elements))
        indices_missing = np.setdiff1d(indices_mol, indices_all)
        indices.append(indices_missing)
    for indices_ in indices:
        fragments.append((elements[indices_], coordinates[indices_]))

    return fragments, indices


def two_frags_from_bo(
    bo_matrix: Array2D, start: float = 0.3, stop: float = 0.7, n_steps: int = 5
) -> tuple[Array1D, Array1D]:
    """Attempts to fragment molecule into two fragments and return indices.

    The bond order threshold will be scanned from start to stop in n_steps. If two
    separate fragments are obtained during this procedure, they will be returned and the
    scan stopped.

    Args:
        bo_matrix: Bond order matrix
        start: Starting threshold
        stop: Stopping threshold
        n_steps: Number of steps

    Returns:
        indices_1: Indices of fragment 1 (0-indexed)
        indices_2: Indices of fragment 2 (0-indexed)

    Raises:
        ValueError: When only one or more than two fragments found.
    """
    for wbo_thres in np.linspace(start, stop, n_steps):
        connectivity_matrix = connectivity_from_bo(bo_matrix, wbo_thres)
        graph = csr_matrix(connectivity_matrix)
        n_components, labels = connected_components(
            graph, directed=False, return_labels=True
        )
        if n_components == 2:
            break
        if n_components > 2:
            raise ValueError(f"{n_components} fragment(s) found.")
    if n_components == 1:
        raise ValueError(f"Only {n_components} fragment found.")

    indices_1 = np.where(labels == 0)[0]
    indices_2 = np.where(labels == 1)[0]

    return indices_1, indices_2
