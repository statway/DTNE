"""Bhattacharyya similarities and DTNE manifold distances."""

from __future__ import annotations

from numbers import Real

import numpy as np


def _probability_rows(values, *, name):
    """Validate probability rows and remove small row-sum roundoff errors."""
    rank = np.asarray(values, dtype=np.float64)
    if rank.ndim != 2 or not all(rank.shape):
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if np.any(rank < 0) or not np.isfinite(rank).all():
        raise ValueError(f"{name} must contain finite, non-negative values.")
    row_sums = rank.sum(axis=1, keepdims=True)
    if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"{name} rows must sum to 1 (probability distributions).")
    return rank / row_sums


def _distance_epsilon(epsilon):
    """Choose a positive similarity floor to keep logarithmic distances finite."""
    if epsilon is None:
        return np.finfo(np.float64).tiny
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise TypeError("epsilon must be a finite number between 0 and 1.")
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must be a finite number between 0 and 1.")
    return epsilon


def manifold_distances(rank_matrix, *, epsilon: float | None = None, distance_only=False):
    """Return similarity, squared-distance, and distance matrices.

    The paper defines ``G = sqrt(R) @ sqrt(R).T`` and
    ``D^2 = -2 log(G)``. Both distance forms are retained explicitly so
    downstream algorithms cannot accidentally mix them.
    Input rows must sum to one; only small numerical drift is normalized.
    ``distance_only=True`` converts the similarity buffer in place and returns
    just ordinary distances, avoiding two additional square matrices.
    """
    rank = _probability_rows(rank_matrix, name="rank_matrix")
    epsilon = _distance_epsilon(epsilon)
    # Validation returns an owned buffer, so its square root can reuse storage.
    np.sqrt(rank, out=rank)
    root = rank
    del rank
    similarities = root @ root.T
    del root
    # A probability distribution has unit self-similarity; remove roundoff drift.
    np.fill_diagonal(similarities, 1.0)
    np.clip(similarities, epsilon, 1.0, out=similarities)
    if distance_only:
        np.log(similarities, out=similarities)
        similarities *= -2.0
        np.maximum(similarities, 0.0, out=similarities)
        np.fill_diagonal(similarities, 0.0)
        np.sqrt(similarities, out=similarities)
        return similarities
    squared = np.maximum(-2.0 * np.log(similarities), 0.0)
    np.fill_diagonal(squared, 0.0)
    distances = np.sqrt(squared)
    return similarities, squared, distances


def cross_manifold_distances(left_rank, right_rank, *, epsilon: float | None = None):
    """Compute distances between probability rows, normalizing only roundoff drift."""
    left = _probability_rows(left_rank, name="left_rank")
    right = _probability_rows(right_rank, name="right_rank")
    epsilon = _distance_epsilon(epsilon)
    if left.shape[1] != right.shape[1]:
        raise ValueError("left_rank and right_rank must share their feature dimension.")
    np.sqrt(left, out=left)
    np.sqrt(right, out=right)
    similarities = left @ right.T
    del left, right
    np.clip(similarities, epsilon, 1.0, out=similarities)
    np.log(similarities, out=similarities)
    similarities *= -2.0
    np.maximum(similarities, 0.0, out=similarities)
    np.sqrt(similarities, out=similarities)
    return similarities


__all__ = ["cross_manifold_distances", "manifold_distances"]
