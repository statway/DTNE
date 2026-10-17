"""Elastic Embedding implementation used as an optional DTNE solver."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy import linalg, spatial

from ._logging import log, stage


def calc_affinity(
    data,
    sigmas=1,
    zeroify_diag=True,
    normalize=True,
    metric="sqeuclidean",
):
    """Return pairwise distances and exponential affinities for EE coordinates.

    ``normalize`` is a compatibility argument and does not rescale the output.
    The default metric returns squared Euclidean distances.
    """
    del normalize  # Retained for compatibility with the historical signature.
    squared_distances = spatial.distance.cdist(data, data, metric=metric)
    affinities = np.exp(-squared_distances / sigmas)
    if zeroify_diag:
        np.fill_diagonal(squared_distances, 0)
        np.fill_diagonal(affinities, 0)
    return squared_distances, affinities


def calc_ee_loss(output_data, attractive_weights, repulsive_weights, lamb):
    """Return EE loss, squared coordinate distances, and repulsive affinities."""
    squared_distances, affinities = calc_affinity(
        output_data, zeroify_diag=False, normalize=False
    )
    loss = np.sum(
        attractive_weights * squared_distances
        + lamb * repulsive_weights * affinities
    )
    return loss, squared_distances, affinities


def get_laplacians(weights):
    """Return the unnormalized graph Laplacian for symmetric weights."""
    degree = np.diag(np.sum(weights, axis=0))
    return degree - weights


def _check_iteration_count(value, *, name):
    """Require a positive integer loop limit or coordinate dimension."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _check_ee_controls(lamb, step_size):
    """Validate finite repulsion strength and a strictly positive step size."""
    for name, value in (("lamb", lamb), ("step_size", step_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number.")
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    if lamb < 0 or step_size <= 0:
        raise ValueError("lamb must be non-negative and step_size must be positive.")


def ee_linear_search(
    output_data,
    attractive_weights,
    repulsive_weights,
    lamb,
    step_size,
    spectral_direction,
    gradients,
    loss,
    *,
    max_iter: int = 60,
):
    """Backtrack until the Armijo decrease condition holds or the search fails."""
    _check_iteration_count(max_iter, name="max_iter")
    _check_ee_controls(lamb, step_size)
    armijo = 0.1 * gradients.ravel().dot(spectral_direction.ravel())
    current_loss = np.inf
    affinities = None
    for _ in range(max_iter):
        candidate = output_data + step_size * spectral_direction
        current_loss, _, affinities = calc_ee_loss(
            candidate, attractive_weights, repulsive_weights, lamb
        )
        del _  # Pairwise coordinate distances are not needed by the line search.
        if current_loss <= loss + step_size * armijo:
            return candidate, current_loss, affinities, step_size
        step_size *= 0.8
        if step_size <= np.finfo(np.float64).eps:
            break
    raise FloatingPointError("Elastic Embedding line search failed to find a descent step.")


def elastic_embedding(
    attractive_weights,
    repulsive_weights,
    n_components=2,
    lamb=1,
    step_size=0.1,
    Y_init=None,
    num_iters=100,
    random_state=None,
    verbose=0,
):
    """Optimize elastic embedding coordinates from symmetric affinity weights.

    Parameters
    ----------
    attractive_weights, repulsive_weights : array-like, shape (N, N)
        Finite, non-negative, symmetric matrices.
    Y_init : array-like, shape (N, n_components), optional
        Initial coordinates, copied before optimization.
    lamb : float, default=1
        Non-negative strength of the repulsive loss and gradient.

    Returns
    -------
    numpy.ndarray
        Optimized coordinates with shape ``(N, n_components)``.
    """
    _check_iteration_count(num_iters, name="num_iters")
    _check_ee_controls(lamb, step_size)
    attractive_weights = np.asarray(attractive_weights, dtype=np.float64)
    repulsive_weights = np.asarray(repulsive_weights, dtype=np.float64)
    if attractive_weights.ndim != 2 or attractive_weights.shape[0] == 0 or attractive_weights.shape[0] != attractive_weights.shape[1]:
        raise ValueError("Weights must be non-empty square matrices.")
    if attractive_weights.shape != repulsive_weights.shape:
        raise ValueError("attractive_weights and repulsive_weights must share a shape.")
    for name, weights in (("attractive_weights", attractive_weights), ("repulsive_weights", repulsive_weights)):
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"{name} must contain finite, non-negative values.")
        if not np.allclose(weights, weights.T, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{name} must be symmetric.")
    _check_iteration_count(n_components, name="n_components")
    rng = np.random.default_rng(random_state)
    embedding = (
        np.asarray(Y_init, dtype=np.float64).copy()
        if Y_init is not None
        else 1e-5 * rng.standard_normal((attractive_weights.shape[0], n_components))
    )
    if embedding.shape != (attractive_weights.shape[0], n_components):
        raise ValueError("Y_init must have shape (n_samples, n_components).")
    if not np.isfinite(embedding).all():
        raise ValueError("Y_init must contain finite values.")

    with stage(verbose, "EE optimization", cells=embedding.shape[0], components=n_components):
        loss, _, affinities = calc_ee_loss(
            embedding, attractive_weights, repulsive_weights, lamb
        )
        del _
        attractive_laplacian = get_laplacians(attractive_weights)
        regularization = 1e-10 * np.eye(attractive_laplacian.shape[0])
        # Regularization makes the Laplacian's translational null mode invertible.
        upper = np.linalg.cholesky(attractive_laplacian + regularization).T
        del regularization

        for iteration in range(num_iters):
            repulsive_laplacian = get_laplacians(repulsive_weights * affinities)
            del affinities
            gradients = 4 * (
                attractive_laplacian - lamb * repulsive_laplacian
            ).dot(embedding)
            del repulsive_laplacian
            # Reuse the Cholesky factors with triangular solves instead of
            # refactorizing them through a general solver on every iteration.
            direction = -linalg.solve_triangular(
                upper, linalg.solve_triangular(upper.T, gradients, lower=True),
                lower=False,
            )
            embedding, loss, affinities, step_size = ee_linear_search(
                embedding,
                attractive_weights,
                repulsive_weights,
                lamb,
                step_size,
                direction,
                gradients,
                loss,
            )
            step_norm = linalg.norm(step_size * direction, "fro")
            if verbose >= 2:
                log(verbose, f"EE iteration {iteration + 1}: loss={loss:.6g}, step={step_size:.4g}, step_norm={step_norm:.4g}", level=2)
            if step_norm < 1e-2:
                break
        log(verbose, f"EE optimization summary: iterations={iteration + 1}, step_norm={step_norm:.4g}")
    return embedding


__all__ = [
    "calc_affinity",
    "calc_ee_loss",
    "ee_linear_search",
    "elastic_embedding",
    "get_laplacians",
]
