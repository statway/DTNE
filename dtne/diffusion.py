"""Diffusion operators and adaptive personalized PageRank for DTNE."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg, sparse, stats
from sklearn import preprocessing
from ._validation import check_positive_real
from ._logging import log, stage


def eigen_kernel(kernel):
    """Factorize the row Markov operator through a symmetric similar matrix.

    Notes
    -----
    For symmetric ``K`` and its degree matrix ``D``, diagonalize
    ``D**(-1/2) @ K @ D**(-1/2)`` to recover ``P = D**(-1) @ K``.
    Returned factors satisfy ``P = phi @ diag(eigenvalues) @ psi`` and
    ``psi @ phi = I``. Eigenvalues retain their signs for matrix powers.
    """
    matrix = kernel.toarray() if sparse.issparse(kernel) else np.asarray(kernel, dtype=np.float64)
    degree = np.asarray(matrix.sum(axis=0)).reshape(-1)
    if np.any(degree <= 0) or not np.isfinite(degree).all():
        raise ValueError("Kernel degrees must be positive and finite.")
    inv_sqrt = 1.0 / np.sqrt(degree)
    normalized = inv_sqrt[:, None] * matrix * inv_sqrt[None, :]
    del matrix
    eigenvalues, eigenvectors = linalg.eigh(normalized, check_finite=True)
    del normalized
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # PageRank requires signed eigenvalues and mutually inverse factors.
    phi = inv_sqrt[:, None] * eigenvectors
    psi = eigenvectors.T * np.sqrt(degree)[None, :]
    return phi, eigenvalues, psi


def eigen_kernel2(matrix, *, degree=None):
    """Factorize a landmark Markov operator in its original coordinates.

    With landmark degrees, use the symmetric similarity transform
    ``S = sqrt(D) @ Pmm @ invsqrt(D)``. Without degrees, retain the
    general eigendecomposition for compatibility with standalone callers.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if degree is not None:
        degree = np.asarray(degree, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape != (degree.size, degree.size) or degree.ndim != 1:
            raise ValueError("Landmark degrees must match the square operator shape.")
        if not np.isfinite(degree).all() or np.any(degree <= 0):
            raise ValueError("Landmark degrees must be positive and finite.")
        sqrt_degree = np.sqrt(degree)
        symmetric = sqrt_degree[:, None] * matrix / sqrt_degree[None, :]
        if not np.allclose(symmetric, symmetric.T, rtol=1e-12, atol=1e-14):
            raise ValueError("Landmark degrees must make the operator reversible.")
        eigenvalues, eigenvectors = linalg.eigh(symmetric, check_finite=True)
        del symmetric
        order = np.abs(eigenvalues).argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        # Undo the similarity transform, keeping signed eigenvalues and
        # mutually inverse factors for powers and PageRank of the original Pmm.
        phi = eigenvectors / sqrt_degree[:, None]
        psi = eigenvectors.T * sqrt_degree[None, :]
        return phi, eigenvalues, psi
    eigenvalues, phi = linalg.eig(matrix, check_finite=True)
    phi = np.real_if_close(phi, tol=1000)
    eigenvalues = np.real_if_close(eigenvalues, tol=1000)
    if np.iscomplexobj(phi) or np.iscomplexobj(eigenvalues):
        raise ValueError("Landmark diffusion operator has significant complex eigenpairs.")
    order = np.abs(eigenvalues).argsort()[::-1]
    # Magnitude controls ordering only; it must not replace signed eigenvalues.
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    phi = np.asarray(phi[:, order], dtype=np.float64)
    psi = linalg.inv(phi, check_finite=True)
    return (
        phi,
        eigenvalues,
        psi,
    )


def _landmark_operator_from_mapping(K, point_to_landmark):
    """Rebuild Pmm from an existing Pnm without rebuilding the sparse mapping.

    For point degrees Dn and landmark degrees Dm,
    ``Pmm = inv(Dm) @ Pnm.T @ Dn @ Pnm``. Only Pmm is dense.
    """
    kernel = sparse.csr_matrix(K)
    point_degree = np.asarray(kernel.sum(axis=0)).reshape(-1)
    weighted_mapping = point_to_landmark.multiply(point_degree[:, None]).tocsr()
    degree = np.asarray(weighted_mapping.sum(axis=0)).reshape(-1)
    if not np.isfinite(degree).all() or np.any(degree <= 0):
        raise ValueError("Landmark degrees must be positive and finite.")
    p_mm = (point_to_landmark.T @ weighted_mapping).toarray()
    p_mm /= degree[:, None]
    return p_mm, degree


def compute_landmark_operator(K, labels, random_state=None, *, return_degree=False):
    """Construct operators with shapes ``Pmm: (M, M)`` and ``Pnm: (N, M)``.

    Rows of both returned operators sum to one. ``Pnm`` stays sparse to avoid
    allocating a dense sample-to-landmark matrix. ``return_degree=True`` adds
    the row sums of the aggregated kernel as a third item, for the symmetric
    similarity transform of ``Pmm``.
    """
    del random_state  # Kept for API compatibility.
    kernel = sparse.csr_matrix(K)
    labels = np.asarray(labels)
    landmarks, group_indices = np.unique(labels, return_inverse=True)
    # Each column assigns one sample to its group; aggregate without dense rows.
    membership = sparse.csr_matrix(
        (np.ones(kernel.shape[0]), (group_indices, np.arange(kernel.shape[0]))),
        shape=(landmarks.size, kernel.shape[0]),
    )
    landmark_to_point = membership @ kernel
    p_nm = sparse.csr_matrix(preprocessing.normalize(landmark_to_point.T, norm="l1", axis=1))
    del landmark_to_point
    p_mm, degree = _landmark_operator_from_mapping(kernel, p_nm)
    if return_degree:
        return p_mm, p_nm, degree
    return p_mm, p_nm


def calc_l(eigenvalues):
    """Estimate the start step from entropy of spectral magnitudes.

    Magnitudes are used only for this non-negative entropy distribution;
    matrix powers and PageRank must continue to use signed eigenvalues.
    """
    eigenvalues = np.abs(np.asarray(eigenvalues))
    entropy_values = [stats.entropy(np.power(eigenvalues, power)) for power in range(30)]
    first = np.gradient(entropy_values)
    second = np.gradient(first)
    changes = np.where(np.diff(np.sign(second)) != 0)[0]
    return 2 if changes.size == 0 else int(changes[0] + 2)


def compute_transform_matrix(X, X_center, metric="sqeuclidean"):
    """Map samples to centers with a normalized radial similarity matrix."""
    from scipy import spatial

    distance_matrix = spatial.distance.cdist(X, X_center, metric=metric)
    minimum = np.min(distance_matrix, axis=1, keepdims=True)
    safe_minimum = np.maximum(minimum, np.finfo(np.float64).eps)
    similarity = np.exp(-(distance_matrix - minimum) / safe_minimum)
    return preprocessing.normalize(similarity, norm="l1", axis=1)


def compute_local_transform_matrix(
    X,
    X_center,
    X_label,
    knn_indices,
    sqsigmas,
    k_neighbors,
    delta=2,
    alpha=1,
    metric="sqeuclidean",
):
    """Map samples to their clusters' neighboring centers."""
    from scipy import spatial

    n_samples = X.shape[0]
    n_centers = X_center.shape[0]
    local_distances = np.zeros((n_samples, k_neighbors))
    local_kernel = np.zeros((n_samples, k_neighbors))
    sample_sigmas = np.maximum(
        np.asarray(sqsigmas)[np.asarray(X_label)], np.finfo(np.float64).eps
    )
    for index in range(n_samples):
        center_indices = knn_indices[X_label[index]]
        local_distances[index] = spatial.distance.cdist(
            np.asarray(X[index]).reshape(1, -1),
            np.asarray(X_center)[center_indices],
            metric=metric,
        )
        local_kernel[index] = np.exp(
            -np.power(
                local_distances[index] / (delta * np.sqrt(sample_sigmas[index])),
                alpha,
            )
        )
    full_kernel = np.zeros((n_samples, n_centers))
    for index in range(n_samples):
        full_kernel[index, knn_indices[X_label[index]]] = local_kernel[index]
    return preprocessing.normalize(full_kernel, norm="l1", axis=1)


def _check_diffusion_step(value, *, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer.")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def compute_infty_R(phi, eigenvalues, psi, damping, diffusion_step, *, return_derivative=True):
    """Compute limiting PageRank distributions and their damping derivatives.

    Notes
    -----
    Each spectral coefficient is ``(1-c_i) * lambda_k**l / (1-c_i*lambda_k)``.
    The derivative is row-local: only ``c_i`` affects row ``i`` of ``R``.
    ``return_derivative=False`` skips its allocation and returns ``None``
    as the second tuple item.
    """
    diffusion_step = _check_diffusion_step(diffusion_step, name="diffusion_step")
    n_samples = len(damping)
    powered = np.power(eigenvalues, diffusion_step)
    denominator = 1 - np.outer(damping, eigenvalues)
    safe = np.where(
        np.abs(denominator) < np.finfo(np.float64).eps,
        np.copysign(np.finfo(np.float64).eps, denominator),
        denominator,
    )
    sigma = np.outer(1 - damping, powered) / safe
    rank = (phi * sigma) @ psi
    del sigma

    if not return_derivative:
        rank = np.real_if_close(rank).astype(np.float64, copy=False)
        rank[rank < 0] = 0
        return preprocessing.normalize(rank, norm="l1", axis=1, copy=False), None

    derivative_sigma = np.tile(
        powered * (eigenvalues - 1), (n_samples, 1)
    ) / np.power(safe, 2)
    derivative = (phi * derivative_sigma) @ psi

    rank = np.real_if_close(rank).astype(np.float64, copy=False)
    derivative = np.real_if_close(derivative).astype(np.float64, copy=False)
    rank[rank < 0] = 0
    derivative[rank == 0] = 0
    rank = preprocessing.normalize(rank, norm="l1", axis=1, copy=False)
    return rank, derivative


def compute_rank_matrix(
    K,
    l1=None,
    l2=None,
    mode=1,
    beta=0.1,
    epsilon=1e-2,
    random_state=None,
    verbose=0,
    max_iter=30,
    *,
    return_info=False,
    landmark_degree=None,
):
    """Optimize node damping factors against the target diffusion distribution.

    Parameters
    ----------
    l1, l2 : int or None
        Restart step and target step. Defaults use spectral entropy for ``l1``
        and ``l1 + 1`` for ``l2``.
    mode : {1, 2}
        Full symmetric-kernel or landmark Markov-operator factorization.
    return_info : bool, default=False
        Return ``(R, info)`` with iteration count, convergence, and maximum
        damping change. The default returns only ``R``.
    landmark_degree : array-like, optional
        Aggregated kernel row sums for stable reversible factorization in
        landmark mode. The original transition matrix is preserved.
    """
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be a positive integer.")
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    check_positive_real(beta, name="beta")
    check_positive_real(epsilon, name="epsilon")
    if l1 is not None:
        l1 = _check_diffusion_step(l1, name="l1")
    if l2 is not None:
        l2 = _check_diffusion_step(l2, name="l2")
    with stage(verbose, "Spectral decomposition", shape=K.shape, mode=mode):
        if mode == 1:
            phi, eigenvalues, psi = eigen_kernel(K)
        elif mode == 2:
            phi, eigenvalues, psi = eigen_kernel2(K, degree=landmark_degree)
        else:
            raise ValueError("mode must be 1 (full) or 2 (landmark).")

    if l1 is None:
        l1 = calc_l(eigenvalues)
    if l2 is None:
        l2 = l1 + 1
    log(verbose, f"Diffusion steps: restart={l1}, target={l2}", level=2)
    with stage(verbose, "Damping optimization", max_iter=max_iter, tolerance=epsilon):
        target = (phi * np.power(eigenvalues, l2)) @ psi
        target = np.real_if_close(target).astype(np.float64, copy=False)

        rng = np.random.default_rng(random_state)
        damping = np.round(rng.uniform(0.5, 0.7, size=K.shape[0]), 2)
        rank = None
        converged = False
        for iteration in range(1, max_iter + 1):
            rank, derivative = compute_infty_R(phi, eigenvalues, psi, damping, l1)
            denominator = rank.copy()
            denominator[denominator == 0] = np.inf
            # Differentiate KL(target || R); zero entries avoid division by zero.
            gradient = (-target / denominator * derivative).sum(axis=1)
            updated = np.clip(damping - beta * gradient, 0.01, 0.99)
            difference = updated - damping
            damping = updated
            # Only the target and spectral factors are needed by the next iteration.
            del rank, derivative, denominator

            if verbose >= 2:
                log(verbose, f"Damping iteration {iteration}: mean={np.mean(damping):.4f}, max_change={np.max(np.abs(difference)):.4g}", level=2)
            if not np.isfinite(damping).all():
                raise FloatingPointError("Damping optimization produced non-finite values.")
            max_change = float(np.max(np.abs(difference)))
            if max_change < epsilon:
                converged = True
                break
        else:
            warnings.warn(
                f"Damping optimization reached max_iter={max_iter} before convergence.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Return the matrix for the final damping values, not the previous iterate.
        del target
        rank, _ = compute_infty_R(phi, eigenvalues, psi, damping, l1, return_derivative=False)
        log(verbose, f"Damping optimization summary: iterations={iteration}, converged={converged}, max_change={max_change:.4g}")
    if return_info:
        return rank, {"n_iter": iteration, "converged": converged, "max_change": max_change}
    return rank


__all__ = [
    "calc_l",
    "compute_infty_R",
    "compute_landmark_operator",
    "compute_local_transform_matrix",
    "compute_rank_matrix",
    "compute_transform_matrix",
    "eigen_kernel",
    "eigen_kernel2",
]
