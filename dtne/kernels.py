"""Neighborhood graph kernels used by DTNE."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn import neighbors

from ._validation import check_feature_matrix, check_neighbors, check_positive_real


def _nearest_neighbors(data, k_neighbors: int, include_self: bool):
    """Return Euclidean kNN distances with explicit self-inclusion semantics."""
    data = check_feature_matrix(data, name="data")
    n_samples = data.shape[0]
    k_neighbors = check_neighbors(k_neighbors, n_samples, include_self)
    query_k = min(n_samples, k_neighbors + 1)
    # Request one extra neighbor so removing the sample itself leaves k entries.
    model = neighbors.NearestNeighbors(
        n_neighbors=query_k,
        metric="euclidean",
        n_jobs=-2,
    ).fit(data)
    raw_distances, raw_indices = model.kneighbors(data)

    distances = np.empty((n_samples, k_neighbors), dtype=np.float64)
    indices = np.empty((n_samples, k_neighbors), dtype=np.int64)
    for row in range(n_samples):
        keep = raw_indices[row] != row
        other_indices = raw_indices[row, keep]
        other_distances = raw_distances[row, keep]
        if include_self:
            # Insert self explicitly because duplicate points can change tie order.
            indices[row, 0] = row
            distances[row, 0] = 0.0
            needed = k_neighbors - 1
            if needed:
                indices[row, 1:] = other_indices[:needed]
                distances[row, 1:] = other_distances[:needed]
        else:
            indices[row] = other_indices[:k_neighbors]
            distances[row] = other_distances[:k_neighbors]
    return distances, indices


def _csr_from_neighbors(values: np.ndarray, indices: np.ndarray, n_samples: int):
    """Store one fixed-length neighbor list per row in a CSR matrix."""
    k_neighbors = indices.shape[1]
    indptr = np.arange(0, (n_samples + 1) * k_neighbors, k_neighbors)
    return sparse.csr_matrix(
        (values.reshape(-1), indices.reshape(-1), indptr),
        shape=(n_samples, n_samples),
    )


def _symmetric_normalize(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    """Apply symmetric degree normalization ``D**(-1/2) @ K @ D**(-1/2)``."""
    matrix = sparse.csr_matrix(matrix)
    degree = np.asarray(matrix.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0) or not np.isfinite(degree).all():
        raise ValueError("The neighborhood graph contains an empty or invalid row.")
    inv_sqrt = 1.0 / np.sqrt(degree)
    scaling = sparse.diags(inv_sqrt, format="csr")
    normalized = scaling @ matrix @ scaling
    normalized.eliminate_zeros()
    return sparse.csr_matrix(normalized)


def box(d, local_sigma):
    """Evaluate the rectangular (box) kernel."""
    return np.where(np.asarray(d) <= local_sigma, 1.0, 0.0)


def box_kernel(data, k_neighbors, include_self: bool = True):
    """Return a normalized binary kNN graph, neighbor indices, and bandwidths."""
    distances, knn_indices = _nearest_neighbors(data, k_neighbors, include_self)
    n_samples = distances.shape[0]
    values = np.ones_like(distances, dtype=np.float64)
    directed = _csr_from_neighbors(values, knn_indices, n_samples)
    symmetric = directed.maximum(directed.T)
    return _symmetric_normalize(symmetric), knn_indices, distances[:, -1]


def gauss_kernel(
    data,
    k_neighbors,
    delta: float = 2,
    alpha: float = 1,
    include_self: bool = True,
):
    """Build a Gaussian graph with sample-neighbor adaptive bandwidth products.

    Notes
    -----
    Before symmetrization, edge weights are
    ``exp(-(d_ij**2 / (delta * sigma_i * sigma_j))**alpha)``.
    Each ``sigma_i`` is the farthest selected neighbor's Euclidean distance.
    Returns a normalized CSR graph, kNN indices, and Euclidean bandwidths.
    """
    for name, value in (("delta", delta), ("alpha", alpha)):
        check_positive_real(value, name=name)
    distances, knn_indices = _nearest_neighbors(data, k_neighbors, include_self)
    n_samples = distances.shape[0]
    sigmas = distances[:, -1]
    # Work in log space to retain scale invariance without squaring tiny
    # distances or forming potentially overflowing bandwidth products.
    weights = np.ones_like(distances)
    positive = distances > 0
    valid = positive & (sigmas[:, None] > 0) & (sigmas[knn_indices] > 0)
    weights[positive & ~valid] = 0.0  # Zero-bandwidth limit at nonzero distance.
    rows, columns = np.nonzero(valid)
    log_ratio = (
        2 * np.log(distances[valid]) - np.log(sigmas[rows])
        - np.log(sigmas[knn_indices[rows, columns]]) - np.log(delta)
    )
    with np.errstate(over="ignore", under="ignore"):
        weights[valid] = np.exp(-np.exp(alpha * log_ratio))
    directed = _csr_from_neighbors(weights, knn_indices, n_samples)
    transpose = directed.T.tocsr()
    # Fuzzy union strengthens mutual edges while keeping weights in [0, 1].
    symmetric = directed + transpose - directed.multiply(transpose)
    symmetric.data = np.clip(symmetric.data, 0.0, 1.0)
    return _symmetric_normalize(symmetric), knn_indices, sigmas


def box_kernel2(
    data,
    k_neighbors,
    delta: float = 1,
    include_self: bool = True,
):
    """Build a box graph with adaptive cutoffs within the selected kNN edges."""
    check_positive_real(delta, name="delta")
    distances, knn_indices = _nearest_neighbors(data, k_neighbors, include_self)
    n_samples = distances.shape[0]
    sigmas = distances[:, -1]
    squared_distances = distances**2
    local_sigmas = delta * sigmas * sigmas[knn_indices[:, -1]]
    floor_column = min(3, k_neighbors - 1)
    # Keep a small local neighborhood even when the adaptive cutoff is narrow.
    local_sigmas = np.maximum(local_sigmas, squared_distances[:, floor_column])
    weights = box(squared_distances, local_sigmas[:, None])
    directed = _csr_from_neighbors(weights, knn_indices, n_samples)
    symmetric = directed.maximum(directed.T)
    symmetric.eliminate_zeros()
    return _symmetric_normalize(symmetric), knn_indices, sigmas


def mix_decay(d, local_sigma, alpha: float = 1):
    """Use unit weights inside the cutoff and exponential decay outside it."""
    check_positive_real(alpha, name="alpha")
    local_sigma = np.maximum(local_sigma, np.finfo(np.float64).eps)
    return np.where(
        np.asarray(d) <= local_sigma,
        1.0,
        np.exp(-np.power(np.asarray(d) / local_sigma, alpha)),
    )


def mix_kernel(
    data,
    k_neighbors,
    delta: float = 1,
    alpha: float = 1,
    include_self: bool = True,
):
    """Build a normalized graph combining box weights and exponential tails."""
    check_positive_real(delta, name="delta")
    check_positive_real(alpha, name="alpha")
    distances, knn_indices = _nearest_neighbors(data, k_neighbors, include_self)
    n_samples = distances.shape[0]
    sigmas = distances[:, -1]
    squared_distances = distances**2
    local_sigmas = delta * sigmas * sigmas[knn_indices[:, -1]]
    weights = mix_decay(squared_distances, local_sigmas[:, None], alpha)
    directed = _csr_from_neighbors(weights, knn_indices, n_samples)
    symmetric = directed.maximum(directed.T)
    return _symmetric_normalize(symmetric), knn_indices, sigmas


def scanpy_kernel(data, knn: int = 5, method: str = "umap", random_state=123):
    """Build Scanpy connectivities from the supplied features and random seed.

    The estimator owns dimensionality reduction; ``use_rep='X'`` prevents
    Scanpy from automatically replacing these features with another PCA.
    """
    try:
        import graphtools
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "scanpy and graphtools are required for Scanpy kernels. "
            "Install DTNE with its 'single-cell' optional dependencies."
        ) from exc

    adata = sc.AnnData(data)
    if method == "rapids":
        try:
            sc.pp.neighbors(
                adata,
                n_neighbors=knn,
                method="umap",
                transformer="rapids",
                use_rep="X",
                random_state=random_state,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "The RAPIDS neighbor backend requires the RAPIDS single-cell "
                "dependencies supported by the installed Scanpy version."
            ) from exc
    else:
        sc.pp.neighbors(adata, n_neighbors=knn, method=method,
                        use_rep="X", random_state=random_state)
    distances = adata.obsp["distances"]
    connectivities = adata.obsp["connectivities"]
    kernel = graphtools.matrix.set_diagonal(connectivities, 1)
    return sparse.csr_matrix(kernel), None, distances


def phate_kernel(
    data,
    knn: int = 5,
    decay: float = 40.0,
    anisotropy: float = 0,
    random_state: int | None = 123,
    **kwargs: Any,
):
    """Build a PHATE-style graph and return its bandwidths through graphtools."""
    try:
        import graphtools
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise ImportError(
            "graphtools is required for the PHATE kernel. "
            "Install DTNE with its 'single-cell' optional dependencies."
        ) from exc

    graph = graphtools.Graph(
        data,
        knn=knn,
        decay=decay,
        anisotropy=anisotropy,
        use_pygsp=True,
        random_state=random_state,
        **kwargs,
    )
    return sparse.csr_matrix(graph.kernel), None, np.asarray(graph.bandwidth)


__all__ = [
    "box",
    "box_kernel",
    "box_kernel2",
    "gauss_kernel",
    "mix_decay",
    "mix_kernel",
    "phate_kernel",
    "scanpy_kernel",
]
