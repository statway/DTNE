"""Pseudotime and clustering helpers for DTNE."""

from __future__ import annotations

import inspect

import numpy as np
from scipy import sparse
from sklearn import cluster

from ._validation import normalize_unit_interval, check_integer
from .distances import cross_manifold_distances
from ._logging import Progress


def pseudotime_from_distances(
    distances,
    root_cells,
    terminal_cells=None,
):
    """Compute pseudotime in [0, 1] from ordinary cell-to-cell distances.

    Root distances are summed across endpoints. Optional terminal distances
    define a backward score, combined as ``forward / (forward + backward)``.
    """
    matrix = np.asarray(distances, dtype=np.float64)
    root_distance = matrix[np.asarray(root_cells)].sum(axis=0)
    forward = normalize_unit_interval(root_distance)
    if terminal_cells is None:
        return forward
    terminal_distance = matrix[np.asarray(terminal_cells)].sum(axis=0)
    backward = normalize_unit_interval(terminal_distance)
    denominator = forward + backward
    ratio = np.divide(
        forward,
        denominator,
        out=np.zeros_like(forward),
        where=denominator > np.finfo(np.float64).eps,
    )
    return normalize_unit_interval(ratio)


def landmark_pseudotime(
    point_to_landmark,
    landmark_rank,
    root_cells,
    terminal_cells=None,
    *,
    batch_size: int = 4096,
    verbose: int = 0,
):
    """Compute cell-level pseudotime from landmark distributions in batches.

    Each batch uses ``Pnm_batch @ Rmm``. Only endpoint distributions and one
    sample batch are materialized, avoiding a full dense ``(N, M)`` matrix.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise TypeError("batch_size must be a positive integer.")
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    operator = sparse.csr_matrix(point_to_landmark)
    n_samples = operator.shape[0]
    roots = np.asarray(operator[np.asarray(root_cells)] @ landmark_rank)
    terminals = (
        None
        if terminal_cells is None
        else np.asarray(operator[np.asarray(terminal_cells)] @ landmark_rank)
    )
    root_sums = np.empty(n_samples, dtype=np.float64)
    terminal_sums = None if terminals is None else np.empty(n_samples, dtype=np.float64)
    progress = Progress(verbose, "Pseudotime cells", n_samples)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        batch = np.asarray(operator[start:stop] @ landmark_rank)
        root_sums[start:stop] = cross_manifold_distances(roots, batch).sum(axis=0)
        if terminals is not None:
            terminal_sums[start:stop] = cross_manifold_distances(terminals, batch).sum(axis=0)
        progress.update(stop)

    forward = normalize_unit_interval(root_sums)
    if terminal_sums is None:
        return forward
    backward = normalize_unit_interval(terminal_sums)
    denominator = forward + backward
    ratio = np.divide(
        forward,
        denominator,
        out=np.zeros_like(forward),
        where=denominator > np.finfo(np.float64).eps,
    )
    return normalize_unit_interval(ratio)


def cluster_precomputed(
    distances,
    *,
    method: str = "agglo",
    n_clusters: int | None = None,
    eps: float = 0.5,
    min_samples: int = 5,
    random_state=None,
    sample_weight=None,
):
    """Cluster distances with at most eight default clusters.

    Explicit cluster counts cannot exceed the matrix size. Sample weights
    are used only for DBSCAN.
    """
    matrix = np.asarray(distances, dtype=np.float64)
    method = method.lower()
    if method in {"agglo", "hiera", "kmedoids"}:
        n_clusters = min(8, matrix.shape[0]) if n_clusters is None else n_clusters
        check_integer(n_clusters, name="n_clusters")
        if n_clusters > matrix.shape[0]:
            raise ValueError(f"n_clusters={n_clusters} exceeds the {matrix.shape[0]} available distance-matrix objects.")
        if matrix.shape[0] == 1:
            return np.zeros(1, dtype=np.int64)
    if method in {"agglo", "hiera"}:
        parameters = inspect.signature(cluster.AgglomerativeClustering).parameters
        kwargs = {"n_clusters": n_clusters, "linkage": "average"}
        if "metric" in parameters:
            kwargs["metric"] = "precomputed"
        else:  # scikit-learn < 1.2
            kwargs["affinity"] = "precomputed"
        return cluster.AgglomerativeClustering(**kwargs).fit_predict(matrix)
    if method == "dbscan":
        return cluster.DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="precomputed",
        ).fit(matrix, sample_weight=sample_weight).labels_
    if method == "kmedoids":
        try:
            from sklearn_extra.cluster import KMedoids
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn-extra is required for cluster_method='kmedoids'."
            ) from exc
        return KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            random_state=random_state,
        ).fit_predict(matrix)
    raise ValueError("cluster_method must be 'agglo', 'hiera', 'dbscan', or 'kmedoids'.")


__all__ = [
    "cluster_precomputed",
    "landmark_pseudotime",
    "pseudotime_from_distances",
]
