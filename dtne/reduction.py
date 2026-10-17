"""Low-dimensional reductions of DTNE manifold distances."""

from __future__ import annotations

import inspect

import numpy as np
from scipy import linalg
from sklearn import manifold

from ._validation import check_integer


def _mds_distances(D, n_components):
    """Check the input shape and output dimension before any MDS work."""
    distances = np.asarray(D, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("D must be a square distance matrix.")
    check_integer(n_components, name="n_components")
    if n_components > distances.shape[0]:
        raise ValueError("n_components must be between 1 and the number of samples.")
    return distances


def classic(D, n_components: int = 2, random_state=None):
    """Classical MDS coordinates used to initialize metric MDS."""
    distances = _mds_distances(D, n_components)
    gram = np.square(distances)
    # Compute means before centering, then reuse the owned square buffer.
    row_mean = gram.mean(axis=1, keepdims=True)
    column_mean = gram.mean(axis=0, keepdims=True)
    grand_mean = gram.mean()
    gram -= row_mean
    gram -= column_mean
    gram += grand_mean
    gram *= -0.5
    eigenvalues, eigenvectors = linalg.eigh(
        gram, check_finite=True,
        subset_by_index=(distances.shape[0] - n_components, distances.shape[0] - 1),
    )
    selected = np.maximum(eigenvalues[::-1], 0.0)
    # Negative modes cannot contribute real Euclidean coordinates.
    return eigenvectors[:, ::-1] * np.sqrt(selected)[None, :]


def metric_mds(
    distances,
    *,
    n_components: int = 2,
    random_state=None,
    verbose: int = 0,
):
    """Run metric MDS across both old and new scikit-learn APIs."""
    distances = _mds_distances(distances, n_components)
    if not np.any(distances):
        # Coincident samples need neither spectral initialization nor optimization.
        return np.zeros((distances.shape[0], n_components), dtype=np.float64)
    init = classic(distances, n_components=n_components, random_state=random_state)
    parameters = inspect.signature(manifold.MDS).parameters
    kwargs = {
        "n_components": n_components,
        "n_init": 1,
        "random_state": random_state,
        "verbose": verbose,
    }
    if "normalized_stress" in parameters:
        kwargs["normalized_stress"] = False
    if "metric_mds" in parameters:
        kwargs["metric_mds"] = True
        kwargs["metric"] = "precomputed"
    else:  # scikit-learn < 1.9
        kwargs["metric"] = True
        kwargs["dissimilarity"] = "precomputed"
    if "init" in parameters:
        # scikit-learn 1.9 validates the constructor value as a strategy name,
        # while still accepting explicit coordinates in fit_transform.
        kwargs["init"] = "random"
    reducer = manifold.MDS(**kwargs)
    return reducer.fit_transform(np.asarray(distances), init=init)


def umap_from_distances(
    distances,
    *,
    n_components: int = 2,
    min_dist: float = 0.3,
    random_state=None,
    verbose: int = 0,
):
    """Embed ordinary distances through the optional UMAP backend."""
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "umap-learn is required when solver='umap'. Install DTNE with "
            "its 'visualization' optional dependencies."
        ) from exc
    return umap.UMAP(
        metric="precomputed",
        n_components=n_components,
        min_dist=min_dist,
        random_state=random_state,
        verbose=verbose >= 2,
    ).fit_transform(np.asarray(distances))


__all__ = ["classic", "metric_mds", "umap_from_distances"]
