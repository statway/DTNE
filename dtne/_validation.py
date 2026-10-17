"""Input and parameter validation shared by DTNE modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from numbers import Real
import sys

import numpy as np
from scipy import sparse


def is_missing_label(value):
    """Recognize scalar missing labels without requiring pandas."""
    pandas_na = getattr(sys.modules.get("pandas"), "NA", None)
    if value is None or value is pandas_na:
        return True
    try:
        return bool(value != value)  # NaN and NaT are not equal to themselves.
    except (TypeError, ValueError):
        return False


def check_feature_matrix(X: Any, *, name: str = "X", copy: bool = False):
    """Return finite float64 features as a dense array or CSR matrix.

    ``copy=True`` establishes model ownership when accepting new input;
    internal checks use ``copy=False`` to avoid repeated full-size copies.
    """
    if sparse.issparse(X):
        checked = sparse.csr_matrix(X, dtype=np.float64, copy=copy)
        if checked.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix.")
        if checked.shape[0] < 2 or checked.shape[1] < 1:
            raise ValueError(
                f"{name} must contain at least 2 samples and 1 feature; "
                f"got shape {checked.shape}."
            )
        if not np.isfinite(checked.data).all():
            raise ValueError(f"{name} contains NaN or infinite values.")
        return checked

    checked = np.asarray(X, dtype=np.float64)
    if checked.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix; got {checked.ndim} dimensions.")
    if checked.shape[0] < 2 or checked.shape[1] < 1:
        raise ValueError(
            f"{name} must contain at least 2 samples and 1 feature; "
            f"got shape {checked.shape}."
        )
    if not np.isfinite(checked).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return checked.copy() if copy else checked


def check_integer(value: Any, *, name: str, minimum: int = 1) -> None:
    """Require an integer lower bound, explicitly excluding boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def check_labels(labels: Any, n_samples: int, *, name: str = "labels") -> np.ndarray | None:
    """Copy one label per sample, keeping labels separate from cell identifiers."""
    if labels is None:
        return None
    if hasattr(labels, "to_numpy"):
        labels = labels.to_numpy()
    values = np.asarray(labels)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.shape[0] != n_samples:
        raise ValueError(
            f"{name} must contain one value per sample; expected {n_samples}, "
            f"got {values.shape[0]}."
        )
    return values.copy()


def check_names(names: Any, n_items: int, *, name: str) -> np.ndarray | None:
    """Validate unique cell or gene names and normalize them to strings."""
    if names is None:
        return None
    if hasattr(names, "to_numpy"):
        names = names.to_numpy()
    values = np.asarray(names)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if values.shape[0] != n_items:
        raise ValueError(
            f"{name} must contain one name per item; expected {n_items}, "
            f"got {values.shape[0]}."
        )
    normalized = np.asarray([str(value) for value in values.tolist()])
    if np.any(normalized == ""):
        raise ValueError(f"{name} must not contain empty names.")
    if np.unique(normalized).size != normalized.size:
        raise ValueError(f"{name} must contain unique names.")
    return normalized


def check_positive_real(value: Any, *, name: str) -> None:
    """Require a finite positive scalar, excluding strings and booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number.")


def check_boolean(value: Any, *, name: str) -> bool:
    """Accept Python or NumPy booleans without truth-value coercion of other types."""
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")
    return bool(value)


def check_neighbors(k_neighbors: int, n_samples: int, include_self: bool) -> int:
    """Validate the total selected neighbor count, including self when requested."""
    include_self = check_boolean(include_self, name="include_self")
    if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, (int, np.integer)):
        raise TypeError("k_neighbors must be an integer.")
    k_neighbors = int(k_neighbors)
    maximum = n_samples if include_self else n_samples - 1
    if not 1 <= k_neighbors <= maximum:
        relation = "including" if include_self else "excluding"
        raise ValueError(
            f"k_neighbors must be between 1 and {maximum} when {relation} self; "
            f"got {k_neighbors}."
        )
    return k_neighbors


def check_choice(value: str, choices: Iterable[str], *, name: str) -> str:
    """Validate an enumerated option and return its lowercase representation."""
    normalized = str(value).lower()
    allowed = tuple(str(choice).lower() for choice in choices)
    if normalized not in allowed:
        raise ValueError(
            f"Unknown {name} {value!r}. Expected one of: {', '.join(allowed)}."
        )
    return normalized


def check_cell_identifier_types(cells: Any, *, name: str) -> None:
    """Reject boolean and floating identifiers before mixed-input coercion."""
    values = np.asarray(cells, dtype=object).reshape(-1)
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise TypeError(f"{name} must contain cell indices or names, not booleans or boolean masks.")
    if any(isinstance(value, (float, np.floating)) for value in values):
        raise TypeError(f"{name} must contain integer indices or names, not floating-point values.")


def check_cell_indices(cells: Any, n_samples: int, *, name: str) -> np.ndarray:
    """Return in-range integer indices without converting floats or booleans."""
    if cells is None:
        raise ValueError(f"{name} must be specified.")
    check_cell_identifier_types(cells, name=name)
    values = np.asarray(cells)
    if values.ndim == 0:
        values = values.reshape(1)
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must contain at least one cell index.")
    if values.dtype.kind not in "iu":
        if not all(isinstance(value, (int, np.integer)) for value in values):
            raise TypeError(f"{name} must contain integer cell indices.")
    values = values.astype(np.int64, copy=False)
    if ((values < 0) | (values >= n_samples)).any():
        raise IndexError(
            f"{name} contains an index outside [0, {n_samples - 1}]."
        )
    return values


def normalize_unit_interval(values: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1], returning zeros when the range is degenerate."""
    values = np.asarray(values, dtype=np.float64)
    lower = np.nanmin(values)
    upper = np.nanmax(values)
    width = upper - lower
    if not np.isfinite(width) or width <= np.finfo(np.float64).eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - lower) / width
