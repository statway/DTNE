"""Optional plotting utilities for DTNE results."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from ._validation import is_missing_label


def _matplotlib():
    """Load plotting dependencies only when a plot is requested."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "matplotlib is required for DTNE plotting. Install DTNE with its "
            "'visualization' optional dependencies."
        ) from exc
    return plt, Line2D


def _check_embedding(embedding, vis_dim: int):
    """Validate that finite coordinates support the requested 2D or 3D view."""
    points = np.asarray(embedding, dtype=np.float64)
    if vis_dim not in (2, 3):
        raise ValueError("vis_dim must be 2 or 3.")
    if points.ndim != 2 or points.shape[1] < vis_dim:
        raise ValueError(f"embedding must have shape (n_samples, >={vis_dim}).")
    if not np.isfinite(points).all():
        raise ValueError("embedding contains NaN or infinite values.")
    return points


def _stable_factorize(values):
    """Encode categories in first-occurrence order, grouping missing labels."""
    labels = np.asarray(values).reshape(-1)
    mapping: dict[Any, int] = {}
    names: list[str] = []
    codes = np.empty(labels.shape[0], dtype=np.int64)
    missing_key = object()
    for index, value in enumerate(labels.tolist()):
        missing = is_missing_label(value)
        if missing:
            key = missing_key
        else:
            try:
                key = value if hash(value) is not None else str(value)
            except TypeError:
                key = str(value)
        if key not in mapping:
            mapping[key] = len(mapping)
            names.append("Missing" if missing else str(value))
        codes[index] = mapping[key]
    return codes, names


def plot_embedding(
    embedding,
    values=None,
    *,
    kind: str = "auto",
    vis_dim: int = 2,
    ax=None,
    title: str | None = None,
    cmap: str | None = None,
    point_size: float = 8,
    alpha: float = 0.75,
    legend: bool = True,
    colorbar: bool = True,
    show_axes: bool = False,
    save_path: str | PathLike[str] | None = None,
    dpi: int = 150,
):
    """Plot a 2D or 3D embedding with categorical or continuous colors."""
    plt, Line2D = _matplotlib()
    points = _check_embedding(embedding, vis_dim)
    if ax is None:
        figure = plt.figure(figsize=(7, 6))
        ax = figure.add_subplot(111, projection="3d" if vis_dim == 3 else None)
    figure = ax.figure
    coordinates = (points[:, 0], points[:, 1])
    if vis_dim == 3:
        coordinates += (points[:, 2],)

    if values is None:
        ax.scatter(
            *coordinates, s=point_size, alpha=alpha, edgecolors="none"
        )
    else:
        values_array = np.asarray(values).reshape(-1)
        if values_array.shape[0] != points.shape[0]:
            raise ValueError("values must contain one entry per embedded sample.")
        if kind == "auto":
            kind = (
                "continuous"
                if np.issubdtype(values_array.dtype, np.floating)
                and np.unique(values_array).size > min(20, max(5, points.shape[0] // 20))
                else "categorical"
            )
        if kind == "continuous":
            color_map = cmap or "viridis"
            artist = ax.scatter(
                *coordinates,
                c=values_array.astype(float),
                cmap=color_map,
                s=point_size,
                alpha=alpha,
                edgecolors="none",
            )
            if colorbar:
                figure.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
        elif kind == "categorical":
            codes, names = _stable_factorize(values_array)
            color_map = plt.get_cmap(cmap or "tab20", max(len(names), 1))
            ax.scatter(
                *coordinates,
                c=codes,
                cmap=color_map,
                vmin=-0.5,
                vmax=max(len(names) - 0.5, 0.5),
                s=point_size,
                alpha=alpha,
                edgecolors="none",
            )
            if legend and len(names) <= 30:
                handles = [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markerfacecolor=color_map(index),
                        markeredgecolor="none",
                        markersize=6,
                    )
                    for index in range(len(names))
                ]
                ax.legend(
                    handles,
                    names,
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                    fontsize=8,
                )
        else:
            raise ValueError("kind must be 'auto', 'categorical', or 'continuous'.")

    if title:
        ax.set_title(title)
    if not show_axes:
        ax.set_axis_off()
    if vis_dim == 2:
        ax.set_aspect("equal", adjustable="datalim")
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return ax


def plot_dtne_results(
    embedding,
    *,
    labels=None,
    pseudotime=None,
    clusters=None,
    vis_dim: int = 2,
    figsize=(18, 5),
    point_size: float = 8,
    alpha: float = 0.75,
    save_path: str | PathLike[str] | None = None,
    dpi: int = 150,
):
    """Plot available original labels, pseudotime, and clusters side by side."""
    plt, _ = _matplotlib()
    panels = [
        ("Original labels", labels, "auto", "tab20"),
        ("DTNE pseudotime", pseudotime, "continuous", "viridis"),
        ("DTNE clusters", clusters, "categorical", "tab20"),
    ]
    panels = [panel for panel in panels if panel[1] is not None]
    if not panels:
        raise ValueError("At least one of labels, pseudotime, or clusters is required.")
    subplot_kw = {"projection": "3d"} if vis_dim == 3 else {}
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=figsize,
        squeeze=False,
        subplot_kw=subplot_kw,
    )
    for axis, (title, values, kind, cmap) in zip(axes[0], panels):
        plot_embedding(
            embedding,
            values,
            kind=kind,
            vis_dim=vis_dim,
            ax=axis,
            title=title,
            cmap=cmap,
            point_size=point_size,
            alpha=alpha,
        )
    figure.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    return figure, axes[0]


__all__ = ["plot_dtne_results", "plot_embedding"]
