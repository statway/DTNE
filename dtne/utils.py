"""Backward-compatible utility imports.

New code should import from :mod:`dtne.kernels`, :mod:`dtne.diffusion`, or
:mod:`dtne.reduction` directly. This module keeps the 0.9 public names stable.
"""

from .diffusion import (
    calc_l,
    compute_infty_R,
    compute_landmark_operator,
    compute_local_transform_matrix,
    compute_rank_matrix,
    compute_transform_matrix as compute_global_transform_matrix,
    eigen_kernel,
    eigen_kernel2,
)
from .kernels import (
    box,
    box_kernel,
    box_kernel2,
    gauss_kernel,
    mix_decay,
    mix_kernel,
    phate_kernel,
    scanpy_kernel,
)
from .reduction import classic

# The historical module exposed the local, cluster-aware implementation under
# this name because it silently replaced an earlier duplicate definition.
compute_transform_matrix = compute_local_transform_matrix

__all__ = [
    "box",
    "box_kernel",
    "box_kernel2",
    "calc_l",
    "classic",
    "compute_global_transform_matrix",
    "compute_infty_R",
    "compute_landmark_operator",
    "compute_local_transform_matrix",
    "compute_rank_matrix",
    "compute_transform_matrix",
    "eigen_kernel",
    "eigen_kernel2",
    "gauss_kernel",
    "mix_decay",
    "mix_kernel",
    "phate_kernel",
    "scanpy_kernel",
]
