"""Diffusive Topology Neighbor Embedding."""

from .ee import (
    calc_affinity,
    calc_ee_loss,
    ee_linear_search,
    elastic_embedding,
    get_laplacians,
)
from ._version import __version__
from .estimator import DTNE
from .plotting import plot_dtne_results, plot_embedding
from .utils import (
    box,
    box_kernel,
    box_kernel2,
    calc_l,
    classic,
    compute_global_transform_matrix,
    compute_infty_R,
    compute_landmark_operator,
    compute_local_transform_matrix,
    compute_rank_matrix,
    compute_transform_matrix,
    eigen_kernel,
    eigen_kernel2,
    gauss_kernel,
    mix_decay,
    mix_kernel,
    phate_kernel,
    scanpy_kernel,
)

__all__ = [
    "DTNE",
    "box",
    "box_kernel",
    "box_kernel2",
    "calc_affinity",
    "calc_ee_loss",
    "calc_l",
    "classic",
    "compute_global_transform_matrix",
    "compute_infty_R",
    "compute_landmark_operator",
    "compute_local_transform_matrix",
    "compute_rank_matrix",
    "compute_transform_matrix",
    "ee_linear_search",
    "eigen_kernel",
    "eigen_kernel2",
    "elastic_embedding",
    "gauss_kernel",
    "get_laplacians",
    "mix_decay",
    "mix_kernel",
    "phate_kernel",
    "plot_dtne_results",
    "plot_embedding",
    "scanpy_kernel",
]
