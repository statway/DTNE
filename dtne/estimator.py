"""Public DTNE estimator."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any
from numbers import Real

import numpy as np
from scipy import sparse
from sklearn import cluster, decomposition, preprocessing

from ._logging import log, stage, Progress

from ._validation import (
    check_boolean,
    check_cell_indices,
    check_choice,
    check_feature_matrix,
    check_labels,
    check_integer,
    check_positive_real,
    check_names,
    check_neighbors,
    check_cell_identifier_types,
    is_missing_label,
)
from .analysis import (
    cluster_precomputed,
    landmark_pseudotime,
    pseudotime_from_distances,
)
from .diffusion import (
    compute_landmark_operator,
    compute_rank_matrix,
    _landmark_operator_from_mapping,
)
from .distances import cross_manifold_distances, manifold_distances
from .ee import elastic_embedding
from .kernels import (
    box_kernel,
    box_kernel2,
    gauss_kernel,
    mix_kernel,
    phate_kernel,
    scanpy_kernel,
)
from .reduction import classic, metric_mds, umap_from_distances


class DTNE:
    """Diffusive Topology Neighbor Embedding.

    Notes
    -----
    Graph parameters persist after successful construction. Overrides of
    ``beta``, ``epsilon``, ``max_iter``, and runtime settings are call-local.
    Large inputs use the landmark approximation unless ``n_landmark`` is
    specified explicitly.
    Fitted attributes ending in ``_`` describe computed results; ``K``, ``R``,
    and ``Pnm`` retain the historical matrix names.
    """

    _KERNELS = ("box", "box2", "gauss", "mix", "umap", "rapids", "scanpy_gauss", "phate")
    _SOLVERS = ("mds", "umap", "ee")
    _GRAPH_PARAMS = {"kernel", "k_neighbors", "include_self", "n_landmark", "n_pca", "delta", "alpha"}
    _RUNTIME_PARAMS = {"random_state", "verbose"}
    _DIFFUSION_PARAMS = _GRAPH_PARAMS | _RUNTIME_PARAMS | {"X", "l", "l2", "beta", "epsilon", "max_iter"}
    _REDUCTION_PARAMS = _RUNTIME_PARAMS | {"n_components", "solver", "min_dist", "lamb", "num_iters"}

    def __init__(
        self,
        n_components: int = 2,
        k_neighbors: int = 15,
        include_self: bool = True,
        kernel: str = "box",
        random_state: int | None = 123,
        beta: float = 0.1,
        epsilon: float = 1e-2,
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs = self._normalize_diffusion_alias(kwargs)
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.include_self = check_boolean(include_self, name="include_self")
        self.kernel = kernel
        self.random_state = random_state
        self.beta = beta
        self.epsilon = epsilon
        self.verbose = verbose

        self.delta = kwargs.pop("delta", 2)
        self.alpha = kwargs.pop("alpha", 1)
        self.solver = kwargs.pop("solver", "mds")
        check_choice(self.solver, self._SOLVERS, name="solver")
        self.n_landmark = kwargs.pop("n_landmark", None)
        self.l = kwargs.pop("l", None)
        self.l2 = kwargs.pop("l2", None)
        self.n_pca = kwargs.pop("n_pca", None)
        self.min_dist = kwargs.pop("min_dist", None)
        self.pseudotime_batch_size = kwargs.pop("pseudotime_batch_size", 4096)
        self.max_iter = kwargs.pop("max_iter", 30)
        if kwargs:
            warnings.warn(
                "Ignoring unsupported DTNE parameters: " + ", ".join(sorted(kwargs)),
                UserWarning,
                stacklevel=2,
            )
        self._validate_parameter_values(self.get_params())
        self._reset_fitted_state()

    def _reset_fitted_state(self) -> None:
        """Clear the input, its metadata, and all dependent results."""
        self.X = None
        self.X_labels = None
        self.labels_ = None
        self.cell_names_ = None
        self.gene_names_ = None
        self._reset_computed_state()

    def _reset_computed_state(self) -> None:
        """Invalidate results derived from the input while retaining its metadata."""
        self.X_pca = None
        self.n_samples = None
        self.n_features = None
        self.n_landmark_ = None
        self.n_pca_ = None
        self.mode = None
        self.mode_ = None
        self.K = None
        self.Pnm = None
        self.cluster_labels = None
        self._graph_params_ = None
        self.adjacency_knn_indices = None
        self.sigmas = None
        self._graph_data = None
        self._pca_model = None
        self._reset_diffusion_state()

    def _reset_diffusion_state(self):
        """Invalidate diffusion and downstream results while retaining the graph."""
        self.mode = None
        self.mode_ = None
        self.K2 = None
        self.R = None
        self._diffusion_params_ = None
        self.n_iter_ = None
        self.converged_ = None
        self.max_change_ = None
        self.distances_ = None
        self.dists = None
        self.Y_ = None
        self.Ym = None
        self._reduction_params_ = None
        self.root_cells = None
        self.terminal_cells = None
        self.pseudotime_ = None
        self.df_times = None
        self.clusters_ = None

    @property
    def squared_distances_(self):
        """Generate squared distances on demand without retaining a matrix."""
        return None if self.distances_ is None else np.square(self.distances_)

    @property
    def similarities_(self):
        """Reconstruct similarities on demand from ordinary distances."""
        values = self.squared_distances_
        if values is not None:
            values *= -0.5
            np.exp(values, out=values)
        return values

    @property
    def sims(self):
        """Retain the historical similarity interface without a matrix cache."""
        return self.similarities_

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return constructor parameters with canonical aliases for cloning."""
        del deep
        return {
            "n_components": self.n_components,
            "k_neighbors": self.k_neighbors,
            "include_self": self.include_self,
            "kernel": self.kernel,
            "random_state": self.random_state,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "verbose": self.verbose,
            "delta": self.delta,
            "alpha": self.alpha,
            "solver": self.solver,
            "n_landmark": self.n_landmark,
            "l": self.l,
            "l2": self.l2,
            "n_pca": self.n_pca,
            "min_dist": self.min_dist,
            "pseudotime_batch_size": self.pseudotime_batch_size,
            "max_iter": self.max_iter,
        }

    @staticmethod
    def _normalize_diffusion_alias(params):
        """Resolve l1 to l consistently, rejecting conflicting explicit values."""
        params = dict(params)
        if "l1" in params:
            if "l" in params and not np.array_equal(params["l"], params["l1"]):
                raise ValueError("l and its alias l1 must have the same value when both are provided.")
            params.setdefault("l", params.pop("l1"))
        return params

    def set_params(self, **params: Any):
        """Validate updates and invalidate only results that depend on changes."""
        params = self._normalize_diffusion_alias(params)
        self._validate_parameter_values(params)
        if "include_self" in params:
            params["include_self"] = check_boolean(params["include_self"], name="include_self")
        valid = self.get_params()
        unknown = sorted(set(params) - set(valid))
        if unknown:
            raise ValueError("Unknown DTNE parameters: " + ", ".join(unknown))
        changed = {name for name, value in params.items() if not np.array_equal(value, valid[name])}
        for name, value in params.items():
            setattr(self, name, value)
        # Changing a default need not invalidate results already computed
        # with that value through a temporary override.
        graph_settings = self._graph_params_ or valid
        diffusion_settings = self._diffusion_params_ or valid
        reduction_settings = self._reduction_params_ or valid
        graph_changed = any(
            not np.array_equal(params[name], graph_settings.get(name, valid[name]))
            for name in changed & self._graph_dependencies()
        )
        diffusion_changed = any(
            not np.array_equal(params[name], diffusion_settings.get(name, valid[name]))
            for name in changed & {"l", "l2", "beta", "epsilon", "max_iter", "random_state"}
        )
        reduction_dependencies = {"n_components", "solver", "random_state"}
        if self.solver == "umap":
            reduction_dependencies.add("min_dist")
        reduction_changed = any(
            not np.array_equal(
                0.3 if name == "min_dist" and params[name] is None else params[name],
                reduction_settings.get(name, valid[name]),
            )
            for name in changed & reduction_dependencies
        )
        # Invalidate from the earliest changed stage: graph -> diffusion -> embedding.
        if graph_changed:
            self._reset_computed_state()
        elif diffusion_changed:
            self._remember_graph_params(params)
            self._reset_diffusion_state()
        else:
            self._remember_graph_params(params)
            if reduction_changed:
                self.Y_ = self.Ym = self._reduction_params_ = None
        return self

    def _graph_dependencies(self, settings=None):
        """Return parameters used by the selected kernel and preprocessing."""
        settings = {**self.get_params(), **(settings or {})}
        dependencies = self._GRAPH_PARAMS - {"delta", "alpha"}
        if settings["kernel"] in {"box2", "gauss", "mix"}:
            dependencies = dependencies | {"delta"}
        if settings["kernel"] in {"gauss", "mix"}:
            dependencies = dependencies | {"alpha"}
        uses_seed = (
            settings["kernel"] in {"umap", "rapids", "scanpy_gauss", "phate"}
            or settings["n_pca"] is not None
            or (self.n_features is not None and self.n_features > 500)
        )
        return dependencies | ({"random_state"} if uses_seed else set())

    def _remember_graph_params(self, params):
        """Keep unused overrides for a later kernel or preprocessing change."""
        for name in self._GRAPH_PARAMS & params.keys():
            setattr(self, name, params[name])
            if self._graph_params_ is not None:
                self._graph_params_[name] = params[name]
        # A retained stochastic graph must keep its actual construction seed,
        # even when public defaults differ because of a temporary fit override.
        if ("random_state" in params and self._graph_params_ is not None
                and "random_state" not in self._graph_dependencies()):
            self._graph_params_["random_state"] = params["random_state"]

    def _can_reuse_diffusion(self, params):
        """Compare only graph dependencies and effective diffusion settings."""
        if "X" in params or self.K is None or self._diffusion_params_ is None:
            return False
        dependencies = self._graph_dependencies({**(self._graph_params_ or {}), **params})
        return all(
            np.array_equal(value, (self._graph_params_ or {}).get(name))
            for name, value in params.items() if name in dependencies
        ) and all(
            np.array_equal(params.get(name, getattr(self, name)), value)
            for name, value in self._diffusion_params_.items()
        )

    @classmethod
    def _validate_parameter_values(cls, params):
        """Shared scalar contracts for construction, updates, and computation."""
        for name in ("n_components", "k_neighbors", "n_pca", "pseudotime_batch_size", "max_iter", "num_iters"):
            if name in params and not (name == "n_pca" and params[name] is None):
                check_integer(params[name], name=name)
        for name in ("l", "l2", "verbose"):
            if name in params and not (name != "verbose" and params[name] is None):
                check_integer(params[name], name=name, minimum=0)
        for name in ("beta", "epsilon", "delta", "alpha"):
            if name in params:
                check_positive_real(params[name], name=name)
        for name, choices in (("kernel", cls._KERNELS), ("solver", cls._SOLVERS)):
            if name in params:
                check_choice(params[name], choices, name=name)
        if "include_self" in params:
            check_boolean(params["include_self"], name="include_self")
        if "n_landmark" in params:
            value = params["n_landmark"]
            if value is not None and not (isinstance(value, str) and value == "auto"):
                check_integer(value, name="n_landmark")
        for name in ("min_dist", "lamb"):
            if name in params and not (name == "min_dist" and params[name] is None):
                value = params[name]
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError(f"{name} must be a finite non-negative number.")
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{name} must be a finite non-negative number.")

    @classmethod
    def _check_method_params(cls, params, allowed, method):
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(f"Unknown DTNE {method} parameters: " + ", ".join(unknown))
        cls._validate_parameter_values(params)

    @contextmanager
    def _runtime_settings(self, params):
        """Apply call-local runtime settings and restore them even on failure."""
        random_state, verbose = self.random_state, self.verbose
        self.random_state = params.get("random_state", random_state)
        self.verbose = params.get("verbose", verbose)
        try:
            yield
        finally:
            self.random_state, self.verbose = random_state, verbose

    def __set_params__(self, **params: Any):
        """Compatibility wrapper for the historical private method."""
        return self.set_params(**params)

    @staticmethod
    def _extract_input(
        X,
        labels=None,
        label_key: str | None = None,
        cell_names=None,
        gene_names=None,
    ):
        """Separate features from labels and names, then own a copy of the input."""
        if labels is None and isinstance(X, tuple) and len(X) == 2:
            X, labels = X

        if hasattr(X, "X") and hasattr(X, "obs"):
            if cell_names is None and hasattr(X, "obs_names"):
                cell_names = X.obs_names
            if gene_names is None and hasattr(X, "var_names"):
                gene_names = X.var_names
            if label_key is not None:
                if label_key not in X.obs:
                    raise KeyError(f"AnnData.obs does not contain {label_key!r}.")
                if labels is not None:
                    raise ValueError("Provide either labels or label_key, not both.")
                labels = X.obs[label_key]
            X = X.X
        elif hasattr(X, "columns") and hasattr(X, "index") and hasattr(X, "drop"):
            if cell_names is None:
                cell_names = X.index
            if label_key is not None:
                if label_key not in X.columns:
                    raise KeyError(
                        f"DataFrame does not contain label column {label_key!r}."
                    )
                if labels is not None:
                    raise ValueError("Provide either labels or label_key, not both.")
                labels = X[label_key]
                X = X.drop(columns=[label_key])
            if gene_names is None:
                gene_names = X.columns
        elif label_key is not None:
            raise TypeError("label_key is supported only for DataFrame or AnnData inputs.")

        X = check_feature_matrix(X, copy=True)
        labels = check_labels(labels, X.shape[0])
        cell_names = check_names(cell_names, X.shape[0], name="cell_names")
        gene_names = check_names(gene_names, X.shape[1], name="gene_names")
        return X, labels, cell_names, gene_names

    @staticmethod
    def _resolve_landmarks(requested, n_samples: int) -> int:
        """Resolve automatic or explicit landmark counts for this dataset."""
        if requested is None or requested == "auto":
            if n_samples >= 10_000:
                return min(2_000, n_samples)
            if n_samples >= 5_000:
                return min(1_000, n_samples)
            return n_samples
        if isinstance(requested, bool) or not isinstance(requested, (int, np.integer)):
            raise TypeError("n_landmark must be an integer, None, or 'auto'.")
        requested = int(requested)
        if not 1 <= requested <= n_samples:
            raise ValueError(
                f"n_landmark must be between 1 and {n_samples}; got {requested}."
            )
        return requested

    def _prepare_graph_data(self, data, n_pca):
        """Prepare dense PCA or sparse SVD features used by graph construction."""
        n_samples, n_features = data.shape
        if n_pca is None and n_features > 500:
            n_pca = min(100, n_samples - 1, n_features)
        if n_pca is None:
            self.n_pca_ = None
            self.X_pca = None
            self._pca_model = None
            return data
        if isinstance(n_pca, bool) or not isinstance(n_pca, (int, np.integer)):
            raise TypeError("n_pca must be an integer or None.")
        n_pca = int(n_pca)
        maximum = min(n_samples - 1, n_features)
        if not 1 <= n_pca <= maximum:
            raise ValueError(f"n_pca must be between 1 and {maximum}; got {n_pca}.")
        self.n_pca_ = n_pca
        if sparse.issparse(data):
            if n_features == 1:
                # TruncatedSVD requires two features; this input already has
                # the requested dimension and needs no reduction.
                self.X_pca = None
                self._pca_model = None
                return data
            reducer = decomposition.TruncatedSVD(
                n_components=n_pca, random_state=self.random_state
            )
        else:
            reducer = decomposition.PCA(
                n_components=n_pca, random_state=self.random_state
            )
        transformed = reducer.fit_transform(data)
        self._pca_model = reducer
        self.X_pca = np.asarray(transformed)
        return self.X_pca

    def compute_kernel_matrix(self, data: np.ndarray, **params):
        """Rebuild the graph and invalidate all downstream results.

        Passing the current ``self.X`` preserves input metadata; a replacement
        matrix becomes the new input and clears its previous labels and names.
        """
        self._check_method_params(params, self._GRAPH_PARAMS | self._RUNTIME_PARAMS, "compute_kernel_matrix")
        if "include_self" in params:
            check_boolean(params["include_self"], name="include_self")
        same_input = data is self.X
        data = check_feature_matrix(data, name="data", copy=not same_input)
        if same_input:
            params = {**(self._graph_params_ or {}), **params}
            self._reset_computed_state()
        else:
            self._reset_fitted_state()
        self.X = data
        with self._runtime_settings(params):
            return self._compute_kernel_matrix(data, **params)

    def _compute_kernel_matrix(self, data, **params):
        """Build a graph after the caller has invalidated derived state."""
        data = check_feature_matrix(data, name="data")
        self.n_samples, self.n_features = data.shape
        kernel_name = check_choice(
            params.get("kernel", self.kernel), self._KERNELS, name="kernel"
        )
        k_neighbors = params.get("k_neighbors", self.k_neighbors)
        include_self = params.get("include_self", self.include_self)
        self.kernel = kernel_name
        self.include_self = check_boolean(include_self, name="include_self")
        self.k_neighbors = check_neighbors(
            k_neighbors, self.n_samples, self.include_self
        )
        self.n_landmark_ = self._resolve_landmarks(
            params.get("n_landmark", self.n_landmark), self.n_samples
        )
        n_pca = params.get("n_pca", self.n_pca)
        if n_pca is not None or self.n_features > 500:
            with stage(self.verbose, "PCA/SVD preprocessing", cells=self.n_samples, features=self.n_features):
                graph_data = self._prepare_graph_data(data, n_pca)
        else:
            graph_data = self._prepare_graph_data(data, n_pca)
            log(self.verbose, "Using input features without PCA/SVD", level=2)
        self._graph_data = graph_data

        with stage(self.verbose, "Graph construction", cells=self.n_samples, features=graph_data.shape[1], k=self.k_neighbors, kernel=kernel_name):
            if kernel_name == "box":
                kernel, indices, sigmas = box_kernel(
                    graph_data, self.k_neighbors, include_self=self.include_self
                )
            elif kernel_name == "box2":
                kernel, indices, sigmas = box_kernel2(
                    graph_data,
                    self.k_neighbors,
                    delta=params.get("delta", self.delta),
                    include_self=self.include_self,
                )
            elif kernel_name == "gauss":
                kernel, indices, sigmas = gauss_kernel(
                    graph_data,
                    self.k_neighbors,
                    delta=params.get("delta", self.delta),
                    alpha=params.get("alpha", self.alpha),
                    include_self=self.include_self,
                )
            elif kernel_name == "mix":
                kernel, indices, sigmas = mix_kernel(
                    graph_data,
                    self.k_neighbors,
                    delta=params.get("delta", self.delta),
                    alpha=params.get("alpha", self.alpha),
                    include_self=self.include_self,
                )
            elif kernel_name in {"umap", "rapids", "scanpy_gauss"}:
                method = {
                    "umap": "umap",
                    "rapids": "rapids",
                    "scanpy_gauss": "gauss",
                }[kernel_name]
                kernel, indices, sigmas = scanpy_kernel(
                    graph_data, knn=self.k_neighbors, method=method,
                    random_state=self.random_state,
                )
            else:
                kernel, indices, sigmas = phate_kernel(
                    graph_data,
                    knn=self.k_neighbors,
                    random_state=self.random_state,
                )

            kernel = sparse.csr_matrix(kernel)
            if kernel_name in {"umap", "rapids", "scanpy_gauss", "phate"}:
                if self.include_self:
                    kernel.setdiag(1.0)
                else:
                    kernel.setdiag(0.0)
                    kernel.eliminate_zeros()
            self.adjacency_knn_indices = indices
            self.sigmas = sigmas
            # Keep the graph sparse even in full mode; spectral decomposition
            # creates its dense working matrix only for the duration of that call.
            self.K = kernel
            # Persist the effective graph configuration in the public parameters.
            # The cached snapshot describes this graph, rather than another set
            # of defaults that can disagree with get_params().
            for name in self._GRAPH_PARAMS - {"kernel", "k_neighbors", "include_self"}:
                setattr(self, name, params.get(name, getattr(self, name)))
            self._graph_params_ = {name: getattr(self, name) for name in self._GRAPH_PARAMS}
            self._graph_params_["random_state"] = self.random_state
            log(self.verbose, f"Graph details: shape={self.K.shape}, nonzeros={self.K.nnz}, landmarks={self.n_landmark_}", level=2)
        return self.K, self.sigmas

    def learn_vectors(self, **params):
        """Learn diffusion distributions from the current or provided input."""
        params = self._normalize_diffusion_alias(params)
        self._check_method_params(params, self._DIFFUSION_PARAMS, "learn_vectors")
        if "include_self" in params:
            check_boolean(params["include_self"], name="include_self")
        with self._runtime_settings(params):
            if self._can_reuse_diffusion(params):
                if self.R is not None:
                    log(self.verbose, "Reusing cached diffusion rank")
                    self._remember_graph_params(params)
                    return self.R
                if (self.mode_ == 1 and self.distances_ is not None
                        and isinstance(self.random_state, (int, np.integer))):
                    # Restore a released rank without invalidating results
                    # computed from the same deterministic diffusion settings.
                    log(self.verbose, "Restoring released diffusion rank; preserving downstream results")
                    rank = self._compute_diffusion_rank(self.K, params)
                    self._remember_graph_params(params)
                    return rank
            return self._learn_vectors(**params)

    def _learn_vectors(self, **params):
        """Reuse or rebuild the graph, then optimize its diffusion distributions."""
        if "X" in params:
            data = check_feature_matrix(params.pop("X"), copy=True)
            # An explicit replacement input has no guaranteed correspondence
            # with the previous labels, names, or endpoints, even at equal size.
            self._reset_fitted_state()
            self.X = data
        if self.X is None:
            raise RuntimeError("No input data is available. Call fit(X) or pass X=...")
        graph_params = dict(self._graph_params_ or {})
        dependencies = self._graph_dependencies({**graph_params, **params})
        # Reuse the graph when explicit overrides match its effective settings.
        # New input clears K above and always requires graph reconstruction.
        rebuild = self.K is None or any(
            not np.array_equal(value, graph_params.get(name))
            for name, value in params.items()
            if name in dependencies
        )
        graph_params.update({name: params[name] for name in self._GRAPH_PARAMS if name in params})
        if "random_state" in params:
            graph_params["random_state"] = params["random_state"]
        if rebuild:
            self._reset_computed_state()
        else:
            self._reset_diffusion_state()
        if "l" in params:
            self.l = params["l"]
        if "l2" in params:
            self.l2 = params["l2"]

        if rebuild:
            with self._runtime_settings(graph_params):
                kernel, _ = self._compute_kernel_matrix(self.X, **graph_params)
        else:
            kernel = self.K
            log(self.verbose, "Reusing cached graph")
            self._remember_graph_params(params)

        landmark_degree = None
        if self.n_landmark_ == self.n_samples:
            self.mode = self.mode_ = 1
            diffusion_kernel = kernel
        else:
            self.mode = self.mode_ = 2
            if self.cluster_labels is None:
                with stage(self.verbose, "Landmark grouping", cells=self.n_samples, landmarks=self.n_landmark_):
                    row_markov = preprocessing.normalize(kernel, norm="l1", axis=1)
                    # Grouping belongs to the graph stage and survives diffusion updates.
                    clustering_data = row_markov @ self._graph_data
                    if sparse.issparse(clustering_data):
                        clustering_data = clustering_data.toarray()
                    landmark_clusterer = cluster.AgglomerativeClustering(
                        n_clusters=self.n_landmark_, connectivity=kernel
                    )
                    self.cluster_labels = landmark_clusterer.fit_predict(clustering_data)
                    del row_markov, clustering_data, landmark_clusterer
            else:
                log(self.verbose, "Reusing cached landmark grouping")
            if self.Pnm is None:
                with stage(self.verbose, "Landmark mapping construction", cells=self.n_samples, landmarks=self.n_landmark_):
                    diffusion_kernel, self.Pnm, landmark_degree = compute_landmark_operator(
                        kernel, self.cluster_labels, random_state=self.random_state,
                        return_degree=True,
                    )
            else:
                log(self.verbose, "Reusing cached landmark mapping")
                # The mapping belongs to the graph stage; diffusion updates
                # require only a new small operator, not another sparse mapping.
                with stage(self.verbose, "Landmark operator reconstruction", landmarks=self.n_landmark_):
                    diffusion_kernel, landmark_degree = _landmark_operator_from_mapping(kernel, self.Pnm)

        # Once graph/group construction is complete, features and neighbor
        # metadata are unused. Release them before the dense diffusion work.
        self.X_pca = self._graph_data = self._pca_model = None
        self.adjacency_knn_indices = self.sigmas = None
        return self._compute_diffusion_rank(diffusion_kernel, params, landmark_degree=landmark_degree)

    def _compute_diffusion_rank(self, diffusion_kernel, params, *, landmark_degree=None):
        """Publish rank and optimization metadata after successful computation."""
        self.R, optimization = compute_rank_matrix(
            diffusion_kernel,
            l1=self.l,
            l2=self.l2,
            mode=self.mode_,
            beta=params.get("beta", self.beta),
            epsilon=params.get("epsilon", self.epsilon),
            random_state=self.random_state,
            verbose=self.verbose,
            max_iter=params.get("max_iter", self.max_iter),
            return_info=True,
            landmark_degree=landmark_degree,
        )
        self.n_iter_ = optimization["n_iter"]
        self.converged_ = optimization["converged"]
        self.max_change_ = optimization["max_change"]
        # Retain effective call settings, including temporary fit overrides,
        # so distance-cache checks do not mistake constructor defaults for them.
        self._diffusion_params_ = {
            name: params.get(name, getattr(self, name))
            for name in ("l", "l2", "beta", "epsilon", "max_iter", "random_state")
        }
        return self.R

    def compute_dist_matrix(self, **params):
        """Compute and retain only ordinary manifold distances.

        Returns
        -------
        numpy.ndarray
            Ordinary distances with shape ``(N, N)`` in full mode or ``(M, M)``
            in landmark mode. Squared distances and similarities are lazy forms
            of these values, generated as independent arrays on access.
        """
        params = self._normalize_diffusion_alias(params)
        self._check_method_params(params, self._DIFFUSION_PARAMS, "compute_dist_matrix")
        with self._runtime_settings(params):
            if self.distances_ is not None:
                if not params:
                    log(self.verbose, "Reusing cached distances")
                    if self.mode_ == 1:
                        self.R = None
                    return self.distances_
                if self._can_reuse_diffusion(params):
                    log(params.get("verbose", self.verbose), "Reusing cached distances")
                    self._remember_graph_params(params)
                    if self.mode_ == 1:
                        self.R = None
                    return self.distances_
            rank = self.R if self.R is not None and not params else self.learn_vectors(**params)
            with stage(self.verbose, "Distance calculation", objects=rank.shape[0]):
                self.distances_ = manifold_distances(rank, distance_only=True)
            if self.mode_ == 1:
                # Full-mode analyses use distances directly. Landmark analyses
                # still need R and Pnm to reconstruct cell-level distributions.
                self.R = None
            del rank
            # The compatibility alias exposes ordinary distances, not squared distances.
            self.dists = self.distances_
            return self.dists

    def reduce_dim(self, **params):
        """Embed ordinary manifold distances in the requested dimensions."""
        self._check_method_params(params, self._REDUCTION_PARAMS, "reduce_dim")
        with self._runtime_settings(params):
            return self._reduce_dim(**params)

    def _reduce_dim(self, **params):
        """Compute coordinates locally and publish settings only after success."""
        if self.distances_ is None:
            raise RuntimeError("No distance matrix is available. Call fit first.")
        n_components = params.get("n_components", self.n_components)
        if not isinstance(n_components, (int, np.integer)) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        n_components = int(n_components)
        solver = check_choice(
            params.get("solver", self.solver), self._SOLVERS, name="solver"
        )
        min_dist = params.get("min_dist", self.min_dist)
        reduction_params = {
            "solver": solver, "n_components": n_components,
            "random_state": self.random_state,
        }
        if solver == "umap":
            min_dist = 0.3 if min_dist is None else min_dist
            reduction_params["min_dist"] = min_dist
        elif solver == "ee":
            reduction_params.update(lamb=params.get("lamb", 1), num_iters=params.get("num_iters", 100))
        if self.Y_ is not None and self._reduction_params_ is not None and all(
            np.array_equal(value, self._reduction_params_.get(name))
            for name, value in reduction_params.items()
        ):
            log(self.verbose, f"Reusing cached {solver.upper()} coordinates")
            self.min_dist = min_dist
            return self.Y_

        with stage(self.verbose, f"{solver.upper()} reduction", objects=self.distances_.shape[0], components=n_components):
            if solver == "mds":
                landmark_embedding = metric_mds(
                    self.distances_,
                    n_components=n_components,
                    random_state=self.random_state,
                    verbose=max(self.verbose - 1, 0),
                )
            elif solver == "umap":
                landmark_embedding = umap_from_distances(
                    self.distances_,
                    n_components=n_components,
                    min_dist=min_dist,
                    random_state=self.random_state,
                    verbose=self.verbose,
                )
            else:
                initial = classic(self.distances_, n_components=n_components,
                                  random_state=self.random_state)
                squared = self.squared_distances_
                similarities = squared.copy()
                similarities *= -0.5
                np.exp(similarities, out=similarities)
                landmark_embedding = elastic_embedding(
                    similarities,
                    squared,
                    n_components=n_components,
                    Y_init=initial,
                    lamb=params.get("lamb", 1),
                    num_iters=params.get("num_iters", 100),
                    random_state=self.random_state,
                    verbose=self.verbose,
                )
                del initial, similarities, squared

        if self.mode_ == 2:
            # Interpolate landmark coordinates back to all original samples.
            with stage(self.verbose, "Landmark coordinate interpolation", cells=self.n_samples):
                embedding = np.asarray(self.Pnm @ landmark_embedding)
        else:
            embedding = np.asarray(landmark_embedding)
        self.n_components = n_components
        self.solver = solver
        self.min_dist = min_dist
        self.Ym = landmark_embedding if self.mode_ == 2 else None
        self.Y_ = embedding
        self._reduction_params_ = reduction_params
        return self.Y_

    def _pseudotime_cells(self, cells, *, name: str):
        """Resolve endpoint identifiers to unique indices without neighbor expansion."""
        check_cell_identifier_types(cells, name=name)
        values = None if cells is None else np.asarray(cells)
        if values is not None and values.dtype.kind == "O":
            elements = values.reshape(-1)
            if all(isinstance(value, (int, np.integer)) for value in elements):
                values = np.asarray(elements, dtype=np.int64)
        if values is not None and values.dtype.kind in "OUS":
            checked = self._cell_indices_from_names(cells, name=name)
        else:
            checked = check_cell_indices(values, self.n_samples, name=name)
        checked = np.unique(checked)
        return checked, checked

    def _cell_indices_from_names(self, cell_names, *, name: str):
        """Look up fitted cell names independently of cell-type labels."""
        if self.cell_names_ is None:
            raise ValueError(
                f"{name} contains cell names, but no cell_names were provided "
                "when fitting DTNE."
            )
        requested = np.asarray(cell_names)
        if requested.ndim == 0:
            requested = requested.reshape(1)
        requested = np.asarray(
            [str(value) for value in requested.reshape(-1).tolist()]
        )
        if requested.size == 0:
            raise ValueError(f"{name} must contain at least one cell name.")

        available = np.asarray(self.cell_names_)
        found = np.isin(requested, available)
        if not found.all():
            missing = ", ".join(repr(value) for value in requested[~found].tolist())
            raise ValueError(
                f"{name} contains unknown cell names: {missing}."
            )
        return np.flatnonzero(np.isin(available, requested))

    def order_cells(self, **params):
        """Infer pseudotime from root and optional terminal cells.

        ``root_cells`` and ``terminal_cells`` accept integer cell indices or
        string/object cell names supplied during fit. Integer inputs are always
        interpreted as indices. Both full and landmark modes use exactly the
        specified cells, without expanding single endpoints to their neighbors.
        Repeated names or indices are counted only once.
        Boolean values, boolean masks, and floating-point indices are rejected.
        """
        if self.distances_ is None:
            raise RuntimeError("DTNE must be fitted before ordering cells.")
        unsupported = {"root_labels", "terminal_labels"}.intersection(params)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(
                f"{names} is not supported; pass cell names through "
                "root_cells or terminal_cells."
            )
        self._check_method_params(params, {"root_cells", "terminal_cells", "batch_size"}, "order_cells")
        batch_size = params.get("batch_size", self.pseudotime_batch_size)
        check_integer(batch_size, name="batch_size")

        root_value = params.get("root_cells", self.root_cells)
        root_cells, roots = self._pseudotime_cells(
            root_value, name="root_cells"
        )

        terminal_value = params.get("terminal_cells", self.terminal_cells)
        if terminal_value is None:
            terminal_cells = terminals = None
        else:
            terminal_cells, terminals = self._pseudotime_cells(
                terminal_value, name="terminal_cells"
            )

        with stage(self.verbose, "Pseudotime calculation", cells=self.n_samples, roots=len(roots), terminals=0 if terminals is None else len(terminals)):
            if self.mode_ == 1:
                pseudotime = pseudotime_from_distances(
                    self.distances_, roots, terminals
                )
            else:
                pseudotime = landmark_pseudotime(
                    self.Pnm,
                    self.R,
                    roots,
                    terminals,
                    batch_size=batch_size,
                    verbose=self.verbose,
                )
        # Commit endpoints together with their result; failed calls keep prior state.
        self.root_cells = root_cells
        self.terminal_cells = terminal_cells
        self.pseudotime_ = self.df_times = pseudotime
        return pseudotime

    def _extreme_cell_scores(
        self,
        candidates,
        references,
        *,
        aggregation: str,
        batch_size: int,
    ):
        """Aggregate candidate-reference distances in bounded memory blocks."""
        candidates = np.asarray(candidates, dtype=np.int64)
        references = np.asarray(references, dtype=np.int64)
        scores = np.empty(candidates.size, dtype=np.float64)
        operator = None if self.mode_ == 1 else sparse.csr_matrix(self.Pnm)
        progress = Progress(self.verbose, "Endpoint scoring pairs", candidates.size * references.size)
        processed = 0

        for candidate_start in range(0, candidates.size, batch_size):
            candidate_stop = min(candidate_start + batch_size, candidates.size)
            candidate_batch = candidates[candidate_start:candidate_stop]
            accumulated = (
                np.zeros(candidate_batch.size, dtype=np.float64)
                if aggregation == "mean"
                else np.full(candidate_batch.size, -np.inf, dtype=np.float64)
            )
            candidate_rank = (
                None
                if self.mode_ == 1
                else np.asarray(operator[candidate_batch] @ self.R)
            )

            for reference_start in range(0, references.size, batch_size):
                reference_stop = min(reference_start + batch_size, references.size)
                reference_batch = references[reference_start:reference_stop]
                if self.mode_ == 1:
                    block = self.distances_[
                        np.ix_(candidate_batch, reference_batch)
                    ]
                else:
                    reference_rank = np.asarray(operator[reference_batch] @ self.R)
                    block = cross_manifold_distances(candidate_rank, reference_rank)
                if aggregation == "mean":
                    accumulated += block.sum(axis=1)
                else:
                    accumulated = np.maximum(accumulated, block.max(axis=1))
                processed += candidate_batch.size * reference_batch.size
                progress.update(processed)

            if aggregation == "mean":
                accumulated /= references.size
            scores[candidate_start:candidate_stop] = accumulated
        return scores

    @staticmethod
    def _label_mask(labels, value):
        """Match a scalar label, treating all missing values as one category."""
        missing = np.fromiter(
            (is_missing_label(item) for item in labels), dtype=bool, count=labels.size
        )
        if is_missing_label(value):
            return missing
        matched = np.zeros(labels.size, dtype=bool)
        matched[~missing] = np.isin(labels[~missing], [value])
        return matched

    def select_extreme_cells(
        self,
        label,
        n_cells: int = 3,
        *,
        reference_label=None,
        aggregation: str = "mean",
        return_type: str = "auto",
        return_scores: bool = False,
        batch_size: int = 1024,
    ):
        """Select labeled cells farthest from a reference population.

        Candidates are cells whose fitted label equals ``label``. By default,
        all cells with other labels form the reference population. Set
        ``reference_label`` to compare against one specific population.
        Missing labels (None, NaN, NaT, or pandas.NA) form one category.
        Use NaN as ``reference_label`` to select that category; None retains
        its default meaning of comparing against all other labels.

        ``return_type`` may be ``"auto"``, ``"names"``, ``"indices"``, or
        ``"both"``. Auto returns cell names when available and indices
        otherwise. Both returns a dictionary containing names, indices, and
        distance scores.
        """
        if self.distances_ is None:
            raise RuntimeError("DTNE must be fitted before selecting cells.")
        if self.labels_ is None:
            raise ValueError(
                "Cell labels are required. Provide labels or label_key when fitting."
            )
        if isinstance(n_cells, bool) or not isinstance(n_cells, (int, np.integer)):
            raise TypeError("n_cells must be a positive integer.")
        n_cells = int(n_cells)
        if n_cells < 1:
            raise ValueError("n_cells must be a positive integer.")
        if isinstance(batch_size, bool) or not isinstance(
            batch_size, (int, np.integer)
        ):
            raise TypeError("batch_size must be a positive integer.")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(return_scores, (bool, np.bool_)):
            raise TypeError("return_scores must be a boolean.")
        aggregation = check_choice(
            aggregation, ("mean", "max"), name="aggregation"
        )
        return_type = check_choice(
            return_type,
            ("auto", "names", "indices", "both"),
            name="return_type",
        )
        if return_type == "names" and self.cell_names_ is None:
            raise ValueError(
                "return_type='names' requires cell_names to be provided "
                "when fitting DTNE."
            )

        label_value = np.asarray(label)
        if label_value.ndim != 0:
            raise ValueError("label must be a single fitted label value.")
        candidate_mask = self._label_mask(self.labels_, label_value.item())
        candidates = np.flatnonzero(candidate_mask)
        if candidates.size == 0:
            raise ValueError(f"Unknown fitted label: {label_value.item()!r}.")
        if n_cells > candidates.size:
            raise ValueError(
                f"n_cells cannot exceed the {candidates.size} cells with label "
                f"{label_value.item()!r}."
            )

        if reference_label is None:
            references = np.flatnonzero(~candidate_mask)
            reference_description = "cells with other labels"
        else:
            reference_value = np.asarray(reference_label)
            if reference_value.ndim != 0:
                raise ValueError(
                    "reference_label must be a single fitted label value."
                )
            reference_mask = self._label_mask(self.labels_, reference_value.item())
            references = np.flatnonzero(reference_mask)
            if references.size == 0:
                raise ValueError(
                    f"Unknown reference_label: {reference_value.item()!r}."
                )
            if np.array_equal(candidate_mask, reference_mask):
                raise ValueError("reference_label must differ from label.")
            reference_description = f"reference_label={reference_value.item()!r}"
        if references.size == 0:
            raise ValueError(
                f"Cannot score label {label_value.item()!r}: no "
                f"{reference_description} are available."
            )

        with stage(self.verbose, "Endpoint selection", candidates=candidates.size, references=references.size, selected=n_cells):
            scores = self._extreme_cell_scores(
                candidates,
                references,
                aggregation=aggregation,
                batch_size=batch_size,
            )
        order = np.lexsort((candidates, -scores))[:n_cells]
        selected_indices = candidates[order].copy()
        selected_scores = scores[order].copy()
        selected_names = (
            None
            if self.cell_names_ is None
            else np.asarray(self.cell_names_)[selected_indices].copy()
        )

        if return_type == "both":
            return {
                "indices": selected_indices,
                "names": selected_names,
                "scores": selected_scores,
            }
        if return_type == "names":
            selected = selected_names
        elif return_type == "indices":
            selected = selected_indices
        else:
            selected = (
                selected_names
                if selected_names is not None
                else selected_indices
            )
        return (selected, selected_scores) if return_scores else selected

    def cluster_cells(self, **params):
        """Cluster manifold distances and map landmark labels back to cells.

        Notes
        -----
        Landmark DBSCAN uses group sizes as weights, so ``min_samples`` counts
        represented cells. Distances and cluster assignments remain approximate.
        The default cluster count is at most eight and never exceeds the
        number of available samples or landmarks.
        """
        self._check_method_params(params, {"cluster_method", "n_clusters", "eps", "min_samples", "random_state"}, "cluster_cells")
        if self.distances_ is None:
            raise RuntimeError("DTNE must be fitted before clustering cells.")
        method = check_choice(params.get("cluster_method", "agglo"),
                              ("agglo", "hiera", "kmedoids", "dbscan"), name="cluster_method")
        n_clusters = params.get("n_clusters", min(8, self.distances_.shape[0]))
        if method != "dbscan":
            check_integer(n_clusters, name="n_clusters")
            if n_clusters > self.distances_.shape[0]:
                objects = "landmarks" if self.mode_ == 2 else "samples"
                raise ValueError(f"n_clusters={n_clusters} exceeds the {self.distances_.shape[0]} available {objects}.")
        with stage(self.verbose, "Clustering", method=method, objects=self.distances_.shape[0]):
            labels = cluster_precomputed(
                self.distances_,
                method=method,
                n_clusters=n_clusters,
                eps=params.get("eps", 0.5),
                min_samples=params.get("min_samples", 5),
                random_state=params.get("random_state", self.random_state),
                sample_weight=(
                    np.bincount(self.cluster_labels, minlength=self.n_landmark_)
                    if self.mode_ == 2 and method == "dbscan"
                    else None
                ),
            )
        clusters = (
            labels
            if self.mode_ == 1
            else np.asarray(labels)[np.asarray(self.cluster_labels)]
        )
        self.clusters_ = np.asarray(clusters)
        return self.clusters_

    def fit(
        self,
        X,
        labels=None,
        *,
        cell_names=None,
        gene_names=None,
        **params,
    ):
        """Fit DTNE and retain optional cell and gene names.

        ``beta``, ``epsilon``, ``max_iter``, ``random_state``, and ``verbose``
        in ``params`` override the constructor settings for this fit only.
        Unknown parameters are rejected.
        """
        params = self._normalize_diffusion_alias(params)
        allowed = (self._DIFFUSION_PARAMS - {"X"}) | self._REDUCTION_PARAMS | {"label_key"}
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError("Unknown DTNE fit parameters: " + ", ".join(unknown))
        self._validate_parameter_values(params)
        if "include_self" in params:
            check_boolean(params["include_self"], name="include_self")
        solver = check_choice(params.get("solver", self.solver), self._SOLVERS, name="solver")
        with stage(params.get("verbose", self.verbose), "Fit"):
            label_key = params.pop("label_key", None)
            with stage(params.get("verbose", self.verbose), "Input validation"):
                X, checked_labels, checked_cell_names, checked_gene_names = (
                    self._extract_input(
                        X,
                        labels,
                        label_key,
                        cell_names,
                        gene_names,
                    )
                )
            # Reject impossible output dimensions before constructing the graph
            # or discarding results from a previous successful fit.
            n_landmarks = self._resolve_landmarks(
                params.get("n_landmark", self.n_landmark), X.shape[0]
            )
            n_components = params.get("n_components", self.n_components)
            if solver in {"mds", "ee"} and n_components > n_landmarks:
                raise ValueError(
                    f"n_components={n_components} exceeds the {n_landmarks} available "
                    "samples or landmarks for MDS/EE."
                )
            self._reset_fitted_state()
            self.X = X
            self.labels_ = self.X_labels = checked_labels
            self.cell_names_ = checked_cell_names
            self.gene_names_ = checked_gene_names
            random_state, verbose = self.random_state, self.verbose
            self.random_state = params.pop("random_state", random_state)
            self.verbose = params.pop("verbose", verbose)
            try:
                self.compute_dist_matrix(**{key: value for key, value in params.items() if key in self._DIFFUSION_PARAMS})
                self.reduce_dim(**{key: value for key, value in params.items() if key in self._REDUCTION_PARAMS})
            finally:
                self.random_state, self.verbose = random_state, verbose
            return self

    def fit_transform(
        self,
        X,
        labels=None,
        *,
        cell_names=None,
        gene_names=None,
        **params,
    ):
        """Fit DTNE and return its low-dimensional coordinates."""
        return self.fit(
            X,
            labels=labels,
            cell_names=cell_names,
            gene_names=gene_names,
            **params,
        ).Y_

    def plot(self, color_by: str = "labels", values=None, **kwargs):
        """Plot the embedding colored by labels, pseudotime, or clusters.

        Parameters
        ----------
        color_by : {"labels", "pseudotime", "clusters"}, default="labels"
            Fitted result used to color the embedding.
        values : array-like, optional
            Values that override the fitted result selected by ``color_by``.
        **kwargs
            Additional arguments passed to :func:`dtne.plotting.plot_embedding`.
        """
        from .plotting import plot_embedding

        if self.Y_ is None:
            raise RuntimeError("DTNE must be fitted before plotting.")
        color_by = check_choice(
            color_by,
            ("labels", "pseudotime", "clusters"),
            name="color_by",
        )
        configurations = {
            "labels": (
                self.labels_,
                "auto",
                "Original labels",
                "No original labels are available.",
            ),
            "pseudotime": (
                self.pseudotime_,
                "continuous",
                "DTNE pseudotime",
                "No pseudotime is available. Call order_cells first.",
            ),
            "clusters": (
                self.clusters_,
                "categorical",
                "DTNE clusters",
                "No clusters are available. Call cluster_cells first.",
            ),
        }
        fitted_values, kind, title, missing_message = configurations[color_by]
        values = (
            fitted_values
            if values is None
            else check_labels(values, self.n_samples, name="values")
        )
        if values is None:
            raise ValueError(missing_message)
        kwargs.setdefault("kind", kind)
        kwargs.setdefault("title", title)
        return plot_embedding(self.Y_, values, **kwargs)

    def plot_results(self, *, labels=None, pseudotime=None, clusters=None, **kwargs):
        """Plot available labels, pseudotime, and clusters beside each other."""
        from .plotting import plot_dtne_results

        if self.Y_ is None:
            raise RuntimeError("DTNE must be fitted before plotting.")
        return plot_dtne_results(
            self.Y_,
            labels=self.labels_ if labels is None else labels,
            pseudotime=self.pseudotime_ if pseudotime is None else pseudotime,
            clusters=self.clusters_ if clusters is None else clusters,
            **kwargs,
        )


__all__ = ["DTNE"]
