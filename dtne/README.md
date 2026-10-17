# DTNE

DTNE (Diffusive Topology Neighbor Embedding) builds a diffusion manifold
distance for visualization, pseudotime inference, and clustering of
single-cell data.

## Installation and packaging

From the repository root, install the source package with:

```bash
python -m pip install ./dtne
```

Optional functionality can be installed with `./dtne[visualization]`,
`./dtne[single-cell]`, `./dtne[clustering]`, or `./dtne[all]` (quote these
paths when using shells that interpret square brackets).

Build a wheel and source distribution with:

```bash
python -m pip install build
python -m build ./dtne
```

The artifacts are written to `dtne/dist/`. Install the version 0.9.0 wheel with:

```bash
python -m pip install ./dtne/dist/dtne-0.9.0-py3-none-any.whl
```

## Basic usage

Prepare a numeric feature matrix `X` with shape `(n_cells, n_features)` and
a matching one-dimensional `labels` array. Rows must refer to the same
cells in the same order. Dense NumPy arrays, SciPy sparse matrices, pandas
DataFrames, and AnnData objects are supported. DTNE builds the graph,
optimizes diffusion distributions, computes manifold distances, and embeds
the cells in one call:

```python
from dtne import DTNE

model = DTNE(
    n_components=2,
    k_neighbors=15,
    kernel="box",
    solver="mds",
    l=2,
    n_landmark=None,
    random_state=123,
    verbose=1,
)
embedding = model.fit_transform(X, labels=labels)
```

`embedding` and `model.Y_` contain coordinates for every input cell, including
when landmarks are used. `fit(X, ...)` performs the same computation and
returns the model. Labels are optional for fitting; provide them when selecting
endpoints by label or plotting annotated groups. Pass
`cell_names=cell_names` to `fit_transform()` when cell names are available.

The example explicitly sets the diffusion start step to `l=8`; the default
`l=None` estimates it from the spectrum. With `l2=None`, the target step is
`l + 1`. `n_landmark=None` selects full mode below 5,000 cells, 1,000 landmarks
for 5,000–9,999 cells, and 2,000 landmarks for larger inputs. Set
`n_landmark=X.shape[0]` to request full mode explicitly. Landmark mode still
uses every input cell; it approximates the diffusion calculation.

For UMAP coordinates, use `solver="umap"` and optionally `min_dist=0.3`.
Install its dependencies with `python -m pip install "./dtne[visualization]"`
from the repository root. For input features with at most 500
dimensions, the default `n_pca=None` leaves the features unchanged; above
500 dimensions it automatically applies PCA or sparse SVD before graph
construction.

### Select endpoints and infer pseudotime

Choose two distinct values from the fitted labels as `root_label` and
`terminal_label`. Select one endpoint from each group by its mean DTNE
distance from the opposite group:

```python
roots = model.select_extreme_cells(
    root_label,
    n_cells=1,
    reference_label=terminal_label,
    aggregation="mean",
    return_type="indices",
)
terminals = model.select_extreme_cells(
    terminal_label,
    n_cells=1,
    reference_label=root_label,
    aggregation="mean",
    return_type="indices",
)
pseudotime = model.order_cells(
    root_cells=roots,
    terminal_cells=terminals,
    batch_size=2048,
)
```

Selection returns the cells farthest from the reference group. The endpoint
labels are supplied by the caller. `pseudotime` and `model.pseudotime_` contain one value
per input cell in `[0, 1]`. Landmark mode computes cell-level distances in
batches rather than allocating a full cell-by-cell distance matrix.

When endpoints are already known, pass their indices or fitted cell names
directly. To use only a root, explicitly clear any terminal cells retained
from an earlier call:

```python
pseudotime = model.order_cells(root_cells=[0], terminal_cells=None)
```

### Cluster and visualize

```python
clusters = model.cluster_cells(n_clusters=8)
model.plot("labels")
model.plot("pseudotime")
model.plot("clusters")
```

The example uses the default agglomerative clustering. The explicit cluster
count must not exceed the number of samples in full mode or the number of
landmarks in landmark mode. Omit `n_clusters` to use `min(8, n_objects)`.
Plotting requires Matplotlib, included in the `visualization` extra.
`model.plot_results()` combines the available views.

## Parameters and execution

| Parameter | Default | Meaning |
| --- | --- | --- |
| `solver` | `"mds"` | `"mds"`, `"umap"`, or `"ee"` |
| `l` / `l2` | `None` / `None` | Estimated start step / start step + 1 |
| `beta` | `0.1` | Damping optimization learning rate |
| `epsilon` | `0.01` | Damping convergence tolerance |
| `max_iter` | `30` | Maximum damping iterations |
| `random_state` | `123` | Random seed forwarded to the backends |
| `min_dist` | `None` | UMAP minimum distance; `None` resolves to `0.3` |
| `pseudotime_batch_size` | `4096` | Cell batch size for landmark pseudotime |

`l1` aliases `l`; conflicting values raise an error. Fit-time `beta`,
`epsilon`, `max_iter`, `random_state`, and `verbose` apply only to that fit.
Use `set_params()` to update defaults and `get_params()` to inspect them.
Computational methods reject unknown parameters.

`verbose=0` is quiet, `1` reports stages, timing, cache reuse, and optimization
summaries, and `2` adds iteration and batch details. Warnings remain enabled.
Check `n_iter_`, `converged_`, and `max_change_` for damping status; reaching
`max_iter` without convergence emits a warning.

## Labels and endpoints

Names are inferred from DataFrame indices/columns or AnnData names. For array
inputs, supply `cell_names` and `gene_names` explicitly. Use `label_key="label"`
to read labels from a DataFrame column or AnnData observation column.

Endpoints accept integer indices or fitted cell names, independently of
labels. Duplicates are removed; floats and booleans are rejected. Convert
boolean masks with `np.flatnonzero(mask)`.

`select_extreme_cells()` compares against all other labels by default.
Its default return is names when available, otherwise indices; select
`return_type="indices"`, `"names"`, or `"both"` explicitly as needed.

## Reuse and memory

`set_params()` invalidates results from the earliest affected stage:

| Changed setting | Invalidated results |
| --- | --- |
| Active graph settings | Graph, diffusion, coordinates, pseudotime, clusters |
| Diffusion settings | Diffusion, coordinates, pseudotime, clusters |
| Reduction settings | Coordinates only |
| Unchanged or unused settings | None |

Dependencies follow the active kernel and solver: `delta` affects `box2`,
`gauss`, and `mix`; `alpha` affects `gauss` and `mix`; `min_dist` affects UMAP.
A seed change rebuilds stochastic preprocessing/graphs; otherwise it starts
invalidation at diffusion. Unused settings are saved for later use.

`learn_vectors()`, `compute_dist_matrix()`, and `reduce_dim()` reuse matching
results. A parameter-free distance request returns the current cache; explicit
requests compare effective settings, with omitted options using current
defaults. `compute_kernel_matrix(X)` always rebuilds the graph. Replacement
input clears prior metadata; passing the current `model.X` retains it.

`distances_` (also `dists`) stores ordinary manifold distances. The
`squared_distances_` and `similarities_` properties generate independent
arrays on access rather than retaining additional square matrices.

The graph `K` and landmark mapping `Pnm` stay sparse. Full-mode `R` is
released after distance conversion; `learn_vectors()` can restore it.
Landmark mode retains `R` and `Pnm` for batched cell-level analysis and
reuses grouping for diffusion-only updates. Matching full-mode rank
restoration with a fixed integer seed preserves downstream results.
External array references remain alive until released by the caller.
