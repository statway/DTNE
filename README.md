# DTNE - Diffusive Topology Neighbor Embedding

DTNE (Diffusive Topology Neighbor Embedding) is a Python tool that implements a novel manifold learning framework. By leveraging a diffusive process, DTNE constructs a manifold distance matrix, enabling key analyses such as dimensionality reduction, pseudotime ordering, and cluster identification, providing valuable insights into complex datasets.

## Features

* **Dimensionality Reduction**: By preserving manifold geodesic distances, DTNE provides accurate low-dimensional projections of high-dimensional single-cell data.
* **Pseudotime Inference**: Infer developmental trajectories by leveraging the manifold distance matrix, improving the identification of lineage progression.
* **Clustering**: DTNE enables clustering by utilizing the manifold distance matrix, which captures both local and global data structures.

## Installation

install the package DTNE by running the following command in the terminal: `pip install .`

## Quick Start

DTNE requires input data in the form of a high-dimensional matrix. Suppose you have loaded a single-cell data `X` in Python.

1. Dimensionality Reduction

    DTNE can be used for dimensionality reduction:
    ```
    from dtne import *
    dtne_operator = DTNE(k_neighbors = 10,l=2) 
    Y = dtne_operator.fit_transform(X)
    ```
2. Pseudotime Inference

    DTNE allows for pseudotime inference to analyze cell development trajectories:
    ```
    dtne_pseudotime = dtne_operator.order_cells(root_cells=[0])
    ```
3. Clustering
    DTNE supports clustering based on the computed manifold distances:
    ```
    dtne_cluster = dtne_operator.cluster_cells(n_clusters=8)
    ```

## Tutorial and Example Notebooks
Several example Jupyter notebooks are provided in the `notebooks/` directory, demonstrating DTNE’s usage on various single-cell datasets.

## Citation
If you use DTNE in your research, please cite the paper.

## License
DTNE is licensed under the MIT License.