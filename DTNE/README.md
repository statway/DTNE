# Diffusive Topology Neighbor Embedding


DTNE (diffusive topology neighbor embedding) is a python tool which establish a novel manifold learning framework to leverage the constructed diffusion manifold matrix for three key tasks: low-dimensional visualization, pseudotime inference, and cluster identification.

### Installation

install the package DTNE by running the following command in the terminal:
"pip install .'

### Quick Start

you can run DTNE as follows If you have loaded a data matrix 'X' in Python :
```
from dtne import *

embedding = DTNE(k_neighbors = 10,l=2) 
Y = embedding.fit_transform(X)

dtne_pseudotime = embedding.order_cells(root_cells=[0])
dtne_cluster = embedding.cluster_cells(n_clusters=8)
```

## Guided Tutorial

For more details on DTNE, see 'notebooks'.