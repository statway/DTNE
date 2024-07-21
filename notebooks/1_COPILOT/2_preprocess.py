import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

adata = sc.read_h5ad('rc.integrated.scale.data.h5ad')
adata = sc.AnnData(X=adata.X.T)

pca = sc.read_h5ad('rc.integrated.pca.data.h5ad')
pca.X.shape

umap = sc.read_h5ad('rc.integrated.umap.data.h5ad')
umap.X.shape

features = pd.read_csv("features.csv", sep=",",index_col=0)
cell_names = pd.read_csv("cell_names.csv",index_col=0)

adata.var_names = features.values.reshape(-1)
adata.obs_names = cell_names.values.reshape(-1)

consensus_time = pd.read_csv("rc.integrated.consensus.time.csv",index_col=0)
adata.uns["consensus_time"] = consensus_time.values.reshape(-1)

cell_type = pd.read_csv("cell_type.csv",index_col=0)
sample_labels = cell_type.values.reshape(-1)
data_clusters_set = set(sample_labels)
zip_types = zip(sorted(data_clusters_set),range(len(data_clusters_set)))
dict_types = dict(zip_types)
cell_clusters  =  [dict_types[i] for i in sample_labels]

adata.obsm["X_pca"] = pca.X
adata.obsm["umap"] = umap.X
adata.obs["cell_type"] = cell_type.values.reshape(-1)
adata.obs["cell_type2"] = np.array(cell_clusters)

adata.uns["iroot"] = 3631 # 59365,82261,106047

umap = adata.obsm['umap']
consensus_time = adata.uns['consensus_time']
plt.scatter(umap[:,0],umap[:,1], c=consensus_time,s =1,cmap="Spectral")

adata.write('rc.integrated.data.h5ad', compression="gzip")