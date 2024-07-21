setwd('E:/DTNE/notebooks/1_Paul')

suppressMessages(library(monocle3, quietly=TRUE))
suppressMessages(library(ggplot2))
suppressMessages(library(anndata))


adata <- read_h5ad('paul15.h5ad')
X <- as.matrix(t(adata$X))
dim(X)
X[1:5,1:5]

cell_names <- colnames(X)
gene_names <- row.names(X)
pd = data.frame(rownames = cell_names)
rownames(pd) = cell_names
fd = data.frame(gene_short_name = gene_names)
rownames(fd) = gene_names


cds <- new_cell_data_set(X,
                         cell_metadata = pd,
                         gene_metadata = fd)

cds <- preprocess_cds(cds, method = 'PCA',num_dim = 50,norm_method='none')
cds <- reduce_dimension(cds,reduction_method = 'UMAP', umap.min_dist = 0.5, umap.n_neighbors = 10,preprocess_method ='PCA')
cds <- cluster_cells(cds,cluster_method = 'leiden',resolution=1)
cds <- learn_graph(cds,use_partition =FALSE)
cds <- order_cells(cds,root_cells=rownames(pd)[841])
pseudotime <- pseudotime(cds)/max(pseudotime(cds))

plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5)

write.csv(pseudotime(cds)/max(pseudotime(cds)),'paul.monocle_pseudotime.csv')
