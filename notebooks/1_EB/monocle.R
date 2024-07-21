rm(list=ls())
setwd('D:\\0_dtne_demo\\1_EB')

suppressMessages(library(monocle3, quietly=TRUE))
suppressMessages(library(ggplot2))
suppressMessages(library(anndata))

EBT_counts <- read_h5ad('EBT_counts_sqrt.h5ad')
X <- as.matrix(t(EBT_counts$X))
dim(X)
X[1:5,1:5]

cell_names <- colnames(X)
gene_names <- row.names(X)
pd = data.frame(rownames = cell_names)
rownames(pd) = cell_names
fd = data.frame(gene_short_name = gene_names)
rownames(fd) = gene_names
cds <- new_cell_data_set(X, cell_metadata = pd, gene_metadata = fd)

cds <- preprocess_cds(cds, method = 'PCA',num_dim = 50,norm_method='none',scaling= FALSE)
cds <- reduce_dimension(cds,reduction_method = 'UMAP', preprocess_method ='PCA')
cds <- cluster_cells(cds,cluster_method = 'leiden',resolution=0.0001)
cds@clusters$UMAP$clusters

cds <- learn_graph(cds,use_partition =FALSE) # learn_graph_control = list(ncenter=1000)
plot_cells(cds, color_cells_by = "cluster", group_cells_by = "cluster", group_label_size = 4)


cds <- order_cells(cds,root_cells=colnames(cds)[1258])
plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5)

write.csv(as.array(cds@clusters$UMAP$clusters),'eb.monocle_leiden.csv')
write.csv(pseudotime(cds)/max(pseudotime(cds)),'eb.monocle_pseudotime.csv')
