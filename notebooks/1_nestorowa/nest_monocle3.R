suppressMessages(library(monocle3, quietly=TRUE))
suppressMessages(library(ggplot2))

setwd('E:/DTNE/notebooks/1_nestorowa/')

sce.nest = readRDS('sce.nest.rds')
sce.nest

label <- sce.nest@colData$label
gene_names <- rownames(sce.nest)
cell_names <- colnames(sce.nest)

logcounts <- sce.nest@assays@data$logcounts
X_pca <- sce.nest@int_colData$reducedDims$PCA

pd = data.frame(rownames = colnames(logcounts))
rownames(pd) = colnames(logcounts)

fd = data.frame(gene_short_name = rownames(logcounts))
rownames(fd) = rownames(logcounts)

cds <- new_cell_data_set(logcounts,
                         cell_metadata = pd,
                         gene_metadata = fd)

cds <- preprocess_cds(cds, method = 'PCA',num_dim = 50,norm_method='none')
cds <- reduce_dimension(cds,reduction_method = 'UMAP', preprocess_method ='PCA')
cds <- cluster_cells(cds,cluster_method = 'leiden',resolution=0.005)

cds <- learn_graph(cds,use_partition =FALSE)

cds <- order_cells(cds,root_cells=rownames(pd)[658])

plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5)

plot_cells(cds,
           color_cells_by = "cluster",
           label_cell_groups=FALSE,
           label_leaves=TRUE,
           label_branch_points=TRUE,
           graph_label_size=1.5)

write.csv(as.array(cds@clusters$UMAP$clusters),'nest.monocle_leiden.csv')
write.csv(pseudotime(cds)/max(pseudotime(cds)),'nest.monocle_pseudotime.csv')
