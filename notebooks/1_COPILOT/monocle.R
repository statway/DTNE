setwd('D:\\0_dtne_demo\\1_COPILOT')

rm(list=ls())

# Read in atlas
# rc.integrated <- readRDS('./COPILOT_RDS/Root_Atlas.rds')

suppressMessages(library(monocle3, quietly=TRUE))
suppressMessages(library(ggplot2))
# suppressMessages(library(anndata))

cds <- load_monocle_objects(directory_path='./my_cds')
cds <- preprocess_cds(cds, method = 'PCA',num_dim = 50,norm_method='none',scaling= FALSE)

cds <- reduce_dimension(cds,reduction_method = 'UMAP', preprocess_method ='PCA')
cds <- cluster_cells(cds,cluster_method = 'leiden',resolution=0.00001)
cds <- learn_graph(cds,use_partition =FALSE, learn_graph_control = list(ncenter=1000))

plot_cells(cds, color_cells_by = "cluster", group_cells_by = "cluster", 
           +            group_label_size = 4)

cds <- order_cells(cds,root_cells=colnames(cds)[3632])

plot_cells(cds,
           color_cells_by = "pseudotime",
           label_cell_groups=FALSE,
           label_leaves=FALSE,
           label_branch_points=FALSE,
           graph_label_size=1.5)

save_monocle_objects(cds=cds, directory_path='./my_cds',hdf5_assays = TRUE)

pseudotime(cds)

write.csv(as.array(cds@clusters$UMAP$clusters),'rc.integrated.monocle_leiden.csv')
write.csv(as.array(cds@clusters$UMAP$partitions),'rc.integrated.monocle_partitions.csv')
write.csv(pseudotime(cds)/max(pseudotime(cds)),'rc.integrated.monocle_pseudotime.csv')
          