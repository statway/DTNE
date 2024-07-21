setwd('E:/DTNE/notebooks/1_lymphoid')

library(Signac)
library(Seurat)
library(SeuratWrappers)
library(monocle3)
library(Matrix)
library(ggplot2)
library(patchwork)
library(magrittr)
set.seed(1234)

filepath <- "./GSE129785/GSE129785_scATAC-Hematopoiesis-CD34"

# peaks <- read.table("./GSE129785/GSE129785_scATAC-Hematopoiesis-All.peaks.txt.gz", header = TRUE)

peaks <- read.table(paste0(filepath, ".peaks.txt.gz"), header = TRUE)
cells <- read.table(paste0(filepath, ".cell_barcodes.txt.gz"), header = TRUE, stringsAsFactors = FALSE)
rownames(cells) <- make.unique(cells$Barcodes)

mtx <- readMM(file = paste0(filepath, ".mtx"))
# mtx <- as(object = mtx, Class = "dgCMatrix")   #########################################
mtx <- as(object = mtx, Class = "CsparseMatrix") 


colnames(mtx) <- rownames(cells)
rownames(mtx) <- peaks$Feature


bone_assay <- CreateChromatinAssay(
  counts = mtx,
  min.cells = 5,
  fragments = "./GSE129785/GSM3722029_CD34_Progenitors_Rep1_fragments.tsv.gz",
  sep = c("_", "_"),
  genome = "hg19"
)
bone <- CreateSeuratObject(
  counts = bone_assay,
  meta.data = cells,
  assay = "ATAC"
)

# The dataset contains multiple cell types
# We can subset to include just one replicate of CD34+ progenitor cells
bone <- bone[, bone$Group_Barcode == "CD34_Progenitors_Rep1"]

# add cell type annotations from the original paper
cluster_names <- c("HSC",   "MEP",  "CMP-BMP",  "LMPP", "CLP",  "Pro-B",    "Pre-B",    "GMP", "MDP",    "pDC",  "cDC",  "Monocyte-1",   "Monocyte-2",   "Naive-B",  "Memory-B",
                   "Plasma-cell",    "Basophil", "Immature-NK",  "Mature-NK1",   "Mature-NK2",   "Naive-CD4-T1","Naive-CD4-T2",   "Naive-Treg",   "Memory-CD4-T", "Treg", "Naive-CD8-T1", "Naive-CD8-T2",
                   "Naive-CD8-T3",   "Central-memory-CD8-T", "Effector-memory-CD8-T",    "Gamma delta T")
num.labels <- length(cluster_names)
names(cluster_names) <- paste0( rep("Cluster", num.labels), seq(num.labels) )
bone$celltype <- cluster_names[as.character(bone$Clusters)] %>%  unname() #######################################

bone[["ATAC"]]




library(EnsDb.Hsapiens.v75)

# extract gene annotations from EnsDb
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v75) # BiocManager::install("Biostrings")

# change to UCSC style since the data was mapped to hg19
seqlevels(annotations) <- paste0('chr', seqlevels(annotations))
genome(annotations) <- "hg19"

# add the gene information to the object
Annotation(bone) <- annotations


# Quality control

bone <- TSSEnrichment(bone)
bone <- NucleosomeSignal(bone)
bone$blacklist_fraction <- FractionCountsInRegion(bone, regions = blacklist_hg19)

VlnPlot(
  object = bone,
  features = c("nCount_ATAC", "TSS.enrichment", "nucleosome_signal", "blacklist_fraction"),
  pt.size = 0.1,
  ncol = 4
)


bone <- bone[, (bone$nCount_ATAC < 50000) &
               (bone$TSS.enrichment > 2) & 
               (bone$nucleosome_signal < 5)]



# Dataset preprocessing

bone <- FindTopFeatures(bone, min.cells = 10)
bone <- RunTFIDF(bone)
bone <- RunSVD(bone, n = 100)
DepthCor(bone)

# the first LSI dimension is highly correlated with sequencing depth in that case, as sometimes happens with scATAC-seq data. 
# For this reason we discard the first component for downstream analysis. 
bone <- RunUMAP(
  bone,
  reduction = "lsi",
  dims = 2:50,
  reduction.name = "UMAP"
)

bone <- FindNeighbors(bone, dims = 2:50, reduction = "lsi")
bone <- FindClusters(bone, resolution = 0.8, algorithm = 3)

DimPlot(bone, label = TRUE) + NoLegend()


for(i in levels(bone)) {
  cells_to_reid <- WhichCells(bone, idents = i)
  newid <- names(sort(table(bone$celltype[cells_to_reid]),decreasing=TRUE))[1]
  Idents(bone, cells = cells_to_reid) <- newid
}
bone$assigned_celltype <- Idents(bone)

DimPlot(bone, label = TRUE)


DefaultAssay(bone) <- "ATAC"

erythroid <- bone[,  bone$assigned_celltype %in% c("HSC", "MEP", "CMP-BMP")]
lymphoid <- bone[, bone$assigned_celltype %in% c("HSC", "LMPP", "GMP", "CLP", "Pro-B", "pDC", "MDP", "GMP")]



## Building trajectories with Monocle 3

erythroid.cds <- as.cell_data_set(erythroid)
erythroid.cds <- cluster_cells(cds = erythroid.cds, reduction_method = "UMAP")
erythroid.cds <- learn_graph(erythroid.cds, use_partition = TRUE)

lymphoid.cds <- as.cell_data_set(lymphoid)
lymphoid.cds <- cluster_cells(cds = lymphoid.cds, reduction_method = "UMAP")
lymphoid.cds <- learn_graph(lymphoid.cds, use_partition = TRUE)


# load the pre-selected HSCs
hsc <- readLines("./GSE129785/hsc_cells.txt")


# order cells
erythroid.cds <- order_cells(erythroid.cds, reduction_method = "UMAP", root_cells = hsc)
lymphoid.cds <- order_cells(lymphoid.cds, reduction_method = "UMAP", root_cells = hsc)

# plot trajectories colored by pseudotime
plot_cells(
  cds = erythroid.cds,
  color_cells_by = "pseudotime",
  show_trajectory_graph = TRUE
)


plot_cells(
  cds = lymphoid.cds,
  color_cells_by = "pseudotime",
  show_trajectory_graph = TRUE
)


bone <- AddMetaData(
  object = bone,
  metadata = erythroid.cds@principal_graph_aux@listData$UMAP$pseudotime,
  col.name = "Erythroid"
)

bone <- AddMetaData(
  object = bone,
  metadata = lymphoid.cds@principal_graph_aux@listData$UMAP$pseudotime,
  col.name = "Lymphoid"
)

FeaturePlot(bone, c("Erythroid", "Lymphoid"), pt.size = 0.1) & scale_color_viridis_c()


write.csv(lymphoid$assigned_celltype,'lymphoid_celltype.csv')
saveRDS(lymphoid.cds, file = "lymphoid.rds")


#######################################################


lymphoid.cds <- readRDS(file = "lymphoid.rds")

plot_cells(
  cds = lymphoid.cds,
  color_cells_by = "pseudotime",
  show_trajectory_graph = TRUE
)

plot_cells(cds = lymphoid.cds,
           color_cells_by = "cluster",
           label_cell_groups=FALSE,
           label_leaves=TRUE,
           label_branch_points=TRUE,
           graph_label_size=1.5)


lymphoid.monocle_pseudotime<- pseudotime(lymphoid.cds)/max(pseudotime(lymphoid.cds))
lymphoid.logcounts <- lymphoid.cds@assays@data$logcounts
lymphoid.lsi <- reducedDims(lymphoid.cds)[['LSI']]

write.csv(lymphoid.monocle_pseudotime,'lymphoid.monocle_pseudotime.csv')
write.csv(lymphoid.lsi,'lymphoid_lsi.csv')


which.min(lymphoid.monocle_pseudotime)

