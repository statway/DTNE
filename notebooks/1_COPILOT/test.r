setwd('D:\\0_dtne_demo\\1_COPILOT')
rm(list=ls())

# Load libraries
suppressMessages(library(Matrix))
suppressMessages(library(ggplot2))
suppressMessages(library(scales))
suppressMessages(library(Seurat))
suppressMessages(library(plotly))
suppressMessages(library(grid))
suppressMessages(library(tidyverse))


pp.genes <- as.character(read.table("./supp_data/Protoplasting_DEgene_FC2_list.txt", header=F)$V1)

# Load the color scheme for time and cell type annotation
load("./supp_data/color_scheme_at.RData")

# Prepare a list of sample names that is going to be used for integration
use.sample <- c("sc_12","sc_11","sc_30","sc_31","sc_37","sc_40","sc_51","sc_9_at","sc_10_at","sc_1","tnw1","tnw2","col0","pp1","dc1","dc2")

# Read in Seurat objects and make a list out of them, here we put all the Seurat objects under the foler ./COPILOT_RDS 

read_seu <- function(dir,sample.name) { 
  seu <- readRDS(dir) 
  # remove unused data to save some memory (optional)
  seu@assays$spliced_RNA <- NULL
  seu@assays$spliced_SCT <- NULL
  seu@assays$unspliced_RNA <- NULL
  seu@assays$unspliced_SCT <- NULL
  return(seu)
}

list.filenames <- list.files(path = "./COPILOT_RDS/",pattern=".rds$") %>% .[match(use.sample, gsub("_COPILOT.rds","",.))]

rc.list <- list()

for (i in 1:length(list.filenames))
{
  rc.list[[i]]<-read_seu(dir = paste0("./COPILOT_RDS/",list.filenames[i]), sample.name = use.sample[i])
}

names(rc.list) <- list.filenames %>% gsub("_COPILOT.rds","",.)

# Select genes shared among samples that will be used for integration, for Arabidopsis, 25000 genes is about the upper limit of genes a single cell can have
rc.features <- SelectIntegrationFeatures(object.list = rc.list, nfeatures = 25000)
length(rc.features)

# Remove mitochondrial, chloroplast and protoplasting-induced genes from the shared gene list
rc.features <- rc.features[-c(grep("ATMG",rc.features),grep("ATCG",rc.features),sort(match(pp.genes, rc.features)))]
length(rc.features)

# Prepare for integration
rc.list <- PrepSCTIntegration(object.list = rc.list, anchor.features = rc.features, verbose = TRUE)

# Find integration anchors. Here we use sc_12, the sample with best quality, as reference, which is the first object in the Seuart object list
rc.anchors <- suppressMessages(FindIntegrationAnchors(object.list = rc.list, normalization.method = "SCT", 
                                                      anchor.features = rc.features, verbose = TRUE, reference=1))

# Running integration
rc.integrated <- suppressMessages(IntegrateData(anchorset = rc.anchors, normalization.method = "SCT", verbose = TRUE))


suppressMessages(library(monocle3, quietly=TRUE))
suppressMessages(library(ggplot2))

rc.integrated <- readRDS('./COPILOT_RDS/Root_Atlas.rds')
X <- rc.integrated @assays$integrated$scale.data
features  <-  row.names(X) 
obs_names <- colnames(X)

pd = data.frame(rownames = obs_names)
rownames(pd) = obs_names
fd = data.frame(gene_short_name = features)
rownames(fd) = features

cds <- new_cell_data_set(as.matrix(X),
                         cell_metadata = pd,
                         gene_metadata = fd)

save_monocle_objects(cds=cds, directory_path='./my_cds')




options(Seurat.object.assay.version = "v3")
rm(list=ls())

setwd('D:\\0_dtne_demo\\1_COPILOT')

# Read in atlas
rc.integrated <- readRDS('./COPILOT_RDS/Root_Atlas.rds')

# Load libraries
suppressMessages(library(Seurat))
suppressMessages(library(CytoTRACE))
suppressMessages(library(RColorBrewer))
suppressMessages(library(ggplot2))
suppressMessages(library(grid))


# Plotting function for cell tyepes and time zone
plot_anno <- function(rc.integrated){
  order <- c("Putative Quiescent Center", "Stem Cell Niche", "Columella", "Lateral Root Cap", "Atrichoblast", "Trichoblast", "Cortex", "Endodermis", "Pericycle", "Phloem", "Xylem", "Procambium", "Unknown")
  palette <- c("#9400D3","#dcd0ff", "#5ab953", "#bfef45", "#008080", "#21B6A8", "#82b6ff", "#0000FF","#ff9900","#e6194b", "#9a6324", "#ffe119","#EEEEEE")
  rc.integrated$celltype.anno <- factor(rc.integrated$celltype.anno, levels = order[sort(match(unique(rc.integrated$celltype.anno),order))]) 
  color <- palette[sort(match(unique(rc.integrated$celltype.anno),order))]
  p1 <- DimPlot(rc.integrated, reduction = "umap", group.by = "celltype.anno", cols=color)
  p2 <- DimPlot(rc.integrated, reduction = "umap", group.by = "time.anno", order = c("Maturation","Elongation","Meristem"),cols = c("#DCEDC8", "#42B3D5", "#1A237E"))
  options(repr.plot.width=20, repr.plot.height=8)
  gl <- lapply(list(p1, p2), ggplotGrob)
  gwidth <- do.call(unit.pmax, lapply(gl, "[[", "widths"))
  gl <- lapply(gl, "[[<-", "widths", value = gwidth)
  gridExtra::grid.arrange(grobs=gl, ncol=2)
}
# Load 4 tissue/lineages
end.cor.integrated <- readRDS('./COPILOT_RDS/Ground_Tissue_Atlas.rds')
epi.integrated <- readRDS('./COPILOT_RDS/Epidermis_LRC_Atlas.rds')
col.integrated <- readRDS('./COPILOT_RDS/Columella_Atlas.rds')
stl.integrated <- readRDS('./COPILOT_RDS/Stele_Atlas.rds')



rm(list=ls())

# Load libraries
suppressMessages(library(Seurat))
suppressMessages(library(CytoTRACE))
suppressMessages(library(RColorBrewer))
suppressMessages(library(ggplot2))
suppressMessages(library(grid))

# Read in atlas
rc.integrated <- readRDS('./COPILOT_RDS/Root_Atlas.rds')
rc.su.counts <- readRDS('./COPILOT_RDS/Root_Atlas_spliced_unspliced_raw_counts.rds')

library(anndata)
ad <- AnnData(
  X = rc.integrated@assays$integrated@scale.data,
  obs = data.frame(group =  names(rc.integrated$celltype.anno), row.names = rc.integrated@assays$integrated@var.features),
  obsm = list( cell_type = as.character(rc.integrated$celltype.anno) ),
)

ad <- AnnData(
  X = rc.integrated@assays$integrated@scale.data,
  # obs =  rc.integrated@assays$integrated@var.features, 
  # var = names(rc.integrated$celltype.anno),
  # obsm = list( cell_type = list(as.character(rc.integrated$celltype.anno)), group = NA)
)

pca <- AnnData(
  X = rc.integrated[['pca']]@cell.embeddings,
  # obs =  rc.integrated@assays$integrated@var.features, 
  # var = names(rc.integrated$celltype.anno),
  # obsm = list( cell_type = list(as.character(rc.integrated$celltype.anno)), group = NA)
)

umap <- AnnData(
  X = rc.integrated[['umap']]@cell.embeddings,
  # obs =  rc.integrated@assays$integrated@var.features, 
  # var = names(rc.integrated$celltype.anno),
  # obsm = list( cell_type = list(as.character(rc.integrated$celltype.anno)), group = NA)
)

write_h5ad(umap, "rc.integrated.umap.data.h5ad")

obsm = data.frame( cell_type = as.character(rc.integrated$celltype.anno) )
obsm = list( cell_type = as.character(rc.integrated$celltype.anno) )

ad$obs_names <-  rc.integrated@assays$integrated@var.features 
ad

write_h5ad(ad, "rc.integrated.scale.data.h5ad")

write_h5ad(pca, "rc.integrated.pca.data.h5ad")

dimnames(rc.integrated@assays$integrated[2])

data1 <- AnnData(
  X = t(rc.integrated@assays$integrated@data)
  # obs =  rc.integrated@assays$integrated@var.features, 
  # var = names(rc.integrated$celltype.anno),
  # obsm = list( cell_type = list(as.character(rc.integrated$celltype.anno)), group = NA)
)
write_h5ad(data1, "rc.integrated.data.h5ad")

ad <- AnnData(
  X = matrix(1:6, nrow = 2),
  obs = data.frame(group = c("a", "b"), row.names = c("s1", "s2")),
  var = data.frame(type = c(1L, 2L, 3L), row.names = c("var1", "var2", "var3"))
)

# Plotting function for cell tyepes and time zone
plot_anno <- function(rc.integrated){
  order <- c("Putative Quiescent Center", "Stem Cell Niche", "Columella", "Lateral Root Cap", "Atrichoblast", "Trichoblast", "Cortex", "Endodermis", "Pericycle", "Phloem", "Xylem", "Procambium", "Unknown")
  palette <- c("#9400D3","#dcd0ff", "#5ab953", "#bfef45", "#008080", "#21B6A8", "#82b6ff", "#0000FF","#ff9900","#e6194b", "#9a6324", "#ffe119","#EEEEEE")
  rc.integrated$celltype.anno <- factor(rc.integrated$celltype.anno, levels = order[sort(match(unique(rc.integrated$celltype.anno),order))]) 
  color <- palette[sort(match(unique(rc.integrated$celltype.anno),order))]
  p1 <- DimPlot(rc.integrated, reduction = "umap", group.by = "celltype.anno", cols=color)
  p2 <- DimPlot(rc.integrated, reduction = "umap", group.by = "time.anno", order = c("Maturation","Elongation","Meristem"),cols = c("#DCEDC8", "#42B3D5", "#1A237E"))
  options(repr.plot.width=20, repr.plot.height=8)
  gl <- lapply(list(p1, p2), ggplotGrob)
  gwidth <- do.call(unit.pmax, lapply(gl, "[[", "widths"))
  gl <- lapply(gl, "[[<-", "widths", value = gwidth)
  gridExtra::grid.arrange(grobs=gl, ncol=2)
}

# Index for ground tissue trajectory
end.cor.traj.idx <- which(rc.integrated$celltype.anno == "Endodermis" | rc.integrated$celltype.anno == "Cortex" | rc.integrated$celltype.anno == "Putative Quiescent Center"| rc.integrated$celltype.anno == "Stem Cell Niche") 

# Extract ground tissue
end.cor.integrated <- subset(rc.integrated, cells = colnames(rc.integrated)[end.cor.traj.idx])

# Run UMAP
end.cor.integrated <- RunUMAP(end.cor.integrated, reduction = "pca", dims = 1:50, umap.method = "umap-learn", metric = "correlation", n.components = 50)
end.cor.integrated@reductions$umap_50 <- end.cor.integrated@reductions$umap
end.cor.integrated <- RunUMAP(end.cor.integrated, reduction = "pca", dims = 1:50, umap.method = "umap-learn", metric = "correlation")
end.cor.integrated@reductions$umap_2D <- end.cor.integrated@reductions$umap
end.cor.integrated <- FindNeighbors(end.cor.integrated, reduction = "umap_50",dims = 1:50)

plot_anno(end.cor.integrated)


# Prepare expression matrix for CytoTRACE
expression_matrix <- end.cor.integrated@assays$integrated@data
expression_matrix[which(expression_matrix < 0)]=0
expression_matrix <- as(expression_matrix, "dgCMatrix")
end.cor.integrated@assays$integrated@counts <- expression_matrix

# Run CytoTRACE
results <- CytoTRACE(as.matrix(end.cor.integrated@assays$integrated@counts),  subsamplesize = 1000)
end.cor.integrated$CytoTRACE <- results$CytoTRACE 


options(repr.plot.width=8, repr.plot.height=8)
FeaturePlot(end.cor.integrated, features = "CytoTRACE", pt.size=0.5)+ scale_colour_gradientn(colours = rev(brewer.pal(11,"Spectral")))


# Save Seurat object
saveRDS(end.cor.integrated,'./supp_data/0_Ground_Tissue_Atlas.rds')


# Prepare scVelo input
rc.su.counts <- subset(rc.su.counts, cells = colnames(rc.su.counts)[end.cor.traj.idx]) 
sr <- rc.su.counts@assays$spliced_RNA@counts
sr <- sr[rownames(end.cor.integrated@assays$integrated@data),]
ur <- rc.su.counts@assays$unspliced_RNA@counts
ur <- ur[rownames(end.cor.integrated@assays$integrated@data),]
ar <- end.cor.integrated@assays$RNA@counts
ar <- ar[rownames(end.cor.integrated@assays$integrated@data),]
sr <- sr/ar;
ur <- ur/ar;
sr <- as.matrix(sr)
ur <- as.matrix(ur)
sr[is.nan(sr)] = 0;
ur[is.nan(ur)] = 0;
colnames(sr) <- colnames(end.cor.integrated)
colnames(ur) <- colnames(end.cor.integrated)
int <- as.matrix(end.cor.integrated@assays$integrated@counts)
spliced <- int*sr;
unspliced <- int*ur;
sg <- intersect(rownames(spliced), rownames(unspliced));
spliced <- spliced[match(sg, rownames(spliced)),];
unspliced <- unspliced[match(sg, rownames(unspliced)),];
meta <- end.cor.integrated@meta.data[,grep("time.anno|celltype.anno|time.celltype.anno|CytoTRACE",colnames(end.cor.integrated@meta.data))];
var <- sg;
pca_int <- end.cor.integrated@reductions$pca@cell.embeddings;
umap_int <- end.cor.integrated@reductions$umap@cell.embeddings;
save(spliced, unspliced, meta, var, pca_int, umap_int, file = "./supp_data/0_Ground_Tissue_Atlas_scVelo_input.RData")



# https://bioconductor.org/books/3.14/OSCA.workflows/nestorowa-mouse-hsc-smart-seq2.html#nestorowa-mouse-hsc-smart-seq2

# https://bioconductor.org/books/3.14/OSCA.advanced/trajectory-analysis.html
rm(list=ls())

setwd('D:\\0\\nestorowa')
saveRDS(sce.nest,'./sce.nest.rds')

sce.nest <- readRDS('sce.nest.rds')
library(anndata)

pca <- AnnData(
  X = sce.nest@int_colData@listData$reducedDims$PCA
)

write_h5ad(pca, "sce.nest.pca.h5ad")

logcounts <- AnnData(
  X = sce.nest@assays@data$logcounts
)

write_h5ad(logcounts, "sce.nest.logcounts.h5ad")



library(scRNAseq)
sce.nest <- NestorowaHSCData()

library(AnnotationHub)
ens.mm.v97 <- AnnotationHub()[["AH73905"]]
anno <- select(ens.mm.v97, keys=rownames(sce.nest), 
               keytype="GENEID", columns=c("SYMBOL", "SEQNAME"))
rowData(sce.nest) <- anno[match(rownames(sce.nest), anno$GENEID),]

unfiltered <- sce.nest

library(scater)
stats <- perCellQCMetrics(sce.nest)
qc <- quickPerCellQC(stats, percent_subsets="altexps_ERCC_percent")
sce.nest <- sce.nest[,!qc$discard]
colSums(as.matrix(qc))

colData(unfiltered) <- cbind(colData(unfiltered), stats)
unfiltered$discard <- qc$discard

gridExtra::grid.arrange(
  plotColData(unfiltered, y="sum", colour_by="discard") +
    scale_y_log10() + ggtitle("Total count"),
  plotColData(unfiltered, y="detected", colour_by="discard") +
    scale_y_log10() + ggtitle("Detected features"),
  plotColData(unfiltered, y="altexps_ERCC_percent",
              colour_by="discard") + ggtitle("ERCC percent"),
  ncol=2
)

library(scran)
set.seed(101000110)
clusters <- quickCluster(sce.nest)
sce.nest <- computeSumFactors(sce.nest, clusters=clusters)
sce.nest <- logNormCounts(sce.nest)

summary(sizeFactors(sce.nest))

plot(librarySizeFactors(sce.nest), sizeFactors(sce.nest), pch=16,
     xlab="Library size factors", ylab="Deconvolution factors", log="xy")

set.seed(00010101)
dec.nest <- modelGeneVarWithSpikes(sce.nest, "ERCC")
top.nest <- getTopHVGs(dec.nest, prop=0.1)

plot(dec.nest$mean, dec.nest$total, pch=16, cex=0.5,
     xlab="Mean of log-expression", ylab="Variance of log-expression")
curfit <- metadata(dec.nest)
curve(curfit$trend(x), col='dodgerblue', add=TRUE, lwd=2)
points(curfit$mean, curfit$var, col="red")

set.seed(101010011)
sce.nest <- denoisePCA(sce.nest, technical=dec.nest, subset.row=top.nest)
sce.nest <- runTSNE(sce.nest, dimred="PCA")


snn.gr <- buildSNNGraph(sce.nest, use.dimred="PCA")
colLabels(sce.nest) <- factor(igraph::cluster_walktrap(snn.gr)$membership)
plotTSNE(sce.nest, colour_by="label")

markers <- findMarkers(sce.nest, colLabels(sce.nest), 
                       test.type="wilcox", direction="up", lfc=0.5,
                       row.data=rowData(sce.nest)[,"SYMBOL",drop=FALSE])

chosen <- markers[['8']]
best <- chosen[chosen$Top <= 10,]
aucs <- getMarkerEffects(best, prefix="AUC")
rownames(aucs) <- best$SYMBOL

library(pheatmap)
pheatmap(aucs, color=viridis::plasma(100))

library(SingleR)
library(celldex)
mm.ref <- MouseRNAseqData()

# Renaming to symbols to match with reference row names.
renamed <- sce.nest
rownames(renamed) <- uniquifyFeatureNames(rownames(renamed),
                                          rowData(sce.nest)$SYMBOL)
labels <- SingleR(renamed, mm.ref, labels=mm.ref$label.fine)

tab <- table(labels$labels, colLabels(sce.nest))
pheatmap(log10(tab+10), color=viridis::viridis(100))



Y <- colData(sce.nest)$FACS
keep <- rowSums(is.na(Y))==0 # Removing NA intensities.

se.averaged <- sumCountsAcrossCells(t(Y[keep,]), 
                                    colLabels(sce.nest)[keep], average=TRUE)
averaged <- assay(se.averaged)

log.intensities <- log2(averaged+1)
centered <- log.intensities - rowMeans(log.intensities)
pheatmap(centered, breaks=seq(-1, 1, length.out=101))








library(AnnotationHub)
ens.mm.v97 <- AnnotationHub()[["AH73905"]]
anno <- select(ens.mm.v97, keys=rownames(sce.nest), 
               keytype="GENEID", columns=c("SYMBOL", "SEQNAME"))
rowData(sce.nest) <- anno[match(rownames(sce.nest), anno$GENEID),]

sce.nest
