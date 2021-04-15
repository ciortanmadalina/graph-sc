
library(splatter)
library(rhdf5)


setwd("/home/rstudio/projects/contrastive-sc/R")
category <- "imbalanced_data"
ifelse (!dir.exists(paste0("simulated_data/", category)), 
        dir.create(paste0("simulated_data/", category)), FALSE)
simulate <- function(groups, nGenes=2500, batchCells=1500, dropout=2) # change dropout to simulate various dropout rates
{
  params <- newSplatParams()
  group.prob <- groups
  nGroups = length(group.prob)
  dropout.type ="experiment"
  dropout.mid=dropout
  params <- setParams(params, batchCells=batchCells, nGenes=nGenes,
                      group.prob = unlist(group.prob) ,# de.prob=0.5, 
                      de.facScale=0.2, 
                      seed=100, dropout.shape=-1,
                      dropout.type=dropout.type, dropout.mid= dropout)
  sce <- splatSimulate(params, method = "groups", verbose = FALSE)
  
  foldername <-paste0("simulated_data/" , category, "/")
  fname= paste0("data_", dropout, "c", nGroups)
  counts     <- as.data.frame(t(counts(sce)))
  truecounts <- as.data.frame(t(assays(sce)$TrueCounts))
  cellinfo   <- as.data.frame(colData(sce))
  geneinfo   <- as.data.frame(rowData(sce))
  dropout.rate <- (sum(counts==0)-sum(truecounts==0))/sum(truecounts>0)
  print(paste0("Dropout rate ", dropout.rate))
  
  X <- t(counts) ## counts with dropout
  Y <- as.integer(substring(cellinfo$Group,6))
  Y <- Y-1
  print(paste0("Creating ... ", foldername, fname,".h5"))
  h5createFile(paste0(foldername, fname,".h5"))
  h5write(X,paste0(foldername, fname,".h5"),"X")
  h5write(Y, paste0(foldername, fname,".h5"),"Y")
  h5write(dropout.rate, paste0(foldername, fname,".h5"),"dropout")
  
  rowData(sce)$feature_symbol <- rownames(sce)
  logcounts(sce) <- log2(counts(sce)+1)
  sce <- splatter:::splatSimDropout(sce, setParam(params, "dropout.mid", dropout.mid))
  
  # save simulated data
  
  save(sce, file=paste0(foldername,  fname, ".Rdata"))
  print(paste0("Writing file to ... ", paste0("simulated_data/", category, "/", fname, ".Rdata")))
  
}

# simulate(nGroups=4, nGenes=2500, batchCells=1500, dropout=2) TEST

group.probs=list(
  c(0.05, 0.05, 0.4, 0.5), 
  c(0.01, 0.01, 0.04 , 0.04, 0.2, 0.2, 0.2, 0.3),
  c(0.01, 0.01, 0.04 , 0.04, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.04 , 0.04, 0.1, 0.1, 0.1, 0.1),
  c(0.02, 0.04, 0.03, 0.03, 0.02, 0.03, 0.02, 0.03, 0.03, 0.04, 0.03,
    0.03, 0.03, 0.03, 0.04, 0.03, 0.04, 0.02, 0.06, 0.04, 0.02, 0.02,
    0.04, 0.03, 0.03, 0.03, 0.02, 0.04, 0.04, 0.03, 0.03, 0.03))




dropouts = c(-1, 0, 1, 1.5)
#dropouts = c(3)
for(i in 4:length(group.probs)){
  for(j in 1:length(dropouts)){
    print(paste0("Simulating ",length(group.probs[[i]]), " clusters and ", dropouts[j], "  dropout level  " ))
    simulate(group.probs[[i]], nGenes=2500, batchCells=3000, dropout=dropouts[j])
  }
}
