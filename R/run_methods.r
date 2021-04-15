#R.Version()
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("impute")
#install.packages("devtools")
#library("devtools")
#devtools::install_github("lingxuez/SOUP")
#require(SOUP)
#packageVersion("SOUP")
require(rhdf5)
require(igraph)
require(SIMLR)
require(SOUP)
require(RaceID)
library(SOUP)
library(cidr)
read_clean <- function(data) {  
  if (length(dim(data)) == 1) {
    data <- as.vector(data)
  } else if (length(dim(data)) == 2) {
    data <- as.matrix(data)
  }
  data
}


trans_label<-function(cell_type){
  all_type<-unique(cell_type)
  label<-rep(1,length(cell_type))
  for(j in 1:length(all_type)){
    label[which(cell_type==all_type[j])]=j
  }
  return (label)
}

read_expr_mat_copy = function(file){
  fileinfor = H5Fopen(file)
  exprs_handle = H5Oopen(fileinfor, "exprs")
  if (H5Iget_type(exprs_handle) == "H5I_GROUP"){
    print("H5I_GROUP")
    mat = new("dgCMatrix", x = read_clean(h5read(exprs_handle, "data")),
              i = read_clean(h5read(exprs_handle, "indices")),
              p = read_clean(h5read(exprs_handle, "indptr")),
              Dim = rev(read_clean(h5read(exprs_handle, "shape"))))
  }else if (H5Iget_type(exprs_handle) == "H5I_DATASET"){
    print("NOT H5I_GROUP")
    mat = read_clean(H5Dread(exprs_handle))  ##gene * cell
  }
  obs = H5Oopen(fileinfor, "obs")
  cell_type = h5read(obs, "cell_type1")
  cell_type = as.vector(cell_type)    
  
  label<-trans_label(cell_type) 
  return(list(data_matrix = mat, cell_type = cell_type, cell_label = label))
}


######## find CIDR cluster  ######  

CIDR_cluster = function(data, label){
  sc_cidr = scDataConstructor(data)
  sc_cidr = determineDropoutCandidates(sc_cidr)
  sc_cidr = wThreshold(sc_cidr)
  sc_cidr = scDissim(sc_cidr)
  sc_cidr = scPCA(sc_cidr,plotPC = FALSE)
  sc_cidr = nPC(sc_cidr)
  sc_cidr = scCluster(sc_cidr, nCluster = max(label) - min(label) + 1)
  nmi = compare(label, sc_cidr@clusters, method = "nmi")
  ari = compare(label, sc_cidr@clusters, method = "adjusted.rand")
  return(c(ari,nmi, sc_cidr@clusters))
}


######## find SOUP cluster  ######      
scaleRowSums <- function(x) {
  return (t(scale(t(x), center=FALSE, scale=rowSums(x))))
}

SOUP_cluster=function(data,label){
  data = t(data)
  colnames(data) = as.vector(seq(ncol(data)))
  data<-log2(scaleRowSums(data)*10^4 + 1)
  select.out = selectGenes(data, DESCEND = FALSE, type = "log")
  select.genes = as.vector(select.out$select.genes)
  data = data[, colnames(data) %in% select.genes]
  soup = SOUP(data, Ks = c(max(label) - min(label) + 1), type = "log")
  pred = as.vector(soup$major.labels[[1]])
  nmi=compare(label, pred, method = "nmi")
  ari=compare(label, pred, method = "adjusted.rand")
  return(c(ari,nmi, pred))
}


######## find SIMLR cluster  ######  

SIMLR_cluster_large = function(data, label){
  res_large_scale = SIMLR_Large_Scale(X = data, c = length(unique(label)),normalize = TRUE)
  nmi = compare(label, res_large_scale$y$cluster, method = "nmi")
  ari = compare(label, res_large_scale$y$cluster, method = "adjusted.rand")
  return(c(ari,nmi, res_large_scale$y$cluster))
}




######## find RaceID cluster  ###### 

datascale<-function(x){
  if(x>10000) {
    return(20)
  }else if(2000<x& x<=10000) {
    return(30)
  }else  {return(50)}
}
RaceID_cluster<-function(data,label){
  
  scale<-datascale(ncol(data))
  sc <- SCseq(data)
  sc <- filterdata(sc,mintotal = 1)
  
  sc <- compdist(sc,metric="pearson")
  
  sc<-clustexp(sc,cln=(max(label)-min(label)+1),sat=FALSE,bootnr=scale,FUNcluster = "kmeans")
  nmi<-compare(as.numeric(sc@cluster$kpart),label,method="nmi")
  ari<-compare(as.numeric(sc@cluster$kpart),label,method="adjusted.rand")
  
  return(c(ari,nmi, as.numeric(sc@cluster$kpart)))  
}  

get_input_data<-function(data_path){
  h5closeAll()
  datacount <- h5read(file, "X")
  print(paste0(file, " nrows ", nrow(datacount), " ncols ", ncol(datacount)))
  cell_label<- h5read(file, "Y")
  colnames(datacount) <-seq(1, ncol(datacount), length.out=ncol(datacount))
  rownames(datacount) <-seq(1, nrow(datacount), length.out=nrow(datacount))

  print(nrow(datacount)) # 23000
  print(ncol(datacount)) # 3660
  return(list(datacount, cell_label))
}  

##################################
#####   main program   ###########

setwd("/home/rstudio/projects/contrastive-sc/R")

######   Analyze real data ######

i = 1
category = "real_data"
data_list = list.files("../real_data",full.names = FALSE, recursive = FALSE)
print(data_list)

for(i in 1:length(data_list)){
  cur_data = data_list[i]
  file = paste0("../real_data/", cur_data)
  print(i)
  print(cur_data)
  print(file)
  output = get_input_data(file)
  datacount = output[[1]]
  cell_label =output[[2]]
  print(nrow(datacount)) # 23000
  print(ncol(datacount)) # 3660
  for(run in 1:2){
    print("begin CIDR cluster")
    if (file.exists(paste0("results/",category, "/", cur_data, "_cidr_", run,".csv")) == FALSE){
      cidr_list = CIDR_cluster(datacount, cell_label)
      print(cidr_list)
      print("finish CIDR cluster")
      write.csv(cidr_list, paste0("results/",category, "/", cur_data, "_cidr_", run,".csv"))
    }
    
    print("begin SOUP cluster")
    if (file.exists(paste0("results/",category, "/", cur_data, "_soup_", run,".csv")) == FALSE){
      soup_list = SOUP_cluster(datacount, cell_label)
      print(soup_list)
      print("finish SOUP cluster")
      write.csv(soup_list, paste0("results/",category, "/", cur_data, "_soup_", run,".csv"))
    }
    
    print("begin RaceID cluster")
    if (file.exists(paste0("results/",category, "/", cur_data,  "_raceid_", run,".csv")) == FALSE){
      race_list = RaceID_cluster(datacount, cell_label)
      print(race_list)
      print("finish RaceID cluster")
      write.csv(race_list, paste0("results/",category, "/", cur_data,  "_raceid_", run,".csv"))
    }

  }
  
}


print("begin SIMLR cluster")
if (file.exists(paste0("results/", category, "/",cur_data, "_simlr_", run,".csv")) == FALSE){
  simlr_list = SIMLR_cluster_large(datacount, cell_label)
  print(simlr_list)
  print("finish SIMLR cluster")
  write.csv(simlr_list, paste0("results/", category, "/",cur_data, "_simlr_", run,".csv"))
}

######   Analyze simulated data ######

category = "balanced_data"
data_list = list.files(paste0("simulated_data/", category),full.names = FALSE, recursive = FALSE)
print(data_list)

for(i in 15:length(data_list)){
  cur_data = data_list[i]
  file = paste0("simulated_data/", category,"/", cur_data)
  print(cur_data)
  print(file)
  output = get_input_data(file)
  datacount = output[[1]]
  cell_label =output[[2]]
  print(nrow(datacount)) # 23000
  print(ncol(datacount)) # 3660
  for(run in 1:2){
    print("begin CIDR cluster")
    if (file.exists(paste0("results/",category, "/", cur_data, "_cidr_", run,".csv")) == FALSE){
      cidr_list = CIDR_cluster(datacount, cell_label)
      print(cidr_list)
      print("finish CIDR cluster")
      write.csv(cidr_list, paste0("results/",category, "/", cur_data, "_cidr_", run,".csv"))
    }
    
    print("begin SOUP cluster")
    if (file.exists(paste0("results/",category, "/", cur_data, "_soup_", run,".csv")) == FALSE){
      soup_list = SOUP_cluster(datacount, cell_label)
      print(soup_list)
      print("finish SOUP cluster")
      write.csv(soup_list, paste0("results/",category, "/", cur_data, "_soup_", run,".csv"))
    }
    
    print("begin RaceID cluster")
    if (file.exists(paste0("results/",category, "/", cur_data,  "_raceid_", run,".csv")) == FALSE){
      race_list = RaceID_cluster(datacount, cell_label)
      print(race_list)
      print("finish RaceID cluster")
      write.csv(race_list, paste0("results/",category, "/", cur_data,  "_raceid_", run,".csv"))
    }
    
  }
}
