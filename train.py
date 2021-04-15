import os
import pickle
import time
from collections import Counter

import dgl
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy.api as sc
import scipy as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import (accuracy_score, adjusted_rand_score,
                             calinski_harabasz_score,
                             normalized_mutual_info_score, silhouette_score)
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from tqdm.notebook import tqdm


def filter_data(X,  highly_genes=500):
    """
    Remove less variable genes

    Args:
        X ([type]): [description]
        highly_genes (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """

    X = np.ceil(X).astype(np.int)
    adata = sc.AnnData(X)

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4,
                                min_disp=0.5, n_top_genes=highly_genes, subset=True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx


def make_graph(X, Y=None, threshold=0, dense_dim=100,  gene_data={},
               normalize_weights="log_per_cell", nb_edges=1,
               node_features="scale", same_edge_values=False,
               edge_norm=True):
    """
    Create DGL graph model from single cell data

    Args:
        X ([type]): [description]
        Y ([type], optional): [description]. Defaults to None.
        threshold (int, optional): [description]. Defaults to 0.
        dense_dim (int, optional): [description]. Defaults to 100.
        gene_data (dict, optional): [description]. Defaults to {}.
        normalize_weights (str, optional): [description]. Defaults to "log_per_cell".
        nb_edges (int, optional): [description]. Defaults to 1.
        node_features (str, optional): [description]. Defaults to "scale".
        same_edge_values (bool, optional): [description]. Defaults to False.
        edge_norm (bool, optional): [description]. Defaults to True.
    """
    num_genes = X.shape[1]

    graph = dgl.DGLGraph()
    gene_ids = torch.arange(X.shape[1], dtype=torch.int32).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    row_idx, gene_idx = np.nonzero(X > threshold)  # intra-dataset index

    if normalize_weights == "none":
        X1 = X
    if normalize_weights == "log_per_cell":
        X1 = np.log1p(X)
        X1 = X1 / (np.sum(X1, axis=1, keepdims=True) + 1e-6)

    if normalize_weights == "per_cell":
        X1 = X / (np.sum(X, axis=1, keepdims=True) + 1e-6)

    non_zeros = X1[(row_idx, gene_idx)]  # non-zero values

    cell_idx = row_idx + graph.number_of_nodes()  # cell_index
    cell_nodes = torch.tensor([-1] * len(X), dtype=torch.int32).unsqueeze(-1)

    graph.add_nodes(len(cell_nodes), {'id': cell_nodes})
    if nb_edges > 0:
        edge_ids = np.argsort(non_zeros)[::-1]
    else:
        edge_ids = np.argsort(non_zeros)
        nb_edges = abs(nb_edges)
        print(f"selecting weakest edges {int(len(edge_ids) *nb_edges)}")
    edge_ids = edge_ids[:int(len(edge_ids) * nb_edges)]
    cell_idx = cell_idx[edge_ids]
    gene_idx = gene_idx[edge_ids]
    non_zeros = non_zeros[edge_ids]

    if same_edge_values:
        graph.add_edges(
            gene_idx, cell_idx, {
                'weight':
                torch.tensor(np.ones_like(non_zeros),
                             dtype=torch.float32).unsqueeze(1)
            })
    else:
        graph.add_edges(
            gene_idx, cell_idx, {
                'weight':
                torch.tensor(non_zeros, dtype=torch.float32).unsqueeze(1)
            })

    if node_features == "scale":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        gene_feat = PCA(dense_dim, random_state=1).fit_transform(
            nX.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)
    if node_features == "scale_by_cell":
        nX = ((X1 - np.mean(X1, axis=0))/np.std(X1, axis=0))
        cell_feat = PCA(dense_dim, random_state=1).fit_transform(
            nX).astype(float)
        gene_feat = X1.T.dot(cell_feat).astype(float)
    if node_features == "none":
        gene_feat = PCA(dense_dim, random_state=1).fit_transform(
            X1.T).astype(float)
        cell_feat = X1.dot(gene_feat).astype(float)

    graph.ndata['features'] = torch.cat([torch.from_numpy(gene_feat),
                                         torch.from_numpy(cell_feat)],
                                        dim=0).type(torch.float)

    graph.ndata['order'] = torch.tensor([-1] * num_genes + list(np.arange(len(X))),
                                        dtype=torch.long)  # [gene_num+train_num]
    if Y is not None:
        graph.ndata['label'] = torch.tensor([-1] * num_genes + list(np.array(Y).astype(int)),
                                            dtype=torch.long)  # [gene_num+train_num]
    else:
        graph.ndata['label'] = torch.tensor(
            [-1] * num_genes + [np.nan] * len(X))
    nb_edges = graph.num_edges()

    if len(gene_data) != 0 and len(gene_data['gene1']) > 0:
        graph = external_data_connections(
            graph, gene_data, X, gene_idx, cell_idx)
    in_degrees = graph.in_degrees()
    # Edge normalization
    if edge_norm:
        for i in range(graph.number_of_nodes()):
            src, dst, in_edge_id = graph.in_edges(i, form='all')
            if src.shape[0] == 0:
                continue
            edge_w = graph.edata['weight'][in_edge_id]
            graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(
                edge_w)

    graph.add_edges(
        graph.nodes(), graph.nodes(), {
            'weight':
            torch.ones(graph.number_of_nodes(),
                       dtype=torch.float).unsqueeze(1)
        })
    return graph


def run_leiden(data, leiden_n_neighbors=300):
    """
    Performs Leiden community detection on given data.

    Args:
        data ([type]): [description]
        n_neighbors (int, optional): [description]. Defaults to 10.
        n_pcs (int, optional): [description]. Defaults to 40.

    Returns:
        [type]: [description]
    """
    import scanpy.api as sc
    n_pcs = 0
    adata = sc.AnnData(data)
    sc.pp.neighbors(adata, n_neighbors=leiden_n_neighbors,
                    n_pcs=n_pcs, use_rep='X')
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred


def evaluate(model, dataloader, n_clusters, plot=False, save=False, cluster=["KMeans"], use_cpu=False):
    """
    Test the graph autoencoder model.

    Args:
        model ([type]): [description]
        dataloader ([type]): [description]
        n_clusters ([type]): [description]
        plot (bool, optional): [description]. Defaults to False.
        save (bool, optional): [description]. Defaults to False.
        cluster (list, optional): [description]. Defaults to ["KMeans"].
        use_cpu (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    device = get_device(use_cpu=use_cpu)
    model.eval()
    z = []
    y = []
    order = []  # the dataloader shuffles samples
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        input_features = blocks[0].srcdata['features']
        adj_logits, emb = model.forward(blocks, input_features)
        z.extend(emb.detach().cpu().numpy())
        if "label" in blocks[-1].dstdata:
            y.extend(blocks[-1].dstdata["label"].cpu().numpy())
        order.extend(blocks[-1].dstdata["order"].cpu().numpy())

    z = np.array(z)
    y = np.array(y)
    order = np.array(order)
    order = np.argsort(order)
    z = z[order]
    y = y[order]
    if pd.isnull(y[0]):
        y = None

    k_start = time.time()
    scores = {"ae_end": k_start}
    if save:
        scores["features"] = z
        scores["y"] = y[order] if y is not None else None

    if "KMeans" in cluster:
        kmeans = KMeans(n_clusters=n_clusters,
                        init="k-means++", random_state=5)
        kmeans_pred = kmeans.fit_predict(z)
        ari_k = None
        nmi_k = None
        if y is not None:
            ari_k = round(adjusted_rand_score(y, kmeans_pred), 4)
            nmi_k = round(normalized_mutual_info_score(y, kmeans_pred), 4)
        sil_k = silhouette_score(z, kmeans_pred)
        cal_k = calinski_harabasz_score(z, kmeans_pred)
        k_end = time.time()
        scores_k = {
            "kmeans_ari": ari_k,
            "kmeans_nmi": nmi_k,
            "kmeans_sil": sil_k,
            "kmeans_cal": cal_k,
            "kmeans_pred": kmeans_pred,
            "kmeans_time": k_end - k_start,
        }
        scores = {**scores, **scores_k}

    # Leiden
    if "Leiden" in cluster:
        l_start = time.time()
        leiden_pred = run_leiden(z)
        ari_l = None
        nmi_l = None
        if y is not None:
            ari_l = round(adjusted_rand_score(y, leiden_pred), 4)
            nmi_l = round(normalized_mutual_info_score(y, leiden_pred), 4)
        sil_l = silhouette_score(z, leiden_pred)
        cal_l = calinski_harabasz_score(z, leiden_pred)
        l_end = time.time()
        scores_l = {
            "leiden_ari": ari_l,
            "leiden_nmi": nmi_l,
            "leiden_sil": sil_l,
            "leiden_cal": cal_l,
            "leiden_pred": leiden_pred,
            "leiden_time": l_end - l_start,
            "ae_end": k_start
        }
        scores = {**scores, **scores_l}

    if plot:
        pca = PCA(2).fit_transform(z)
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.title("Ground truth")
        plt.scatter(pca[:, 0], pca[:, 1], c=y, s=4)

        plt.subplot(132)
        plt.title("K-Means pred")
        plt.scatter(pca[:, 0], pca[:, 1], c=kmeans_pred, s=4)

        plt.subplot(133)
        plt.title("Leiden pred")
        plt.scatter(pca[:, 0], pca[:, 1], c=pred, s=4)
        plt.show()
    return scores


def train(model, optim, n_epochs, dataloader, n_clusters, plot=False, save=False, cluster=["KMeans"], use_cpu=False):
    """
    Train the graph autoencoder model (model) with the given optimizer (optim)
    for n_epochs.

    Args:
        model ([type]): [description]
        optim ([type]): [description]
        n_epochs ([type]): [description]
        dataloader ([type]): [description]
        n_clusters ([type]): [description]
        plot (bool, optional): [description]. Defaults to False.
        save (bool, optional): [description]. Defaults to False.
        cluster (list, optional): [description]. Defaults to ["KMeans"].
        use_cpu (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    device = get_device(use_cpu=use_cpu)
    losses = []
    aris_kmeans = []
    for epoch in tqdm(range(n_epochs)):
        # normalization
        model.train()
        for input_nodes, output_nodes, blocks in dataloader:
            blocks = [b.to(device) for b in blocks]
            input_features = blocks[0].srcdata['features']
            g = blocks[-1]
            degs = g.in_degrees().float()

            adj = g.adjacency_matrix().to_dense()
            adj = adj[g.dstnodes()]
            pos_weight = torch.Tensor(
                [float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
            factor = float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            if factor == 0:
                factor = 1
            norm = adj.shape[0] * adj.shape[0] / factor
            adj_logits, _ = model.forward(blocks, input_features)
            loss = norm * BCELoss(adj_logits, adj.to(device),
                                  pos_weight=pos_weight.to(device))
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        if plot == False:
            continue
        elif epoch % plot == 0:
            score = evaluate(model, dataloader, n_clusters,
                             cluster=cluster, use_cpu=use_cpu)
            print(f'ARI {score.get("kmeans_ari")}, {score.get("kmeans_sil")}')
            aris_kmeans.append(score["kmeans_ari"])

    if plot:
        plt.figure()
        plt.plot(aris_kmeans, label="kmeans")
        plt.legend()
        plt.show()
    # return model

    score = evaluate(model, dataloader, n_clusters, save=save,
                     cluster=cluster, use_cpu=use_cpu)
    score["aris_kmeans"] = aris_kmeans
    print(f'ARI {score.get("kmeans_ari")}, {score.get("kmeans_sil")}')
    return score


def get_device(use_cpu=False):
    """[summary]

    Returns:
        [type]: [description]
    """
    if torch.cuda.is_available() and use_cpu == False:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def external_data_connections(graph, gene_data, X, gene_idx, cell_idx):
    num_genes = X.shape[1]
    initial_nb_edges = graph.num_edges()
    if gene_data.get("single_layer", False) == True:
        sel_cell_idx = np.argsort((X > 0).sum(axis=1))[
            :int(len(X) * gene_data["select_cells"])]
        sel_cell_idx += X.shape[1]

        normalized_w_values = graph.edata["weight"].numpy().reshape(-1)
        exclude_high_genes = []

        for cell_id in tqdm(sel_cell_idx):
            all_existing_genes = gene_idx[np.where(cell_idx == cell_id)[0]]
            existing_genes_w = normalized_w_values[np.where(cell_idx == cell_id)[
                0]]
            keep_idx = np.where(
                ~np.isin(all_existing_genes, exclude_high_genes))[0]
            existing_genes = all_existing_genes[keep_idx]
            existing_genes_w = existing_genes_w[keep_idx]
            # select random genes
            strond_id = np.random.choice(np.arange(len(existing_genes)),
                                         gene_data["select_genes_threshold"], replace=False)

            existing_genes = existing_genes[strond_id]
            existing_genes_w = existing_genes_w[strond_id]

            for i, g in enumerate(existing_genes):
                correlated_ids = np.where(gene_data['gene2'] == g)[0]
                correlated_genes = gene_data['gene1'][correlated_ids]
                ii = np.where(
                    ~np.isin(correlated_genes, all_existing_genes))[0]
                correlated_ids = correlated_ids[ii]
                correlated_genes = correlated_genes[ii]
                correlated_weights = gene_data['gene_weights'][correlated_ids]
                if len(correlated_genes) > 0:
                    best_id = np.argsort(correlated_weights)[
                        ::-1][:gene_data["nb_correlated_genes"]]
                    graph.add_edges(
                        correlated_genes[best_id], [cell_id] * len(best_id), {
                            'weight':
                            torch.tensor(existing_genes_w[i]*np.ones_like(best_id),
                                         dtype=torch.float32).unsqueeze(1)
                        })
    else:

        weights = torch.from_numpy(gene_data['gene_weights'].astype(
            np.float32) * gene_data['weight']).unsqueeze(1)
        graph.add_edges(gene_data['gene1'], gene_data['gene2'],
                        {'weight': weights})
        print("Adding gene to gene relations", gene_data['gene_weights'].shape, gene_data['weight'],
              weights.max(), weights.min())
    gene_data["extra_edges"] = (
        graph.num_edges() - initial_nb_edges)/graph.num_edges()
    return graph


def tissue_data(gene_names,
                filename="../gene_network/41598_2017_4520_MOESM2_ESM.pkl",
                threshold=0.8,
                method='pearson',
                plot=False,
                max_size=100000):
    """[summary]

    Args:
        gene_names ([type]): [description]
        filename (str, optional): [description]. Defaults to "../gene_network/41598_2017_4520_MOESM2_ESM.pkl".
        threshold (float, optional): [description]. Defaults to 0.8.
        method (str, optional): [description]. Defaults to 'pearson'.
        plot (bool, optional): [description]. Defaults to False.
        max_size (int, optional): [description]. Defaults to 100000.
    """
    df = pd.read_pickle(filename)
    existing_genes = np.intersect1d(gene_names, df.index.values)
    print(f">> Existing {len(existing_genes)}, {len(gene_names)}")
    gene_df = pd.DataFrame(
        data={"gene": gene_names, "id": np.arange(len(gene_names))})
    gene_df = pd.merge(gene_df, df, left_on="gene",
                       right_index=True, how="left").dropna()

    gene_df.drop("gene", axis=1, inplace=True)

    gene_df.set_index("id", inplace=True)
    print(f"Nb common genes {gene_df.shape[0]}")

    cor_tpms = gene_df.T.corr(method=method).fillna(0).values
    cor_tpms = cor_tpms - np.eye(len(cor_tpms))
    cor_tpms = np.abs(cor_tpms)
    gene1, gene2 = np.where(cor_tpms > threshold)
    if plot:
        plt.figure()
        plt.hist(cor_tpms.reshape(-1), bins=30)
        plt.axvline(x=threshold, c="red", linestyle="--")
        plt.show()
    print(f"Selecting {len(gene1)} from {len(cor_tpms)*len(cor_tpms)}")
    gene_weights = cor_tpms[gene1, gene2]

    gene1 = gene_df.index.values[gene1]
    gene2 = gene_df.index.values[gene2]
    print(f"Unique genes {np.unique(gene1).shape} ")

    if max_size < len(gene_weights):
        print(f"Restricting {len(gene_weights)} corr edges to {max_size}")
        ordered_id = np.argsort(gene_weights)[::-1]
        ordered_id = ordered_id[:max_size]
        gene_weights = gene_weights[ordered_id]
        gene1 = gene1[ordered_id]
        gene2 = gene2[ordered_id]

    gene_data = {"gene1": gene1, "gene2": gene2, "gene_weights": gene_weights}
    return gene_data
