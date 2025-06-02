
'''
Implements a
'''


import os, sys
import numpy as np
import torch
import scanpy as sc
from scipy import sparse
import squidpy as sq
import torch_geometric as pyg
from torch_geometric.utils.convert import from_scipy_sparse_matrix

from tqdm.autonotebook import tqdm
from sklearn.linear_model import LinearRegression
import time


def func_get_map_geneidx_to_R2(
    fname_adata,
    obskey_spatial_x,
    obskey_spatial_y,
    kwargs_compute_graph,
    flag_drop_the_targetgene_from_input:bool
):
    """
    :param fname_adata:
    :param obskey_spatial_x:
    :param obskey_spatial_y:
    :param kwargs_compute_graph
    :param flag_drop_the_targetgene_from_input: if set to True, when predicting gene `g` it is dropped from neighbours' expression vectors.
    :return:
    """
    # read the anndata object and create neigh graph
    adata = sc.read_h5ad(fname_adata)
    adata.obsm['spatial'] = np.stack(
        [np.array(adata.obs['obskey_spatial_x'].tolist()), np.array(adata.obs['obskey_spatial_y'].tolist())],
        1
    )
    sq.gr.spatial_neighbors(
        adata=adata,
        **kwargs_compute_graph
    )
    with torch.no_grad():
        edge_index, _ = from_scipy_sparse_matrix(adata.obsp['spatial_connectivities'])  # [2, num_edges]
        edge_index = torch.Tensor(pyg.utils.remove_self_loops(pyg.utils.to_undirected(edge_index))[0])

    np_edge_index = edge_index.detach().cpu().numpy()  # [2 x num_edges]  and for each i,j it contains both [i,j] and [j,i]

    # compute `dict_nodeindex_to_listX` and `dict_nodeindex_to_nodedegree`
    set_ij = set([
        "{}_{}".format(np_edge_index[0, n], np_edge_index[1, n]) for n in range(np_edge_index.shape[1])
    ])
    dict_nodeindex_to_listX = {nodeindex: [] for nodeindex in range(adata.shape[0])}
    for ij in tqdm(set_ij):
        i, j = ij.split("_")
        i, j = int(i), int(j)
        dict_nodeindex_to_listX[i].append(adata.X[j, :])

    dict_nodeindex_to_nodedegree = {
        nodeindex: len(dict_nodeindex_to_listX[nodeindex])
        for nodeindex in range(adata.shape[0])
    }

    for nodeindex in tqdm(range(adata.shape[0])):
        dict_nodeindex_to_listX[nodeindex] = sparse.vstack(dict_nodeindex_to_listX[nodeindex])

    # loop over genes and compute R2 scores
    for idx_gene in tqdm(range(adata.shape[1])):
        t_begin = time.time()

        all_X = sparse.vstack(
            [dict_nodeindex_to_listX[n] for n in range(adata.shape[0])]
        ).toarray()

        if flag_drop_the_targetgene_from_input:
            all_X = np.delete(all_X, idx_gene, 1)

        print("all_X.shape = {}".format(all_X.shape))

        reg = LinearRegression()
        reg.fit(
            all_X.toarray(),
            [adata.X[n, idx_gene] for n in range(adata.X.shape[0]) for _ in range(dict_nodeindex_to_nodedegree[n])]
        )
        r2_score = reg.score(
            all_X.toarray(),
            [adata.X[n, idx_gene] for n in range(adata.X.shape[0]) for _ in range(dict_nodeindex_to_nodedegree[n])]
        )
        print(r2_score)

        print("Took {} seconds.".format(time.time() - t_begin))
        assert False
