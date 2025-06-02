
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
    adata,
    obskey_spatial_x,
    obskey_spatial_y,
    kwargs_compute_graph,
    flag_drop_the_targetgene_from_input:bool,
    perc_trainsplit:int=50
):
    """
    :param adata:
    :param obskey_spatial_x:
    :param obskey_spatial_y:
    :param kwargs_compute_graph
    :param flag_drop_the_targetgene_from_input: if set to True, when predicting gene `g` it is dropped from neighbours' expression vectors.
    :return:
    """
    # read the anndata object and create neigh graph
    # adata = sc.read_h5ad(fname_adata)

    adata.obsm['spatial'] = np.stack(
        [np.array(adata.obs[obskey_spatial_x].tolist()), np.array(adata.obs[obskey_spatial_y].tolist())],
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
    for ij in tqdm(set_ij, desc="Analysing the neighbourhood graph"):
        i, j = ij.split("_")
        i, j = int(i), int(j)
        dict_nodeindex_to_listX[i].append(adata.X[j, :])

    dict_nodeindex_to_nodedegree = {
        nodeindex: len(dict_nodeindex_to_listX[nodeindex])
        for nodeindex in range(adata.shape[0])
    }

    for nodeindex in tqdm(range(adata.shape[0]), desc='Precomputing regression input'):
        dict_nodeindex_to_listX[nodeindex] = sparse.vstack(dict_nodeindex_to_listX[nodeindex])

    # loop over genes and compute R2 scores
    list_r2score = []
    for idx_gene in tqdm(range(adata.shape[1])):
        t_begin = time.time()

        # create all_X and all_Y
        all_X = sparse.vstack(
            [dict_nodeindex_to_listX[n] for n in range(adata.shape[0])]
        ).toarray()

        if flag_drop_the_targetgene_from_input:
            all_X = np.delete(all_X, idx_gene, 1)

        all_Y = np.array([float(adata.X[n, idx_gene]) for n in range(adata.X.shape[0]) for _ in range(dict_nodeindex_to_nodedegree[n])])

        # split X and Y to train/test
        randperm_N = np.random.permutation(adata.shape[0])
        N_train = int((perc_trainsplit/100.0) * adata.shape[0])
        list_idx_train = randperm_N[0:N_train]
        list_idx_test  = randperm_N[N_train:]

        # print("all_X.shape = {}".format(all_X.shape))

        reg = LinearRegression()
        reg.fit(
            all_X[list_idx_train, :],
            all_Y[list_idx_train]
        )
        r2_score = reg.score(
            all_X[list_idx_test, :],
            all_Y[list_idx_test]
        )

        list_r2score.append(r2_score)

        #print(r2_score)

        # print("Took {} seconds.".format(time.time() - t_begin))
        # assert False

    return list_r2score

