
'''
Implements a
'''
from typing import List
import gc
import os, sys
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from scipy import sparse
import squidpy as sq
import torch_geometric as pyg
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import pickle

from tqdm.autonotebook import tqdm
from sklearn.linear_model import LinearRegression
import time
from dataclasses import dataclass

from . import base_evaluation

@dataclass
class GeneMicScore:
    """
    A gene ensemble ID, it's score, tissue_info_scoreomputed, and optionally the gene-name.
    """
    ens_ID:str | None
    score:float
    tissue_info_scoreomputed:str
    gene_name: str | None = None

class ListGeneMicScore:
    def __init__(self, list_genemicscore:List[GeneMicScore]):
        assert isinstance(list_genemicscore, list)
        for u in list_genemicscore:
            assert isinstance(u, GeneMicScore)

        self.list_genemicscore = list_genemicscore

    def retrieve_existing_genes(
        self,
        list_ens_ID: list | None,
        list_gene_name: list | None
    ):
        """
        Tries to find genes in the collectio by first checking the ensemble IDs and then gene names.
        :param list_ens_ID: feed None if EnsIDs are not available.
        :param list_gene_name:
        :return:
        """
        if list_ens_ID is None:
            assert list_gene_name is not None
            assert isinstance(list_gene_name, list)
        else:
            assert isinstance(list_ens_ID, list)

        list_idx_toret = []
        dict_map_idxincollection_to_idxininput = {}
        dict_map_idxininput_to_idxincollection = {}

        # attemp to search by EnsID
        if list_ens_ID is not None:
            for idx_ininput, input_ens_ID in enumerate(list_ens_ID):
                for idx_incollection, u in enumerate(self.list_genemicscore):
                    if u.ens_ID == input_ens_ID:
                        dict_map_idxincollection_to_idxininput[idx_incollection] = idx_ininput
                        dict_map_idxininput_to_idxincollection[idx_ininput] = idx_incollection
        else:
            # attemp to search by gene name
            for idx_ininput, input_gene_name in enumerate(list_gene_name):
                for idx_incollection, u in enumerate(self.list_genemicscore):
                    if u.gene_name == input_gene_name:
                        dict_map_idxincollection_to_idxininput[idx_incollection] = idx_ininput
                        dict_map_idxininput_to_idxincollection[idx_ininput] = idx_incollection

        return dict_map_idxincollection_to_idxininput, dict_map_idxininput_to_idxincollection



    def score_Xmic_Xint(
        self,
        list_ens_ID,
        list_gene_name,
        Xint_before_scppnormalizetotal,
        Xmic_before_scppnormalizetotal
    ):
        assert sparse.issparse(Xint_before_scppnormalizetotal)
        assert sparse.issparse(Xmic_before_scppnormalizetotal)

        # query genes in this collection
        dict_map_idxincollection_to_idxininput, dict_map_idxininput_to_idxincollection = self.retrieve_existing_genes(
            list_ens_ID=list_ens_ID,
            list_gene_name=list_gene_name
        )


        # subselect the genes which are found in the collection
        list_idx_selgene = list(dict_map_idxininput_to_idxincollection.keys())
        list_idx_selgene.sort()

        if len(list_idx_selgene) == 0:
            print("No gene was found in the collection.")
            return

        Xint_before_scppnormalizetotal = Xint_before_scppnormalizetotal[:, list_idx_selgene]
        Xmic_before_scppnormalizetotal = Xmic_before_scppnormalizetotal[:, list_idx_selgene]
        X_before_scppnormalizetotal = Xint_before_scppnormalizetotal + Xmic_before_scppnormalizetotal
        mask_readcount = (X_before_scppnormalizetotal > 0).toarray()  # [N x num_selgenes] and dense

        # compute r2scores
        np_r2score_amongfoundgenes = np.array([
            self.list_genemicscore[dict_map_idxininput_to_idxincollection[idx_ininput]].score for idx_ininput in list_idx_selgene
        ])  # [num_selgenes]
        np_r2score_amongfoundgenes = np.stack(
            X_before_scppnormalizetotal.shape[0]*[np_r2score_amongfoundgenes],
            0
        )  # [N x num_selgenes] and dense

        # compute fraction of readcount assigned to Xmic
        fraction_Xmic = np.array(
            Xmic_before_scppnormalizetotal / (Xint_before_scppnormalizetotal + Xmic_before_scppnormalizetotal)
        )  # [N x num_selgenes] and dense

        # get ens_ID-s and gene_name-s
        list_idxincollection = [
            dict_map_idxininput_to_idxincollection[idx_ininput]
            for idx_ininput in list_idx_selgene
        ]
        np_ens_ID_s = np.array(
            [self.list_genemicscore[idxincollection].ens_ID for idxincollection in list_idxincollection]
        )  # [num_selgenes]
        np_gene_name_s = np.array(
            [self.list_genemicscore[idxincollection].gene_name for idxincollection in list_idxincollection]
        )  # [num_selgenes]
        np_ens_ID_s = np.stack(
            X_before_scppnormalizetotal.shape[0] * [np_ens_ID_s],
            0
        )  # [N x num_selgenes] and dense
        np_gene_name_s = np.stack(
            X_before_scppnormalizetotal.shape[0] * [np_gene_name_s],
            0
        )  # [N x num_selgenes] and dense





        # create the dataframe toreturn


        df_toret = pd.DataFrame(
            np.stack([
                X_before_scppnormalizetotal.toarray()[mask_readcount],
                fraction_Xmic[mask_readcount],
                np_r2score_amongfoundgenes[mask_readcount],
                np_ens_ID_s[mask_readcount],
                np_gene_name_s[mask_readcount]
            ],
            1),
            columns=[
                base_evaluation.EvalDFColname.readcount.value,
                base_evaluation.EvalDFColname.fraction_Xmic.value,
                base_evaluation.EvalDFColname.gene_spatial_score.value,
                base_evaluation.EvalDFColname.gene_ens_ID,
                base_evaluation.EvalDFColname.gene_name
            ]
        )

        # correct the dtype of each column
        df_toret['base_evaluation.EvalDFColname.readcount.value'] = df_toret['base_evaluation.EvalDFColname.readcount.value'].astype(float)
        df_toret['base_evaluation.EvalDFColname.fraction_Xmic.value'] = df_toret['base_evaluation.EvalDFColname.fraction_Xmic.value'].astype(float)
        df_toret['base_evaluation.EvalDFColname.gene_spatial_score.value'] = df_toret['base_evaluation.EvalDFColname.gene_spatial_score.value'].astype(float)
        df_toret['base_evaluation.EvalDFColname.gene_ens_ID'] = df_toret['base_evaluation.EvalDFColname.gene_ens_ID'].astype('category')
        df_toret['base_evaluation.EvalDFColname.gene_name'] = df_toret['base_evaluation.EvalDFColname.gene_name'].astype('category')


        return df_toret






def func_get_map_geneidx_to_R2(
    adata,
    obskey_spatial_x,
    obskey_spatial_y,
    kwargs_compute_graph,
    flag_drop_the_targetgene_from_input:bool,
    perc_trainsplit:int=50,
    path_incremental_dump=None
):
    """
    :param adata:
    :param obskey_spatial_x:
    :param obskey_spatial_y:
    :param kwargs_compute_graph
    :param flag_drop_the_targetgene_from_input: if set to True, when predicting gene `g` it is dropped from neighbours' expression vectors.
    :param path_incremental_dump: if it's not None, it incrementally (i.e. gene by gene) dumps the scores into that folder.
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
        dict_nodeindex_to_listX[i].append(
            adata.X[j, :]
        )

    dict_nodeindex_to_nodedegree = {
        nodeindex: len(dict_nodeindex_to_listX[nodeindex])
        for nodeindex in range(adata.shape[0])
    }

    for nodeindex in tqdm(range(adata.shape[0]), desc='Precomputing regression input'):
        dict_nodeindex_to_listX[nodeindex] = sparse.hstack(dict_nodeindex_to_listX[nodeindex])[:, 0:adata.shape[1]*kwargs_compute_graph['n_neighs']]  # [1 x num_genes*num_NNs]



    # loop over genes and compute R2 scores
    list_r2score = []
    for idx_gene in tqdm(range(adata.shape[1])):
        t_begin = time.time()

        # deterimine if calculation has to be done.
        if path_incremental_dump is None:
           flag_hastodo_calculation = True
        else:
            # the incrementatl output path is not None
            flag_hastodo_calculation = not os.path.isfile(os.path.join(path_incremental_dump, '{}.pkl'.format(idx_gene)))

        if not flag_hastodo_calculation:
            continue

        # create all_X and all_Y
        all_X = sparse.vstack(
            [dict_nodeindex_to_listX[n] for n in range(adata.shape[0])]
        ).toarray()  # [N x num_genes*num_NNs]

        if flag_drop_the_targetgene_from_input:
            list_idx_keep = [u for u in set(range(all_X.shape[1])) if u%adata.shape[1] != idx_gene]
            # print("len(list_idx_keep) = {}".format(len(list_idx_keep)))
            all_X = all_X[:, list_idx_keep]

        all_Y = adata.X[:, idx_gene].toarray() # np.array([float(adata.X[n, idx_gene]) for n in range(adata.X.shape[0])])

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

        if path_incremental_dump is None:
            list_r2score.append(r2_score)
        else:
            with open(
                os.path.join(path_incremental_dump, '{}.pkl'.format(idx_gene)),
                'wb'
            ) as f:
                pickle.dump(
                    {
                        'r2_score':r2_score,
                        'idx_gene':idx_gene,
                        'gene_name':adata.var.index.tolist()[idx_gene]
                    },
                    f
                )

        del all_X
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()

    return list_r2score if (path_incremental_dump is not None) else None

