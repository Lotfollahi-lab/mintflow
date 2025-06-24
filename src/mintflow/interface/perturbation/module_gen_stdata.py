


import random
from typing import Dict, List

import anndata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import gc
import torch
import torch_geometric as pyg
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from tqdm.autonotebook import tqdm

from . import module_gen_micsizefactor

from .. import base_interface
from ...evaluation import base_evaluation

from ... import vardist, utils_multislice
from .. import module_predict

from ...modules.gnn import KhopAvgPoolWithoutselfloop


dict_generate_oldvarname_to_newvarname = {
    'z':'MintFlow_Generated_Z',
    's_out':'MintFLow_Generated_S_out',
    's_in':'MintFLow_Generated_S_in',
    'xbar_int':'MintFlow_Generated_Xbar_int',
    'xbar_spl':'MintFlow_Generated_Xbar_mic',
    'x_int':'MintFlow_Generated_Xint',
    'x_spl':'MintFLow_Generated_Xmic',
    'x_int_softmax':'MintFlow_Generated_Xint_softmax_output',
    'x_spl_softmax':'MintFlow_Generated_Xmic_softmax_output',
    'ten_u_int':'MintFlow_Cond_int',
    'ten_u_spl':'MintFlow_Cond_mic'
}

@torch.no_grad()
def generate_insilico_ST_data(
    adata:anndata.AnnData,
    obskey_celltype:str,
    obspkey_neighbourhood_graph:str,
    device,
    batch_index_trainingdata:int,
    num_generated_realisations:int,
    model:vardist.InFlowVarDist,
    data_mintflow:Dict,
    dict_all4_configs:Dict,
    evalulate_on_sections: List[int] | List[str] | str,
    kwargs_Kmeans_MCC=None,
    kwargs_pygdl_computeMCC=None,
):
    """

    :param adata:
    :param obskey_celltype:
    :param batch_index_trainingdata:
    :param num_generated_realisations:
    :param obspkey_neighbourhood_graph:
    :param device
    :param model:
    :param data_mintflow:
    :param dict_all4_configs:
    :param evalulate_on_sections:
    :param kwargs_Kmeans_MCC:
    :param kwargs_pygdl_computeMCC
    :return:
    """

    if kwargs_Kmeans_MCC is None:
        kwargs_Kmeans_MCC = {'n_clusters': 10, 'random_state': 0, 'n_init': "auto"}

    if kwargs_pygdl_computeMCC is None:
        kwargs_pygdl_computeMCC = {
            'batch_size': 10,
            'num_workers': 0,
            'num_neighbors': [-1]
        }



    # check args
    base_interface.check_arg_data_mintflow(data_mintflow=data_mintflow)
    base_interface.checkif_4configs_are_verified(dict_all4_configs=dict_all4_configs)

    obj_sizefacgenerator = module_gen_micsizefactor.GeneratorMicSizeFactor(
        model=model,
        data_mintflow=data_mintflow,
        dict_all4_configs=dict_all4_configs,
        evalulate_on_sections=evalulate_on_sections,
        kwargs_Kmeans_MCC=kwargs_Kmeans_MCC
    )

    # check that there are no novel cell types in the CT col
    if obskey_celltype not in adata.obs.columns:
        raise Exception(
            "The provided `obskey_celltype = {}` is not among the columns of the `.obs` field of the provided anndata object.".format(
                obskey_celltype
            )
        )

    if set(adata.obs[obskey_celltype]).difference(set(data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT.keys())) != set([]):
        raise Exception(
            "The `{}` column of the `.obs` field of the provided anndata object contains the following cell types which weren't present in training set: {}".format(
                obskey_celltype,
                set(adata.obs[obskey_celltype]).difference(
                    set(data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT.keys())
                )
            )
        )
    list_CTindex = [
        int(
            data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT[
                adata.obs.iloc[n][obskey_celltype]
            ].split("_")[1]
        )
        for n in range(adata.shape[0])
    ]
    ten_CT = torch.eye(len(set(data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT.keys())))[
        list_CTindex,
        :
    ]


    # compute edge_index from the neighbourhood graph
    edge_index, _ = from_scipy_sparse_matrix(
        adata.obsp[obspkey_neighbourhood_graph]
    )  # [2, num_edges]
    edge_index = torch.Tensor(
        pyg.utils.remove_self_loops(pyg.utils.to_undirected(edge_index))[0]
    )

    # compute np_MCC (needed for generating micenv size factors)
    module_compNCC = KhopAvgPoolWithoutselfloop(
        num_hops=dict_all4_configs['config_model']['num_graph_hops'],
        dim_input=None,
        dim_output=None
    )
    module_compNCC = module_compNCC.to(device)
    ten_MCC = module_compNCC.evaluate_layered(
        x=ten_CT.to(device),
        edge_index=edge_index.to(device),
        kwargs_dl=kwargs_pygdl_computeMCC
    )



    # generate realisations
    list_idx_MCCcluster = obj_sizefacgenerator.kmeans.predict(ten_MCC.detach().cpu().numpy()).tolist()

    ten_BatchEmb_in = torch.eye(len(set(data_mintflow['train_list_tissue_section'].map_Batchname_to_inflowBatchID.keys())))[
        len(list_CTindex) * [batch_index_trainingdata],
        :
    ]
    if len(data_mintflow['train_list_tissue_section'].list_slice) == 1:
        ten_BatchEmb_in = ten_BatchEmb_in * 0.0  # when a single tissue section is used for training, the batch identifier is all-zero

    model.to(device)

    list_generated_realisations = []
    for idx_realisation in tqdm(
        range(num_generated_realisations),
        desc='Generating the realisations of the expression data (i.e. generative samples) for the provided in silico tissue'
    ):
        list_micenv_sizefactors = obj_sizefacgenerator.gen_sizefactors(
            list_idx_CT=list_CTindex,
            list_idx_MCCcluster=list_idx_MCCcluster
        )


        generated_realisation = model.module_genmodel.sample_withZINB(
            edge_index=edge_index.to(device),
            t_num_steps=dict_all4_configs['config_model']['neuralODE_t_num_steps'],
            device=device,
            batch_size_feedforward=10,  # local settings (TODO:modify if needed) ===
            kwargs_dl_neighbourloader={
                'num_neighbors': [-1] * dict_all4_configs['config_model']['num_graph_hops'],
                'batch_size': 5,  # local settings (TODO:modify if needed) ===
                'shuffle': False,
                'num_workers': 0
            },
            ten_CT=ten_CT.to(device),
            ten_BatchEmb_in=ten_BatchEmb_in.to(device),
            sizefactor_int=dict_all4_configs['config_training']['val_scppnorm_total'] - np.array(list_micenv_sizefactors),
            sizefactor_spl=np.array(list_micenv_sizefactors)
        )

        # print(set(generated_realisation.keys()))
        # prints:    {'xbar_spl', 'z', 'x_spl_softmax', 's_out', 's_in', 'x_spl', 'x_int', 'ten_u_spl', 'x_int_softmax', 'xbar_int', 'ten_u_int'}

        # replace the keys in dictionary
        for k_old, k_new in dict_generate_oldvarname_to_newvarname.items():
            generated_realisation[k_new] = generated_realisation.pop(k_old).detach().cpu().numpy()

        list_generated_realisations.append(generated_realisation)


    return list_generated_realisations




