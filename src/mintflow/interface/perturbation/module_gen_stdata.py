


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
    estimate_spatial_sizefactors_on_sections: List[int] | List[str] | str,
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
    :param estimate_spatial_sizefactors_on_sections:
    :param kwargs_Kmeans_MCC:
    :param kwargs_pygdl_computeMCC
    :return:
    """

    model.eval()

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
        device=device,
        data_mintflow=data_mintflow,
        dict_all4_configs=dict_all4_configs,
        evalulate_on_sections=estimate_spatial_sizefactors_on_sections,
        kwargs_Kmeans_MCC=kwargs_Kmeans_MCC
    )
    model.eval()

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

    list_generated_realisations, list_generated_mic_sizefactors = [], []
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
            sizefactor_int=dict_all4_configs['config_training']['val_scppnorm_total'] - np.array(list_micenv_sizefactors)*dict_all4_configs['config_training']['val_scppnorm_total'],
            sizefactor_spl=np.array(list_micenv_sizefactors)*dict_all4_configs['config_training']['val_scppnorm_total']
        )



        # replace the keys in dictionary
        for k_old, k_new in dict_generate_oldvarname_to_newvarname.items():
            generated_realisation[k_new] = generated_realisation.pop(k_old).detach().cpu().numpy()

        list_generated_realisations.append(generated_realisation)
        list_generated_mic_sizefactors.append(list_micenv_sizefactors)

    model.train()

    return dict(
        list_generated_realisations_ie_expressions=list_generated_realisations,
        list_generated_microenv_sizefactors=list_generated_mic_sizefactors,
        np_CT=ten_CT.detach().cpu().numpy(),
        np_MCC=ten_MCC.detach().cpu().numpy()
    )





def generate_insilico_ST_data_with_gene_perturbation(
    adata_reference_expression:anndata.AnnData | None,
    df_gene_perturbation:pd.DataFrame,
    adata:anndata.AnnData,
    obskey_celltype:str,
    obspkey_neighbourhood_graph:str,
    device,
    batch_index_trainingdata:int,
    num_generated_realisations:int,
    model:vardist.InFlowVarDist,
    data_mintflow:Dict,
    dict_all4_configs:Dict,
    estimate_spatial_sizefactors_on_sections: List[int] | List[str] | str,
    kwargs_Kmeans_MCC=None,
    kwargs_pygdl_computeMCC=None,
):
    """
    :param adata_reference_expression: an anndata object with the same gene panel as the 3rd input argument, `adata`. 
    The upregulation perturbations are specified relative to the expression in this anndata object, in `adata_reference_expression.X`.
    Refrain from normalising `adata_reference_expression.X`, i.e., `adata_reference_expression.X` is required to containg raw counts.
    If set to None, then the tissue sections used for training are concatanted and used as `adata_reference_expression`.

    :param df_gene_perturbation: A panda's dataframe that specifies the gene perturbation relative to `adata_reference_expression.X`, with the following requirements:
    - The dataframe has as many rows as the number of cells, and as many columns os the number of genes.
    - Each column of the dataframe is a sparse pandas Series. Please refer to sample notebooks to see how such a dataframe can be created.
    - The column names have to match the gene panel of both `adata_reference_expression` (if not set to None) and that of `adata`.
    - Each element of `df_gene_perturbation` is required to be one of these values
        - 'DC': short for "Don't Care", meaning there is no preference about the expression of that gene in that cell.
        - 'KO': short for 'Knock Out", meaning that gene in that cell is knocked out.
        - 'UP:LFC': "UP" is for "Upregulate", and "LFC" is a positive floating point number that specifies the log-fold change of upregulation. 
        For example, 'UP:5.0' means that gene in that cell has to be upregulated by a log-fold change of 5.0 compared to the control expression provided in `adata_reference_expression.X`.
    
    :param adata: The anndata object containing (i) cell type labels, (ii) neighbourhood graph, and (iii) the gene panel. Other fields like `adata.X` are ignored.

    :param obskey_celltype:
    :param batch_index_trainingdata:
    :param num_generated_realisations:
    :param obspkey_neighbourhood_graph:
    :param device
    :param model:
    :param data_mintflow:
    :param dict_all4_configs:
    :param estimate_spatial_sizefactors_on_sections:
    :param kwargs_Kmeans_MCC:
    :param kwargs_pygdl_computeMCC
    :return:
    """

    # check adata_reference_expression
    if adata_reference_expression is None:
        adata_reference_expression = anndata.concat([
            sl.adata_before_scppnormalize_total
            for sl in data_mintflow['train_list_tissue_section'].list_slice
        ])
    
    assert np.allclose(
        adata_reference_expression.X.data,
        np.floor(adata_reference_expression.X.data)
    ), print("`adata_reference_expression.X` must contain raw counts, but it doesn't.")

    assert (
        adata_reference_expression.var_names.tolist() == adata.var_names.tolist()
    ), print("The provided `adata_reference_expression` and `adata` have to have the same gene panel, but they don't.")

    # check `df_gene_perturbation`
    assert (df_gene_perturbation.dtypes.apply(lambda x: isinstance(x, pd.SparseDtype)).all()), print(
        "Some columns of `df_gene_perturbation` are not sparse, while they must be."
    )
    assert (df_gene_perturbation.columns.tolist() == adata.var_names.tolist()), print(
        "The columns of `df_gene_perturbation` are different from the gene panel of `adata_reference_expression`."
    )
    set_df_vals = set()
    for g in df_gene_perturbation.columns.tolist():
        set_df_vals = set_df_vals.union(
            set(
                df_gene_perturbation[g].unique()
            )
        )
    
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    for v in set_df_vals:
        if v in ['DC', 'KO']:
            pass
        else:
            flag_is_v_valid = len(v) >= len('UP: ')
            flag_is_v_valid = flag_is_v_valid and (v[0:3] == 'UP:')
            flag_is_v_valid = flag_is_v_valid and is_float(v[3::])
            if not flag_is_v_valid:
                print("Found unexpected value {} in `df_gene_perturbation`".format(
                    v
                ))

    assert (df_gene_perturbation.shape[0] == adata.shape[0]), print(
        "`df_gene_perturbation` has to have as many rows as the number of cells in `adata`, while it doesn't"
    )
    assert (df_gene_perturbation.shape[1] == adata.shape[1]), print(
        "`df_gene_perturbation` has to have as many columns as the number of genes in `adata`, while it doesn't"
    )
    

    


    model.eval()

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
        device=device,
        data_mintflow=data_mintflow,
        dict_all4_configs=dict_all4_configs,
        evalulate_on_sections=estimate_spatial_sizefactors_on_sections,
        kwargs_Kmeans_MCC=kwargs_Kmeans_MCC
    )
    model.eval()

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

    list_generated_realisations, list_generated_mic_sizefactors = [], []
    for idx_realisation in tqdm(
        range(num_generated_realisations),
        desc='Generating the realisations of the expression data (i.e. generative samples) for the provided in silico tissue'
    ):
        list_micenv_sizefactors = obj_sizefacgenerator.gen_sizefactors(
            list_idx_CT=list_CTindex,
            list_idx_MCCcluster=list_idx_MCCcluster
        )


        generated_realisation = model.module_genmodel.sample_withZINB_and_GuidanceLoss(
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
            sizefactor_int=dict_all4_configs['config_training']['val_scppnorm_total'] - np.array(list_micenv_sizefactors)*dict_all4_configs['config_training']['val_scppnorm_total'],
            sizefactor_spl=np.array(list_micenv_sizefactors)*dict_all4_configs['config_training']['val_scppnorm_total'],
            obj_get_loss=None
        )



        # replace the keys in dictionary
        for k_old, k_new in dict_generate_oldvarname_to_newvarname.items():
            generated_realisation[k_new] = generated_realisation.pop(k_old).detach().cpu().numpy()

        list_generated_realisations.append(generated_realisation)
        list_generated_mic_sizefactors.append(list_micenv_sizefactors)

    model.train()

    return dict(
        list_generated_realisations_ie_expressions=list_generated_realisations,
        list_generated_microenv_sizefactors=list_generated_mic_sizefactors,
        np_CT=ten_CT.detach().cpu().numpy(),
        np_MCC=ten_MCC.detach().cpu().numpy()
    )


