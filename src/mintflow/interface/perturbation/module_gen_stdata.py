


import random
from typing import Dict, List
import scanpy as sc

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

from ... import utils_guidance


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
    conf_interval_percelltype_prior:float,
    flag_doublecheck_df:bool,
    adata:anndata.AnnData,
    obskey_celltype:str,
    obspkey_neighbourhood_graph:str,
    device,
    batch_index_trainingdata:int,
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
    - The column names have to match the gene panel of both `adata_reference_expression` (if not set to None) and that of `adata`.
    - Each element of `df_gene_perturbation` is required to be one of these values
        - 'DC': short for "Don't Care", meaning there is no preference about the expression of that gene in that cell.
        - 'KO': short for 'Knock Out", meaning that gene in that cell is knocked out.
        - 'UP:LFC': "UP" is for "Upregulate", and "LFC" is a positive floating point number that specifies the log-fold change of upregulation. 
        For example, 'UP:5.0' means that gene in that cell has to be upregulated by a log-fold change of 5.0 compared to the control expression provided in `adata_reference_expression.X`.
    
    :param conf_interval_percelltype_prior: A floating point number between 0.0 and 1.0. The generated `z` and `s_out` embeddings will be forced to reside 
    in a confidence interval of their corresponding population, and `conf_interval_percelltype_prior` determines the total probability of the confidence interval.
    
        
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
    # assert (df_gene_perturbation.dtypes.apply(lambda x: isinstance(x, pd.SparseDtype)).all()), print(
    #     "Some columns of `df_gene_perturbation` are not sparse, while they must be."
    # )
    # assert (df_gene_perturbation.columns.tolist() == adata.var_names.tolist()), print(
    #     "The columns of `df_gene_perturbation` are different from the gene panel of `adata_reference_expression`."
    # )

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
    

    # create a guidance object =============================
    # create the per-gene reference mean(log(gex))
    sc.pp.normalize_total(adata_reference_expression, target_sum=dict_all4_configs['config_training']['val_scppnorm_total'])
    sc.pp.log1p(adata_reference_expression)
    np_ref_pergene_meanloggex = np.array(adata_reference_expression.X.mean(0)).flatten()  # [num_genes]
    np_ref_pergene_meanloggex = np.expand_dims(np_ref_pergene_meanloggex, 0)  # [1 x num_genes]

    # create the upreg target gex values
    def func_map_to_upregval(val):
        if val=='DC':
            return 0
        elif val == 'KO':
            return 0
        else:
            assert val[0:3] == 'UP:'
            return float(val[3:])
    
    np_targetgexval_upreg = df_gene_perturbation.map(func_map_to_upregval).to_numpy()  # [num_cells x num_genes]
    np_targetgexval_upreg = np_targetgexval_upreg + np_ref_pergene_meanloggex  # [num_cells x num_genes]
    np_targetgexval_upreg = np.expm1(np_targetgexval_upreg)  # [num_cells x num_genes], so far, the actual target gex, but for elements to be upreged


    # zero-out 'KO' and 'DC' elements
    def func_map_to_isUPreg(val):
        if val=='DC':
            return 0
        elif val == 'KO':
            return 0
        else:
            assert val[0:3] == 'UP:'
            return 1
    
    np_targetgexval = np_targetgexval_upreg * df_gene_perturbation.map(func_map_to_isUPreg).to_numpy()  # [num_cells x num_genes], the actual target gex, puting aside 'DC'-s
    del np_targetgexval_upreg

    # create np_flag_KOorUPreg
    def func_map_to_isKOorUPreg(val):
        if val=='DC':
            return 0
        elif val == 'KO':
            return 1
        else:
            assert val[0:3] == 'UP:'
            return 1

    np_flag_KOorUPreg = (df_gene_perturbation.map(func_map_to_isKOorUPreg).to_numpy() == 1)

    # double-check element-by-element if `np_targetgexval` and `np_flag_KOorUPreg` exactly match `df_gene_perturbation`
    if flag_doublecheck_df:
        tmp_npdf_gene_perturbation = df_gene_perturbation.to_numpy()  # a temporary ndarray to avoid calling .loc which is very slow.
        for idx_cell in tqdm(range(df_gene_perturbation.shape[0]), desc='Double-checking the processing of `df_gene_perturbation`'):
            for idx_gene in range(df_gene_perturbation.shape[1]):

                if tmp_npdf_gene_perturbation[idx_cell, idx_gene] == 'DC':
                    assert np_flag_KOorUPreg[idx_cell][idx_gene] == False
                    assert np_targetgexval[idx_cell][idx_gene] == 0.0
                elif tmp_npdf_gene_perturbation[idx_cell, idx_gene] == 'KO':
                    assert np_flag_KOorUPreg[idx_cell][idx_gene] == True
                    assert np_targetgexval[idx_cell][idx_gene] == 0.0
                else:
                    assert tmp_npdf_gene_perturbation[idx_cell, idx_gene][0:3] == 'UP:'
                    assert np_flag_KOorUPreg[idx_cell][idx_gene] == True
                    expected_target_val = np.expm1(
                        float(tmp_npdf_gene_perturbation[idx_cell, idx_gene][3:]) + np_ref_pergene_meanloggex[0, idx_gene]
                    )

                    # if np_targetgexval[idx_cell][idx_gene] != expected_target_val:
                    #     print(">>>>>>>>>>>>>>>>> absolute different = {}".format(
                    #         abs(np_targetgexval[idx_cell][idx_gene] - expected_target_val)
                    #     ))
                    #     print("      expected val = {}".format(
                    #         expected_target_val
                    #     ))
                    #     breakpoint()
                    #     print("DDDDD")

                    relative_error = abs(np_targetgexval[idx_cell][idx_gene] - expected_target_val) / expected_target_val
                    if relative_error > 1e-3:
                        raise Exception("An error occured during element-by-element double-check")

        del tmp_npdf_gene_perturbation
        print("\n\n\nSuccessfuly double-checked the processing of `df_gene_perturbation` !")

    # check if the row-sum-s of `np_targetgexval` exceed 1e4, otherwise normalise and warn
    np_checkrowsum = np.sum(np_targetgexval, 1)  # [num_cells]
    for idx_cell in range(df_gene_perturbation.shape[0]):
        if np_checkrowsum[idx_cell] > 0.0:
            if np_checkrowsum[idx_cell] > dict_all4_configs['config_training']['val_scppnorm_total']:
                # normalise that row, so it sums up to 1e4
                np_targetgexval[idx_cell, :] = (np_targetgexval[idx_cell, :] / np.sum(np_targetgexval[idx_cell, :])) * dict_all4_configs['config_training']['val_scppnorm_total']
                
                # warn about the incident
                print(
                    "**** WARNING: for the {}-th cell (zero-based), the specified log-fold change of upregulation is so big that ".format(
                        idx_cell
                    )+\
                    "it prevents other genes to be expressed due to the `target_sum` value of each row. "+\
                    "To avoid this issue, try reducing the log-fold-change of upregulation."
                )


    obj_guider = utils_guidance.GenePerturbationGuider(
        np_desired_expression=np_targetgexval,
        np_mask=np_flag_KOorUPreg,
        device=device,
        conf_interval=conf_interval_percelltype_prior
    )

    # generate size factors (int and mic)  =======================
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

    list_idx_MCCcluster = obj_sizefacgenerator.kmeans.predict(ten_MCC.detach().cpu().numpy()).tolist()

    ten_BatchEmb_in = torch.eye(len(set(data_mintflow['train_list_tissue_section'].map_Batchname_to_inflowBatchID.keys())))[
        len(list_CTindex) * [batch_index_trainingdata],
        :
    ]
    if len(data_mintflow['train_list_tissue_section'].list_slice) == 1:
        ten_BatchEmb_in = ten_BatchEmb_in * 0.0  # when a single tissue section is used for training, the batch identifier is all-zero

    model.to(device)

    list_micenv_sizefactors = obj_sizefacgenerator.gen_sizefactors(
        list_idx_CT=list_CTindex,
        list_idx_MCCcluster=list_idx_MCCcluster
    )
    

    # having genrated the size factors (int and mic), generate the guided gene expression vectors

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
        obj_get_loss=obj_guider
    )

    breakpoint()
    print("DDD")



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


