
import random
from typing import Dict, List

import anndata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import gc

from .. import base_interface
from ...evaluation import base_evaluation

from ... import vardist, utils_multislice
from .. import module_predict


class GeneratorMicSizeFactor:

    def __init__(
        self,
        model:vardist.InFlowVarDist,
        data_mintflow:Dict,
        dict_all4_configs:Dict,
        evalulate_on_sections: List[int] | List[str] | str,
        kwargs_Kmeans_MCC=None
    ):
        if kwargs_Kmeans_MCC is None:
            kwargs_Kmeans_MCC = {'n_clusters': 10, 'random_state': 0, 'n_init': "auto"}

        # check args
        base_interface.check_arg_data_mintflow(data_mintflow=data_mintflow)
        base_interface.checkif_4configs_are_verified(dict_all4_configs=dict_all4_configs)



        # get list of tissue sections to consider
        list_sliceidx_evalulate_on_sections = base_evaluation.parse_arg_evalulate_on_sections(
            dict_all4_configs=dict_all4_configs,
            data_mintflow=data_mintflow,
            evalulate_on_sections=evalulate_on_sections
        )


        # create an anndata object to be used for size factor conditioning based on both CT and MCC
        adata_cond_CT_MCC = []
        for idx_sl, sl in enumerate(data_mintflow['evaluation_list_tissue_section'].list_slice):
            if idx_sl in list_sliceidx_evalulate_on_sections:
                sl : utils_multislice.Slice

                # get the predictions
                dict_preds = module_predict.predict(
                    dict_all4_configs=dict_all4_configs,
                    data_mintflow=data_mintflow,
                    model=model,
                    evalulate_on_sections=[idx_sl]
                )
                Xmic = dict_preds['TissueSection {} (zero-based)'.format(idx_sl)]['MintFlow_Xmic']
                Xint = dict_preds['TissueSection {} (zero-based)'.format(idx_sl)]['MintFlow_Xint']

                # create/add the anndata
                adata_toadd = sl.adata.copy()
                adata_toadd.obs['MintFLow_signalling_Activity'] = Xmic.sum(1) / (dict_all4_configs['config_training']['val_scppnorm_total'] + 0.0)
                adata_toadd.obsm['MintFlow_MCC'] = sl.ten_NCC.detach().cpu().numpy()
                adata_cond_CT_MCC.append(adata_toadd)

                del dict_preds
                gc.collect()

        adata_cond_CT_MCC = anndata.concat(adata_cond_CT_MCC)

        print(">>>>>>>>>>>>>>>>")
        print("FOR DEBUG: selected tissues to estimate size-factors:")
        print(set(adata_cond_CT_MCC.obs['info_id']))
        print("adata_cond_CT_MCC.shape = {}".format(adata_cond_CT_MCC.shape))
        print(">>>>>>>>>>>>>>>>>>>>>>>>")  # TODO:DELETE

        # get an MCC kmeans index for MCC vectors
        kmeans = KMeans(
            **kwargs_Kmeans_MCC
        ).fit(adata_cond_CT_MCC.obsm['MintFlow_MCC'])
        adata_cond_CT_MCC.obs['MintFlow_MCC_cluster'] = kmeans.labels_.tolist()
        adata_cond_CT_MCC.obs['MintFlow_MCC_cluster'] = adata_cond_CT_MCC.obs['MintFlow_MCC_cluster'].astype('category')


        # for each MintFlow celltype and MCC kmeans cluster, consider a subsect of `adata_cond_CT_MCC`
        dict_4warning_map_mintflowCT_to_strCT = {
            v:k
            for k, v in data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT.items()
        }
        dict_idxct_to_dict_idxcluster_to_adata = {
            idxct: {idxcluster:None for idxcluster in range(kmeans.n_clusters)}
            for idxct in range(len(list(data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT.keys())))
        }
        for idxct in range(len(list(data_mintflow['train_list_tissue_section'].map_CT_to_inflowCT.keys()))):
            for idxcluster in range(kmeans.n_clusters):
                selrow_CT = (adata_cond_CT_MCC.obs['inflow_CT'] == 'inflowCT_{}'.format(idxct)).tolist()
                selrow_MCC = (adata_cond_CT_MCC.obs['MintFlow_MCC_cluster'] == idxcluster).tolist()

                if np.any(np.logical_and(selrow_CT, selrow_MCC)):
                    # filterring based on both CT and MCC is possible
                    dict_idxct_to_dict_idxcluster_to_adata[idxct][idxcluster] = adata_cond_CT_MCC[
                        (adata_cond_CT_MCC.obs['inflow_CT'] == 'inflowCT_{}'.format(idxct)) &\
                        (adata_cond_CT_MCC.obs['MintFlow_MCC_cluster'] == idxcluster)
                    ].obs['MintFLow_signalling_Activity'].tolist()
                elif np.any(selrow_MCC):
                    # filter only based on MCC
                    dict_idxct_to_dict_idxcluster_to_adata[idxct][idxcluster] = adata_cond_CT_MCC[
                        (adata_cond_CT_MCC.obs['MintFlow_MCC_cluster'] == idxcluster)
                    ].obs['MintFLow_signalling_Activity'].tolist()
                    print("Warning: To generate size factors, failed to find cells with same cell type and MCC cluster. Therefore, only MCC cluster was used.".format(
                        dict_4warning_map_mintflowCT_to_strCT['inflowCT_{}'.format(idxct)]
                    ))
                else:
                    pass # never reaches here since MCC index has to be in range(n_clusters)

                '''
                TODO: check if CT-only filterring can be preferable to MCC-only.
                '''



        self.dict_idxct_to_dict_idxcluster_to_libsizepopulation = dict_idxct_to_dict_idxcluster_to_adata
        self.kmeans = kmeans





    def gen_sizefactors(
        self,
        list_idx_CT: List[int],
        list_idx_MCCcluster: List[int]
    ):
        """
        :param list_idx_CT:
        :param list_idx_MCCcluster:
        :return:
        """
        assert len(list_idx_CT) == len(list_idx_MCCcluster)

        return [random.choice(self.dict_idxct_to_dict_idxcluster_to_libsizepopulation[list_idx_CT[n]][list_idx_MCCcluster[n]]) for n in range(len(list_idx_CT))]





