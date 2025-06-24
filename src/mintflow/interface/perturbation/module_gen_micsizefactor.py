


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

def generate_mic_sizefactors(
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
            adata_cond_CT_MCC.append(adata_toadd)

            del dict_preds
            gc.collect()

    adata_cond_CT_MCC = anndata.concat(adata_cond_CT_MCC)



    # np_MCC = np.concatenate(np_MCC, 0)  # [num_cells x num_CT]
    # kmeans = KMeans(
    #     **kwargs_Kmeans_MCC
    # ).fit(np_MCC)

    return adata_cond_CT_MCC


