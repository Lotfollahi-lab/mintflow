
from typing import List
import numpy as np
import torch

from ..evaluation import base_evaluation
from .. import vardist
from ..interface import base_interface

dict_oldvarname_to_newvarname = {
    'muxint':'MintFlow_X_int',
    'muxspl':'MintFlow_X_mic',
    'muxbar_int':'MintFlow_Xbar_int',
    'muxbar_spl':'MintFlow_Xbar_mic',
    'mu_sin':'MintFlow_S_in',
    'mu_sout':'MintFlow_S_out',
    'mu_z':'MintFlow_Z',
    'muxint_before_sc_pp_normalize_total':'MintFlow_Xint (before_sc_pp_normalize_total)',
    'muxspl_before_sc_pp_normalize_total':'MintFlow_Xmic (before_sc_pp_normalize_total)'
}  # map names according to the latest glossery of the manuscript.

@torch.no_grad()
def predict(
    dict_all4_configs:dict,
    data_mintflow:dict,
    model:vardist.InFlowVarDist,
    evalulate_on_sections: List[int] | List[str] | str
):
    model.eval()

    # check arg `data_mintflow`
    base_interface.check_arg_data_mintflow(data_mintflow)

    # get list of evaluation tissue sections to pick
    list_sliceidx_evalulate_on_sections = base_evaluation.parse_arg_evalulate_on_sections(
        dict_all4_configs=dict_all4_configs,
        data_mintflow=data_mintflow,
        evalulate_on_sections=evalulate_on_sections
    )

    dict_sliceid_to_dict_predictions = {}
    for idx_sl, sl in enumerate(data_mintflow['evaluation_list_tissue_section'].list_slice):
        if idx_sl in list_sliceidx_evalulate_on_sections:
            anal_dict_varname_to_output_slice = model.eval_on_pygneighloader_dense(
                dl=sl.pyg_dl_test,
                # this is correct, because all neighbours are to be included (not a subset of neighbours).
                ten_xy_absolute=sl.ten_xy_absolute,
                tqdm_desc="Tissue section: {}".format(idx_sl)
            )

            # remove redundant fields ===
            anal_dict_varname_to_output_slice.pop('output_imputer', None)
            anal_dict_varname_to_output_slice.pop('x_int', None)
            anal_dict_varname_to_output_slice.pop('x_spl', None)

            # get pred_Xspl and pred_Xint before row normalisation on adata.X
            rowcoef_correct4scppnormtotal = (np.array(sl.adata_before_scppnormalize_total.X.sum(1).tolist()) + 0.0) / \
                                            (dict_all4_configs['config_training']['val_scppnorm_total'] + 0.0)
            if len(rowcoef_correct4scppnormtotal.shape) == 1:
                rowcoef_correct4scppnormtotal = np.expand_dims(rowcoef_correct4scppnormtotal, -1)  # [N x 1]

            assert rowcoef_correct4scppnormtotal.shape[0] == sl.adata_before_scppnormalize_total.shape[0]
            assert rowcoef_correct4scppnormtotal.shape[1] == 1

            anal_dict_varname_to_output_slice['muxint_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxint'].multiply(
                rowcoef_correct4scppnormtotal
            )
            anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxspl'].multiply(
                rowcoef_correct4scppnormtotal
            )

            # convert from coo to csr, so they can be saved in anndata object.
            anal_dict_varname_to_output_slice['muxint_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxint_before_sc_pp_normalize_total'].tocsr()
            anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'].tocsr()

            # replace the keys in dictionary
            for k_old, k_new in dict_oldvarname_to_newvarname.items():
                anal_dict_varname_to_output_slice[k_new] = anal_dict_varname_to_output_slice.pop(k_old)

            dict_sliceid_to_dict_predictions['TissueSection {} (zero-based)'.format(idx_sl)] = anal_dict_varname_to_output_slice

    model.train()

    return dict_sliceid_to_dict_predictions



