
from typing import List

import pandas
import pandas as pd
import importlib
import importlib.resources
import anndata
import numpy as np

from .. import vardist
from . import base_evaluation

from ..interface import module_predict


def _create_eval_df(
    idx_sl,
    dict_all4_configs,
    anal_dict_varname_to_output,
    list_known_LRgenes_inDB,
    adata_before_scppnormalizetotal:anndata.AnnData
):
    # find UID of the tissue section
    UID_tissue_section = list(
        set(
            adata_before_scppnormalizetotal.obs[
                dict_all4_configs['config_data_evaluation'][idx_sl]['obskey_sliceid_to_checkUnique']
            ]
        )
    )[0]

    # split genes
    num_found_in_LRDB = len(set(adata_before_scppnormalizetotal.var.index.tolist()).intersection(set(list_known_LRgenes_inDB)))
    print("In the gene panel {} genes were found in the list of known signalling genes.".format(num_found_in_LRDB))
    if num_found_in_LRDB == 0:
        return

    X_inLRDB = adata_before_scppnormalizetotal[
        :,
        adata_before_scppnormalizetotal.var.index.isin(set(list_known_LRgenes_inDB))
    ]
    X_notinLRDB = adata_before_scppnormalizetotal[
        :,
        ~adata_before_scppnormalizetotal.var.index.isin(set(list_known_LRgenes_inDB))
    ]

    df_toret = []
    for idx_colsel, colsel in enumerate([
        adata_before_scppnormalizetotal.var.index.isin(set(list_known_LRgenes_inDB)),
        ~adata_before_scppnormalizetotal.var.index.isin(set(list_known_LRgenes_inDB))
    ]):
        pass
        X = adata_before_scppnormalizetotal[:, colsel].X.copy().toarray()
        np_mask_readcount_gt_zero = X > 0.0
        X_mic_beforescppnormalizetotal = anal_dict_varname_to_output['MintFlow_Xmic (before_sc_pp_normalize_total)'][:, colsel].toarray()

        np_read_count = X[np_mask_readcount_gt_zero] + 0.0
        np_count_Xmic = X_mic_beforescppnormalizetotal[np_mask_readcount_gt_zero] + 0.0
        np_fraction_Xmic = np_count_Xmic / np_read_count

        flag_is_among_signalling_genes = [True, False][idx_colsel]
        df_toret.append(
            pandas.DataFrame(
                np.stack([
                        np.array(
                            np_read_count.shape[0]*[UID_tissue_section]
                        ),
                        np_read_count, np_count_Xmic, np_fraction_Xmic,
                        np.array(np_read_count.shape[0] * [flag_is_among_signalling_genes])
                    ],
                    -1
                ),  # [N x 3]
                columns=[
                    base_evaluation.EvalDFColname.tissue_section_unique_ID.value,
                    base_evaluation.EvalDFColname.readcount.value,
                    base_evaluation.EvalDFColname.count_Xmic.value,
                    base_evaluation.EvalDFColname.fraction_Xmic.value,
                    base_evaluation.EvalDFColname.among_signalling_genes.value
                ]
            )
        )

    df_toret = pd.concat(df_toret)

    # modify dtype of columns
    df_toret[base_evaluation.EvalDFColname.tissue_section_unique_ID.value] = df_toret[base_evaluation.EvalDFColname.tissue_section_unique_ID.value].astype('category')
    for c in [
        base_evaluation.EvalDFColname.readcount.value,
        base_evaluation.EvalDFColname.count_Xmic.value,
        base_evaluation.EvalDFColname.fraction_Xmic.value,
    ]:
        df_toret[c] = df_toret[c].astype(float)

    df_toret[base_evaluation.EvalDFColname.among_signalling_genes.value] = df_toret[base_evaluation.EvalDFColname.among_signalling_genes.value].astype('category')

    return df_toret





def evaluate_by_DB_signalling_genes(
    dict_all4_configs:dict,
    data_mintflow:dict,
    model:vardist.InFlowVarDist,
    evalulate_on_sections: List[int] | List[str] | str,
    optional_list_colvaltype_toadd:List[list] = None
):
    """
    :param dict_all4_configs:
    :param model: the mintflow model.
    :param data_mintflow: MintFlow data, as returned by `mintflow.setup_data`
    :param evalulate_on_sections: Specifies whcih evaluation tissue sections to choose and evaluate on.
    This argument can either be
    - A list of integers: the indices of the evaluation tissue sections, on which evaluation is done.
    For example if `evalulate_on_sections` equqls [0,1] then the evaluation will be done on the first two tissue sections.
    - Or a list of tissue section IDs, as you've specificed in `config_data_train` and `config_data_evaluation` in `obskey_sliceid_to_checkUnique`.
    For example if `obskey_sliceid_to_checkUnique` is set to "info_id" in the config files and the passed argument `evalulate_on_sections` equals
    ['my_sample_1', 'my_sample_15'], then the evaluation is done on evaluation anndata objects whose `adata.obs['info_id']`
    is either 'my_sample_1' or'my_sample_15'.
    - Or "all": in this case evaluation is done on all evaluation tissue sections.
    :param optional_list_colvaltype_toadd: can be used to add additional info to the returned dataframe.
    For example one can pass [['training_epoch', 8, 'category']] to specify that this evaluation is done in the 8th training epoch, and the dtype of the added column is 'category'.

    :return: A pandas dataframe that contains the evaluation result for each tissue section.
    """

    # get list of evaluation tissue sections to pick
    list_sliceidx_evalulate_on_sections = base_evaluation.parse_arg_evalulate_on_sections(
        dict_all4_configs=dict_all4_configs,
        data_mintflow=data_mintflow,
        evalulate_on_sections=evalulate_on_sections
    )

    # get the known signalling genes in the database
    f = importlib.resources.open_binary(
        "mintflow.data.for_evaluation.db_signalling_genes",
        "df_LRpairs_Armingoletal.txt"
    )
    df_LRpairs = pd.read_csv(f)
    f.close()

    list_known_LRgenes_inDB = [
        genename
        for colname in ['LigName', 'RecName'] for group in df_LRpairs[colname].tolist() for genename in str(group).split("__")
    ]
    list_known_LRgenes_inDB = set(list_known_LRgenes_inDB)

    # evaluate tissue sections one by one (the ones picked by `list_sliceidx_evalulate_on_sections`)
    dict_sliceid_to_evaldf = {}
    for idx_sl, sl in enumerate(data_mintflow['evaluation_list_tissue_section'].list_slice):
        if idx_sl in list_sliceidx_evalulate_on_sections:
            anal_dict_varname_to_output = list(
                module_predict.predict(
                    dict_all4_configs=dict_all4_configs,
                    data_mintflow=data_mintflow,
                    model=model,
                    evalulate_on_sections=[idx_sl]
                ).items()
            )[0][1]

            dict_sliceid_to_evaldf['TissueSection {} (zero-based)'.format(idx_sl)] = _create_eval_df(
                idx_sl=idx_sl,
                dict_all4_configs=dict_all4_configs,
                anal_dict_varname_to_output=anal_dict_varname_to_output,
                list_known_LRgenes_inDB=list_known_LRgenes_inDB,
                adata_before_scppnormalizetotal=sl.adata_before_scppnormalize_total
            )


    df_toret = pandas.concat(
        [dict_sliceid_to_evaldf[k] for k in dict_sliceid_to_evaldf.keys()]
    )

    if optional_list_colvaltype_toadd is not None:
        for colvaldtype in optional_list_colvaltype_toadd:
            col, val, dtype = colvaldtype
            df_toret[col] = val
            df_toret[col] = df_toret[col].astype(dtype)


    return df_toret








