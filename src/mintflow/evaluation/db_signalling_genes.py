
from typing import List
import pandas as pd
import importlib
import importlib.resources

from .. import vardist

def evaluate_by_DB_signalling_genes(
    model:vardist.InFlowVarDist,
    data_mintflow:dict,
    evalulate_on_sections: List[int] | List[str] | str
):
    """

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
    :return:
    """

    # get list of evaluation tissue sections to pick
    if isinstance(evalulate_on_sections)

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

    for idx_sl, sl in enumerate(list_slice.list_slice):
        anal_dict_varname_to_output = module_vardist.eval_on_pygneighloader_dense(
            dl=sl.pyg_dl_test,
            # this is correct, because all neighbours are to be included (not a subset of neighbours).
            ten_xy_absolute=sl.ten_xy_absolute,
            tqdm_desc="Tissue {}".format(idx_sl)
        )








