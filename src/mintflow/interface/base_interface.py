
from .. import utils_multislice

dict_oldvarname_to_newvarname = {
    'muxint':'MintFlow_Xint',
    'muxspl':'MintFlow_Xmic',
    'muxbar_int':'MintFlow_Xbar_int',
    'muxbar_spl':'MintFlow_Xbar_mic',
    'mu_sin':'MintFlow_S_in',
    'mu_sout':'MintFlow_S_out',
    'mu_z':'MintFlow_Z',
    'muxint_before_sc_pp_normalize_total':'MintFlow_Xint (before_sc_pp_normalize_total)',
    'muxspl_before_sc_pp_normalize_total':'MintFlow_Xmic (before_sc_pp_normalize_total)'
}  # map names according to the latest glossery of the manuscript.


def check_arg_data_mintflow(data_mintflow):
    flag_isvalid = True
    flag_isvalid = flag_isvalid and isinstance(data_mintflow, dict)
    flag_isvalid = flag_isvalid and set(data_mintflow.keys()) == {
        'train_list_tissue_section',
        'evaluation_list_tissue_section',
        'maxsize_subgraph'
    }
    flag_isvalid = flag_isvalid and isinstance(
        data_mintflow['train_list_tissue_section'],
        utils_multislice.ListSlice
    )
    flag_isvalid = flag_isvalid and isinstance(
        data_mintflow['evaluation_list_tissue_section'],
        utils_multislice.ListSlice
    )

    if not flag_isvalid:
        raise Exception(
            "There is an issue with the passed argument `data_mintflow`. " +\
            "Please make sure it is returned by the function `mintflow.setup_data`."
        )



