
from .. import utils_multislice

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



