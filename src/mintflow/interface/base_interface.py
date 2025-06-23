

from typing import Dict
import torch
from .. import utils_multislice
from .. import vardist

from .module_parse_config_data_train import get_defaultconfig_data_train, verify_and_postprocess_config_data_train
from .module_parse_config_data_evaluation import get_defaultconfig_data_evaluation, verify_and_postprocess_config_data_evaluation
from .module_parse_config_model import get_defaultconfig_model, verify_and_postprocess_config_model
from .module_parse_config_training import get_defaultconfig_training, verify_and_postprocess_config_training


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


def checkif_4configs_are_verified(dict_all4_configs: Dict):
    assert set(dict_all4_configs.keys()) == {
        'config_data_train', 'config_data_evaluation', 'config_model', 'config_training'
    }
    config_data_train = dict_all4_configs['config_data_train']
    config_data_evaluation = dict_all4_configs['config_data_evaluation']
    config_model = dict_all4_configs['config_model']
    config_training = dict_all4_configs['config_training']

    if not isinstance(config_data_train, list):
        raise Exception(
            "`config_data_train` is of invalid type. Make sure it is verified and post-processed by `config_data_train = mintflow.verify_and_postprocess_config_data_train(config_data_train)`"
        )
    if not isinstance(config_data_evaluation, list):
        raise Exception(
            "`config_data_evaluation` is of invalid type. Make sure it is verified and post-processed by `config_data_evaluation = mintflow.verify_and_postprocess_config_data_evaluation(config_data_evaluation)`"
        )

    if 'CONFIG_MODEL_VERIFIED' not in config_model.keys():
        raise Exception(
            "It seems `config_model` is not post-processed and verified by the function `mintflow.verify_and_postprocess_config_model`."
        )

    if 'CONFIG_TRAINING_VERIFIED' not in config_training.keys():
        raise Exception(
            "It seems `config_training` is not post-processed and verified by the function `mintflow.verify_and_postprocess_config_training`."
        )




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



def dump_model(
    model,
    path_dump
):
    """
    Dumps a MintFlow model.
    Params
    :param model:
    :param path_dump:
    :return:
    """
    if not isinstance(model, vardist.InFlowVarDist):
        raise Exception(
            "The argument `model` is of incorrect type.\n"+\
            "This function is meant to be used for dumping an object returned by `mintflow.setup_model`."
        )

    model: vardist.InFlowVarDist
    # save to tmp variables
    torevert_module_annealing = model.module_annealing  # to restore after dump.
    torevert_module_annealing_decoderXintXspl = model.module_annealing_decoderXintXspl  # to restore after dump.

    # dump
    model.module_annealing = "NONE"  # so it can be dumped.
    model.module_annealing_decoderXintXspl = "NONE"  # so it can be dumped.
    torch.save(
        model,
        path_dump
    )

    # revert back
    model.module_annealing = torevert_module_annealing  # restore after dump.
    model.module_annealing_decoderXintXspl = torevert_module_annealing_decoderXintXspl  # restore after dump.


def dump_checkpoint(
    model: vardist.InFlowVarDist,
    data_mintflow: Dict,
    dict_all4_configs: Dict,
    path_dump
):
    # check input args
    if not isinstance(model, vardist.InFlowVarDist):
        raise Exception(
            "The passed argument `model` is of incorrect type. Make sure `model` is returned by the function `mintflow.setup_model`."
        )
    check_arg_data_mintflow(data_mintflow=data_mintflow)
    checkif_4configs_are_verified(dict_all4_configs=dict_all4_configs)

    # make `model` serialisable =====================
    model: vardist.InFlowVarDist
    # save to tmp variables
    torevert_module_annealing = model.module_annealing  # to restore after dump.
    torevert_module_annealing_decoderXintXspl = model.module_annealing_decoderXintXspl  # to restore after dump.

    # dump
    model.module_annealing = "NONE"  # so it can be dumped.
    model.module_annealing_decoderXintXspl = "NONE"  # so it can be dumped.
    # torch.save(
    #     model,
    #     path_dump
    # )

    dict_todump = {
        'model':model,
        'data_mintflow':data_mintflow,
        'dict_all4_configs':dict_all4_configs
    }

    torch.save(
        dict_todump,
        path_dump
    )


    # ======================= revert back
    model.module_annealing = torevert_module_annealing  # restore after dump.
    model.module_annealing_decoderXintXspl = torevert_module_annealing_decoderXintXspl  # restore after dump.




def get_default_configurations(
    num_tissue_sections_training: int,
    num_tissue_sections_evaluation: int
):
    """
    Creates and returns 4 default configuration objects.
    :param num_tissue_sections_training: Number of tissue sections for training.
    :param num_tissue_sections_evaluation: Number of tissue sections for evaluation. If its set to
    :return: For objects
    - config_data_train
    - config_data_evaluation
    - config_model
    - config_training
    """
    config_data_train = get_defaultconfig_data_train(num_tissue_sections=num_tissue_sections_training)
    config_data_evaluation = get_defaultconfig_data_evaluation(num_tissue_sections=num_tissue_sections_evaluation)
    config_model = get_defaultconfig_model()
    config_training = get_defaultconfig_training()
    return config_data_train, config_data_evaluation, config_model, config_training











