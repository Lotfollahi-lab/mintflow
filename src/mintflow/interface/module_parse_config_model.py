

import os, sys
import yaml
import importlib
import importlib.resources
from .. import utils_multislice

def _correct_booleans(fname_config, dict_config):
    '''
    Yaml may set the True/False or true/false values as string.
    This function replaces the boolean values with python True/False boolean values.
    :param dict_config:
    :return:
    '''

    set_keys_boolean = []
    for k in dict_config.keys():
        if len(k) >= len("flag_"):
            if k[0:len("flag_")] == "flag_":
                set_keys_boolean.append(k)
    set_keys_boolean = list(set(set_keys_boolean))


    for k in set_keys_boolean:
        if not isinstance(dict_config[k], str):
            raise Exception(
                "In the provided model config file, key {} starts seems to be a boolean flag, but the value is not a string ['True', 'False'].\n"+
                "We require you True/False values be provided as a string (i.e. True or False with quoation or double-quotaitons on both sides) in the yaml files."
            )
        assert dict_config[k] in ["True", "False"]
        dict_config[k] = dict_config[k] == "True"

    for k in set_keys_boolean:
        assert isinstance(dict_config[k], bool)

    return dict_config


def get_defaultconfig_model():
    """
    Returns the default file for `config_model.yml`
    :return:
    """
    f = importlib.resources.open_binary(
        "mintflow.data.default_config_files",
        "config_model.yml"
    )
    try:
        config_model = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print("Something went wrong when attempting to read the default `config_data_train.yml.`")
        print(exc)

    return config_model

def verify_and_postprocess_config_model(dict_config_model, data_mintflow, fname_config_model=""):

    # # load config_trianing.yml
    # with open(fname_config_model, 'rb') as f:
    #     try:
    #         dict_config_model = yaml.safe_load(f)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #         raise Exception(
    #             "Something went wrong when reading the config file for training. (backtrace printed above).\n" +
    #             "Please refer to TODO: for sample file config_training.yml"
    #         )

    # check the 2nd arg
    flag_isvalid_arg_mintflow_data = True
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and isinstance(data_mintflow, dict)
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and set(data_mintflow.keys()) == {
        'train_list_tissue_section', 'evaluation_list_tissue_section', 'maxsize_subgraph'}
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and isinstance(
        data_mintflow['train_list_tissue_section'], utils_multislice.ListSlice)
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and isinstance(
        data_mintflow['evaluation_list_tissue_section'], utils_multislice.ListSlice)


    if not flag_isvalid_arg_mintflow_data:
        raise Exception(
            "Something is wrong with the argument `mintflow_data` passed to `verify_and_postprocess_config_model`. Please make sure that argument `mintflow_data` is the output from the function `mintflow.setup_data`.")

    # check if the keys in the yaml file are correct.
    expected_set_keys_config_model = {
         'CTNCC_usage_modulecond4flow',
         'CTNCC_usage_moduledisent',
         'anneal_logp_ZSout_coef_max',
         'anneal_logp_ZSout_coef_min',
         'anneal_logp_ZSout_num_cycles',
         'anneal_logp_ZSout_numepochs_in_cycle',
         'arch_module_encoder_X2Xbar',
         'args_list_adjmatloss',
         'clipval_cov_noncentralnodes',
         'coef_flowmatchingloss',
         'coef_loss_CTpredfromZ',
         'coef_loss_NCCpredfromSin',
         'coef_loss_closeness_xbarintxbarint',
         'coef_loss_closeness_xintxint',
         'coef_loss_closeness_zz',
         'coef_rankloss_Z',
         'coef_rankloss_xbarint',
         'coef_xbarint2notNCC_loss',
         'coef_xbarint2notbatchID_loss',
         'coef_xbarintCT_loss',
         'coef_xbarspl2notbatchID_loss',
         'coef_xbarsplNCC_loss',
         'coef_z2notNCC_loss',
         'dict_pname_to_scaleandunweighted',
         'dict_qname_to_scaleandunweighted',
         'dim_sz',
         'enc3_encSin_list_dim_hidden',
         'enc3_encSout_list_dim_hidden',
         'enc3_encZ_list_dim_hidden',
         'flag_detach_mu_u_int',
         'flag_detach_mu_u_spl',
         'flag_drop_loss_logQdisentangler',
         'flag_enable_batchtoken_disentangler',
         'flag_enable_batchtoken_encxbar',
         'flag_enable_batchtoken_flowmodule',
         'flag_train_negbintheta_int',
         'flag_train_negbintheta_spl',
         'flag_use_dropout_cond4flow_enc3',
         'flag_use_dropout_disentangler_enc1',
         'flag_use_int_u',
         'flag_use_layernorm_cond4flow_enc3',
         'flag_use_layernorm_dec2',
         'flag_use_layernorm_dimreduction_enc2',
         'flag_use_layernorm_disentangler_enc1',
         'flag_use_spl_u',
         'flag_zinbdec_endswith_softmax',
         'flag_zinbdec_endswith_softplus',
         'flowmatching_mode_fmloss',
         'flowmatching_mode_minibatchper',
         'flowmatching_mode_samplex0',
         'flowmatching_mode_timesched',
         'flowmatching_sigma',
         'initval_thetanegbin_int',
         'initval_thetanegbin_spl',
         'lowerbound_cov_u',
         'module_classifier_P1loss',
         'module_classifier_xbarintCT',
         'module_predictor_P3loss',
         'module_predictor_xbarint2notBatchID',
         'module_predictor_xbarint2notNCC',
         'module_predictor_xbarspl2notBatchID',
         'module_predictor_xbarsplNCC',
         'module_predictor_z2notNCC',
         'negbintheta_int_clamp_minmax',
         'negbintheta_spl_clamp_minmax',
         'neuralODE_t_num_steps',
         'num_graph_hops',
         'num_subsample_XYrankloss',
         'std_maxval_finalclip',
         'std_minval_finalclip',
         'str_listdimhidden_dec2',
         'str_modeP3loss_regorcls',
         'str_mode_headxint_headxspl_headboth_twosep',
         'str_modexbarint2notNCCloss_regorclsorwassdist',
         'str_modexbarsplNCCloss_regorcls',
         'str_modez2notNCCloss_regorclsorwassdist',
         'upperbound_cov_u',
         'weight_logprob_zinbpos',
         'weight_logprob_zinbzero',
         'zi_probsetto0_int',
         'zi_probsetto0_spl'
    }


    dict_config_model = _correct_booleans(
        fname_config=fname_config_model,
        dict_config=dict_config_model
    )

    if set(dict_config_model.keys()) != expected_set_keys_config_model:
        set_1m2 = set(dict_config_model.keys()).difference(expected_set_keys_config_model)
        set_2m1 = expected_set_keys_config_model.difference(set(dict_config_model.keys()))

        if len(set_1m2) > 0:
            raise Exception(
                "In config_model, the following unexpected keys were found in the config file: {}".format(set_1m2)
            )
        elif len(set_2m1) > 0:
            raise Exception(
                "In config_model, the following keys and their corresponding values are missing: {}".format(set_2m1)
            )

    # if there's one tissue --> set batch mixing coefficients to zero
    if len(data_mintflow['train_list_tissue_section'].list_slice) == 1:
        dict_config_model['coef_xbarint2notbatchID_loss'] = 0
        dict_config_model['coef_xbarspl2notbatchID_loss'] = 0


    return dict_config_model
