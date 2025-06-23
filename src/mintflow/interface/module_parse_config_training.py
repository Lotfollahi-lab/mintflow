

import os, sys
import yaml
import importlib
import importlib.resources

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
                "In the provided training config file, the key {} seems to be a boolean flag, but the value is not a string ['True', 'False'].\n"+
                "We require that True/False values be provided as a string (i.e. True or False with quoation or double-quotaitons on both sides) in the yaml files."
            )
        assert dict_config[k] in ["True", "False"]
        dict_config[k] = dict_config[k] == "True"

    for k in set_keys_boolean:
        assert isinstance(dict_config[k], bool)

    # check the annealing numbers for decoder Xint and Xspl ===
    assert dict_config['annealing_decoder_XintXspl_fractionepochs_phase1'] >= 0.0, "In the config file for training, annealing_decoder_XintXspl_fractionepochs_phase1 cannot be negative."
    assert dict_config['annealing_decoder_XintXspl_fractionepochs_phase2'] >= 0.0, "In the config file for training, annealing_decoder_XintXspl_fractionepochs_phase2 cannot be negative."

    assert dict_config['annealing_decoder_XintXspl_fractionepochs_phase1'] <= 1.0, "In the config file for training, annealing_decoder_XintXspl_fractionepochs_phase1 cannot be larger than 1.0."
    assert dict_config['annealing_decoder_XintXspl_fractionepochs_phase2'] <= 1.0, "In the config file for training, annealing_decoder_XintXspl_fractionepochs_phase2 cannot be larger than 1.0."

    assert (dict_config['annealing_decoder_XintXspl_fractionepochs_phase1'] + dict_config['annealing_decoder_XintXspl_fractionepochs_phase2']) <= 1.0, \
        "In the config file for training the sum of `annealing_decoder_XintXspl_fractionepochs_phase1` and `annealing_decoder_XintXspl_fractionepochs_phase2` cannot be larger than 1.0"

    return dict_config


def get_defaultconfig_training():
    """
    Returns the default file for `config_model.yml`
    :return:
    """
    f = importlib.resources.open_binary(
        "mintflow.data.default_config_files",
        "config_training.yml"
    )
    try:
        config_training = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print("Something went wrong when attempting to read the default `config_data_train.yml.`")
        print(exc)

    return config_training


def verify_and_postprocess_config_training(dict_config_training, fname_config_training=""):

    # # load config_trianing.yml
    # with open(fname_config_training, 'rb') as f:
    #     try:
    #         dict_config_training = yaml.safe_load(f)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    #         raise Exception(
    #             "Something went wrong when reading the config file for training. (backtrace printed above).\n" +
    #             "Please refer to TODO: for sample file config_training.yml"
    #         )

    # check if the keys in the yaml file are correct.
    expected_set_keys_config_training = {
         'annealing_decoder_XintXspl_coef_max',
         'annealing_decoder_XintXspl_coef_min',
         'annealing_decoder_XintXspl_fractionepochs_phase1',
         'annealing_decoder_XintXspl_fractionepochs_phase2',
         'batchsize_updateduals_seprately_perepoch',
         'flag_enable_wandb',
         'flag_finaleval_createanndata_alltissuescombined',
         'flag_finaleval_enable_alltissue_violinplot',
         'flag_finaleval_enable_alltissuecombined_eval',
         'flag_finaleval_enable_pertissue_violinplot',
         'flag_use_GPU',
         'lr_training',
         'method_ODE_solver',
         'num_training_epochs',
         'num_updateseparate_afterGRLs',
         'numiters_updateduals_seprately_perepoch',
         'numsteps_accumgrad',
         'sleeptime_gccollect_aftertraining',
         'sleeptime_gccollect_dumpOnePred',
         'val_scppnorm_total',
         'wandb_project_name',
         'wandb_run_name',
         'wandb_stepsize_log'
    }

    dict_config_training = _correct_booleans(
        fname_config=fname_config_training,
        dict_config=dict_config_training
    )

    if set(dict_config_training.keys()) != expected_set_keys_config_training:
        set_1m2 = set(dict_config_training.keys()).difference(expected_set_keys_config_training)
        set_2m1 = expected_set_keys_config_training.difference(set(dict_config_training.keys()))

        if len(set_1m2) > 0:
            raise Exception(
                "In config_training, the following unexpected keys were found in the config file: {}".format(
                    set_1m2)
            )
        elif len(set_2m1) > 0:
            raise Exception(
                "In config_model, the following keys and their corresponding values are missing: {}".format(set_2m1)
            )

    dict_config_training['CONFIG_TRAINING_VERIFIED'] = "DONE"
    
    return dict_config_training
