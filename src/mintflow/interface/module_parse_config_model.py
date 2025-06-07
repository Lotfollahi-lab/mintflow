

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

def verify_and_postprocess_config_model(dict_config_model, fname_config_model=""):

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

    # TODO: check if the keys in the yaml file are correct.

    dict_config_model = _correct_booleans(
        fname_config=fname_config_model,
        dict_config=dict_config_model
    )

    return dict_config_model
