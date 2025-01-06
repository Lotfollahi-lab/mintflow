

import os, sys
import yaml

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
                "In the provided training config file, key {} starts seems to be a boolean flag, but the value is not a string ['True', 'False'].\n"+
                "We require you True/False values be provided as a string (i.e. True or False with quoation or double-quotaitons on both sides) in the yaml files."
            )
        assert dict_config[k] in ["True", "False"]
        dict_config[k] = dict_config[k] == "True"

    for k in set_keys_boolean:
        assert isinstance(dict_config[k], bool)

    return dict_config


def parse(fname_config_training):

    # load config_trianing.yml
    with open(fname_config_training, 'rb') as f:
        try:
            dict_config_training = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise Exception(
                "Something went wrong when reading the config file for training. (backtrace printed above).\n" +
                "Please refer to TODO: for sample file config_training.yml"
            )

    dict_config_training = _correct_booleans(
        fname_config=fname_config_training,
        dict_config=dict_config_training
    )

    return dict_config_training
