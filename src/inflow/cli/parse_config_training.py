

import os, sys
import yaml

def _correct_booleans(fname_config, dict_config, set_keys_boolean):
    '''
    Yaml may set the True/False or true/false values as string.
    This function replaces the boolean values with python True/False boolean values.
    :param dict_config:
    :return:
    '''

    # check if `set_keys_boolean` contains all keys starting with flag_...
    for k in dict_config.keys():
        if len(k) >= len("flag_"):
            if k[0:len("flag_")] == "flag_":
                if k not in set_keys_boolean:
                    raise Exception(
                        "In file {} the key {} seems to be a boolean (starts with 'flag_'), but it's not provided in the priori known list of booleans.\n".format(
                            fname_config,
                            k
                        ) + "This may result problems when parsing {}".format(fname_config)
                    )

    for k in set_keys_boolean:
        assert isinstance(dict_config[k], str)
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
        dict_config=dict_config_training,
        set_keys_boolean={
            'flag_use_GPU'
        }  # TODO: complete set of booleans
    )

    return dict_config_training
