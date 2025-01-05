

import os, sys
import yaml

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

    return dict_config_training
