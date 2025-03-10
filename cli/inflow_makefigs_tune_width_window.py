
"""
Creates figures that help in tuning the parameters `width_window` in config_data_train and config_data_test.
"""



#use inflow or inflow_synth
STR_INFLOW_OR_INFLOW_SYNTH = "inflow"  # in ['inflow', 'inflow_synth']
assert(
    STR_INFLOW_OR_INFLOW_SYNTH == 'inflow' #  in ['inflow', 'inflow_synth']
)

import os, sys
import warnings
import scipy
from scipy.sparse import coo_matrix, issparse
import yaml
import gc
from IPython.utils import io
from pprint import pprint
import time
# import scib
from sklearn.metrics import r2_score
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from collections import Counter
import types
import argparse
from argparse import RawTextHelpFormatter
import wandb
from datetime import datetime
import PIL
import PIL.Image
import anndata
import pandas as pd
import scanpy as sc
import gc
# import dask
import squidpy as sq
import numpy as np
import pickle
from scipy import sparse
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torchdyn
from torchdyn.core import NeuralODE
import torchcfm
from torchcfm.models import MLP
from torchcfm.utils import plot_trajectories, torch_wrapper
from torch_geometric.loader import NeighborLoader
from tqdm.autonotebook import tqdm
import gdown

list_pathstoadd = [
    "../",
    "../src/{}/".format(STR_INFLOW_OR_INFLOW_SYNTH),
    "../src/"
]
for path in list_pathstoadd:
    if(path not in sys.path):
        sys.path.append(path)

exec(
    'import {}'.format(STR_INFLOW_OR_INFLOW_SYNTH)
)

exec('import {}.utils'.format(STR_INFLOW_OR_INFLOW_SYNTH))
BASE_PATH = inflow if(STR_INFLOW_OR_INFLOW_SYNTH=='inflow') else inflow_synth
exec('import {}.generativemodel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('import {}.modules.gnn'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('import {}.modules.neuralODE'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('import {}.modules.mlp'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('import {}.modules.disentonly'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.modules.impanddisentgl import MaskLabel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('import {}.vardist'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('import {}.masking'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.modules.impanddisentgl import ImputerAndDisentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.modules.disentonly import Disentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.modules.disentonly_twosep import DisentanglerTwoSep'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.zs_samplers import RandomZSSampler, PerCelltypeZSSampler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.predadjmat import ListAdjMatPredLoss, AdjMatPredLoss'.format(STR_INFLOW_OR_INFLOW_SYNTH))
exec('from {}.utils_flowmatching import ModeSampleX0, ModeMinibatchPerm, ModeTimeSched, ModeFMLoss, ConditionalFlowMatcher'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.modules.cond4flow import Cond4FlowVarphi0'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.modules.cond4flow_simple3mpls import Cond4FlowVarphi0SimpleMLPs'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.utils_pyg import PygSTDataGridBatchSampler'.format(STR_INFLOW_OR_INFLOW_SYNTH))

exec('from {}.evaluation.bioconsv import EvaluatorKmeans, EvaluatorLeiden'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.evaluation.predxspl import EvalXsplpred, EvalLargeReadoutsXsplpred, EvalOnHVGsXsplpred'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.modules.gnn_disentangler import GNNDisentangler'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.kl_annealing import LinearAnnealingSchedule'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.modules.predictorperCT import PredictorPerCT'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.utils_multislice import ListSlice, Slice'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.modules.varphienc4xbar import EncX2Xbar'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.modules.predictorbatchID import PredictorBatchID'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.cli import parse_config_data_train, parse_config_data_test, parse_config_training, parse_config_model, check_listtissue_trtest'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.cli.auxiliary_modules import *'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.cli.analresults import disentanglement_jointplot'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.cli.analresults import disentanglement_violinplot'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))
exec('from {}.anneal_decoder_xintxspl import AnnealingDecoderXintXspl'.format(
    STR_INFLOW_OR_INFLOW_SYNTH
))

# parse arguments ========================================
parser = argparse.ArgumentParser(
    description='Inflow command-line to create figures that help tuning width_window arguments in config_data_train and config_data_test.\n This sciprts takes in some paths to some config yaml files.',
    formatter_class=RawTextHelpFormatter
)

parser.add_argument(
    '--file_config_data_train',
    type=str,
    help='The yaml file to configure how inflow is expected to read training data.\n' +\
    'Please refere to TODO: for an example of such file.'
)

parser.add_argument(
    '--file_config_data_test',
    type=str,
    help='The yaml file to configure how inflow is expected to read testing data.\n' +\
    'Please refere to TODO: for an example of such file.'
)

parser.add_argument(
    '--file_config_model',
    type=str,
    help='The yaml file to configure architecture of inflow modules (e.g. encoders etc.) .\n' +\
    'Please refere to TODO: for an example such file.'
)

parser.add_argument(
    '--file_config_training',
    type=str,
    help="The yaml file to configure inflow's training (e.g. learning rate, number of epochs, etc.) .\n" +\
    'Please refere to TODO: for an example such file.'
)

parser.add_argument(
    '--list_potential_width_window',
    type=str,
    help="List of potential values for width_window, separated by underscore. For example: 500_600_700_800"
)

parser.add_argument(
    '--path_output',
    type=str,
    help="The output path where inflow will dump the output (some figures, embeddings, etc.)"
)

parser.add_argument(
    '--flag_verbose',
    type=str,
    help="Whether the script is verbose, a string in ['True', 'False']"
)

parser.add_argument(
    '--flag_use_GPU',
    type=str,
    help="Whether GPU is used, a string in ['True', 'False']"
)

args = parser.parse_args()
print("args = {}".format(args)) # ======================================================


def try_mkdir(path_in):
    if not os.path.isdir(path_in):
        os.mkdir(path_in)


# modify/check args ===
assert isinstance(args.flag_verbose, str)
assert args.flag_verbose in ['True', 'False']
args.flag_verbose = (args.flag_verbose == 'True')

assert isinstance(args.flag_use_GPU, str)
assert args.flag_use_GPU in ['True', 'False']
args.flag_use_GPU = (args.flag_use_GPU == 'True')



list_potential_width_window = []
for idx_u, u in enumerate(args.list_potential_width_window.split("_")):
    if not u.isdigit():
        raise Exception(
            "In the provided argument `list_potential_width_window`, the {}-th lement is equal to {} and cannot be interpreted as an integer value.".format(
                idx_u,
                u
            )
        )
    list_potential_width_window.append(
        int(u)
    )


for current_width_window in list_potential_width_window:


    # parse the config files ===
    config_data_train = parse_config_data_train.parse(
        args.file_config_data_train
    )
    config_data_test = parse_config_data_test.parse(
        args.file_config_data_test
    )
    config_training = parse_config_training.parse(
        args.file_config_training
    )
    config_model = parse_config_model.parse(
        args.file_config_model
    )

    # modify width_window fields in `config_data_train` and `config_data_test` ===
    for idx_config in range(len(config_data_train)):
        config_data_train[idx_config]['config_dataloader_train']['width_window'] = current_width_window

    for idx_config in range(len(config_data_test)):
        config_data_test[idx_config]['config_dataloader_test']['width_window'] = current_width_window




    # check if the provided anndata-s share the same gene panel and they all contain count values ===========
    fname_adata0, adata0 = config_data_train[0]['file'], sc.read_h5ad(config_data_train[0]['file'])
    for config_temp in config_data_train + config_data_test:
        if args.flag_verbose:
            print("checking if {} and {} share the same gene panel".format(
                fname_adata0,
                config_temp['file']
            ))

        fname_adata_temp, adata_temp = config_temp['file'], sc.read_h5ad(config_temp['file'])
        if adata_temp.var_names.tolist() != adata0.var_names.tolist():
            raise Exception(
                "Anndata-s {} and {} do not have the same gene panel.".format(
                    fname_adata0,
                    fname_adata_temp
                )
            )

        if not sc._utils.check_nonnegative_integers(adata_temp.X):  # grabbed from https://github.com/scverse/scanpy/blob/0cfd0224f8b0a90317b0f1a61562f62eea2c2927/src/scanpy/preprocessing/_highly_variable_genes.py#L74
            raise Exception(
                "Inflow requires count data, but the anndata in {} seems to have non-count values in adata.X. ".format(
                    fname_adata_temp
                ) + " This could also be due to the dtype of adata.X being, e.g., float, in which case you can solve it by, e.g., `adata.X = adata.X.astype(int)`."
            )
        else:
            if args.flag_verbose:
                print("    also checked that the 2nd anndata has count data in adata.X")

        del fname_adata_temp, adata_temp
        gc.collect()

    del fname_adata0, adata0, config_temp
    gc.collect()

    # set device ===
    if args.flag_use_GPU:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("Although flag_use_GPU is set True by {}, but cuda is not available --> falling back to CPU.".format(
                'args.flag_use_GPU'
            ))
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if args.flag_verbose:
        print("\n\nDevice is set to {}.\n\n".format(device))

    # Create list tissue training =========

    def _convert_TrueFalse_to_bool(dict_input):
        '''
        Given an input dictionary, if any element is "True" or "Fasle" it converts them to booleans.
        :param dict_input:
        :return: the modified dictionary.
        '''
        for k in dict_input:
            if isinstance(dict_input[k], bool):
                raise Exception(
                    'Found {} of type bool. Did you write "true" or "flase" (or True or False without double-quotation-s) instead of "True" or "False" in any of the config files? If so, please modify to latter.'.format(
                        k
                    )
                )

            if dict_input[k] in ['True', 'False']:
                dict_input[k] = (dict_input[k] == 'True')

        return dict_input

    list_slice = []
    for dict_current_anndata in config_data_train:
        current_anndata = sc.read_h5ad(dict_current_anndata['file'])
        unnorm_anndata = current_anndata.copy()
        sc.pp.normalize_total(
            current_anndata,
            target_sum=10000,  # In this script it doesn't matter if it's different from the actual value in config files.
            inplace=True
        )

        list_slice.append(
            Slice(
                adata=current_anndata,
                adata_before_scppnormalize_total=unnorm_anndata,
                dict_obskey={
                    'cell_type':dict_current_anndata['obskey_cell_type'],
                    'sliceid_to_checkUnique':dict_current_anndata['obskey_sliceid_to_checkUnique'],
                    'x':dict_current_anndata['obskey_x'],
                    'y':dict_current_anndata['obskey_y'],
                    'biological_batch_key':dict_current_anndata['obskey_biological_batch_key']
                },
                kwargs_compute_graph={
                    'spatial_key': 'spatial',
                    'library_key': None,
                     **_convert_TrueFalse_to_bool(dict_current_anndata['config_neighbourhood_graph'])
                },
                flag_use_custompygsampler=True,
                kwargs_pygdl_train={
                    'num_neighbors': [dict_current_anndata['config_dataloader_train']['num_neighbors']] * config_model['num_graph_hops'],
                    'width_window': dict_current_anndata['config_dataloader_train']['width_window'],
                    'min_numpoints_ingrid': dict_current_anndata['config_dataloader_train']['min_numpoints_ingrid'],
                    'flag_disable_randoffset': False
                },
                kwargs_pygdl_test={
                    'num_neighbors': [-1] * config_model['num_graph_hops'],
                    'width_window': dict_current_anndata['config_dataloader_train']['width_window'],
                    'min_numpoints_ingrid': 1,
                    'flag_disable_randoffset': True
                },
                neighgraph_num_hops_computeNCC=1,
                kwargs_sq_pl_spatial_scatter=_convert_TrueFalse_to_bool(dict_current_anndata['config_sq_pl_spatial_scatter']),
                device=device,
                batchsize_compute_NCC=dict_current_anndata['batchsize_compute_NCC']
            )
        )


    list_slice = ListSlice(
        list_slice=list_slice
    )

    if args.flag_verbose:
        print("\n\ncreated list_slice for training.")
        for sl in list_slice.list_slice:
            print("Tissue {} --> {} cells".format(
                set(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']]),
                sl.adata.shape[0]
            ))
        print("\n\n")


    # create test_list_slice for evaluation ===
    test_list_slice = []
    for dict_current_anndata in config_data_test:
        current_anndata = sc.read_h5ad(dict_current_anndata['file'])
        unnorm_anndata = current_anndata.copy()
        sc.pp.normalize_total(
            current_anndata,
            target_sum=10000,  # In this script it doesn't matter if it's different from the actual value in config files.
            inplace=True
        )

        test_list_slice.append(
            Slice(
                adata=current_anndata,
                adata_before_scppnormalize_total=unnorm_anndata,
                dict_obskey={
                    'cell_type':dict_current_anndata['obskey_cell_type'],
                    'sliceid_to_checkUnique':dict_current_anndata['obskey_sliceid_to_checkUnique'],
                    'x':dict_current_anndata['obskey_x'],
                    'y':dict_current_anndata['obskey_y'],
                    'biological_batch_key':dict_current_anndata['obskey_biological_batch_key']
                },
                kwargs_compute_graph={
                    'spatial_key': 'spatial',
                    'library_key': None,
                     **_convert_TrueFalse_to_bool(dict_current_anndata['config_neighbourhood_graph'])
                },
                flag_use_custompygsampler=True,
                kwargs_pygdl_train={
                    'num_neighbors': [dict_current_anndata['config_dataloader_test']['num_neighbors']] * config_model['num_graph_hops'],
                    'width_window': dict_current_anndata['config_dataloader_test']['width_window'],
                    'min_numpoints_ingrid': dict_current_anndata['config_dataloader_test']['min_numpoints_ingrid'],
                    'flag_disable_randoffset': False
                },  # dummy training dl's which are never used.
                kwargs_pygdl_test={
                    'num_neighbors': [-1] * config_model['num_graph_hops'],
                    'width_window': dict_current_anndata['config_dataloader_test']['width_window'],
                    'min_numpoints_ingrid': 1,
                    'flag_disable_randoffset': True
                },
                neighgraph_num_hops_computeNCC=1,
                kwargs_sq_pl_spatial_scatter=_convert_TrueFalse_to_bool(dict_current_anndata['config_sq_pl_spatial_scatter']),
                device=device,
                batchsize_compute_NCC=dict_current_anndata['batchsize_compute_NCC']
            )
        )

    test_list_slice = ListSlice(
        list_slice=test_list_slice,
        prev_list_slice_to_imitate=list_slice
    )

    check_listtissue_trtest.check(
        train_list_slice=list_slice,
        test_list_slice=test_list_slice
    )

    if args.flag_verbose:
        print("\n\ncreated list_slice for evaluation.")
        for sl in test_list_slice.list_slice:
            print("Tissue {} --> {} cells".format(
                set(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']]),
                sl.adata.shape[0]
            ))
        print("\n\n")


    # check if path_output is not emtpy, and raise warnings otherwise ===
    if len(os.listdir(args.path_output)) != 0:
        warnings.warn(
            "\n\nThe specified path_output {} is not empty. It's recommeneded to empty path_output so files from different runs are not mixed.\n\n".format(args.path_output)
        )


    if args.flag_verbose:
        print("\n\n\n")
        print("The provided cell types are aggregated/mapped to inflow cell types as follow:")
        pprint(list_slice.map_CT_to_inflowCT)
        print("\n\n")

    if args.flag_verbose:
        print("\n\n\n")
        print("The provided biological batch IDs are aggregated/mapped to inflow batch IDs as follows")
        pprint(list_slice.map_Batchname_to_inflowBatchID)
        print("\n\n")


    if args.flag_verbose:
        with torch.no_grad():
            print("One-hot encoded batch ID for each sample (tissue):")
            for sl in list_slice.list_slice:
                print(
                    "     sample {} --> batch ID {}".format(
                        list(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']])[0],
                        set(sl.ten_BatchEmb.argmax(1).tolist())
                    )
                )

    # TODO: assert that the 1st tissue is assigned batch ID '0' ===

    for kv in config_model['dict_pname_to_scaleandunweighted'].split("&"):
        assert (
            kv.split('#')[2] in ['True', 'False', 'None']
        )
    dict_pname_to_scaleandunweighted = {
        kv.split('#')[0]:[float(kv.split('#')[1]), (kv.split('#')[2]=='True') ]
        for kv in config_model['dict_pname_to_scaleandunweighted'].split("&")
    }

    if(config_model['str_mode_headxint_headxspl_headboth_twosep'] in ['headxint', 'headxspl']):
        dict_pname_to_scaleandunweighted['x'] = [None, None]

    if args.flag_verbose:
        print("\n\ndict_pname_to_scaleandunweighted is set to:")
        pprint(dict_pname_to_scaleandunweighted)
        print("\n\n")


    # compute/report the maximum number of subgraphs
    print("Computing some initial stats (max number of central nodes, etc) for each tissue.")
    list_maxsize_subgraph = []
    dict_slideID_to_maxnumcentralnodes, dict_slideID_to_worsecasebatchsize = {}, {}
    for idx_sl, sl in enumerate(list_slice.list_slice):
        print("Tissue # {}".format(idx_sl+1))
        sl: Slice
        pyg_neighloader_train, pyg_neighloader_test = sl.pyg_dl_train, sl.pyg_dl_test

        if pyg_neighloader_train.batch_size is None:
            batch_size = pyg_neighloader_train.batch_sampler.get_maxnumpoints_insquare()
            maxsize_subgraph = 10 + np.sum(
                np.cumprod([batch_size] + pyg_neighloader_train.node_sampler.num_neighbors.values)
            )
            maxsize_subgraph = int(maxsize_subgraph)

        else:
            maxsize_subgraph = 10 + np.sum(
                np.cumprod([pyg_neighloader_train.batch_size] + pyg_neighloader_train.node_sampler.num_neighbors.values))
            maxsize_subgraph = int(maxsize_subgraph)

            print("maxsize_subgraph is equal to {}. If it's too big, try reducing width_window.".format(
                maxsize_subgraph
            ))

        list_maxsize_subgraph.append(maxsize_subgraph)

        if pyg_neighloader_train.batch_size is None:
            print("    width_window={} --> [maxnum_centralnodes:{},    worse-case batchsize:{}]".format(
                sl.kwargs_pygdl_train['width_window'],
                pyg_neighloader_train.batch_sampler.get_maxnumpoints_insquare(),
                maxsize_subgraph
            ))

            dict_slideID_to_maxnumcentralnodes[list(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']])[0]] = pyg_neighloader_train.batch_sampler.get_maxnumpoints_insquare()
            dict_slideID_to_worsecasebatchsize[list(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']])[0]] = maxsize_subgraph


    maxsize_subgraph = max(list_maxsize_subgraph)


    # dump the window sizes =========
    # Train
    path_window_overlays = os.path.join(
        args.path_output,
        'Figures_To_Tune_Width_Window'
    )
    try_mkdir(path_window_overlays)

    try_mkdir(os.path.join(path_window_overlays, 'Train'))
    try_mkdir(os.path.join(path_window_overlays, 'Test'))

    for idx_sl in range(len(list_slice.list_slice)):
        try_mkdir(os.path.join(path_window_overlays, 'Train', 'Slice_{}'.format(idx_sl+1)))

    for idx_sl in range(len(test_list_slice.list_slice)):
        try_mkdir(os.path.join(path_window_overlays, 'Test', 'Slice_{}'.format(idx_sl+1)))

    for idx_sl, sl in enumerate(list_slice.list_slice):
        sl._show_pygbatch_window_4cli(
            fname_output=os.path.join(
                path_window_overlays,
                'Train',
                'Slice_{}'.format(idx_sl + 1),
                'width_window_{}.png'.format(current_width_window)
            ),
            str_train_or_test='train',
            dict_slideID_to_maxnumcentralnodes=dict_slideID_to_maxnumcentralnodes,
            dict_slideID_to_worsecasebatchsize=dict_slideID_to_worsecasebatchsize
        )


    # Test
    for idx_sl, sl in enumerate(test_list_slice.list_slice):
        sl._show_pygbatch_window_4cli(
            fname_output=os.path.join(
                path_window_overlays,
                'Test',
                'Slice_{}'.format(idx_sl + 1),
                'width_window_{}.png'.format(current_width_window)
            ),
            str_train_or_test='test',
            dict_slideID_to_maxnumcentralnodes=dict_slideID_to_maxnumcentralnodes,
            dict_slideID_to_worsecasebatchsize=dict_slideID_to_worsecasebatchsize
        )


print("Finished running the script.")



# TODO: check if any of the config files contain true, flase or True/False without double-quotation s.

# TODO: check if all tissue-s (including the ones in training and testing) share the same gene panel.

# TODO: check if all tissue-s (training/testing) share the same set of cell types.


# TODO: copy the config files to args.path_output.



# TODO: in the above part `correct for ...` add also x_int and x_spl as well.


