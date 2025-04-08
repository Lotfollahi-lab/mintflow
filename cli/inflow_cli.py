

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
    description='Inflow command-line interface.\n This sciprts takes in some paths to some config yaml files.',
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
    '--path_output',
    type=str,
    help="The output path where inflow will dump the output (some figures, embeddings, etc.)"
)

parser.add_argument(
    '--flag_verbose',
    type=str,
    help="Whether the script is verbose, a string in ['True', 'False']"
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
# TODO: parse other config files ===


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
            "Inflow requires count data, but the anndata in {} seems to have non-count values in adata.X".format(
                fname_adata_temp
            )
        )
    else:
        if args.flag_verbose:
            print("    also checked that the 2nd anndata has count data in adata.X")

    del fname_adata_temp, adata_temp
    gc.collect()

del fname_adata0, adata0, config_temp
gc.collect()

# set device ===
if config_training['flag_use_GPU']:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Although flag_use_GPU is set True in {}, but cuda is not available --> falling back to CPU.".format(
            args.file_config_training
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
        target_sum=config_training['val_scppnorm_total'],
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
        target_sum=config_training['val_scppnorm_total'],
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



# dump the scatters (i.e. neighbourhood graphs) ======
# Train
path_scatters = os.path.join(
    args.path_output,
    'Toinspect_NeighbourhoodGraphs'
)
if not os.path.isdir(path_scatters):
    os.mkdir(path_scatters)

path_scatters = os.path.join(
    path_scatters,
    'Train'
)
if not os.path.isdir(path_scatters):
    os.mkdir(path_scatters)
list_slice.show_scatters_4cli(path_output=path_scatters)

# Test
path_scatters = os.path.join(
    args.path_output,
    'Toinspect_NeighbourhoodGraphs'
)
if not os.path.isdir(path_scatters):
    os.mkdir(path_scatters)

path_scatters = os.path.join(
    path_scatters,
    'Test'
)
if not os.path.isdir(path_scatters):
    os.mkdir(path_scatters)
test_list_slice.show_scatters_4cli(path_output=path_scatters)



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



dict_varname_to_dim = {
    's': config_model['dim_sz'],
    'z': config_model['dim_sz'],
    'x': list_slice.list_slice[0].adata.X.shape[1],  # i.e. number of genes
    'xbar': config_model['dim_sz'],
    'vardist': {
        'impanddisent': {
            'em1': {
                'dim_embedding': 100,
                'dim_em_iscentralnode': 10,
                'dim_em_blankorobserved': 10
            }  # dummy, not used in the latest code
        },
        'cond4flowvarphi0': {
            'em1': {
                'dim_embedding': 20,
                'dim_em_iscentralnode': 10,
                'dim_em_blankorobserved': 10
            }
        }  # dummy, not used in the latest code
    }
}
dict_varname_to_dim['xbar'] = dict_varname_to_dim['z']
assert (
        dict_varname_to_dim['s'] == dict_varname_to_dim['z']
)

kwargs_genmodel = {
    'num_cells': 1234,  # TODO: is it needed? dummy for training, only used when generating expression vectors.
    'dict_varname_to_dim': {
        's': dict_varname_to_dim['s'],
        'z': dict_varname_to_dim['z'],
        'x': dict_varname_to_dim['x'],
        'cell-types': list_slice.list_slice[0]._global_num_CT,
        'u_int': list_slice.list_slice[0]._global_num_CT,
        'u_spl': list_slice.list_slice[0]._global_num_CT,
        'CT': list_slice.list_slice[0]._global_num_CT,
        'NCC': list_slice.list_slice[0]._global_num_CT,
        'BatchEmb': list_slice.list_slice[0]._global_num_Batch
    },
    'dict_pname_to_scaleandunweighted': dict_pname_to_scaleandunweighted,
    'type_theta_aggr': BASE_PATH.modules.gnn.KhopAvgPoolWithoutselfloop,
    'kwargs_theta_aggr': {'num_hops': config_model['num_graph_hops']},
    'type_moduleflow': BASE_PATH.modules.neuralODE.MLP,
    'kwargs_moduleflow': {'w': 64},
    'type_w_dec': BASE_PATH.modules.mlp.SimpleMLP,
    'kwargs_w_dec': {
        'list_dim_hidden': [] if (config_model['str_listdimhidden_dec2'] == '') else [int(u) for u in config_model['str_listdimhidden_dec2'].split('_')],
        # it was [100, 200, 300], changed to [] or tunable to have a simpler decoder to avoid collapse.
        'bias': True,
        'flag_startswithReLU': True,
        'flag_endwithReLU': False,
        'flag_endswithSoftmax': config_model['flag_zinbdec_endswith_softmax'],
        'flag_endswith_softplus': config_model['flag_zinbdec_endswith_softplus'],
        'flag_use_layernorm': config_model['flag_use_layernorm_dec2']
    },
    'kwargs_negbin_int': {'zi_logits': scipy.special.logit(config_model['zi_probsetto0_int'])},
    'kwargs_negbin_spl': {'zi_logits': scipy.special.logit(config_model['zi_probsetto0_spl'])},
    'initval_thetanegbin_int': float(config_model['initval_thetanegbin_int']) if (config_model['initval_thetanegbin_int'] != 'rand') else config_model['initval_thetanegbin_int'],
    'flag_train_negbintheta_int': config_model['flag_train_negbintheta_int'],
    'negbintheta_int_clamp_minmax': None if (config_model['negbintheta_int_clamp_minmax'] in ["None", None]) else [float(u) for u in config_model['negbintheta_int_clamp_minmax'].split('_')],
    'initval_thetanegbin_spl': float(config_model['initval_thetanegbin_spl']) if (config_model['initval_thetanegbin_spl'] != 'rand') else config_model['initval_thetanegbin_spl'],
    'flag_train_negbintheta_spl': config_model['flag_train_negbintheta_spl'],
    'negbintheta_spl_clamp_minmax': None if (config_model['negbintheta_spl_clamp_minmax'] in ["None", None]) else [float(u) for u in config_model['negbintheta_spl_clamp_minmax'].split('_')],
    'flag_use_int_u': config_model['flag_use_int_u'],
    'module_int_mu_u': BASE_PATH.modules.mlp.LinearEncoding(
        dim_input=list_slice.list_slice[0]._global_num_CT,
        dim_output=dict_varname_to_dim['z'],
        flag_detach=config_model['flag_detach_mu_u_int']
    ) if (config_model['flag_use_int_u']) else None,
    'module_int_cov_u': BASE_PATH.modules.mlp.SimpleMLPandExp(
        dim_input=list_slice.list_slice[0]._global_num_CT,
        list_dim_hidden=[],
        dim_output=dict_varname_to_dim['z'],
        bias=True,
        min_clipval=config_model['lowerbound_cov_u'],
        max_clipval=config_model['upperbound_cov_u']
    ) if (config_model['flag_use_int_u']) else None,
    'flag_use_spl_u': config_model['flag_use_spl_u'],
    'module_spl_mu_u': BASE_PATH.modules.mlp.LinearEncoding(
        dim_input=list_slice.list_slice[0]._global_num_CT,
        dim_output=dict_varname_to_dim['s'],
        flag_detach=config_model['flag_detach_mu_u_spl']
    ) if (config_model['flag_use_spl_u']) else None,
    'module_spl_cov_u': BASE_PATH.modules.mlp.SimpleMLPandExp(
        dim_input=list_slice.list_slice[0]._global_num_CT,
        list_dim_hidden=[],
        dim_output=dict_varname_to_dim['z'],
        bias=True,
        min_clipval=config_model['lowerbound_cov_u'],
        max_clipval=config_model['upperbound_cov_u']
    ) if (config_model['flag_use_spl_u']) else None,
    'coef_zinb_int_loglik': 1.0,
    'coef_zinb_spl_loglik': 1.0,
    'dict_config_batchtoken': {
        'flag_enable_batchtoken_flowmodule': config_model['flag_enable_batchtoken_flowmodule']
    }
}

# create a list of `AdjMatPredLoss`-s ====
list_ajdmatpredloss = []
if len(config_model['args_list_adjmatloss']) > 0:
    for arg_adjmatloss in config_model['args_list_adjmatloss'].split("&"):
        print(arg_adjmatloss)
        assert (
                len(arg_adjmatloss.split("#")) == 6
        )
        if len(arg_adjmatloss.split('#')[0]) >= len("exec:"):
            if arg_adjmatloss.split('#')[0][0:len("exec:")] == 'exec:':
                # get dim_input with exec
                exec('dim_input=' + arg_adjmatloss.split('#')[0][len("exec:")::])
                print("dim_input is set to {}".format(dim_input))
            else:
                # read the arg as int
                dim_input = int(arg_adjmatloss.split('#')[0])
        else:
            # read the arg as int
            dim_input = int(arg_adjmatloss.split('#')[0])

        assert (
                arg_adjmatloss.split('#')[5] in ['True', 'False']
        )

        kwargs_newadjmatpredloss = {
            'module_predictor': torch.nn.Sequential(
                torch.nn.Linear(dim_input, 200),
                torch.nn.ReLU(),
                torch.nn.Linear(200, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 2)
            ),
            'varname_1': arg_adjmatloss.split('#')[1],
            'varname_2': arg_adjmatloss.split('#')[2],
            'str_predicable_or_unpredictable': arg_adjmatloss.split('#')[3],
            'coef_loss': float(arg_adjmatloss.split('#')[4]),
            'flag_defineloss_onlyon_pygneighinternal': bool(arg_adjmatloss.split('#')[5] == 'True')
        }

        adjmatpredloss_toadd = AdjMatPredLoss(
            **kwargs_newadjmatpredloss
        )

        list_ajdmatpredloss.append(adjmatpredloss_toadd)
        print("added new ajdmatpred loss ")
        pprint(kwargs_newadjmatpredloss)
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


# dump the window sizes ========= TODO:HERE
# Train
path_window_overlays = os.path.join(
    args.path_output,
    'Toinspect_CropsOnTissues'
)
if not os.path.isdir(path_window_overlays):
    os.mkdir(path_window_overlays)
path_window_overlays = os.path.join(
    path_window_overlays,
    'Train'
)
if not os.path.isdir(path_window_overlays):
    os.mkdir(path_window_overlays)
list_slice.show_pygbatch_windows_4cli(
    path_output=path_window_overlays,
    str_train_or_test='train',
    dict_slideID_to_maxnumcentralnodes=dict_slideID_to_maxnumcentralnodes,
    dict_slideID_to_worsecasebatchsize=dict_slideID_to_worsecasebatchsize
)

# Test
path_window_overlays = os.path.join(
    args.path_output,
    'Toinspect_CropsOnTissues'
)
if not os.path.isdir(path_window_overlays):
    os.mkdir(path_window_overlays)
path_window_overlays = os.path.join(
    path_window_overlays,
    'Test'
)
if not os.path.isdir(path_window_overlays):
    os.mkdir(path_window_overlays)
test_list_slice.show_pygbatch_windows_4cli(
    path_output=path_window_overlays,
    str_train_or_test='test',
    dict_slideID_to_maxnumcentralnodes=None,
    dict_slideID_to_worsecasebatchsize=None
)


exec('disent_dict_CTNNC_usage = {}'.format(config_model['CTNCC_usage_moduledisent']))
assert (
    config_model['str_mode_headxint_headxspl_headboth_twosep'] in [
        'headxint', 'headxspl', 'headboth', 'twosep'
    ]
)
dict_temp = {
    'headxint':inflow.modules.gnn_disentangler.ModeArch.HEADXINT,
    'headxspl':inflow.modules.gnn_disentangler.ModeArch.HEADXSPL,
    'headboth':inflow.modules.gnn_disentangler.ModeArch.HEADBOTH,
    'twosep':inflow.modules.gnn_disentangler.ModeArch.TWOSEP,
}

type_impanddisentgl = GNNDisentangler
kwargs_impanddisentgl = {
    'dict_CTNNC_usage':disent_dict_CTNNC_usage,
    'kwargs_genmodel':kwargs_genmodel,
    'clipval_cov_noncentralnodes': config_model['clipval_cov_noncentralnodes'],
    'str_mode_normalizex':'counts',
    'maxsize_subgraph':maxsize_subgraph,
    'dict_general_args':{
        'num_genes':dict_varname_to_dim['x'],
        'num_celltypes':list_slice.list_slice[0]._global_num_CT,
        'flag_use_int_u':config_model['flag_use_int_u'],
        'flag_use_spl_u':config_model['flag_use_spl_u']
    },
    'mode_headxint_headxspl_headboth_twosep':dict_temp[config_model['str_mode_headxint_headxspl_headboth_twosep']],
    'gnn_list_dim_hidden':[],
    'kwargs_sageconv':{},
    'std_minval_finalclip':config_model['std_minval_finalclip'],
    'std_maxval_finalclip':config_model['std_maxval_finalclip'],
    'flag_use_layernorm':config_model['flag_use_layernorm_disentangler_enc1'],
    'flag_use_dropout':config_model['flag_use_dropout_disentangler_enc1'],
    'flag_enable_batchtoken':config_model['flag_enable_batchtoken_disentangler']
}

for kv in config_model['dict_qname_to_scaleandunweighted'].split("&"):
    assert (kv.split("#")[2] in ['True', 'False'])

dict_qname_to_scaleandunweighted = {
    kv.split("#")[0]:{
        'scale':float(kv.split("#")[1]),
        'flag_unweighted':kv.split("#")[2]=='True'
    }
    for kv in config_model['dict_qname_to_scaleandunweighted'].split("&")
}

if args.flag_verbose:
    print("\ndict_qname_to_scaleandunweighted is set to: ")
    pprint(dict_qname_to_scaleandunweighted)


# get flowmatching arguments ===
exec('flowmatching_mode_samplex0 = {}'.format(config_model['flowmatching_mode_samplex0']))
exec('flowmatching_mode_minibatchper = {}'.format(config_model['flowmatching_mode_minibatchper']))
exec('flowmatching_mode_timesched = {}'.format(config_model['flowmatching_mode_timesched']))
exec('flowmatching_mode_fmloss = {}'.format(config_model['flowmatching_mode_fmloss']))


exec(
    "module_encX_int = {}".format(
        config_model['arch_module_encoder_X2Xbar']
    )
)
exec(
    "module_encX_spl = {}".format(
        config_model['arch_module_encoder_X2Xbar']
    )
)

module_varphi_enc_int = EncX2Xbar(
    module_encX=module_encX_int,
    num_batches=kwargs_genmodel['dict_varname_to_dim']['BatchEmb'],
    dim_xbar=kwargs_genmodel['dict_varname_to_dim']['z'],
    flag_enable_batchEmb=config_model['flag_enable_batchtoken_encxbar']
)
module_varphi_enc_spl = EncX2Xbar(
    module_encX=module_encX_spl,
    num_batches=kwargs_genmodel['dict_varname_to_dim']['BatchEmb'],
    dim_xbar=kwargs_genmodel['dict_varname_to_dim']['z'],
    flag_enable_batchEmb=config_model['flag_enable_batchtoken_encxbar']
)

# module_varphi_enc_spl = module_varphi_enc_int


exec('dict_varname_to_takeCT_takeNCC = {}'.format(config_model['CTNCC_usage_modulecond4flow']))
exec('tmp_encZ_list_dim_hidden = {}'.format(
    config_model['enc3_encZ_list_dim_hidden']
))
exec('tmp_encSin_list_dim_hidden = {}'.format(
    config_model['enc3_encSin_list_dim_hidden']
))
exec('tmp_encSout_list_dim_hidden = {}'.format(
    config_model['enc3_encSout_list_dim_hidden']
))

type_cond4flowvarphi0 = Cond4FlowVarphi0SimpleMLPs
kwargs_cond4flowvarphi0 = {
    'kwargs_genmodel':kwargs_genmodel,
    'encZ_list_dim_hidden':tmp_encZ_list_dim_hidden,
    'encSin_list_dim_hidden':tmp_encSin_list_dim_hidden,
    'encSout_list_dim_hidden':tmp_encSout_list_dim_hidden,
    'dict_varname_to_takeCT_takeNCC':dict_varname_to_takeCT_takeNCC,
    'flag_use_layernorm':config_model['flag_use_layernorm_cond4flow_enc3'],
    'flag_use_dropout':config_model['flag_use_dropout_cond4flow_enc3']
}

if args.flag_verbose:
    print("\n\nThe way CTs/NCCs are fed to the 3rd encoder:")
    pprint(dict_varname_to_takeCT_takeNCC)




exec(
    "module_predictor_z2notNCC = {}".format(
        config_model['module_predictor_z2notNCC']
    )
)

exec(
    "module_predictor_xbarint2notNCC = {}".format(
        config_model['module_predictor_xbarint2notNCC']
    )
)

exec(
    "module_predictor_xbarint2notBatchID = {}".format(
        config_model['module_predictor_xbarint2notBatchID']
    )
)

exec(
    "module_predictor_xbarspl2notBatchID = {}".format(
        config_model['module_predictor_xbarspl2notBatchID']
    )
)


# create the vardist ====
exec(
    "module_classifier_P1loss = {}".format(
        config_model['module_classifier_P1loss']
    )
)
exec(
    "module_predictor_P3loss = {}".format(
        config_model['module_predictor_P3loss']
    )
)

exec(
    "module_classifier_xbarintCT = {}".format(
        config_model['module_classifier_xbarintCT']
    )
)

exec(
    "module_predictor_xbarsplNCC = {}".format(
        config_model['module_predictor_xbarsplNCC']
    )
)

dict_m_temp = {
    'module_predictor_z2notNCC':module_predictor_z2notNCC,
    'module_predictor_xbarint2notNCC':module_predictor_xbarint2notNCC,
    'module_predictor_xbarint2notBatchID':module_predictor_xbarint2notBatchID,
    'module_predictor_xbarspl2notBatchID':module_predictor_xbarspl2notBatchID,
    'module_classifier_P1loss':module_classifier_P1loss,
    'module_predictor_P3loss':module_predictor_P3loss,
    'module_classifier_xbarintCT':module_classifier_xbarintCT,
    'module_predictor_xbarsplNCC':module_predictor_xbarsplNCC
}
for k in dict_m_temp.keys():
    if not isinstance(dict_m_temp[k], torch.nn.Module):
        raise Exception(
            "In the provided config model (i.e. file {}), the provided module {} is not a pytorch module and is of type {} instead.\n Maybe a syntax error in the config file?".format(
                args.file_config_model,
                k,
                type(dict_m_temp[k])
            )
        )


module_vardist = BASE_PATH.vardist.InFlowVarDist(
    module_genmodel=BASE_PATH.generativemodel.InFlowGenerativeModel(
        **kwargs_genmodel
    ),
    type_impanddisentgl=type_impanddisentgl,
    kwargs_impanddisentgl=kwargs_impanddisentgl,
    module_varphi_enc_int=module_varphi_enc_int,
    module_varphi_enc_spl=module_varphi_enc_spl,
    type_cond4flowvarphi0=type_cond4flowvarphi0,
    kwargs_cond4flowvarphi0=kwargs_cond4flowvarphi0,
    dict_qname_to_scaleandunweighted=dict_qname_to_scaleandunweighted,
    list_ajdmatpredloss=ListAdjMatPredLoss(list_ajdmatpredloss),
    module_conditionalflowmatcher=ConditionalFlowMatcher(
        mode_samplex0=flowmatching_mode_samplex0,
        mode_minibatchper=flowmatching_mode_minibatchper,
        kwargs_otsampler={}, #TODO: maybe add mini-batch OT
        mode_timesched=flowmatching_mode_timesched,
        sigma=config_model['flowmatching_sigma'],
        mode_fmloss=flowmatching_mode_fmloss
    ),
    coef_P1loss=config_model['coef_loss_CTpredfromZ'], # config_model['coef_P1loss'],
    module_classifier_P1loss=module_classifier_P1loss,
    coef_P3loss=config_model['coef_loss_NCCpredfromSin'],#args.coef_P3loss,
    module_predictor_P3loss=module_predictor_P3loss,
    str_modeP3loss_regorcls=config_model['str_modeP3loss_regorcls'],
    module_annealing=LinearAnnealingSchedule(
        coef_min=config_model['anneal_logp_ZSout_coef_min'],
        coef_max=config_model['anneal_logp_ZSout_coef_max'],
        num_cyles=np.inf if(config_model['anneal_logp_ZSout_num_cycles'] == 'np.inf') else int(config_model['anneal_logp_ZSout_num_cycles']),
        numepochs_in_cycle=config_model['anneal_logp_ZSout_numepochs_in_cycle']
    ),
    weight_logprob_zinbpos=config_model['weight_logprob_zinbpos'],
    weight_logprob_zinbzero=config_model['weight_logprob_zinbzero'],
    flag_drop_loss_logQdisentangler=config_model['flag_drop_loss_logQdisentangler'],
    coef_xbarintCT_loss=config_model['coef_xbarintCT_loss'],
    module_classifier_xbarintCT=module_classifier_xbarintCT,
    coef_xbarsplNCC_loss=config_model['coef_xbarsplNCC_loss'],
    module_predictor_xbarsplNCC=module_predictor_xbarsplNCC,
    str_modexbarsplNCCloss_regorcls=config_model['str_modexbarsplNCCloss_regorcls'],
    coef_rankloss_xbarint=config_model['coef_rankloss_xbarint'],
    module_predictor_ranklossxbarint_X=torch.nn.Sequential(
        torch.nn.Linear(2*kwargs_genmodel['dict_varname_to_dim']['z'], kwargs_genmodel['dict_varname_to_dim']['z']),
        torch.nn.ReLU(),
        torch.nn.Linear(kwargs_genmodel['dict_varname_to_dim']['z'], 2)
    ),  # TODO:check: is linear better than, e.g., 2-layer?
    module_predictor_ranklossxbarint_Y=torch.nn.Sequential(
        torch.nn.Linear(2*kwargs_genmodel['dict_varname_to_dim']['z'], kwargs_genmodel['dict_varname_to_dim']['z']),
        torch.nn.ReLU(),
        torch.nn.Linear(kwargs_genmodel['dict_varname_to_dim']['z'], 2)
    ),  # TODO:check: is linear better than, e.g., 2-layer?,
    num_subsample_XYrankloss=config_model['num_subsample_XYrankloss'],
    coef_xbarint2notNCC_loss=config_model['coef_xbarint2notNCC_loss'],
    module_predictor_xbarint2notNCC=module_predictor_xbarint2notNCC,  # TODO:check: is linear better than, e.g., 2-layer?,
    str_modexbarint2notNCCloss_regorclsorwassdist=config_model['str_modexbarint2notNCCloss_regorclsorwassdist'],
    coef_z2notNCC_loss=config_model['coef_z2notNCC_loss'],
    module_predictor_z2notNCC=module_predictor_z2notNCC,  # TODO:check: is linear better than, e.g., 2-layer?,
    str_modez2notNCCloss_regorclsorwassdist=config_model['str_modez2notNCCloss_regorclsorwassdist'],
    coef_rankloss_Z=config_model['coef_rankloss_Z'],
    module_predictor_ranklossZ_X=torch.nn.Sequential(
        torch.nn.Linear(2*kwargs_genmodel['dict_varname_to_dim']['z'], kwargs_genmodel['dict_varname_to_dim']['z']),
        torch.nn.ReLU(),
        torch.nn.Linear(kwargs_genmodel['dict_varname_to_dim']['z'], 2)
    ),  # TODO:check: is linear better than, e.g., 2-layer?,
    module_predictor_ranklossZ_Y=torch.nn.Sequential(
        torch.nn.Linear(2*kwargs_genmodel['dict_varname_to_dim']['z'], kwargs_genmodel['dict_varname_to_dim']['z']),
        torch.nn.ReLU(),
        torch.nn.Linear(kwargs_genmodel['dict_varname_to_dim']['z'], 2)
    ), # TODO:check: is linear better than, e.g., 2-layer?,
    module_predictor_xbarint2notbatchID=module_predictor_xbarint2notBatchID,
    coef_xbarint2notbatchID_loss=config_model['coef_xbarint2notbatchID_loss'],
    module_predictor_xbarspl2notbatchID=module_predictor_xbarspl2notBatchID,
    coef_xbarspl2notbatchID_loss=config_model['coef_xbarspl2notbatchID_loss'],
    module_annealing_decoderXintXspl=AnnealingDecoderXintXspl(
        coef_min=config_training['annealing_decoder_XintXspl_coef_min'],
        coef_max=config_training['annealing_decoder_XintXspl_coef_max'],
        num_phase1=int(config_training['annealing_decoder_XintXspl_fractionepochs_phase1'] * config_training['num_training_epochs']),
        num_phase2=int(config_training['annealing_decoder_XintXspl_fractionepochs_phase2'] * config_training['num_training_epochs'])
    )
)
module_vardist.to(device)


if args.flag_verbose:
    print("Inflow vardist module was created on {}.".format(device))


# start a new wandb run to track this script
if config_training['flag_enable_wandb']:
    wandb.init(
        project=config_training['wandb_project_name'],
        name=config_training['wandb_run_name'],
        config={
            'dd':'dd'
        }
    )
itrcount_wandbstep = None

paramlist_optim = module_vardist.parameters()
flag_freezeencdec = False
optim_training = torch.optim.Adam(
    params=paramlist_optim,
    lr=config_training['lr_training']
)
optim_training.flag_freezeencdec = flag_freezeencdec

optim_afterGRLpreds = torch.optim.Adam(
    params=list(module_vardist.module_predictor_xbarint2notNCC.parameters()) +\
        list(module_vardist.module_predictor_z2notNCC.parameters()) +\
        list(module_vardist.module_predictor_xbarint2notbatchID.parameters()) +\
        list(module_vardist.module_predictor_xbarspl2notbatchID.parameters()),
    lr=config_training['lr_training']
)  # the optimizer for the dual functions (i.e. predictor Z2NotNCC, xbarint2NotNCC)
# TODO:NOTE:BUG module_predictor_xbarint2notbatchID and module_predictor_xbarspl2notbatchID had not been included,

# log the inflow module
with open(os.path.join(args.path_output, "log_inflow_module.txt"), 'w') as f:
    f.write(str(module_vardist))


if 'dict_measname_to_histmeas' not in globals():
    dict_measname_to_histmeas = {}
    dict_measname_to_evalpredxspl = {}
    total_cnt_epoch = 0
    list_coef_anneal = []

# dump the config dictionaries again, so any inconsistency (e.g. due to boolean variables being treated as str) becomes obvious.
tmp_check_unique = [
    os.path.split(args.file_config_data_train)[1],
    os.path.split(args.file_config_data_test)[1],
    os.path.split(args.file_config_model)[1],
    os.path.split(args.file_config_training)[1]
]
for u in tmp_check_unique:
    if tmp_check_unique.count(u) > 1:
        raise Exception(
            "In the provided config files the file name '{}' is repeated {} times, although probably in different directories. \n Please avoid this repeatition and try again".format(
                u,
                tmp_check_unique.count(u)
            )
        )

try_mkdir(os.path.join(args.path_output, 'ConfigFilesCopiedOver'))
os.system(
    "cp {} {}".format(
        os.path.abspath(args.file_config_data_train),
        os.path.join(args.path_output, 'ConfigFilesCopiedOver')
    )
)
os.system(
    "cp {} {}".format(
        os.path.abspath(args.file_config_data_test),
        os.path.join(args.path_output, 'ConfigFilesCopiedOver')
    )
)
os.system(
    "cp {} {}".format(
        os.path.abspath(args.file_config_model),
        os.path.join(args.path_output, 'ConfigFilesCopiedOver')
    )
)
os.system(
    "cp {} {}".format(
        os.path.abspath(args.file_config_training),
        os.path.join(args.path_output, 'ConfigFilesCopiedOver')
    )
)

with open(os.path.join(args.path_output, 'ConfigFilesCopiedOver', 'args.yml'), 'w') as f:
    yaml.dump(
        {
            'file_config_data_train':os.path.split(args.file_config_data_train)[1],
            'file_config_data_test':os.path.split(args.file_config_data_test)[1],
            'file_config_model':os.path.split(args.file_config_model)[1],
            'file_config_training':os.path.split(args.file_config_training)[1]
        },
        f,
        default_flow_style=False
    )


t_before_training = time.time()

for idx_epoch in range(config_training['num_training_epochs']):
    print("\n\nEpoch {} from {} ================ ".format(
        idx_epoch+1,
        config_training['num_training_epochs']
    ))
    # ten_Z, ten_xbarint, ten_CT, ten_NCC, ten_xy_absolute are obtained using all tissues.
    # update the dual functions separately =============
    with torch.no_grad():
        forduals_ten_Z, forduals_ten_CT, forduals_ten_NCC, forduals_ten_BatchEmb, \
            forduals_ten_xbarint, forduals_ten_xy_absolute, forduals_ten_xbarspl = \
            [], [], [], [], [], [], []

        print("     Getting different embeddings to update the dual functions separately.")
        for idx_sl, sl in enumerate(list_slice.list_slice):
            anal_dict_varname_to_output = module_vardist.eval_on_pygneighloader_dense(
                dl=sl.pyg_dl_test, # this is correct, because all neighbours are to be included (not a subset of neighbours).
                ten_xy_absolute=sl.ten_xy_absolute,
                tqdm_desc="Tissue {}".format(idx_sl)
            )

            forduals_ten_Z.append(
                torch.tensor(anal_dict_varname_to_output['mu_z'] + 0.0)
            )

            forduals_ten_CT.append(
                sl.ten_CT + 0.0
            )

            forduals_ten_NCC.append(
                sl.ten_NCC + 0.0
            )

            forduals_ten_BatchEmb.append(
                sl.ten_BatchEmb + 0.0
            )

            forduals_ten_xbarint.append(
                torch.tensor(anal_dict_varname_to_output['muxbar_int'] + 0.0)
            )

            forduals_ten_xy_absolute.append(
                sl.ten_xy_absolute + 0.0
            )

            forduals_ten_xbarspl.append(
                torch.tensor(anal_dict_varname_to_output['muxbar_spl'] + 0.0)
            )

            del anal_dict_varname_to_output

        forduals_ten_Z = torch.concat(forduals_ten_Z, 0)
        forduals_ten_CT = torch.concat(forduals_ten_CT, 0)
        forduals_ten_NCC = torch.concat(forduals_ten_NCC, 0)
        forduals_ten_BatchEmb = torch.concat(forduals_ten_BatchEmb, 0)
        forduals_ten_xbarint = torch.concat(forduals_ten_xbarint, 0)
        forduals_ten_xbarspl = torch.concat(forduals_ten_xbarspl, 0)
        forduals_ten_xy_absolute = torch.concat(forduals_ten_xy_absolute, 0)

    module_vardist._trainsep_GradRevPreds(
        optim_gradrevpreds=optim_afterGRLpreds,
        numiters=config_training['numiters_updateduals_seprately_perepoch'],
        ten_Z=forduals_ten_Z,
        ten_CT=forduals_ten_CT,
        ten_NCC=forduals_ten_NCC,
        ten_xbarint=forduals_ten_xbarint,
        ten_BatchEmb=forduals_ten_BatchEmb,
        ten_xbarspl=forduals_ten_xbarspl,
        ten_xy_absolute=forduals_ten_xy_absolute,
        # Note: the arg `ten_xy_absolute` is not internally used, but kept for backward comptbility.
        device=device,
        kwargs_dl={
            'batch_size':config_training['batchsize_updateduals_seprately_perepoch']
        }
    )

    # gccollect
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # train all modules ===============
    itrcount_wandbstep, list_coef_anneal_ = module_vardist.training_epoch(
        flag_lockencdec_duringtraining=False,  # unused arg
        dl=[sl.pyg_dl_train for sl in list_slice.list_slice],
        prob_maskknowngenes=0.0,  # unused arg
        t_num_steps=config_model['neuralODE_t_num_steps'],
        ten_xy_absolute=[sl.ten_xy_absolute for sl in list_slice.list_slice],
        optim_training=optim_training,
        tensorboard_stepsize_save=config_training['wandb_stepsize_log'],
        itrcount_wandbstep_input=itrcount_wandbstep,
        list_flag_elboloss_imputationloss=[True, False],  # unused arg
        coef_loss_closeness_zz=config_model['coef_loss_closeness_zz'],
        coef_loss_closeness_xbarintxbarint=config_model['coef_loss_closeness_xbarintxbarint'],
        coef_loss_closeness_xintxint=config_model['coef_loss_closeness_xintxint'],
        prob_applytfm_affinexy=0.0,  # unused arg
        coef_flowmatchingloss=config_model['coef_flowmatchingloss'],
        np_size_factor=[
            np.array(sl.adata.shape[0] * [config_training['val_scppnorm_total']]) for sl in list_slice.list_slice
        ],
        numsteps_accumgrad=config_training['numsteps_accumgrad'],
        num_updateseparate_afterGRLs=config_training['num_updateseparate_afterGRLs'],
        flag_verbose=False,
        flag_enable_wandb=config_training['flag_enable_wandb']
    )
    list_coef_anneal = list_coef_anneal + list_coef_anneal_
    total_cnt_epoch += 1




if args.flag_verbose:
    print("Training for {} epochs took {} seconds.".format(
        config_training['num_training_epochs'],
        time.time() - t_before_training
    ))


# gccollect
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
gc.collect()
time.sleep(config_training['sleeptime_gccollect_aftertraining'])


# load LR-DB and the ones found in the shared gene panel of tissues ===
df_LRpairs = pd.read_csv("./Files2Use_CLI/df_LRpairs_Armingoletal.txt")
list_known_LRgenes_inDB = [
    genename
    for colname in ['LigName', 'RecName'] for group in df_LRpairs[colname].tolist() for genename in str(group).split("__")
]
list_known_LRgenes_inDB = set(list_known_LRgenes_inDB)
list_LR = []
for gene_name in list_slice.list_slice[0].adata.var.index.tolist():
    if gene_name in list_known_LRgenes_inDB:
        list_LR.append(gene_name)

if args.flag_verbose:
    print("\n\n Among the {} genes in tissues' gene panels, {} genes were found in the ligand-receptor database.\n\n".format(
        len(list_slice.list_slice[0].adata.var.index.tolist()),
        len(list_LR)
    ))

# dump the inflow model as well as the inferred latent factors ===
path_dump_checkpoint = os.path.join(
    args.path_output,
    'CheckpointAndPredictions'
)
if not os.path.isdir(path_dump_checkpoint):
    os.mkdir(path_dump_checkpoint)

# dump predictions per-tissue
with torch.no_grad():
    for idx_sl, sl in enumerate(test_list_slice.list_slice):
        print("\n\n")

        anal_dict_varname_to_output_slice = module_vardist.eval_on_pygneighloader_dense(
            dl=test_list_slice.list_slice[idx_sl].pyg_dl_test,
            ten_xy_absolute=test_list_slice.list_slice[idx_sl].ten_xy_absolute,
            tqdm_desc="Evaluating on tissue {}".format(idx_sl+1)
        )
        '''
        anal_dict_varname_to_output_slice is a dict with the following keys:
        ['output_imputer',
         'muxint',
         'muxspl',
         'muxbar_int',
         'muxbar_spl',
         'mu_sin',
         'mu_sout',
         'mu_z',
         'x_int',
         'x_spl']
        '''

        # remove redundant fields ===
        anal_dict_varname_to_output_slice.pop('output_imputer', None)
        anal_dict_varname_to_output_slice.pop('x_int', None)
        anal_dict_varname_to_output_slice.pop('x_spl', None)


        # get pred_Xspl and pred_Xint before row normalisation on adata.X
        rowcoef_correct4scppnormtotal = (np.array(sl.adata_before_scppnormalize_total.X.sum(1).tolist()) + 0.0) / (config_training['val_scppnorm_total'] + 0.0)
        if len(rowcoef_correct4scppnormtotal.shape) == 1:
            rowcoef_correct4scppnormtotal = np.expand_dims(rowcoef_correct4scppnormtotal, -1)  # [N x 1]

        assert rowcoef_correct4scppnormtotal.shape[0] == sl.adata_before_scppnormalize_total.shape[0]
        assert rowcoef_correct4scppnormtotal.shape[1] == 1

        anal_dict_varname_to_output_slice['muxint_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxint'] * rowcoef_correct4scppnormtotal + 0.0
        anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxspl'] * rowcoef_correct4scppnormtotal + 0.0

        '''
        Sparsify the following vars
        - muxint
        - muxspl
        - muxint_before_sc_pp_normalize_total
        - muxspl_before_sc_pp_normalize_total
        -
        '''
        tmp_mask = test_list_slice.list_slice[idx_sl].adata.X + 0
        if issparse(tmp_mask):
            tmp_mask = tmp_mask.toarray()
        tmp_mask = ((tmp_mask > 0) + 0).astype(int)

        for var in [
            'muxint',
            'muxspl',
            'muxint_before_sc_pp_normalize_total',
            'muxspl_before_sc_pp_normalize_total'
        ]:
            anal_dict_varname_to_output_slice[var] = coo_matrix(anal_dict_varname_to_output_slice[var] * tmp_mask)

            '''
            The sparse format may have more 0-s than tmp_mask, so the check below was removed.
            if len(anal_dict_varname_to_output_slice[var].data) == tmp_mask.sum():
                path_debug_output = os.path.join(
                    args.path_output,
                    'DebugInfo'
                )
                try_mkdir(path_debug_output)

                # dump the anndata ===
                test_list_slice.list_slice[idx_sl].adata.write(
                    os.path.join(
                        path_debug_output,
                        'adata.h5ad'
                    )
                )

                # dump `tmp_mask` ===
                with open(os.path.join(path_debug_output, 'tmp_mask.pkl'), 'wb') as f:
                    pickle.dump(tmp_mask, f)

                # dump anal_dict_varname_to_output_slice[var]
                with open(os.path.join(path_debug_output, 'var_{}.pkl'.format(var)), 'wb') as f:
                    pickle.dump(
                        anal_dict_varname_to_output_slice[var],
                        f
                    )

                raise Exception(
                    "Something went wrong when trying to sparsify {}".format(var)
                )
            '''

            gc.collect()


        # dump the predictions
        torch.save(
            anal_dict_varname_to_output_slice,
            os.path.join(path_dump_checkpoint, 'predictions_slice_{}.pt'.format(idx_sl + 1)),
            pickle_protocol=4
        )


        # dump the jointplots ====
        path_result_disent = os.path.join(
            args.path_output,
            "Results"
        )
        try_mkdir(path_result_disent)

        path_result_disent = os.path.join(
            path_result_disent,
            "JointPlots"
        )
        try_mkdir(path_result_disent)

        try_mkdir(os.path.join(path_result_disent, 'Tissue_{}'.format(idx_sl+1)))

        if issparse(anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total']):
            anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'].toarray()
            # TODO:implement visualizations directly for sparse Xspl.

        disentanglement_jointplot.vis(
            adata_unnorm=sl.adata_before_scppnormalize_total,
            pred_Xspl_rownormcorrected=anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'],
            list_LR=list_LR,
            fname_dump_red=os.path.join(path_result_disent, 'Tissue_{}'.format(idx_sl+1), 'jointplot_red_{}.png'.format(idx_sl+1)),
            fname_dump_blue=os.path.join(path_result_disent, 'Tissue_{}'.format(idx_sl + 1), 'jointplot_blue_{}.png'.format(idx_sl+1)),
            str_sampleID=set(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']]),
            str_batchID=set(sl.adata.obs[sl.dict_obskey['biological_batch_key']])
        )

        # dump the violin plots ====
        path_violinplots = os.path.join(
            args.path_output,
            'Results',
            'ViolinPlots'
        )
        try_mkdir(path_violinplots)
        path_violinplots = os.path.join(
            path_violinplots,
            'ViolinPlots_Tissue_{}'.format(idx_sl+1)
        )
        try_mkdir(path_violinplots)

        if config_training['flag_finaleval_enable_pertissue_violinplot']:
            disentanglement_violinplot.vis(
                adata_unnorm=sl.adata_before_scppnormalize_total,
                pred_Xspl_rownormcorrected=anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'],
                min_cnt_vertical_slice=1,
                max_cnt_vertical_slice=int(sl.adata_before_scppnormalize_total.X.max()),
                list_LR=list_LR,
                path_dump=path_violinplots,
                str_sampleID=set(sl.adata.obs[sl.dict_obskey['sliceid_to_checkUnique']]),
                str_batchID=set(sl.adata.obs[sl.dict_obskey['biological_batch_key']]),
                idx_slplus1=idx_sl+1
            )


        if args.flag_verbose:
            print("Dumped predictions for testing tissue {}".format(idx_sl + 1))

        del anal_dict_varname_to_output_slice
        gc.collect()
        time.sleep(config_training['sleeptime_gccollect_dumpOnePred'])

# dump the inflow checkpoint
module_vardist.module_annealing = "NONE"  # so it can be dumped.
module_vardist.module_annealing_decoderXintXspl = "NONE"  # so it can be dumped.
torch.save(
    {
        'module_inflow': module_vardist,
     },
    os.path.join(
        path_dump_checkpoint,
        'inflow_model.pt'
    ),
    pickle_protocol=4
)


# dump the combined jointplots ====
if config_training['flag_finaleval_enable_alltissuecombined_eval']:
    list_predXspl = []
    for idx_sl, sl in enumerate(test_list_slice.list_slice):
        vects_sl = torch.load(
            os.path.join(
                args.path_output,
                'CheckpointAndPredictions',
                'predictions_slice_{}.pt'.format(idx_sl+1)
            )
        )

        if issparse(vects_sl['muxspl']):
            vects_sl['muxspl'] = vects_sl['muxspl'].toarray()  # TODO:implement visualizations directly for sparse Xspl.

        list_predXspl.append(vects_sl['muxspl_before_sc_pp_normalize_total'])

        del vects_sl
        gc.collect()
        gc.collect()
        gc.collect()
        gc.collect()

    alltissue_adata = anndata.concat([sl.adata_before_scppnormalize_total for sl in test_list_slice.list_slice])
    alltissue_pred_Xspl = np.concatenate(list_predXspl, 0)

    disentanglement_jointplot.vis(
        adata_unnorm=alltissue_adata,
        pred_Xspl_rownormcorrected=alltissue_pred_Xspl,
        list_LR=list_LR,
        fname_dump_red=os.path.join(args.path_output, 'Results', 'JointPlots', 'jointplot_alltissuescombined_red.png'),
        fname_dump_blue=os.path.join(args.path_output, 'Results', 'JointPlots', 'jointplot_alltissuescombined_blue.png'),
        str_sampleID='combined (all tissues)',
        str_batchID='combined (all tissues)'
    )

    if config_training['flag_finaleval_enable_alltissue_violinplot']:
        path_combined_violinplots = os.path.join(
            args.path_output,
            'Results',
            'ViolinPlots',
            'AllTissues_Combined'
        )
        try_mkdir(path_combined_violinplots)

        disentanglement_violinplot.vis(
            adata_unnorm=alltissue_adata,
            pred_Xspl_rownormcorrected=alltissue_pred_Xspl,
            min_cnt_vertical_slice=1,
            max_cnt_vertical_slice=int(alltissue_adata.X.max()),
            list_LR=list_LR,
            path_dump=path_combined_violinplots,
            str_sampleID='combined (all tissues)',
            str_batchID='combined (all tissues)',
            idx_slplus1="combined"
        )


# dump the tissue samples ===
path_dump_training_listtissue = os.path.join(
    args.path_output,
    "TrainingListTissue"
)
path_dump_testing_listtissue = os.path.join(
    args.path_output,
    "TestingListTissue"
)
try_mkdir(path_dump_training_listtissue)
try_mkdir(path_dump_testing_listtissue)

for idx_sl, sl in enumerate(list_slice.list_slice):
    # with open(os.path.join(path_dump_training_listtissue, 'tissue_tr_{}.pkl'.format(idx_sl+1)), 'wb') as f:
    #     pickle.dump(sl, f)

    torch.save(
        sl,
        os.path.join(path_dump_training_listtissue, 'tissue_tr_{}.pt'.format(idx_sl + 1)),
        pickle_protocol=4
    )

for idx_sl, sl in enumerate(test_list_slice.list_slice):

    # with open(os.path.join(path_dump_testing_listtissue, 'tissue_test_{}.pkl'.format(idx_sl+1)), 'wb') as f:
    #     pickle.dump(sl, f)

    torch.save(
        sl,
        os.path.join(path_dump_testing_listtissue, 'tissue_test_{}.pt'.format(idx_sl + 1)),
        pickle_protocol=4
    )


# (if enabled) combine all tissues in a single anndata and dump (with predictions in adata.obsm).
if config_training['flag_finaleval_createanndata_alltissuescombined']:
    gc.collect()
    gc.collect()
    time.sleep(1)

    # create the normalised version: adata_inflowOutput_norm
    list_anndata_norm = []
    for idx_sl, sl in enumerate(test_list_slice.list_slice):
        dict_temp = torch.load(
            os.path.join(path_dump_checkpoint, 'predictions_slice_{}.pt'.format(idx_sl + 1))
        )

        for k in dict_temp.keys():
            sl.adata.obsm['inflow_{}'.format(k)] = dict_temp[k] + 0.0

        list_anndata_norm.append(sl.adata)
        del dict_temp
        gc.collect()

    adata_norm = anndata.concat(list_anndata_norm)
    adata_norm.uns['inflow_map_CT_to_inflowCT'] = test_list_slice.map_CT_to_inflowCT
    adata_norm.uns['inflow_map_Batchname_to_inflowBatchID'] = test_list_slice.map_Batchname_to_inflowBatchID

    adata_norm.write_h5ad(
        os.path.join(
            args.path_output,
            'adata_inflowOutput_norm.h5ad'
        )
    )
    del adata_norm

    # create the unnormalised version: adata_inflowOutput_unnorm
    list_anndata_unnorm = []
    for idx_sl, sl in enumerate(test_list_slice.list_slice):
        dict_temp = torch.load(
            os.path.join(path_dump_checkpoint, 'predictions_slice_{}.pt'.format(idx_sl + 1))
        )

        dict_temp['muxint'] = dict_temp['muxint_before_sc_pp_normalize_total'] + 0.0  # muxint for the original adata.X
        dict_temp['muxspl'] = dict_temp['muxspl_before_sc_pp_normalize_total'] + 0.0  # muxspl for the original adata.X

        for k in dict_temp.keys():
            sl.adata_before_scppnormalize_total.obsm['inflow_{}'.format(k)] = dict_temp[k] + 0.0

        list_anndata_unnorm.append(sl.adata_before_scppnormalize_total)
        del dict_temp
        gc.collect()

    adata_unnorm = anndata.concat(list_anndata_unnorm)
    adata_unnorm.uns['inflow_map_CT_to_inflowCT'] = test_list_slice.map_CT_to_inflowCT
    adata_unnorm.uns['inflow_map_Batchname_to_inflowBatchID'] = test_list_slice.map_Batchname_to_inflowBatchID

    adata_unnorm.write_h5ad(
        os.path.join(
            args.path_output,
            'adata_inflowOutput_unnorm.h5ad'
        )
    )


# copy over the README files to each output folder ===
os.system(
    "cp {} {}".format(
        "./Files2Use_CLI/README_OutputPath.md",
        os.path.join(
            os.path.abspath(args.path_output),
            'README_InflowOutput.md'
        )
    )
)

for u in [
    'CheckpointAndPredictions',
    'Results',
    'TestingListTissue',
    'Toinspect_CropsOnTissues',
    'Toinspect_NeighbourhoodGraphs',
    'TrainingListTissue'
]:
    if os.path.isdir(os.path.join(args.path_output, u)):  # because some could be disabled in the training config
        os.system(
            "cp {} {}".format(
                "./Files2Use_CLI/README_{}.md".format(u),
                os.path.join(
                    os.path.abspath(args.path_output),
                    u
                )
            )
        )



print("Finished running the script.")



# TODO: check if any of the config files contain true, flase or True/False without double-quotation s.

# TODO: check if all tissue-s (including the ones in training and testing) share the same gene panel.

# TODO: check if all tissue-s (training/testing) share the same set of cell types.


# TODO: copy the config files to args.path_output.



# TODO: in the above part `correct for ...` add also x_int and x_spl as well.


