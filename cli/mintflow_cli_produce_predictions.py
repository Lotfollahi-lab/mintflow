
"""
After running the CLI via `python mintflow_cli.py ... ` the code may crash due to, e.g., memory issue before some results are dumpued in the specified `path_output`.
The current script goes over the checkpoints in `CheckpointsAndPredictions` in the output path and creates the predictions as well as
"""

#use inflow or inflow_synth
STR_INFLOW_OR_INFLOW_SYNTH = "inflow"  # in ['inflow', 'inflow_synth']
assert(
    STR_INFLOW_OR_INFLOW_SYNTH == 'inflow'  # in ['inflow', 'inflow_synth']
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
    description='For usage intrcutions please refer to the documentation under "Recovering CLI Outputs".',
    formatter_class=RawTextHelpFormatter
)

parser.add_argument(
    '--original_CLI_run_path_output',
    type=str,
    help='The original output path specified when running mintflow CLI.\n' +\
    'In other words, the `path_output` passed to the CLI when running `python mintflow_cli_train_model.py ....`.'
)

parser.add_argument(
    '--flag_use_cuda',
    type=str,
    help="A string in ['True', 'False']"
)

parser.add_argument(
    '--flag_dump_anndata_objects',
    type=str,
    help="If set to True, this scripot dumps the anndata objects with MintFlow predictions in its .obsm field. A string in ['True', 'False']\n"+\
         "If you face out of memory issues, you can set this argument to 'False'."
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

assert isinstance(args.flag_use_cuda, str)
assert args.flag_use_cuda in ['True', 'False']
args.flag_use_cuda = (args.flag_use_cuda == 'True')

assert isinstance(args.flag_dump_anndata_objects, str)
assert args.flag_dump_anndata_objects in ['True', 'False']
args.flag_dump_anndata_objects = (args.flag_dump_anndata_objects == 'True')

# find the mapping of the config file names (important when the config files have been modified and are potentially irrelevant names)
with open(
    os.path.join(
        args.original_CLI_run_path_output,
        'ConfigFilesCopiedOver',
        'args.yml'
    )
) as f:
    try:
        dict_resconfignames_to_actualfnames = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)
print(dict_resconfignames_to_actualfnames)



# parse the config files ===
config_data_train = parse_config_data_train.parse(
    os.path.join(
        args.original_CLI_run_path_output,
        'ConfigFilesCopiedOver',
        dict_resconfignames_to_actualfnames['file_config_data_train']
    )
)
config_data_test = parse_config_data_test.parse(
    os.path.join(
        args.original_CLI_run_path_output,
        'ConfigFilesCopiedOver',
        dict_resconfignames_to_actualfnames['file_config_data_test']
    )
)

config_training = parse_config_training.parse(
    os.path.join(
        args.original_CLI_run_path_output,
        'ConfigFilesCopiedOver',
        dict_resconfignames_to_actualfnames['file_config_training']
    )
)

config_model = parse_config_model.parse(
    os.path.join(
        args.original_CLI_run_path_output,
        'ConfigFilesCopiedOver',
        dict_resconfignames_to_actualfnames['file_config_model']
    )
)


# set device ===
if args.flag_use_cuda: #config_training['flag_use_GPU']:
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
    print("\n\nLoaded the training list of tissue.")
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



if args.flag_verbose:
    print("\n\n\n")
    print("The provided cell types are aggregated/mapped to inflow cell types as follows:")
    pprint(list_slice.map_CT_to_inflowCT)
    print("\n\n")

if args.flag_verbose:
    print("\n\n\n")
    print("The provided biological batch IDs are aggregated/mapped to inflow batch IDs as follows")
    pprint(list_slice.map_Batchname_to_inflowBatchID)
    print("\n\n")

# Note: due to the implementation in `utils_multislice.py` the assigned cell type and batchIDs do not vary in different runs.
# TODO: double-check it via the dumped general info in the output path

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


# Loop over the dumped checkpoints
for fname_checkpoint in os.listdir(os.path.join(args.original_CLI_run_path_output, 'CheckpointAndPredictions')):
    if fname_checkpoint.endswith(".pt"):
        if len(fname_checkpoint) > len('mintflow_checkpoint_epoch_'):
            if fname_checkpoint[0:len('mintflow_checkpoint_epoch_')] == 'mintflow_checkpoint_epoch_':
                # load the checkpoint
                module_vardist = torch.load(
                    os.path.join(
                        args.original_CLI_run_path_output,
                        'CheckpointAndPredictions',
                        fname_checkpoint
                    ),
                    map_location=device
                )['module_inflow']

                print("Loaded the mintflow module on device {} from checkpiont {}".format(
                    device,
                    os.path.join(
                        args.original_CLI_run_path_output,
                        'CheckpointAndPredictions',
                        fname_checkpoint
                    )
                ))

                torch.cuda.empty_cache()
                gc.collect()

                # dump the predictions per-tissue
                dict_oldvarname_to_newvarname = {
                    'muxint':'MintFlow_X_int',
                    'muxspl':'MintFlow_X_mic',
                    'muxbar_int':'MintFlow_Xbar_int',
                    'muxbar_spl':'MintFlow_Xbar_mic',
                    'mu_sin':'MintFlow_S_in',
                    'mu_sout':'MintFlow_S_out',
                    'mu_z':'MintFlow_Z'
                }  # map names according to the latest glossery of the manuscript.
                with torch.no_grad():
                    for idx_sl, sl in enumerate(test_list_slice.list_slice):
                        print("\n\n")

                        anal_dict_varname_to_output_slice = module_vardist.eval_on_pygneighloader_dense(
                            dl=test_list_slice.list_slice[idx_sl].pyg_dl_test,
                            ten_xy_absolute=test_list_slice.list_slice[idx_sl].ten_xy_absolute,
                            tqdm_desc="Evaluating on tissue {}".format(idx_sl + 1)
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

                        anal_dict_varname_to_output_slice['muxint_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxint'].multiply(
                            rowcoef_correct4scppnormtotal
                        )
                        anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'] = anal_dict_varname_to_output_slice['muxspl'].multiply(
                            rowcoef_correct4scppnormtotal
                        )

                        # replace the keys in dictionary
                        for k_old, k_new in dict_oldvarname_to_newvarname.items():
                            anal_dict_varname_to_output_slice[k_new] = anal_dict_varname_to_output_slice.pop(k_old)

                        # dump the predictions
                        try_mkdir(
                            os.path.join(
                                args.original_CLI_run_path_output,
                                'CheckpointAndPredictions',
                                "Predictions_And_Evaluation_{}".format(fname_checkpoint[0:-3])
                            )
                        )

                        torch.save(
                            anal_dict_varname_to_output_slice,
                            os.path.join(
                                args.original_CLI_run_path_output,
                                'CheckpointAndPredictions',
                                "Predictions_And_Evaluation_{}".format(fname_checkpoint[0:-3]),
                                'perdictions_tissue_section_{}.pt'.format(idx_sl+1)
                            ),
                            pickle_protocol=4
                        )



                        # Dump anndata objects if needed
                        if args.flag_dump_anndata_objects:
                            adata_todump = sl.adata
                            for varname in anal_dict_varname_to_output_slice.keys():
                                adata_todump.obsm[varname] = anal_dict_varname_to_output_slice[varname]

                            adata_todump.write_h5ad(
                                os.path.join(
                                    args.original_CLI_run_path_output,
                                    'CheckpointAndPredictions',
                                    "Predictions_And_Evaluation_{}".format(fname_checkpoint[0:-3]),
                                    'anndata_tissue_section_{}.pt'.format(idx_sl + 1)
                                )
                            )



                        del anal_dict_varname_to_output_slice
                        gc.collect()


                del module_vardist
                gc.collect()






torch.cuda.empty_cache()
gc.collect()


print("Finished running the script successfully!")



