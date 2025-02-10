

#use inflow or inflow_synth
STR_INFLOW_OR_INFLOW_SYNTH = "inflow"  # in ['inflow', 'inflow_synth']
assert(
    STR_INFLOW_OR_INFLOW_SYNTH == 'inflow' #  in ['inflow', 'inflow_synth']
)

import os, sys
import warnings
import scipy
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
    description='This script creates (or recreates) the distanglments figures (violin plots/joint plots).\n This sciprts takes in the output path when `inflow_cli.py` was originally run.',
    formatter_class=RawTextHelpFormatter
)

parser.add_argument(
    '--path_output_inflow_cli_dot_py',
    type=str,
    help="The output path when `inflow_cli.py` was originally run."
)

parser.add_argument(
    '--flag_verbose',
    type=str,
    help="Whether the script is verbose, a string in ['True', 'False']"
)

args = parser.parse_args()
print("args = {}".format(args)) # ======================================================

# modify/check args ===
assert isinstance(args.flag_verbose, str)
assert args.flag_verbose in ['True', 'False']
args.flag_verbose = (args.flag_verbose == 'True')


# read the original args when inflow_cli.py script was run ===
with open(os.path.join(path_output_inflow_cli_dot_py, 'ConfigFilesCopiedOver', 'args.yml'), 'r') as f:
    dict_args_inflowclidotpy = yaml.load(f)


def try_mkdir(path_in):
    if not os.path.isdir(path_in):
        os.mkdir(path_in)



# read the test anndata-s 1by1 ===
config_data_test = parse_config_data_test.parse(
    dict_args_inflowclidotpy['file_config_data_test']
)

for idx_sl, config_anndata_test in enumerate(config_data_test):
    # read the anndata and predictions ===
    adata_before_scppnormalize_total = sc.read_h5ad(
        config_anndata_test['file']
    )

    path_dump_checkpoint = os.path.join(
        args.path_output_inflow_cli_dot_py,
        'CheckpointAndPredictions'
    )
    anal_dict_varname_to_output_slice = torch.load(
        os.path.join(path_dump_checkpoint, 'predictions_slice_{}.pt'.format(idx_sl + 1)),
        map_location='cpu'
    )

    if args.flag_verbose:
        print("Loaded checkpoint and predictions for slice {}".format(idx_sl))


    assert (
        anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'].shape[0] == adata_before_scppnormalize_total.shape[0]
    )

    # dump the jointplots ====
    path_result_disent = os.path.join(
        args.path_output_inflow_cli_dot_py,
        "Results"
    )
    try_mkdir(path_result_disent)

    path_result_disent = os.path.join(
        path_result_disent,
        "JointPlots"
    )
    try_mkdir(path_result_disent)

    try_mkdir(os.path.join(path_result_disent, 'Tissue_{}'.format(idx_sl + 1)))

    print("Creating joint plots in {}/Tissue_{}/".format(path_result_disent, idx_sl+1))

    disentanglement_jointplot.vis(
        adata_unnorm=adata_before_scppnormalize_total,
        pred_Xspl_rownormcorrected=anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'],
        list_LR=list_LR,
        fname_dump_red=os.path.join(
            path_result_disent,
            'Tissue_{}'.format(idx_sl + 1),
            'jointplot_red_{}.png'.format(idx_sl + 1)
        ),
        fname_dump_blue=os.path.join(
            path_result_disent,
            'Tissue_{}'.format(idx_sl + 1),
            'jointplot_blue_{}.png'.format(idx_sl + 1)
        ),
        str_sampleID=set(adata_before_scppnormalize_total.obs[config_anndata_test['sliceid_to_checkUnique']]),
        str_batchID=set(adata_before_scppnormalize_total.obs[config_anndata_test['biological_batch_key']])
    )

