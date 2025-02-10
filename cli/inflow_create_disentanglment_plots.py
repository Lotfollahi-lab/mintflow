

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
with open(os.path.join(args.path_output_inflow_cli_dot_py, 'ConfigFilesCopiedOver', 'args.yml'), 'r') as f:
    dict_args_inflowclidotpy = yaml.load(f, Loader=yaml.Loader)  # the path tails are saved only.


def try_mkdir(path_in):
    if not os.path.isdir(path_in):
        os.mkdir(path_in)



# read the test anndata-s 1by1 ===
config_data_test = parse_config_data_test.parse(
    os.path.join(
        args.path_output_inflow_cli_dot_py,
        'ConfigFilesCopiedOver',
        dict_args_inflowclidotpy['file_config_data_test']
    )
)



for idx_sl, config_anndata_test in enumerate(config_data_test):

    # read the anndata and predictions ===
    adata_before_scppnormalize_total = sc.read_h5ad(
        config_anndata_test['file']
    )

    # load LR-DB and the ones found in the shared gene panel of tissues ===
    if 'list_LR' not in globals():
        df_LRpairs = pd.read_csv("./Files2Use_CLI/df_LRpairs_Armingoletal.txt")
        list_known_LRgenes_inDB = [
            genename
            for colname in ['LigName', 'RecName'] for group in df_LRpairs[colname].tolist() for genename in str(group).split("__")
        ]
        list_known_LRgenes_inDB = set(list_known_LRgenes_inDB)
        list_LR = []
        for gene_name in adata_before_scppnormalize_total.var.index.tolist():
            if gene_name in list_known_LRgenes_inDB:
                list_LR.append(gene_name)

        print(">>>>>>>>>>>>>> {} genes were found in the LR-DB.".format(len(list_LR)))

    path_dump_checkpoint = os.path.join(
        args.path_output_inflow_cli_dot_py,
        'CheckpointAndPredictions'
    )
    if args.flag_verbose:
        print("\n\n ..... Loading predictions from {}.\n\n".format(
            os.path.join(path_dump_checkpoint, 'predictions_slice_{}.pt'.format(idx_sl + 1))
        ))

    anal_dict_varname_to_output_slice = torch.load(
        os.path.join(path_dump_checkpoint, 'predictions_slice_{}.pt'.format(idx_sl + 1)),
        map_location='cpu'
    )

    if args.flag_verbose:
        print("Loaded checkpoint and predictions for slice {}".format(idx_sl + 1))


    assert (
        anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'].shape[0] == adata_before_scppnormalize_total.shape[0]
    )

    break  # ---override for debug TODO:revert

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

    print("\n\n .... Creating joint plots in {}/Tissue_{}/".format(path_result_disent, idx_sl+1))

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
        str_sampleID=set(adata_before_scppnormalize_total.obs[config_anndata_test['obskey_sliceid_to_checkUnique']]),
        str_batchID=set(adata_before_scppnormalize_total.obs[config_anndata_test['obskey_biological_batch_key']])
    )

    # dump the violin plots ====
    path_violinplots = os.path.join(
        args.path_output_inflow_cli_dot_py,
        'Results',
        'ViolinPlots'
    )
    try_mkdir(path_violinplots)
    path_violinplots = os.path.join(
        path_violinplots,
        'ViolinPlots_Tissue_{}'.format(idx_sl + 1)
    )
    try_mkdir(path_violinplots)

    print("\n\n .... Creating violin plots in {}.".format(path_violinplots))
    disentanglement_violinplot.vis(
        adata_unnorm=adata_before_scppnormalize_total,
        pred_Xspl_rownormcorrected=anal_dict_varname_to_output_slice['muxspl_before_sc_pp_normalize_total'],
        min_cnt_vertical_slice=1,
        max_cnt_vertical_slice=int(adata_before_scppnormalize_total.X.max()),
        list_LR=list_LR,
        path_dump=path_violinplots,
        str_sampleID=set(adata_before_scppnormalize_total.obs[config_anndata_test['obskey_sliceid_to_checkUnique']]),
        str_batchID=set(adata_before_scppnormalize_total.obs[config_anndata_test['obskey_biological_batch_key']]),
        idx_slplus1=idx_sl + 1
    )

    del adata_before_scppnormalize_total, anal_dict_varname_to_output_slice
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    time.sleep(30)  # TODO: make tunable




# dump the combined jointplots ====
list_predXspl = []
for idx_sl, _ in enumerate(config_data_test):
    vects_sl = torch.load(
        os.path.join(
            args.path_output_inflow_cli_dot_py,
            'CheckpointAndPredictions',
            'predictions_slice_{}.pt'.format(idx_sl+1)
        )
    )
    list_predXspl.append(vects_sl['muxspl_before_sc_pp_normalize_total'])

    del vects_sl
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()

    time.sleep(30)  # TODO: make tunable

alltissue_adata = anndata.concat([sc.read_h5ad(config_anndata_test['file']) for config_anndata_test in config_data_test])
alltissue_pred_Xspl = np.concatenate(list_predXspl, 0)

disentanglement_jointplot.vis(
    adata_unnorm=alltissue_adata,
    pred_Xspl_rownormcorrected=alltissue_pred_Xspl,
    list_LR=list_LR,
    fname_dump_red=os.path.join(args.path_output_inflow_cli_dot_py, 'Results', 'JointPlots', 'jointplot_alltissuescombined_red.png'),
    fname_dump_blue=os.path.join(args.path_output_inflow_cli_dot_py, 'Results', 'JointPlots', 'jointplot_alltissuescombined_blue.png'),
    str_sampleID='combined (all tissues)',
    str_batchID='combined (all tissues)'
)

path_combined_violinplots = os.path.join(
    args.path_output_inflow_cli_dot_py,
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



