

# general imports ====
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

# mintflow imports ====

from .. import utils

from .. import generativemodel # exec('import {}.generativemodel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .. import modules  # exec('import {}.modules.gnn'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..modules import gnn, neuralODE, mlp, disentonly


from ..modules.impanddisentgl import MaskLabel #import exec('from {}.modules.impanddisentgl import MaskLabel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .. import vardist  # exec('import {}.vardist'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .. import masking # exec('import {}.masking'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..modules.impanddisentgl import ImputerAndDisentangler # exec('from {}.modules.impanddisentgl import ImputerAndDisentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..modules.disentonly import Disentangler # exec('from {}.modules.disentonly import Disentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..modules.disentonly_twosep import DisentanglerTwoSep # exec('from {}.modules.disentonly_twosep import DisentanglerTwoSep'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..zs_samplers import RandomZSSampler, PerCelltypeZSSampler #exec('from {}.zs_samplers import RandomZSSampler, PerCelltypeZSSampler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..predadjmat import ListAdjMatPredLoss, AdjMatPredLoss #  exec('from {}.predadjmat import ListAdjMatPredLoss, AdjMatPredLoss'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from ..utils_flowmatching import ModeSampleX0, ModeMinibatchPerm, ModeTimeSched, ModeFMLoss, ConditionalFlowMatcher

from ..modules.cond4flow import Cond4FlowVarphi0

from ..modules.cond4flow_simple3mpls import Cond4FlowVarphi0SimpleMLPs

from ..utils_pyg import PygSTDataGridBatchSampler


from ..evaluation.bioconsv import EvaluatorKmeans, EvaluatorLeiden

from ..evaluation.predxspl import EvalXsplpred, EvalLargeReadoutsXsplpred, EvalOnHVGsXsplpred

from ..modules.gnn_disentangler import GNNDisentangler

from ..kl_annealing import LinearAnnealingSchedule

from ..modules.predictorperCT import PredictorPerCT

from ..utils_multislice import ListSlice, Slice

from ..modules.varphienc4xbar import EncX2Xbar

from ..modules.predictorbatchID import PredictorBatchID


from . import \
    get_defaultconfig_data_train, verify_config_data_train,\
    get_defaultconfig_data_evaluation, verify_config_data_evaluation,\
    get_defaultconfig_model, verify_and_postprocess_config_model,\
    get_defaultconfig_training, verify_and_postprocess_config_training


from .auxiliary_modules import *

#
# from .interface.analresults import disentanglement_jointplot
#
# from .interface.analresults import disentanglement_violinplot

# from . import interface

from ..anneal_decoder_xintxspl import AnnealingDecoderXintXspl



# TODO:modif
args = types.SimpleNamespace()
args.flag_verbose = "DDD"

config_data_train, config_data_test, config_model, config_training = None, None, None, None  # TODO:complete


# TODO:take in `module_vardist`
module_vardist = "DDDD"

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


if 'dict_measname_to_histmeas' not in globals():
    dict_measname_to_histmeas = {}
    dict_measname_to_evalpredxspl = {}
    total_cnt_epoch = 0
    list_coef_anneal = []



