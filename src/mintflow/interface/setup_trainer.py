

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
    get_defaultconfig_data_train, verify_and_postprocess_config_data_train,\
    get_defaultconfig_data_evaluation, verify_and_postprocess_config_data_evaluation,\
    get_defaultconfig_model, verify_and_postprocess_config_model,\
    get_defaultconfig_training, verify_and_postprocess_config_training


from .auxiliary_modules import *

#
# from .interface.analresults import disentanglement_jointplot
#
# from .interface.analresults import disentanglement_violinplot

# from . import interface

from ..anneal_decoder_xintxspl import AnnealingDecoderXintXspl


class Trainer:
    def __init__(self, dict_all4_configs, model, data_mintflow):
        if not isinstance(model, vardist.InFlowVarDist):
            raise Exception(
                "The passed `model` to `Trainer` is not of proper type. Make sure it's created and returned by `mintflow.setup_data`"
            )

        # grab args
        self.config_data_train, self.config_data_test, self.config_model, self.config_training = \
            dict_all4_configs['config_data_train'], \
            dict_all4_configs['config_data_evaluation'], \
            dict_all4_configs['config_model'], \
            dict_all4_configs['config_training']

        self.model = model
        self.data_mintflow = data_mintflow

        # init
        self.init_wandb()
        self.init_optimisers()

    def train_one_epoch(self):

        list_slice = self.data_mintflow['train_list_tissue_section']

        # ten_Z, ten_xbarint, ten_CT, ten_NCC, ten_xy_absolute are obtained using all tissues.
        # update the dual functions separately =============
        with torch.no_grad():
            forduals_ten_Z, forduals_ten_CT, forduals_ten_NCC, forduals_ten_BatchEmb, \
                forduals_ten_xbarint, forduals_ten_xy_absolute, forduals_ten_xbarspl = \
                [], [], [], [], [], [], []

            print("     Getting different embeddings to update the dual functions separately.")
            for idx_sl, sl in enumerate(list_slice.list_slice):
                anal_dict_varname_to_output = self.model.eval_on_pygneighloader_dense(
                    dl=sl.pyg_dl_test,
                    # this is correct, because all neighbours are to be included (not a subset of neighbours).
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

        self.model._trainsep_GradRevPreds(
            optim_gradrevpreds=self.optim_afterGRLpreds,
            numiters=self.config_training['numiters_updateduals_seprately_perepoch'],
            ten_Z=forduals_ten_Z,
            ten_CT=forduals_ten_CT,
            ten_NCC=forduals_ten_NCC,
            ten_xbarint=forduals_ten_xbarint,
            ten_BatchEmb=forduals_ten_BatchEmb,
            ten_xbarspl=forduals_ten_xbarspl,
            ten_xy_absolute=forduals_ten_xy_absolute,
            # Note: the arg `ten_xy_absolute` is not internally used, but kept for backward comptbility.
            device=list_slice.list_slice[0].device,
            kwargs_dl={
                'batch_size': self.config_training['batchsize_updateduals_seprately_perepoch']
            }
        )

        # train all modules ===============
        self.itrcount_wandbstep, list_coef_anneal_ = self.model.training_epoch(
            flag_lockencdec_duringtraining=False,  # unused arg
            dl=[sl.pyg_dl_train for sl in list_slice.list_slice],
            prob_maskknowngenes=0.0,  # unused arg
            t_num_steps=self.config_model['neuralODE_t_num_steps'],
            ten_xy_absolute=[sl.ten_xy_absolute for sl in list_slice.list_slice],
            optim_training=self.optim_training,
            tensorboard_stepsize_save=self.config_training['wandb_stepsize_log'],
            itrcount_wandbstep_input=self.itrcount_wandbstep,
            list_flag_elboloss_imputationloss=[True, False],  # unused arg
            coef_loss_closeness_zz=self.config_model['coef_loss_closeness_zz'],
            coef_loss_closeness_xbarintxbarint=self.config_model['coef_loss_closeness_xbarintxbarint'],
            coef_loss_closeness_xintxint=self.config_model['coef_loss_closeness_xintxint'],
            prob_applytfm_affinexy=0.0,  # unused arg
            coef_flowmatchingloss=self.config_model['coef_flowmatchingloss'],
            np_size_factor=[
                np.array(sl.adata.shape[0] * [self.config_training['val_scppnorm_total']]) for sl in list_slice.list_slice
            ],
            numsteps_accumgrad=self.config_training['numsteps_accumgrad'],
            num_updateseparate_afterGRLs=self.config_training['num_updateseparate_afterGRLs'],
            flag_verbose=False,
            flag_enable_wandb=self.config_training['flag_enable_wandb']
        )


        # gccollect
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def init_wandb(self):
        # start a new wandb run to track this script
        if self.config_training['flag_enable_wandb']:
            wandb.init(
                project=self.config_training['wandb_project_name'],
                name=self.config_training['wandb_run_name'],
                config={
                    'dd': 'dd'
                }
            )

        self.itrcount_wandbstep = None

    def init_optimisers(self):
        paramlist_optim = self.model.parameters()
        flag_freezeencdec = False
        optim_training = torch.optim.Adam(
            params=paramlist_optim,
            lr=self.config_training['lr_training']
        )
        optim_training.flag_freezeencdec = flag_freezeencdec

        optim_afterGRLpreds = torch.optim.Adam(
            params=list(self.model.module_predictor_xbarint2notNCC.parameters()) + \
                   list(self.model.module_predictor_z2notNCC.parameters()) + \
                   list(self.model.module_predictor_xbarint2notbatchID.parameters()) + \
                   list(self.model.module_predictor_xbarspl2notbatchID.parameters()),
            lr=self.config_training['lr_training']
        )  # the optimizer for the dual functions (i.e. predictor Z2NotNCC, xbarint2NotNCC)
        # TODO:NOTE:BUG module_predictor_xbarint2notbatchID and module_predictor_xbarspl2notbatchID had not been included,
        self.optim_training = optim_training
        self.optim_afterGRLpreds = optim_afterGRLpreds








# TODO:modif
args = types.SimpleNamespace()
args.flag_verbose = "DDD"

config_data_train, config_data_test, config_model, config_training = None, None, None, None  # TODO:complete


# TODO:take in `module_vardist`
module_vardist = "DDDD"



# # start a new wandb run to track this script
# if config_training['flag_enable_wandb']:
#     wandb.init(
#         project=config_training['wandb_project_name'],
#         name=config_training['wandb_run_name'],
#         config={
#             'dd':'dd'
#         }
#     )
# itrcount_wandbstep = None

# paramlist_optim = module_vardist.parameters()
# flag_freezeencdec = False
# optim_training = torch.optim.Adam(
#     params=paramlist_optim,
#     lr=config_training['lr_training']
# )
# optim_training.flag_freezeencdec = flag_freezeencdec
#
# optim_afterGRLpreds = torch.optim.Adam(
#     params=list(module_vardist.module_predictor_xbarint2notNCC.parameters()) +\
#         list(module_vardist.module_predictor_z2notNCC.parameters()) +\
#         list(module_vardist.module_predictor_xbarint2notbatchID.parameters()) +\
#         list(module_vardist.module_predictor_xbarspl2notbatchID.parameters()),
#     lr=config_training['lr_training']
# )  # the optimizer for the dual functions (i.e. predictor Z2NotNCC, xbarint2NotNCC)
# # TODO:NOTE:BUG module_predictor_xbarint2notbatchID and module_predictor_xbarspl2notbatchID had not been included,


if 'dict_measname_to_histmeas' not in globals():
    dict_measname_to_histmeas = {}
    dict_measname_to_evalpredxspl = {}
    total_cnt_epoch = 0
    list_coef_anneal = []



