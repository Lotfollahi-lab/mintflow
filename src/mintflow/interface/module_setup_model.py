


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


from ..modules.impanddisentgl import  MaskLabel #import exec('from {}.modules.impanddisentgl import MaskLabel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
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
from .. import modules
from ..modules import *

from .. import utils_multislice



#
# from .interface.analresults import disentanglement_jointplot
#
# from .interface.analresults import disentanglement_violinplot

# from . import interface

from ..anneal_decoder_xintxspl import AnnealingDecoderXintXspl


def setup_model(
    dict_all4_configs,
    data_mintflow,
    flag_verbose=True,
    flag_visualise_tissue_sections=True
):
    """

    :param dict_all4_configs: a dictionary with the following keys and their corresponding values
        - config_data_train
        - config_data_evaluation
        - config_model
        - config_training
    :param data_mintflow: an object returned by the function `mintflow.setup_data`
    :param flag_verbose:
    :param flag_visualise_tissue_sections:
    :return:
    """
    # check the 2nd arg
    flag_isvalid_arg_mintflow_data = True
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and isinstance(data_mintflow, dict)
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and set(data_mintflow.keys()) == {'train_list_tissue_section', 'evaluation_list_tissue_section', 'maxsize_subgraph'}
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and isinstance(data_mintflow['train_list_tissue_section'], utils_multislice.ListSlice)
    flag_isvalid_arg_mintflow_data = flag_isvalid_arg_mintflow_data and isinstance(data_mintflow['evaluation_list_tissue_section'], utils_multislice.ListSlice)
    maxsize_subgraph = data_mintflow['maxsize_subgraph']

    if not flag_isvalid_arg_mintflow_data:
        raise Exception("Something is wrong with the passed argument `mintflow_data`. Make sure that argument `mintflow_data` is the output from the function `mintflow.setup_data`.")

    # grab args
    args = types.SimpleNamespace()
    args.flag_verbose = flag_verbose
    args.flag_visualise_tissue_sections = flag_visualise_tissue_sections

    config_data_train, config_data_test, config_model, config_training = \
        dict_all4_configs['config_data_train'], \
        dict_all4_configs['config_data_evaluation'], \
        dict_all4_configs['config_model'], \
        dict_all4_configs['config_training']

    # set device (duplicate) === TODO:maybe delete
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

    # TODO: take in `list_slice`
    list_slice = data_mintflow['train_list_tissue_section']


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
        'type_theta_aggr': modules.gnn.KhopAvgPoolWithoutselfloop,
        'kwargs_theta_aggr': {'num_hops': config_model['num_graph_hops']},
        'type_moduleflow': modules.neuralODE.MLP,
        'kwargs_moduleflow': {'w': 64},
        'type_w_dec': modules.mlp.SimpleMLP,
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
        'module_int_mu_u': modules.mlp.LinearEncoding(
            dim_input=list_slice.list_slice[0]._global_num_CT,
            dim_output=dict_varname_to_dim['z'],
            flag_detach=config_model['flag_detach_mu_u_int']
        ) if (config_model['flag_use_int_u']) else None,
        'module_int_cov_u': modules.mlp.SimpleMLPandExp(
            dim_input=list_slice.list_slice[0]._global_num_CT,
            list_dim_hidden=[],
            dim_output=dict_varname_to_dim['z'],
            bias=True,
            min_clipval=config_model['lowerbound_cov_u'],
            max_clipval=config_model['upperbound_cov_u']
        ) if (config_model['flag_use_int_u']) else None,
        'flag_use_spl_u': config_model['flag_use_spl_u'],
        'module_spl_mu_u': modules.mlp.LinearEncoding(
            dim_input=list_slice.list_slice[0]._global_num_CT,
            dim_output=dict_varname_to_dim['s'],
            flag_detach=config_model['flag_detach_mu_u_spl']
        ) if (config_model['flag_use_spl_u']) else None,
        'module_spl_cov_u': modules.mlp.SimpleMLPandExp(
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
        },
        'method_ODE_solver':config_training['method_ODE_solver']
    }


    # create a list of `AdjMatPredLoss`-s ====
    dim_input = None  # due to exec limitaiton for locals()
    list_ajdmatpredloss = []
    if len(config_model['args_list_adjmatloss']) > 0:
        raise NotImplementedError("")
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

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'disent_dict_CTNNC_usage = {}'.format(config_model['CTNCC_usage_moduledisent']),
        globals(),
        tmp_ldict
    )
    disent_dict_CTNNC_usage = tmp_ldict['disent_dict_CTNNC_usage']
    gc.collect()

    assert (
        config_model['str_mode_headxint_headxspl_headboth_twosep'] in [
            'headxint', 'headxspl', 'headboth', 'twosep'
        ]
    )
    dict_temp = {
        'headxint':modules.gnn_disentangler.ModeArch.HEADXINT,
        'headxspl':modules.gnn_disentangler.ModeArch.HEADXSPL,
        'headboth':modules.gnn_disentangler.ModeArch.HEADBOTH,
        'twosep':modules.gnn_disentangler.ModeArch.TWOSEP,
    }

    print(disent_dict_CTNNC_usage)

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

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'flowmatching_mode_samplex0 = {}'.format(config_model['flowmatching_mode_samplex0']),
        {**globals(), **locals()},
        tmp_ldict
    )
    flowmatching_mode_samplex0 = tmp_ldict['flowmatching_mode_samplex0']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'flowmatching_mode_minibatchper = {}'.format(config_model['flowmatching_mode_minibatchper']),
        {**globals(), **locals()},
        tmp_ldict
    )
    flowmatching_mode_minibatchper = tmp_ldict['flowmatching_mode_minibatchper']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'flowmatching_mode_timesched = {}'.format(config_model['flowmatching_mode_timesched']),
        {**globals(), **locals()},
        tmp_ldict
    )
    flowmatching_mode_timesched = tmp_ldict['flowmatching_mode_timesched']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'flowmatching_mode_fmloss = {}'.format(config_model['flowmatching_mode_fmloss']),
        {**globals(), **locals()},
        tmp_ldict
    )
    flowmatching_mode_fmloss = tmp_ldict['flowmatching_mode_fmloss']
    gc.collect()



    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_encX_int = {}".format(
            locals()['config_model']['arch_module_encoder_X2Xbar']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_encX_int = tmp_ldict['module_encX_int']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_encX_spl = {}".format(
            config_model['arch_module_encoder_X2Xbar']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_encX_spl = tmp_ldict['module_encX_spl']
    gc.collect()


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

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'dict_varname_to_takeCT_takeNCC = {}'.format(config_model['CTNCC_usage_modulecond4flow']),
        {**globals(), **locals()},
        tmp_ldict
    )
    dict_varname_to_takeCT_takeNCC = tmp_ldict['dict_varname_to_takeCT_takeNCC']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'tmp_encZ_list_dim_hidden = {}'.format(
            config_model['enc3_encZ_list_dim_hidden']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    tmp_encZ_list_dim_hidden = tmp_ldict['tmp_encZ_list_dim_hidden']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'tmp_encSin_list_dim_hidden = {}'.format(
            config_model['enc3_encSin_list_dim_hidden']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    tmp_encSin_list_dim_hidden = tmp_ldict['tmp_encSin_list_dim_hidden']
    gc.collect()

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        'tmp_encSout_list_dim_hidden = {}'.format(
            config_model['enc3_encSout_list_dim_hidden']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    tmp_encSout_list_dim_hidden = tmp_ldict['tmp_encSout_list_dim_hidden']
    gc.collect()

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

    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "print('>>>>>>>>>', 'kwargs_genmodel' in globals())",
        {**globals(), **locals()},
        tmp_ldict
    )
    exec(
        "print('>>>>>>>>>', 'kwargs_genmodel' in locals())",
        {**globals(), **locals()},
        tmp_ldict
    )
    exec(
        "module_predictor_z2notNCC = {}".format(
            config_model['module_predictor_z2notNCC']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_predictor_z2notNCC = tmp_ldict['module_predictor_z2notNCC']
    gc.collect()


    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_predictor_xbarint2notNCC = {}".format(
            config_model['module_predictor_xbarint2notNCC']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_predictor_xbarint2notNCC = tmp_ldict['module_predictor_xbarint2notNCC']
    gc.collect()


    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_predictor_xbarint2notBatchID = {}".format(
            config_model['module_predictor_xbarint2notBatchID']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_predictor_xbarint2notBatchID = tmp_ldict['module_predictor_xbarint2notBatchID']
    gc.collect()


    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_predictor_xbarspl2notBatchID = {}".format(
            config_model['module_predictor_xbarspl2notBatchID']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_predictor_xbarspl2notBatchID = tmp_ldict['module_predictor_xbarspl2notBatchID']
    gc.collect()


    # create the vardist ====
    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_classifier_P1loss = {}".format(
            config_model['module_classifier_P1loss']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_classifier_P1loss = tmp_ldict['module_classifier_P1loss']
    gc.collect()


    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_predictor_P3loss = {}".format(
            config_model['module_predictor_P3loss']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_predictor_P3loss = tmp_ldict['module_predictor_P3loss']
    gc.collect()


    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_classifier_xbarintCT = {}".format(
            config_model['module_classifier_xbarintCT']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    module_classifier_xbarintCT = tmp_ldict['module_classifier_xbarintCT']
    gc.collect()


    # due to exec limitaiton for locals()
    tmp_ldict = locals().copy()
    exec(
        "module_predictor_xbarsplNCC = {}".format(
            config_model['module_predictor_xbarsplNCC']
        ),
        {**globals(), **locals()},
        tmp_ldict
    )
    # TODO: is the change correct?
    # TODO: (cont.) Depending on the python version, it seems, e.g., `kwargs_genmodel` is sometimes local and sometimes global.
    module_predictor_xbarsplNCC = tmp_ldict['module_predictor_xbarsplNCC']
    gc.collect()

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


    module_vardist = vardist.InFlowVarDist(
        module_genmodel=generativemodel.InFlowGenerativeModel(
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
    module_vardist.train()


    if args.flag_verbose:
        print("Inflow vardist module was created on {}.".format(device))

    return module_vardist




