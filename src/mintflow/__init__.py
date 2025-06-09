#
# # general imports ====
#
# # mintflow imports ====
#
#
# from . import utils
#
# from . import generativemodel # exec('import {}.generativemodel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from . import modules  # exec('import {}.modules.gnn'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .modules import gnn, neuralODE, mlp, disentonly
#
#
# from .modules.impanddisentgl import  MaskLabel #import exec('from {}.modules.impanddisentgl import MaskLabel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from . import vardist  # exec('import {}.vardist'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from . import masking # exec('import {}.masking'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .modules.impanddisentgl import ImputerAndDisentangler # exec('from {}.modules.impanddisentgl import ImputerAndDisentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .modules.disentonly import Disentangler # exec('from {}.modules.disentonly import Disentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .modules.disentonly_twosep import DisentanglerTwoSep # exec('from {}.modules.disentonly_twosep import DisentanglerTwoSep'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .zs_samplers import RandomZSSampler, PerCelltypeZSSampler #exec('from {}.zs_samplers import RandomZSSampler, PerCelltypeZSSampler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .predadjmat import ListAdjMatPredLoss, AdjMatPredLoss #  exec('from {}.predadjmat import ListAdjMatPredLoss, AdjMatPredLoss'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# from .utils_flowmatching import ModeSampleX0, ModeMinibatchPerm, ModeTimeSched, ModeFMLoss, ConditionalFlowMatcher
#
# from .modules.cond4flow import Cond4FlowVarphi0
#
# from .modules.cond4flow_simple3mpls import Cond4FlowVarphi0SimpleMLPs
#
# from .utils_pyg import PygSTDataGridBatchSampler
#
#
# from .evaluation.bioconsv import EvaluatorKmeans, EvaluatorLeiden
#
# from .evaluation.predxspl import EvalXsplpred, EvalLargeReadoutsXsplpred, EvalOnHVGsXsplpred
#
# from .modules.gnn_disentangler import GNNDisentangler
#
# from .kl_annealing import LinearAnnealingSchedule
#
# from .modules.predictorperCT import PredictorPerCT
#
# from .utils_multislice import ListSlice, Slice
#
# from .modules.varphienc4xbar import EncX2Xbar
#
# from .modules.predictorbatchID import PredictorBatchID
#

from .interface import \
    get_defaultconfig_data_train, verify_and_postprocess_config_data_train,\
    get_defaultconfig_data_evaluation, verify_and_postprocess_config_data_evaluation,\
    get_defaultconfig_model, verify_and_postprocess_config_model,\
    get_defaultconfig_training, verify_and_postprocess_config_training,\
    setup_data, setup_model, Trainer, predict


from .evaluation import \
    evaluate_by_DB_signalling_genes

# from .interface.auxiliary_modules import *
#
# from .interface.analresults import disentanglement_jointplot
#
# from .interface.analresults import disentanglement_violinplot

# from . import interface

# from .anneal_decoder_xintxspl import AnnealingDecoderXintXspl
#
#
#
#
