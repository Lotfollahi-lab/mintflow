
# general imports ====

# mintflow imports ====


from . import utils

from . import generativemodel # exec('import {}.generativemodel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from . import modules  # exec('import {}.modules.gnn'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .modules import gnn, neuralODE, mlp, disentonly

# exec('import {}.modules.neuralODE'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# exec('import {}.modules.mlp'.format(STR_INFLOW_OR_INFLOW_SYNTH))
# exec('import {}.modules.disentonly'.format(STR_INFLOW_OR_INFLOW_SYNTH))


from .modules.impanddisentgl import  MaskLabel #import exec('from {}.modules.impanddisentgl import MaskLabel'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from . import vardist  # exec('import {}.vardist'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from . import masking # exec('import {}.masking'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .modules.impanddisentgl import ImputerAndDisentangler # exec('from {}.modules.impanddisentgl import ImputerAndDisentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .modules.disentonly import Disentangler # exec('from {}.modules.disentonly import Disentangler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .modules.disentonly_twosep import DisentanglerTwoSep # exec('from {}.modules.disentonly_twosep import DisentanglerTwoSep'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .zs_samplers import RandomZSSampler, PerCelltypeZSSampler #exec('from {}.zs_samplers import RandomZSSampler, PerCelltypeZSSampler'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .predadjmat import ListAdjMatPredLoss, AdjMatPredLoss #  exec('from {}.predadjmat import ListAdjMatPredLoss, AdjMatPredLoss'.format(STR_INFLOW_OR_INFLOW_SYNTH))
from .utils_flowmatching import ModeSampleX0, ModeMinibatchPerm, ModeTimeSched, ModeFMLoss, ConditionalFlowMatcher
# exec('from {}.utils_flowmatching import ModeSampleX0, ModeMinibatchPerm, ModeTimeSched, ModeFMLoss, ConditionalFlowMatcher'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .modules.cond4flow import Cond4FlowVarphi0
# exec('from {}.modules.cond4flow import Cond4FlowVarphi0'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .modules.cond4flow_simple3mpls import Cond4FlowVarphi0SimpleMLPs
# exec('from {}.modules.cond4flow_simple3mpls import Cond4FlowVarphi0SimpleMLPs'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .utils_pyg import PygSTDataGridBatchSampler
# exec('from {}.utils_pyg import PygSTDataGridBatchSampler'.format(STR_INFLOW_OR_INFLOW_SYNTH))


from .evaluation.bioconsv import EvaluatorKmeans, EvaluatorLeiden
# exec('from {}.evaluation.bioconsv import EvaluatorKmeans, EvaluatorLeiden'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))


from .evaluation.predxspl import EvalXsplpred, EvalLargeReadoutsXsplpred, EvalOnHVGsXsplpred
# exec('from {}.evaluation.predxspl import EvalXsplpred, EvalLargeReadoutsXsplpred, EvalOnHVGsXsplpred'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .modules.gnn_disentangler import GNNDisentangler
# exec('from {}.modules.gnn_disentangler import GNNDisentangler'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .kl_annealing import LinearAnnealingSchedule
# exec('from {}.kl_annealing import LinearAnnealingSchedule'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .modules.predictorperCT import PredictorPerCT
# exec('from {}.modules.predictorperCT import PredictorPerCT'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .utils_multislice import ListSlice, Slice
# exec('from {}.utils_multislice import ListSlice, Slice'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .modules.varphienc4xbar import EncX2Xbar
# exec('from {}.modules.varphienc4xbar import EncX2Xbar'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .modules.predictorbatchID import PredictorBatchID
# exec('from {}.modules.predictorbatchID import PredictorBatchID'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .interface import parse_config_data_train, parse_config_data_test, parse_config_training, parse_config_model, check_listtissue_trtest
# exec('from {}.cli import parse_config_data_train, parse_config_data_test, parse_config_training, parse_config_model, check_listtissue_trtest'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .interface.auxiliary_modules import *
# exec('from {}.cli.auxiliary_modules import *'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .interface.analresults import disentanglement_jointplot
# exec('from {}.cli.analresults import disentanglement_jointplot'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .interface.analresults import disentanglement_violinplot
# exec('from {}.cli.analresults import disentanglement_violinplot'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))

from .anneal_decoder_xintxspl import AnnealingDecoderXintXspl
# exec('from {}.anneal_decoder_xintxspl import AnnealingDecoderXintXspl'.format(
#     STR_INFLOW_OR_INFLOW_SYNTH
# ))




