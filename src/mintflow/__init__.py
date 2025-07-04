from importlib.metadata import version

__all__ = ["data", "evaluation", "interface", "lrdb", "modules"]
__version__ = version("mintflow")

from .interface import \
    verify_and_postprocess_config_data_train,\
    verify_and_postprocess_config_data_evaluation,\
    verify_and_postprocess_config_model,\
    verify_and_postprocess_config_training

# from .interface import get_default_configurations


from .interface import \
    get_default_configurations, setup_data, setup_model, Trainer, predict, \
    dump_model, dump_checkpoint


from .evaluation import \
    evaluate_by_known_signalling_genes

from .interface.perturbation import generate_insilico_ST_data

# from .interface import \
#     dump_model, dump_checkpoint

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
