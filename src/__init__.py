from importlib.metadata import version

from . import data, evaluation, interface, lrdb, modules

__all__ = ["data", "evaluation", "interface", "lrdb", "modules"]

__version__ = version("mintflow")
