
'''
NMI/ARI metrics, as implemented by scib-metrics.
'''

from typing import Dict, List
import os, sys
import numpy as np
import scib_metrics
from abc import ABC, abstractmethod


class Evaluator(ABC):

    def __init__(self, func_aggreg, str_varname, obskey_labels):
        '''

        :param func_aggreg: given a list of measures, how to report the best measure.
            - lower the better  -->  func_aggreg could be, e.g., np.min
            - higher the better -->  func_aggreg could be, e.g., np.max
        :param str_varname: the name of the variable in `dict_varname_to_var`
        :param obskey_labels: the colum name of labels in adata.obs
        '''
        self.func_aggreg = func_aggreg
        self.str_varname = str_varname
        self.obskey_labels = obskey_labels

    def _get_list_labels(self, adata):
        df_labels = adata.obs[self.obskey_labels].astype('category')
        return df_labels.cat.codes.tolist()


    @abstractmethod
    def eval(self, dict_varname_to_var, adata):
        pass

    def get_output_keys(self):
        raw_keys = self._get_output_keys()  # like ['nmi', 'ari'], if the output contains these keys.
        return ["{}_{}_{}".format(self.str_varname, self.obskey_labels, k) for k in raw_keys]

    @abstractmethod
    def _get_output_keys(self):
        pass

class EvaluatorKmeans(Evaluator):

    def eval(self, dict_varname_to_var, adata):
        assert (self.str_varname in dict_varname_to_var.keys())
        dict_output_raw = scib_metrics.nmi_ari_cluster_labels_kmeans(
            X=dict_varname_to_var[self.str_varname],
            labels=np.array(self._get_list_labels(adata=adata))
        )
        dict_output = {
            "{}_{}_{}".format(self.str_varname, self.obskey_labels, k):dict_output_raw[k] for k in dict_output_raw.keys()
        }
        assert (
            set(dict_output.keys()) == set(self.get_output_keys())
        )
        return dict_output

    def _get_output_keys(self):
        return ['nmi', 'ari']










