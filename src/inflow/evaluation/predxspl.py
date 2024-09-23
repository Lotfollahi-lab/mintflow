
'''
Measures to evaluate how well a method can predict xspl (i.e. the spatial part of readout).
'''

import numpy as np
from scipy.stats import wasserstein_distance
from scipy import stats

def func_mse(a, b):
    return 'MSE', np.mean((a-b)**2)

def func_mae(a, b):
    return 'MAE', np.mean(np.abs(a-b))

def func_wassdist(a, b):
    return 'EMD', wasserstein_distance(a.flatten(), b.flatten())

def func_pearsoncorrel(a, b):
    try:
        val_corelcoef = stats.pearsonr(a.flatten(), b.flatten()).statistic
    except:
        val_corelcoef = None
    return 'PearsonCorrelation', val_corelcoef


class EvalOnHVGsXsplpred:
    def __init__(self, list_flag_HVG):
        self.list_measures = [func_mse, func_mae, func_wassdist, func_pearsoncorrel]
        self.list_flag_HVG = list_flag_HVG
        assert (isinstance(self.list_flag_HVG, list))
        for u in list_flag_HVG:
            assert (u in [True, False])

        self.np_flag_HVG = np.expand_dims(
            np.array(self.list_flag_HVG),
            0
        )  # [1, num_genes]

    def eval(self, np_xspl_gt:np.ndarray, np_xspl_pred:np.ndarray, np_xobs:np.ndarray, flag_normalize:bool):

        assert (
            isinstance(np_xspl_gt, np.ndarray)
        )
        assert (
            isinstance(np_xspl_pred, np.ndarray)
        )
        assert (
            isinstance(np_xobs, np.ndarray)
        )


        mask_selecteval = (np_xobs > 0.0) * self.np_flag_HVG  # [N x num_genes]
        np_pred = np_xspl_pred + 0.0  # np_xspl_pred[mask_nonzero_exp].flatten() + 0.0
        if flag_normalize:
            try:
                np_pred = np_pred - np.expand_dims(np.min(np_pred, 1), 1)
                np_pred = np_pred / np.expand_dims(np.max(np_pred, 1), 1)
                np_pred = np_pred[mask_selecteval].flatten() * np_xobs[mask_selecteval].flatten()
            except:
                np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0
        else:
            np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0

        np_gt = np_xspl_gt[mask_selecteval].flatten() + 0.0

        dict_toret = {}
        for measure in self.list_measures:
            measname, measval = measure(np_pred, np_gt)
            dict_toret[measname+"_among_{}HVGs".format(np.sum(self.list_flag_HVG))] = measval

        return dict_toret




class EvalXsplpred:
    def __init__(self):
        self.list_measures = [func_mse, func_mae, func_wassdist, func_pearsoncorrel]

    def eval(self, np_xspl_gt:np.ndarray, np_xspl_pred:np.ndarray, np_xobs:np.ndarray, flag_normalize:bool):
        assert (
            isinstance(np_xspl_gt, np.ndarray)
        )
        assert (
            isinstance(np_xspl_pred, np.ndarray)
        )
        assert (
            isinstance(np_xobs, np.ndarray)
        )

        #mask_nonzero_exp = np_xobs > 0.0
        mask_selecteval = np_xobs > 0.0
        np_pred = np_xspl_pred + 0.0  # np_xspl_pred[mask_nonzero_exp].flatten() + 0.0
        if flag_normalize:
            try:
                np_pred = np_pred - np.expand_dims(np.min(np_pred, 1), 1)
                np_pred = np_pred / np.expand_dims(np.max(np_pred, 1), 1)
                np_pred = np_pred[mask_selecteval].flatten() * np_xobs[mask_selecteval].flatten()
            except:
                np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0
        else:
            np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0

        np_gt = np_xspl_gt[mask_selecteval].flatten() + 0.0

        dict_toret = {}
        for measure in self.list_measures:
            measname, measval = measure(np_pred, np_gt)
            dict_toret[measname] = measval

        return dict_toret



class EvalLargeReadoutsXsplpred:
    '''
    Evaluates predXspl on large readouts (i.e. after excluding small readouts).
    '''

    def __init__(self, mincut_readout:int):
        self.mincut_readout = mincut_readout
        self.list_measures = [func_mse, func_mae, func_wassdist, func_pearsoncorrel]

    def eval(self, np_xspl_gt:np.ndarray, np_xspl_pred:np.ndarray, np_xobs:np.ndarray, flag_normalize:bool):
        assert (
            isinstance(np_xspl_gt, np.ndarray)
        )
        assert (
            isinstance(np_xspl_pred, np.ndarray)
        )
        assert (
            isinstance(np_xobs, np.ndarray)
        )

        set_cnts = list(
            set(np_xobs[np_xobs >= self.mincut_readout].flatten().tolist())
        )
        set_cnts.sort()

        dict_toret = {}
        for min_count in set_cnts:
            # mask_min_exp = (np_xobs >= min_count)
            mask_selecteval = (np_xobs >= min_count)
            np_pred = np_xspl_pred + 0.0  # np_xspl_pred[mask_nonzero_exp].flatten() + 0.0
            if flag_normalize:
                try:
                    np_pred = np_pred - np.expand_dims(np.min(np_pred, 1), 1)
                    np_pred = np_pred / np.expand_dims(np.max(np_pred, 1), 1)
                    np_pred = np_pred[mask_selecteval].flatten() * np_xobs[mask_selecteval].flatten()
                except:
                    np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0
            else:
                np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0

            np_gt = np_xspl_gt[mask_selecteval].flatten() + 0.0

            for measure in self.list_measures:
                measname, measval = measure(np_pred, np_gt)
                dict_toret["{} (among readout >= {}, total={})".format(
                    measname, min_count, np.sum(np_xobs >= min_count))
                ] = measval

        return dict_toret



class EvalLargeReadoutsXsplpredExactVersion:
    '''
    Evaluates predXspl on large readouts (i.e. after excluding small readouts) and when the number of readouts is "exactly" equal to mincut_readout.
    '''

    def __init__(self, mincut_readout:int):
        self.mincut_readout = mincut_readout
        self.list_measures = [func_mse, func_mae, func_wassdist, func_pearsoncorrel]

    def eval(self, np_xspl_gt:np.ndarray, np_xspl_pred:np.ndarray, np_xobs:np.ndarray, flag_normalize:bool):
        assert (
            isinstance(np_xspl_gt, np.ndarray)
        )
        assert (
            isinstance(np_xspl_pred, np.ndarray)
        )
        assert (
            isinstance(np_xobs, np.ndarray)
        )

        set_cnts = list(
            set(np_xobs[np_xobs >= self.mincut_readout].flatten().tolist())
        )
        set_cnts.sort()

        dict_toret = {}
        for min_count in set_cnts:
            # mask_min_exp = (np_xobs >= min_count)
            mask_selecteval = (np_xobs == min_count)
            np_pred = np_xspl_pred + 0.0  # np_xspl_pred[mask_nonzero_exp].flatten() + 0.0
            if flag_normalize:
                try:
                    np_pred = np_pred - np.expand_dims(np.min(np_pred, 1), 1)
                    np_pred = np_pred / np.expand_dims(np.max(np_pred, 1), 1)
                    np_pred = np_pred[mask_selecteval].flatten() * np_xobs[mask_selecteval].flatten()
                except:
                    np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0
            else:
                np_pred = np_xspl_pred[mask_selecteval].flatten() + 0.0

            np_gt = np_xspl_gt[mask_selecteval].flatten() + 0.0

            for measure in self.list_measures:
                measname, measval = measure(np_pred, np_gt)
                dict_toret["{} (among readout >= {}, total={})".format(
                    measname, min_count, np.sum(np_xobs >= min_count))
                ] = measval

        return dict_toret

