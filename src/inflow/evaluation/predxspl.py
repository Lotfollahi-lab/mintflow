
'''
Measures to evaluate how well a method can predict xspl (i.e. the spatial part of readout).
'''

import numpy as np

def func_mse(a, b):
    return 'MSE', np.mean((a-b)**2)

def func_mae(a, b):
    return 'MAE', np.mean(np.abs(a-b))


class EvalXsplpred:
    def __init__(self):
        self.list_measures = [func_mse, func_mae]

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

        mask_nonzero_exp = np_xobs > 0.0
        np_pred = np_xspl_pred[mask_nonzero_exp].flatten() + 0.0
        if flag_normalize:
            try:
                np_pred = np_pred - np.min(np_pred)
                np_pred = np_pred / np_pred.max()
                np_pred = np_pred * np_xobs[mask_nonzero_exp].flatten()
            except:
                np_pred = np_xspl_pred[mask_nonzero_exp].flatten() + 0.0

        np_gt = np_xspl_gt[mask_nonzero_exp].flatten() + 0.0

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
        self.list_measures = [func_mse, func_mae]

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

        for min_count in set_cnts:
            mask_min_exp = (np_xobs >= min_count)
            np_pred = np_xspl_pred[mask_min_exp].flatten() + 0.0
            if flag_normalize:
                try:
                    np_pred = np_pred - np.min(np_pred)
                    np_pred = np_pred / np_pred.max()
                    np_pred = np_pred * np_xobs[mask_min_exp].flatten()
                except:
                    np_pred = np_xspl_pred[mask_min_exp].flatten() + 0.0

            np_gt = np_xspl_gt[mask_min_exp].flatten() + 0.0

            dict_toret = {}
            for measure in self.list_measures:
                measname, measval = measure(np_pred, np_gt)
                dict_toret["{} (among readout >= {})".format(measname, min_count)] = measval




