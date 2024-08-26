
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
            except:
                np_pred = np_xspl_pred[mask_nonzero_exp].flatten() + 0.0

        np_gt = np_xspl_gt[mask_nonzero_exp].flatten() + 0.0

        dict_toret = {}
        for measure in self.list_measures:
            measname, measval = measure(np_pred, np_gt)
            dict_toret[measname] = measval

        return dict_toret

