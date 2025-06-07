
'''
wassdist utils but for mixing batches.
The difference to wassdist_utils is that mixing based on BatchIDs is "not" done per CT (but still per batch).
'''

import numpy as np
import torch
import torch.nn as nn
from .modules import predictorbatchID

def _truncated_x2min1(x):
    '''
    :param x:
    :return: x**2-1 outside [-1, 1], and zero inside [-1, 1].
    '''
    with torch.no_grad():
        mask_inside_min1plus1 = (x >= -1.0) & (x <= 1.0)  # [N, D]

    return (x**2 - 1.0) * (~mask_inside_min1plus1 + 0) + 0.0 * (mask_inside_min1plus1 + 0)

def _smoothnessloss_leibcont(z:torch.Tensor, module_BatchIDpredictor:predictorbatchID.PredictorBatchID):
    EPS = 1e-4
    N = z.size()[0]
    z_initpoints = z[
        np.random.permutation(N).tolist(),
        :
    ] + torch.rand(N).unsqueeze(1).to(z.device) * z[
        np.random.permutation(N).tolist(),
        :
    ]  # [N x numBatch]

    z1 = z_initpoints[
        np.random.permutation(N).tolist(),
        :
    ] + 0.0  # [N x numBatch]
    z2 = z_initpoints[
        np.random.permutation(N).tolist(),
        :
    ] + 0.0  # [N x numBatch]

    loss_smoothness = module_BatchIDpredictor(
        x=z1.detach()
    ) - module_BatchIDpredictor(
        x=z2.detach()
    )  # [N x numBatch]
    loss_smoothness = torch.abs(loss_smoothness) / (EPS + torch.norm(z1 - z2, dim=1).unsqueeze(1))  # [N x numBatch]
    loss_smoothness = _truncated_x2min1(loss_smoothness)  # [N x numBatch]

    return loss_smoothness.sum(1).mean()

class WassDistBatchID(nn.Module):
    '''
    Wasserstein distance dual function.

    '''
    def __init__(self, coef_fminf:float, coef_smoothness:float):
        super(WassDistBatchID, self).__init__()
        self.coef_fminf = coef_fminf
        self.coef_smoothness = coef_smoothness


    def forward(self, z:torch.Tensor, module_BatchIDpredictor:predictorbatchID.PredictorBatchID, ten_BatchID:torch.Tensor):
        """

        :param z: the z (or muz) vectors "after" GRL is applied, tensor of shape [N, dimZ].
        :param module_BatchIDpredictor: batch ID predictor.
        :param ten_BatchID: the ground-truth batch associations (with one-hot vector in each row).
        :return:
        """

        # TODO: make sure that the smoothness Libch loss is done on z.detach(), so the grad doesn't go back from GRL.
        assert isinstance(module_BatchIDpredictor, predictorbatchID.PredictorBatchID)
        assert (z.size()[0] == ten_BatchID.size()[0])

        x = module_BatchIDpredictor.forward(
            x=z
        )  # [N, numBatch]

        # separate cells by BatchID
        with torch.no_grad():
            list_BatchID = torch.argmax(ten_BatchID, 1).tolist()

        dict_bid_to_listidxlocal = {}
        for bid in range(ten_BatchID.size()[1]):
            dict_bid_to_listidxlocal[bid] = np.where(np.array(list_BatchID) == bid)[0].tolist()
            dict_bid_to_listidxlocal[bid].sort()

        # compute f(.) - f(.) term
        loss = 0.0
        if len(set(list_BatchID)) > 1:  # if x cells are note merely from a single biobatch
            for bid in range(ten_BatchID.size()[1]):  # for each bacth (i.e. b_head)
                if len(dict_bid_to_listidxlocal[bid]) >= 1:
                    ten_in_batch_b = (ten_BatchID[:, bid] > 0).detach()  # [N]
                    loss = torch.mean(x[ten_in_batch_b, bid]) - torch.mean(x[~ten_in_batch_b, bid])

        # compute the smoothness loss based on x_ctgroup
        loss_smoothness = _smoothnessloss_leibcont(
            z=z.detach(),
            module_BatchIDpredictor=module_BatchIDpredictor
        )

        return {
            'loss_fminf': {
                'val': loss,
                'coef': self.coef_fminf
            },
            'loss_smoothness': {
                'val': loss_smoothness,
                'coef': self.coef_smoothness
            }
        }




