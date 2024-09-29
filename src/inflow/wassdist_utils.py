

import numpy as np
import torch
import torch.nn as nn
from .modules import predictorperCT

def _truncated_x2min1(x):
    '''
    :param x:
    :return: x**2-1 outside [-1, 1], and zero inside [-1, 1].
    '''
    with torch.no_grad():
        mask_inside_min1plus1 = (x >= -1.0) & (x <= 1.0)  # [N, D]

    return (x**2 - 1.0) * (~mask_inside_min1plus1 + 0) + 0.0 * (mask_inside_min1plus1 + 0)

def _smoothnessloss_leibcont(z:torch.Tensor, module_NCCpredictor:predictorperCT.PredictorPerCT, ten_CT: torch.Tensor, ten_NCC: torch.Tensor):
    assert isinstance(z, torch.Tensor)
    assert (z.size()[0] == ten_CT.size()[0])


    # separate cells by CT
    with torch.no_grad():
        list_CT = torch.argmax(ten_CT, 1).tolist()


    dict_ct_to_listidxlocal = {}
    for ct in range(ten_CT.size()[1]):
        dict_ct_to_listidxlocal[ct] = np.where(np.array(list_CT) == ct)[0].tolist()
        dict_ct_to_listidxlocal[ct].sort()

    EPS = 1e-4  # TODO:TUNE
    loss_total = 0.0
    print("z.shape = {}".format(z.shape))

    for ct in range(ten_CT.size()[1]):  # for each CT group

        if len(dict_ct_to_listidxlocal[ct]) >= 2:
            # create z1 and z2 of shape [N, dimZ], the vector sets on which the smoothness loss is defined.
            z_ctgroup = z[dict_ct_to_listidxlocal[ct], :] + 0.0  # [size_CTgroup, dimZ]



            print("z_ctgroup.shape = {}".format(z_ctgroup.shape))

            N = z_ctgroup.size()[0]
            assert (N == len(dict_ct_to_listidxlocal[ct]))
            z_ctgroup = z_ctgroup[
                np.random.permutation(N).tolist(),
                :
            ] + torch.rand(N).unsqueeze(1) * z_ctgroup[
                np.random.permutation(N).tolist(),
                :
            ]
            z1 = z_ctgroup[
                np.random.permutation(N).tolist(),
                :
            ] + 0.0  # [N, dimZ]
            z2 = z_ctgroup[
                np.random.permutation(N).tolist(),
                :
            ] + 0.0  # [N, dimZ]

            loss_smoothness = module_NCCpredictor(
                x=z1.detach(),
                ten_CT=ten_CT[dict_ct_to_listidxlocal[ct], :]
            ) - module_NCCpredictor(
                x=z2.detach(),
                ten_CT=ten_CT[dict_ct_to_listidxlocal[ct], :]
            )  # [N, numCT]
            loss_smoothness = torch.abs(loss_smoothness) / (EPS + torch.norm(z1 - z2, dim=1).unsqueeze(1))  # [N, numCT]
            loss_smoothness = _truncated_x2min1(loss_smoothness)  # [N, numCT]
            loss_total = loss_total + loss_smoothness.sum(1).mean()

    return loss_total





class WassDist(nn.Module):
    '''
    Wasserstein distance dual function.

    '''
    def __init__(self, coef_fminf:float, coef_smoothness:float):
        super(WassDist, self).__init__()
        self.coef_fminf = coef_fminf
        self.coef_smoothness = coef_smoothness



    def forward(self, z:torch.Tensor, module_NCCpredictor:predictorperCT.PredictorPerCT, ten_CT: torch.Tensor, ten_NCC: torch.Tensor):
        '''

        :param z: the z (or muz) vectors "after" GRL is applied, tensor of shape [N, dimZ].
        :param module_NCCpredictor: the output of NCC predictors,
        :param ten_CT
        :param ten_NCC
        :return:
        '''

        #TODO: make sure that the smoothness Libch loss is done on z.detach(), so the grad doesn't go back from GRL.
        assert isinstance(module_NCCpredictor, predictorperCT.PredictorPerCT)
        assert (z.size()[0] == ten_CT.size()[0])
        assert (z.size()[0] == ten_NCC.size()[0])
        assert (ten_CT.size()[0] == ten_NCC.size()[0])

        x = module_NCCpredictor.forward(
            x=z,
            ten_CT=ten_CT
        )  # [N, numCT]

        # separate cells by CT
        with torch.no_grad():
            list_CT = torch.argmax(ten_CT, 1).tolist()

        dict_ct_to_listidxlocal = {}
        for ct in range(ten_CT.size()[1]):
            dict_ct_to_listidxlocal[ct] = np.where(np.array(list_CT) == ct)[0].tolist()
            dict_ct_to_listidxlocal[ct].sort()



        # compute f(.) - f(.) term
        loss = 0.0
        for ct in range(ten_CT.size()[1]):  # for each CT group
            if len(dict_ct_to_listidxlocal[ct]) >= 2:
                x_ctgroup = x[dict_ct_to_listidxlocal[ct], :]  # [size_CTgroup, num_CT]
                y_ctgroup = (ten_NCC[dict_ct_to_listidxlocal[ct], :] > 0) + 0  # [size_CTgroup, num_CT]

                for ncc_head in range(ten_NCC.size()[1]):
                    # add the f(.)-f(.) term
                    if set(y_ctgroup[:, ncc_head].tolist()) == {0, 1}:  # if there are both 0 and 1 in the colunmn of NCC matrix.
                        loss = loss + torch.mean(x_ctgroup[y_ctgroup[:, ncc_head] == 1, ncc_head]) - torch.mean(x_ctgroup[y_ctgroup[:, ncc_head] == 0, ncc_head])



        # compute the smoothness loss based on x_ctgroup
        loss_smoothness = _smoothnessloss_leibcont(
            z=z.detach(),
            module_NCCpredictor=module_NCCpredictor,
            ten_CT=ten_CT,
            ten_NCC=ten_NCC
        )

        return {
            'loss_fminf':{
                'val':loss,
                'coef':self.coef_fminf
            },
            'loss_smoothness':{
                'val':loss_smoothness,
                'coef':self.coef_smoothness
            }
        }





