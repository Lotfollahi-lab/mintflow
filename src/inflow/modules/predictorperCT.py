
from typing import List
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class MinRowLoss(nn.Module):
    '''
    A torch loss, but in each row the minimum value is selected to be maximised.
    Like, intuitively, the maximum values may not be reducible because, e.g., a cell of type A never sits next to a cell of type B.
    '''
    def __init__(self, type_loss):
        super(MinRowLoss, self).__init__()
        self.crit = type_loss(reduction='none')

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert (x.size()[0] == y.size()[0])
        output = self.crit(x, y)
        assert (
            len(output.size()) == 2
        )
        assert (output.size()[0] == x.size()[0])  # i.e. assert no reduction is done.
        assert (output.size()[1] == y.size()[1])  # i.e. assert no reduction is done.

        with torch.no_grad():
            argmin_row = torch.argmin(output, 1).tolist()
            assert (len(argmin_row) == x.size()[0])



        FLAG_DISABLE = True

        if not FLAG_DISABLE:
            return torch.mean(
                torch.stack(
                    [output[n, argmin_row[n]] for n in range(x.size()[0])]
                )
            )
        else:
            return output.sum(1).mean()



class PredictorPerCT(nn.Module):
    def __init__(self, list_modules:List[nn.Module]):
        super(PredictorPerCT, self).__init__()
        assert isinstance(list_modules, list)
        for m in list_modules:
            assert isinstance(m, nn.Module)

        self.list_modules = nn.ModuleList(list_modules)

        # set flag_has_BNlayer
        self.flag_has_BNlayer = False
        for mod in self.list_modules:
            for ch in list(mod.children()):
                if isinstance(ch, nn.BatchNorm1d):
                    self.flag_has_BNlayer = True

    def forward(self, x, ten_CT):
        '''
        :param x: a tensor of shape [N, D].
        :param ten_CT: a tensor of shape [N, #CT].
        :return:
        '''
        assert (len(self.list_modules) == ten_CT.size()[1])
        assert (x.size()[0] == ten_CT.size()[0])

        with torch.no_grad():
            list_CT = torch.argmax(ten_CT, 1).tolist()

        # separate cells by CT
        dict_ct_to_listidxlocal = {}
        for ct in range(ten_CT.size()[1]):
            dict_ct_to_listidxlocal[ct] = np.where(np.array(list_CT) == ct)[0].tolist()
            dict_ct_to_listidxlocal[ct].sort()

        dict_nlocal_to_output = {}
        for ct in range(ten_CT.size()[1]):
            if len(dict_ct_to_listidxlocal[ct]) == 0:
                continue  # no cell of type ct --> continue
            flag_retzero = self.flag_has_BNlayer and (len(dict_ct_to_listidxlocal[ct]) <= 1)

            if flag_retzero:
                assert (len(dict_ct_to_listidxlocal[ct]) == 1)
                assert (
                    dict_ct_to_listidxlocal[ct][0] not in dict_nlocal_to_output.keys()
                )
                dict_nlocal_to_output[dict_ct_to_listidxlocal[ct][0]] = torch.zeros(size=[ten_CT.size()[1]]).to(x.device)
            else:
                # group x rows based on ct
                x_ct = x[
                    dict_ct_to_listidxlocal[ct],
                    :
                ]
                if len(dict_ct_to_listidxlocal[ct]) == 1:
                    x_ct = x_ct.unsqueeze(0)


                netout_ct = self.list_modules[ct](x_ct)
                for idx_inct, nlocal in enumerate(dict_ct_to_listidxlocal[ct]):
                    assert (
                        nlocal not in dict_nlocal_to_output.keys()
                    )
                    dict_nlocal_to_output[nlocal] = netout_ct[idx_inct, :]

        assert (
            set(dict_nlocal_to_output.keys()) == set(range(x.size()[0]))
        )

        '''
        BNlayer could be used --> batch size of size 1 is not doable.
        output = torch.stack(
            [self.list_modules[list_CT[n]](x[n,:].unsqueeze(0))[0,:] for n in range(x.size()[0])],
            dim=0
        )  # [N, Dout]
        '''

        return torch.stack(
            [dict_nlocal_to_output[nlocal] for nlocal in range(x.size()[0])],
            0
        )














