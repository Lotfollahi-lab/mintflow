
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

        return torch.mean(
            torch.stack(
                [output[n, argmin_row[n]] for n in x.size()[0]]
            )
        )



class PredictorPerCT(nn.Module):
    def __init__(self, list_modules:List[nn.Module]):
        super(PredictorPerCT, self).__init__()
        assert isinstance(list_modules, list)
        for m in list_modules:
            assert isinstance(m, nn.Module)

        self.list_modules = nn.ModuleList(list_modules)

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

        output = torch.stack(
            [self.list_modules[list_CT[n]](x[n,:].unsqueeze(0))[0,:] for n in range(x.size()[0])],
            dim=0
        )  # [N, Dout]

        return output







