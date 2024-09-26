
from typing import List
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod



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







