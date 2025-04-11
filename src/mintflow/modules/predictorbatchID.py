

'''
The dual function that predicts the batch token from xbar_int or xbar_spl.
'''
from typing import Union, Callable, Tuple, Any, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class PredictorBatchID(nn.Module):

    """
    The dual function that predicts batch ID (i.e. a num_bachtes-dimensional output).
    In the end no specific functionality was added (unlike, e.g., PredictorPerCT)
    """
    def __init__(self, list_modules, num_batches):
        super(PredictorBatchID, self).__init__()
        self.list_modeuls = nn.ModuleList(list_modules)
        self.num_batches = num_batches
        assert len(list_modules) == num_batches

    def forward(self, x):
        output = []
        for b in range(self.num_batches):
            netout_b = self.list_modeuls[b](x)
            assert netout_b.size()[0] == x.size()[0]
            assert netout_b.size()[1] == 1
            assert torch.all(netout_b <= 1)  # to make sure torch.tanh is applied.
            assert torch.all(netout_b >= -1)  # to make sure torch.tanh is applied.
            output.append(netout_b)

        output = torch.cat(output, 1)  # [N x B]
        assert output.size()[1] == self.num_batches
        return output


