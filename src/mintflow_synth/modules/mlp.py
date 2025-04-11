

from typing import List
import numpy as np
import torch

class SimpleMLP(torch.nn.Module):
    def __init__(self, dim_input:int, list_dim_hidden:List, dim_output:int, bias:bool, flag_endwithReLU:bool):
        super(SimpleMLP, self).__init__()
        #grab args ===
        self.dim_input = dim_input
        self.list_dim = [dim_input] + list_dim_hidden + [dim_output]
        self.dim_output = dim_output
        self.flag_endwithReLU = flag_endwithReLU
        #make internals ==
        list_module = []
        for l in range(len(self.list_dim)-1):
            list_module.append(
                torch.nn.Linear(self.list_dim[l], self.list_dim[l+1], bias=bias)
            )
            if l != len(self.list_dim)-2:
                list_module.append(torch.nn.ReLU())
        if flag_endwithReLU:
            list_module.append(torch.nn.ReLU())
        self.module = torch.nn.Sequential(*list_module)

    def forward(self, x):
        out = self.module(x)
        return out

