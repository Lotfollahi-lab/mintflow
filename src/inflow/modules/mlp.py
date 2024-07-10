

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


class SimpleMLPandExp(torch.nn.Module):
    '''
    MLP ending with .exp(), to be used in, e.g., covariance matrix which is essential for identfiability.
    '''
    def __init__(self, dim_input:int, list_dim_hidden:List, dim_output:int, bias:bool):
        super(SimpleMLPandExp, self).__init__()
        self.module_mlp = SimpleMLP(
            dim_input=dim_input,
            list_dim_hidden=list_dim_hidden,
            dim_output=dim_output,
            bias=bias,
            flag_endwithReLU=False
        )

    def forward(self, x):
        return self.module_mlp(x).exp()





class LinearEncoding(torch.nn.Module):
    '''
    Fixed linear layer (similar to torch.Embedding).
    '''
    def __init__(self, dim_input:int, dim_output:int):
        super(LinearEncoding, self).__init__()
        self.W = torch.nn.Parameter(
            torch.rand(dim_input, dim_output),
            requires_grad=True
        )
        self.flag_endwithReLU = False  # so this module passes an external assertion

    def forward(self, x):
        return torch.matmul(x, self.W)


