

from typing import List
import numpy as np
import torch



class Identity(torch.nn.Module):
    '''
    The identity module (dummy module to simplify handling, e.g., dropout or LayerNorm flags).s
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SimpleMLP(torch.nn.Module):
    def __init__(
        self, dim_input:int, list_dim_hidden:List, dim_output:int, bias:bool, flag_endwithReLU:bool,
        flag_startswithReLU:bool, flag_endswithSoftmax:bool=False, flag_use_layernorm:bool=False,
        flag_endswith_softplus:bool=False
    ):
        super(SimpleMLP, self).__init__()
        #grab args ===
        self.dim_input = dim_input
        self.list_dim = [dim_input] + list_dim_hidden + [dim_output]
        self.dim_output = dim_output
        self.flag_endwithReLU = flag_endwithReLU
        self.flag_startswithReLU = flag_startswithReLU
        self.flag_endswithSoftmax = flag_endswithSoftmax
        self.flag_use_layernorm = flag_use_layernorm
        self.flag_endswith_softplus = flag_endswith_softplus
        self._check_args()

        #make internals ==
        list_module = []
        if flag_startswithReLU:
            list_module.append(torch.nn.ReLU())

        for l in range(len(self.list_dim)-1):
            list_module.append(
                torch.nn.Linear(self.list_dim[l], self.list_dim[l+1], bias=bias)
            )
            if self.flag_use_layernorm and (l != len(self.list_dim)-2):
                list_module.append(
                    torch.nn.LayerNorm(self.list_dim[l+1])
                )
            if l != len(self.list_dim)-2:
                list_module.append(torch.nn.ReLU())

        if self.flag_endwithReLU:
            list_module.append(torch.nn.ReLU())
        if self.flag_endswithSoftmax:
            list_module.append(torch.nn.Softmax(1))
        if self.flag_endswith_softplus:
            list_module.append(torch.nn.Softplus())

        self.module = torch.nn.Sequential(*list_module)

    def _check_args(self):
        assert(
            np.sum(
                [self.flag_endwithReLU, self.flag_endswithSoftmax, self.flag_endswith_softplus]
            ) <= 1
        )

        assert (
            self.flag_endswith_softplus in [True, False]
        )
        assert (
            self.flag_use_layernorm in [True, False]
        )
        if self.flag_endswithSoftmax:
            assert not self.flag_endwithReLU

        if self.flag_endwithReLU:
            assert not self.flag_endswithSoftmax

    def forward(self, x):
        out = self.module(x)
        return out


class SimpleMLPandExp(torch.nn.Module):
    '''
    MLP ending with .exp(), to be used in, e.g., covariance matrix which is essential for identfiability.
    '''
    def __init__(self, dim_input:int, list_dim_hidden:List, dim_output:int, bias:bool, min_clipval:float, max_clipval:float):
        super(SimpleMLPandExp, self).__init__()
        self.min_clipval = min_clipval
        self.max_clipval = max_clipval
        self.module_mlp = SimpleMLP(
            dim_input=dim_input,
            list_dim_hidden=list_dim_hidden,
            dim_output=dim_output,
            bias=bias,
            flag_endwithReLU=False,
            flag_startswithReLU=False
        )

    def forward(self, x):
        return torch.clamp(
            self.module_mlp(x).exp(),
            min=self.min_clipval,
            max=self.max_clipval
        )





class LinearEncoding(torch.nn.Module):
    '''
    Fixed linear layer (similar to torch.Embedding).
    '''
    def __init__(self, dim_input:int, dim_output:int, flag_detach:bool=False):

        super(LinearEncoding, self).__init__()
        self.W = torch.nn.Parameter(
            torch.rand(dim_input, dim_output),
            requires_grad=(not self.flag_detach)
        )
        self.flag_endwithReLU = False  # so this module passes an external assertion
        self.flag_detach = flag_detach

    def forward(self, x):
        if self.flag_detach:
            torch.matmul(x, self.W.detach())
        else:
            return torch.matmul(x, self.W)



class ConstMLP(torch.nn.Module):
    '''
    A constant number to be used for, e.g., the covariance of p(z | u_z)
    '''

    def __init__(self, dim_input:int, dim_output:int, constval:float):
        super(ConstMLP, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.constval = constval

        self._check_args()

    def _check_args(self):
        assert (
            isinstance(self.constval, float)
        )

    def forward(self, x:torch.Tensor):
        return self.constval + 0.0  # torch.tensor([self.constval + 0.0], device=x.device, requires_grad=False)

        '''
        return self.constval * torch.ones(
            size=[x.size()[0], self.dim_output],
            device=x.device,
            requires_grad=False
        )
        '''
