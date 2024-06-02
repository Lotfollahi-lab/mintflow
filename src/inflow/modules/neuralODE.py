
import torch

'''
This code is grabbed and modified from 
https://github.com/atong01/conditional-flow-matching/blob/ec4da0846ddaf77e8406ad2fd592a6f0404ce5ae/torchcfm/models/models.py
'''

class MLP(torch.nn.Module):
    def __init__(self, dim_input, dim_output, w=64):
        super().__init__()
        assert(dim_input == dim_output)
        dim = dim_input
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim),
        )

    def forward(self, x):
        return self.net(x)

