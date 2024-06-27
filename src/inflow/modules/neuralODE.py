
import torch

'''
This code is grabbed and modified from
https://github.com/atong01/conditional-flow-matching/blob/ec4da0846ddaf77e8406ad2fd592a6f0404ce5ae/torchcfm/models/models.py
'''

class MLPDefault(torch.nn.Module):
    '''
    The default mlp which shouldn't be used for inflow.
        Because in the flow the first half of output depends only on the first half of input, but the second half of output depends on the entire input.
    '''
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



class MLP(torch.nn.Module):
    '''
    The first half of output depends only on the first half of input, but the second half of output depends on the entire input.
    '''
    def __init__(self, dim_input, dim_output, w=64):
        super().__init__()
        assert(dim_input == dim_output)
        self.dim = dim_input
        dim = dim_input
        self.net_z = torch.nn.Sequential(
            torch.nn.Linear(dim//2 + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim//2)
        )
        self.net_s = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim//2)
        )

    def forward(self, x):
        dim = self.dim
        output_z = self.net_z(
            torch.cat(
                [x[:, 0:dim//2], x[:,-1].unsqueeze(0)],
                1
            )
        )  # [N, dim//2]
        output_s = self.net_s(x)  # [N, dim//2]
        output = torch.cat([output_z, output_s], 1)  # [N, dim]
        return output
