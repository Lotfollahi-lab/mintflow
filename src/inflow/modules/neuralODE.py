
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
    The input has `dim_b + dim_z + dim_s` dimensions, where `dim_b` is the number of batch tokens.
    The first half of output depends only on the `dim_b + dim_z` dimensions.
    The send half of output depends on the `dim_b + dim_s` dimensions.
    '''
    def __init__(self, dim_b, dim_z, dim_s, w=64):
        # TODO: `dim_input` and `dim_output` arguments were removed after adding batch token. Any issues?
        super().__init__()
        assert(dim_z == dim_s)
        self.dim_z = dim_z
        self.dim_s = dim_s
        self.dim_b = dim_b
        self.net_z = torch.nn.Sequential(
            torch.nn.Linear(dim_b + dim_z + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim_z)
        )
        self.net_s = torch.nn.Sequential(
            torch.nn.Linear(dim_b + dim_z + dim_s + 1, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, dim_s)
        )

    def forward(self, t, x, ten_BatchEmb):
        """
        :param t: a 1-D tensor of time-steps, of shape [b].
        :param x: of shape [b x dim_z+dim_s]
        :param ten_BatchEmb: of shape [b x dim_b]
        :return:
        """
        # torchcfm's sample label-conditioned forward was used
        # https://github.com/atong01/conditional-flow-matching/blob/62c44affd877a01b7838d408b5dc4cbcbf83e3ad/torchcfm/models/unet/unet.py#L599

        assert t.dim() == 1
        assert x.dim() == 2
        assert ten_BatchEmb.dim() == 2
        assert x.size()[1] == (self.dim_z + self.dim_s)
        assert ten_BatchEmb.size()[1] == self.dim_b

        dim_z, dim_s, dim_b = self.dim_z, self.dim_s, self.dim_b
        output_z = self.net_z(
            torch.cat(
                [
                    ten_BatchEmb,
                    x[:, 0:dim_z],
                    t.unsqueeze(1)
                ],
                1
            )
        )  # [N, dim_z]
        output_s = self.net_s(
            torch.cat(
                [
                    ten_BatchEmb,
                    x,
                    t.unsqueeze(1)
                ],
                1
            )
        )  # [N, dim_s]
        output = torch.cat([output_z, output_s], 1)  # [N, dim_z+dim_s]
        return output
