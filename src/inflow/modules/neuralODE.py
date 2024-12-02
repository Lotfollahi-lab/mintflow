
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

    def forward_4torchdiffeq(self, t, x, ten_BatchEmb):
        """
        The forward function compatible with torchdiffeq.
        The main difference is the shape of `t`, where `t`-s dimension doesn't have to be equal to `x.size()[0]`
        :param t: a 0-D tensor (since torchdiffeq calls this func with a single time-step)
        :param x: of shape [b x dim_z+dim_s]
        :param ten_BatchEmb: of shape [b x dim_b]
        :return: a tensor of shape [b x dim_z+dim_s]
        """

        '''
        print("Input shapes (as passed in by torchdiffeq)")
        print("   t.shape = {}".format(t.shape))  # t.shape = torch.Size([]) --> as if dim is zero ???
        print("   t = {}".format(t))
        print("   type(t) = {}".format(type(t)))
        print("   t.shape = {}".format(t.shape))
        print("   x.shape = {}".format(x.shape))  # x.shape = torch.Size([856, 200])
        print("   ten_BatchEmb.shape = {}".format(ten_BatchEmb.shape))  # ten_BatchEmb.shape = torch.Size([856, 4])


        Input shapes (as passed in by torchdiffeq)
           t.shape = torch.Size([])
           t = 0.0
           type(t) = <class 'torch.Tensor'>
           t.shape = torch.Size([])
           x.shape = torch.Size([13, 200])
           ten_BatchEmb.shape = torch.Size([13, 4])
        '''

        assert t.dim() == 0
        batc_size = x.size()[0]


        return self.forward(
                t=t.repeat(batc_size),
                x=x,
                ten_BatchEmb=ten_BatchEmb
            )


    def forward(self, t, x, ten_BatchEmb):
        """
        :param t: a 1-D tensor of time-steps, of shape [b].
        :param x: of shape [b x dim_z+dim_s]
        :param ten_BatchEmb: of shape [b x dim_b]
        :return: of shape [b x dim_z+dim_s]
        """
        # torchcfm's sample label-conditioned forward was used in
        # https://github.com/atong01/conditional-flow-matching/blob/62c44affd877a01b7838d408b5dc4cbcbf83e3ad/torchcfm/models/unet/unet.py#L599
        # in that function `t` is a tensor of dim=0, so here another forward `forward_4torchdiffeq` is created.

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
