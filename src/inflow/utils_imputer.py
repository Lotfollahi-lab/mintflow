'''
Utilities for the imputer, e.g.
- affine transformation on xy coordinates.
- different masking schemes.
'''

from typing import List
import numpy as np
import torch

class RandomGeometricTfm:
    def __init__(self, prob_applytfm:float, rng_00:List[float], rng_01:List[float], rng_10:List[float], rng_11:List[float]):
        '''
        :param prob_applytfm: the probability that random transformation is applied.
        :param rng_00: a list of length 2 containing the min/max values of H[0,0] where H is the affine matrix.
        :param rng_01: sim.
        :param rng_10: sim.
        :param rng_11: sim.
        '''
        self.prob_applytfm = prob_applytfm
        self.rng_00 = rng_00
        self.rng_01 = rng_01
        self.rng_10 = rng_10
        self.rng_11 = rng_11


        # check args
        assert (isinstance(self.rng_00, list))
        assert (isinstance(self.rng_01, list))
        assert (isinstance(self.rng_10, list))
        assert (isinstance(self.rng_11, list))
        assert (len(self.rng_00) == 2)
        assert (len(self.rng_01) == 2)
        assert (len(self.rng_10) == 2)
        assert (len(self.rng_11) == 2)
        assert (isinstance(self.rng_00[0], float) and isinstance(self.rng_00[1], float))
        assert (isinstance(self.rng_01[0], float) and isinstance(self.rng_01[1], float))
        assert (isinstance(self.rng_10[0], float) and isinstance(self.rng_10[1], float))
        assert (isinstance(self.rng_11[0], float) and isinstance(self.rng_11[1], float))
        assert (self.rng_00[1] >= self.rng_00[0])
        assert (self.rng_01[1] >= self.rng_01[0])
        assert (self.rng_10[1] >= self.rng_10[0])
        assert (self.rng_11[1] >= self.rng_11[0])


    @torch.no_grad()
    def forward(self, ten_xy:torch.Tensor):
        '''
        :param ten_xy: a tensor of shape [N, 2] containing the xy positions.
        :return: the modified ten_xy
        '''

        if np.random.rand() > self.prob_applytfm:
            return ten_xy + 0.0  # no transformation

        H_00 = self.rng_00[0] + (self.rng_00[1] - self.rng_00[0]) * np.random.rand()
        H_01 = self.rng_01[0] + (self.rng_01[1] - self.rng_01[0]) * np.random.rand()
        H_10 = self.rng_10[0] + (self.rng_10[1] - self.rng_10[0]) * np.random.rand()
        H_11 = self.rng_11[0] + (self.rng_11[1] - self.rng_11[0]) * np.random.rand()
        H = torch.tensor(
            [[H_00, H_01, 0.0], [H_10, H_11, 0.0], [0.0,  0.0, 1.0]]
        ).to(ten_xy.device)  # [3,3]

        ten_xy_extended = torch.cat(
            [ten_xy, torch.ones(size=[ten_xy.size()[0], 1], dtype=ten_xy.dtype, device=ten_xy.device)],
            1
        ).T  # [3, N]

        output = torch.matmul(H, ten_xy_extended)  # [3, N]
        output = output[0:2, :] / output[2,:].unsqueeze(0)  # [2, N]
        output = output.T + 0.0  # [N, 2]
        print("Random affine tfm was applied.")
        return output




